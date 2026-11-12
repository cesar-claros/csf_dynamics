"""Stage E analysis: gates, roster, and the descriptive analyses E1 to E7.

Run from code/ after copying the run folders from the HPC:

    python -m dynamics.analyze_e --results-root dyn_results

Stage E makes no claim. One training run is one sample, so every summary groups by
run first. Protocol: ../documentation/imagenette_dynamics_protocol_2026-11-10.md
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from dynamics.config import (
    DEFAULT_PRECISION,
    PENULTIMATE,
    PRECISIONS,
    STAGES,
    parse_run,
    results_folder,
)
from dynamics.scores import ALL_CSFS, CSF_GROUP, NO_FIT_CSFS

HERE = Path(__file__).resolve().parent
KEYS = ["run", "recipe", "seed", "held_out", "epoch"]
EXTERNAL = ("imagewoof", "textures")
#: Fixed before any run: which external set each stand-in is compared with.
STAND_IN_FOR = {"blend": "imagewoof", "id_errors": "imagewoof", "heldout": "imagewoof",
                "shuffle": "textures"}
#: Runs of other stages can share the results folder. Stage E reads only its own.
STAGE_E = frozenset(spec.name for stage in ("e_main", "e_heldout")
                    for spec in STAGES[stage])
TOLERANCE = 1e-6
PENALTY = 10.0
MATERIALITY = 1.0
#: A checkpoint counts as converged if its validation accuracy is at most this far
#: below the final validation accuracy of its run (protocol amendments A1 and A2).
CONVERGENCE_MARGIN = 0.01
SUBSET_SIZES = (1, 2, 3, 5, 10)
CLOCK = ["log_epoch", "train_accuracy", "train_loss", "val_accuracy"]
HIGHER_IS_BETTER = {"auroc": 1.0, "fpr95": -1.0, "augrc": -1.0, "aurc": -1.0}


@dataclass
class Tables:
    """Long-format results of every evaluated checkpoint under one tag."""

    metrics: pd.DataFrame
    descriptors: pd.DataFrame
    placement: pd.DataFrame
    checks: pd.DataFrame


def read_runs(root: Path, tag: str) -> Tables:
    """Read <root>/runs/<run>/<tag>/epoch_*.json of every run folder."""
    parts: dict[str, list[dict]] = {"metrics": [], "descriptors": [], "placement": [],
                                    "checks": []}
    for folder in sorted((Path(root) / "runs").glob("*")):
        try:
            spec = parse_run(folder.name)
        except ValueError:
            continue
        for path in sorted((folder / tag).glob("epoch_*.json")):
            out = json.loads(path.read_text())
            key = {"run": spec.name, "recipe": spec.recipe, "seed": spec.seed,
                   "held_out": -1 if spec.held_out is None else spec.held_out,
                   "epoch": out["epoch"]}
            for name in ("metrics", "descriptors", "placement"):
                parts[name] += [{**key, **row} for row in out[name]]
            parts["checks"].append({
                **key, **out["checks"],
                "nonfinite": ",".join(c for c, n in out["nonfinite"].items() if n),
                "failed": ",".join(c for c in out["errors"] if c != "paper_nc")})
    return Tables(**{name: pd.DataFrame(rows) for name, rows in parts.items()})


def stage_e_only(tables: Tables) -> Tables:
    """The same tables without the runs of other stages."""
    return Tables(**{name: frame if frame.empty
                     else frame[frame["run"].isin(STAGE_E)].reset_index(drop=True)
                     for name, frame in vars(tables).items()})


def clock_table(descriptors: pd.DataFrame) -> pd.DataFrame:
    """One row per checkpoint with the clock group as columns."""
    clock = descriptors[descriptors["group"] == "clock"]
    table = clock.set_index(KEYS + ["name"])["value"].unstack("name").reset_index()
    table["log_epoch"] = np.log1p(table["epoch"])
    return table


def converged_checkpoints(clock: pd.DataFrame, margin: float = CONVERGENCE_MARGIN
                          ) -> pd.Series:
    """Per (run, epoch), whether the checkpoint counts as converged.

    A checkpoint is converged if its own validation accuracy is at most `margin`
    below the final validation accuracy of its run (amendment A2). So a checkpoint
    inside a training instability is not converged, at any epoch.
    """
    ordered = clock.sort_values("epoch")
    final = ordered.groupby("run")["val_accuracy"].last()
    flags = ordered["val_accuracy"] >= ordered["run"].map(final) - margin
    index = pd.MultiIndex.from_arrays([ordered["run"], ordered["epoch"]],
                                      names=["run", "epoch"])
    return pd.Series(flags.to_numpy(), index=index, name="converged")


def convergence_epochs(converged: pd.Series) -> pd.Series:
    """Per run, the first converged checkpoint."""
    epochs = converged[converged].reset_index()
    return epochs.groupby("run")["epoch"].min()


def disturbed_checkpoints(converged: pd.Series) -> pd.DataFrame:
    """Checkpoints after the first converged one of their run that are not converged."""
    first = convergence_epochs(converged)
    rows = converged[~converged].reset_index()[["run", "epoch"]]
    return rows[rows["epoch"] > rows["run"].map(first)].reset_index(drop=True)


def is_converged(runs, epochs, converged: pd.Series) -> np.ndarray:
    """Whether each (run, epoch) pair is a converged checkpoint."""
    index = pd.MultiIndex.from_arrays([np.asarray(runs), np.asarray(epochs)])
    return converged.reindex(index, fill_value=False).to_numpy(dtype=bool)


def gates(tables: Tables, recheck: Tables | None, manifest: dict | None
          ) -> tuple[pd.DataFrame, set[str]]:
    """Gates G1 to G5, and the CSFs that fail the determinism gate G3."""
    rows = []
    clock = clock_table(tables.descriptors)
    main = clock[clock["held_out"] < 0]
    final = int(clock["epoch"].max())

    regime_a = main[main["recipe"] == "A"]
    reached = (regime_a[regime_a["train_accuracy"] >= 1.0].groupby("run")["epoch"]
               .min().reindex(sorted(regime_a["run"].unique())))
    detail = "; ".join(f"{run}: {'never' if pd.isna(e) else int(e)}"
                       for run, e in reached.items())
    rows.append(("G1, regime A: 100% training accuracy by epoch 100",
                 "not run" if regime_a.empty else
                 "pass" if (reached <= 100).all() else "FAIL", detail))
    regime_b = main[(main["recipe"] == "B") & (main["epoch"] == final)]
    rows.append(("G1, regime B: validation accuracy of at least 85% at the end",
                 "not run" if regime_b.empty else
                 "pass" if (regime_b["val_accuracy"] >= 0.85).all() else "FAIL",
                 "; ".join(f"{r.run}: {r.val_accuracy:.3f}"
                           for r in regime_b.itertuples())))

    collapse = tables.descriptors[(tables.descriptors["group"] == "nc_paper")
                                  & (tables.descriptors["name"] == "var_collapse")]
    collapse = collapse.set_index(["run", "epoch"])["value"]
    results = []
    for run, first in reached.items():
        if pd.isna(first):
            results.append((run, False, "no 100% training accuracy"))
            continue
        start, end = collapse[(run, int(first))], collapse[(run, final)]
        results.append((run, end < start, f"{start:.4g} to {end:.4g}"))
    rows.append(("G2, regime A: var_collapse falls after 100% training accuracy",
                 "not run" if not results else
                 "pass" if all(ok for _, ok, _ in results) else "FAIL",
                 "; ".join(f"{run}: {text}" for run, _, text in results)))

    unstable: set[str] = set()
    if recheck is None or recheck.metrics.empty:
        rows.append(("G3, determinism of a second extraction", "not run", ""))
    else:
        index = ["run", "epoch", "set", "csf"]
        both = tables.metrics.merge(recheck.metrics, on=index, suffixes=("", "_2"))
        same_nan = both["auroc"].isna() & both["auroc_2"].isna()
        gap = (both["auroc"] - both["auroc_2"]).abs().fillna(np.inf).mask(same_nan, 0)
        worst = gap.groupby(both["csf"]).max()
        unstable = set(worst[worst > TOLERANCE].index)
        rows.append(("G3, determinism of a second extraction",
                     "pass" if not unstable else "FAIL",
                     f"{both[['run', 'epoch']].drop_duplicates().shape[0]} "
                     f"checkpoints, largest AUROC gap {worst.max():.2e}"
                     + (f", unstable: {sorted(unstable)}" if unstable else "")))

    gap = float(np.nanmax(tables.checks["g4_max_gap"]))
    rows.append(("G4, MSR, MLS, and Energy equal an independent computation",
                 "pass" if gap <= TOLERANCE else "FAIL", f"largest AUROC gap {gap:.2e}"))

    if manifest is None or "imagewoof" not in manifest:
        rows.append(("G5, Imagenette and Imagewoof share no class", "not run", ""))
    else:
        nette = set(manifest["imagenette"]["train"])
        woof = set(manifest["imagewoof"]["val"])
        counts = ", ".join(
            f"{name} {split} {sum(manifest[name][split].values())}"
            for name in ("imagenette", "imagewoof") for split in manifest[name])
        if "textures" in manifest:
            counts += f", textures {manifest['textures']['images']}"
        rows.append(("G5, Imagenette and Imagewoof share no class",
                     "pass" if not nette & woof else "FAIL", counts))
    return pd.DataFrame(rows, columns=["gate", "status", "detail"]), unstable


def nc_cross_check(descriptors: pd.DataFrame) -> float:
    """Largest relative gap between our class-mean metrics and the paper's."""
    at_last = descriptors[descriptors["layer"] == PENULTIMATE]
    ours = at_last[at_last["group"] == "nc"].set_index(KEYS + ["name"])["value"]
    paper = at_last[at_last["group"] == "nc_paper"].set_index(KEYS + ["name"])["value"]
    both = pd.concat([ours, paper], axis=1, keys=["ours", "paper"]).dropna()
    return float(((both["ours"] - both["paper"]).abs()
                  / both["paper"].abs().clip(lower=1e-12)).max())


def roster(tables: Tables, unstable: set[str]) -> tuple[list[str], pd.DataFrame]:
    """CSFs that stay finite in every cell from epoch 1 and pass gate G3."""
    reasons: dict[str, list[str]] = {csf: ["fails gate G3"] for csf in unstable}
    checks = tables.checks[tables.checks["epoch"] >= 1]
    for row in checks.itertuples():
        for csf in filter(None, set(row.nonfinite.split(",") + row.failed.split(","))):
            reasons.setdefault(csf, []).append(f"{row.run} epoch {row.epoch}")
    present = set(tables.metrics["csf"])
    kept = [csf for csf in ALL_CSFS if csf in present and csf not in reasons]
    dropped = pd.DataFrame([{"csf": csf, "cells": len(cells), "first": cells[0]}
                            for csf, cells in sorted(reasons.items())],
                           columns=["csf", "cells", "first"])
    return kept, dropped


def wide(metrics: pd.DataFrame, csfs: list[str], value: str = "auroc"
         ) -> pd.DataFrame:
    """One row per (checkpoint, set) from epoch 1, one column per CSF, in points."""
    kept = metrics[metrics["csf"].isin(csfs) & (metrics["epoch"] >= 1)]
    return kept.set_index(KEYS + ["set", "csf"])[value].unstack("csf")[csfs] * 100


def main_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.index.get_level_values("held_out") < 0]


def fixed_best(points: pd.DataFrame, reference: pd.DataFrame | None = None,
               converged: pd.Series | None = None) -> pd.DataFrame:
    """E1: per external set and epoch, the CSF that is best on average (B1).

    Args:
        points: AUROC points of the roster to choose from.
        reference: AUROC points of the full roster. With it, `cost_vs_full` is the
            mean gap between the best CSF of the full roster and of this roster.
        converged: Flags of `converged_checkpoints`. With them, `converged_runs`
            is the share of the runs whose checkpoint at that epoch is converged.
    """
    rows = []
    blocks = main_rows(points).groupby(level=["set", "epoch"])
    for (name, epoch), block in blocks:
        if name not in EXTERNAL:
            continue
        best = block.max(axis=1)
        choice = block.mean(axis=0).idxmax()
        winners = block.idxmax(axis=1)
        row = {
            "set": name, "epoch": epoch, "runs": len(block), "b1": choice,
            "b1_regret": float((best - block[choice]).mean()),
            "b1_regret_max": float((best - block[choice]).max()),
            "random_regret": float((best.to_numpy()[:, None] - block.to_numpy())
                                   .mean()),
            "distinct_winners": int(winners.nunique()),
            "b1_wins": float((winners == choice).mean())}
        if reference is not None:
            row["cost_vs_full"] = float(
                (reference.loc[block.index].max(axis=1) - best).mean())
        if converged is not None:
            level = block.index.get_level_values
            row["converged_runs"] = float(is_converged(level("run"), level("epoch"),
                                                       converged).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def trajectories(points: pd.DataFrame, converged: pd.Series | None = None
                 ) -> pd.DataFrame:
    """E2: mean and spread over seeds of each AUROC, per regime, set, and epoch."""
    long = main_rows(points).stack().rename("auroc").reset_index()
    long["group"] = long["csf"].map(CSF_GROUP)
    long["converged"] = (True if converged is None else
                         is_converged(long["run"], long["epoch"], converged))
    return (long.groupby(["recipe", "set", "epoch", "group", "csf"])
            .agg(mean=("auroc", "mean"), std=("auroc", "std"),
                 count=("auroc", "count"), converged_runs=("converged", "mean"))
            .reset_index())


def design_regime(points: pd.DataFrame, converged: pd.Series) -> pd.DataFrame:
    """Mean regret of each CSF on unconverged and on converged checkpoints.

    The regret is against the best CSF of the roster in the same cell, on the
    external sets. `excess` is the regret on the unconverged checkpoints minus the
    regret on the converged ones, averaged over runs and sets.
    """
    cells = main_rows(points)
    cells = cells[cells.index.get_level_values("set").isin(EXTERNAL)]
    if cells.empty:
        return pd.DataFrame()
    regret = cells.rsub(cells.max(axis=1), axis=0).reset_index()
    long = regret.melt(id_vars=KEYS + ["set"], var_name="csf", value_name="regret")
    long["phase"] = phase_of(long, converged)
    per_run = (long.groupby(["csf", "run", "set", "phase"])["regret"].mean()
               .unstack("phase").reindex(columns=["unconverged", "converged"]))
    per_run["excess"] = per_run["unconverged"] - per_run["converged"]
    table = per_run.groupby("csf").mean().reset_index()
    table.insert(0, "group", table["csf"].map(CSF_GROUP))
    return table.sort_values(["group", "csf"]).reset_index(drop=True)


def phase_of(frame: pd.DataFrame, converged: pd.Series) -> np.ndarray:
    """`converged` or `unconverged` for each row, from its run and epoch."""
    return np.where(is_converged(frame["run"], frame["epoch"], converged),
                    "converged", "unconverged")


def pick_quality(ranking: pd.Series, target: pd.Series) -> tuple[float, float]:
    """Spearman correlation with the target scores, and the regret of the top pick."""
    rho = float(spearmanr(ranking.to_numpy(), target.to_numpy()).statistic)
    return rho, float(target.max() - target[ranking.idxmax()])


def within_model_stand_ins(points: pd.DataFrame) -> pd.DataFrame:
    """E5: each stand-in of a main run against the external set of its type."""
    rows = []
    cells = main_rows(points)
    names = set(cells.index.get_level_values("set"))
    for stand_in, target in STAND_IN_FOR.items():
        if stand_in == "heldout" or not {stand_in, target} <= names:
            continue
        source = cells.xs(stand_in, level="set")
        goal = cells.xs(target, level="set")
        for key in source.index.intersection(goal.index):
            rho, regret = pick_quality(source.loc[key], goal.loc[key])
            rows.append({"stand_in": stand_in, "target": target, "run": key[0],
                         "recipe": key[1], "epoch": key[4], "spearman": rho,
                         "regret": regret})
    return pd.DataFrame(rows)


def held_out_stand_in(points: pd.DataFrame, seed: int = 0
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E5: the held-out-class ranking of a regime against Imagewoof on its main runs.

    Returns:
        Rows per (regime, epoch, main run) for the ranking that averages all
        held-out classes, and rows per (regime, epoch, subset size) with the mean
        regret when only some classes are held out.
    """
    names = set(points.index.get_level_values("set"))
    target = STAND_IN_FOR["heldout"]
    if not {"heldout", target} <= names:
        return pd.DataFrame(), pd.DataFrame()
    held = points.xs("heldout", level="set")
    goal = main_rows(points).xs(target, level="set")
    rng = np.random.default_rng(seed)
    rows, subsets = [], []
    for (recipe, epoch), block in held.groupby(level=["recipe", "epoch"]):
        level = goal.index.get_level_values
        targets = goal[(level("recipe") == recipe) & (level("epoch") == epoch)]
        if targets.empty:
            continue
        ranking = block.mean(axis=0)
        agreement = [float(spearmanr(row.to_numpy(), ranking.to_numpy()).statistic)
                     for _, row in block.iterrows()]
        for key, scores in targets.iterrows():
            rho, regret = pick_quality(ranking, scores)
            rows.append({"recipe": recipe, "epoch": epoch, "run": key[0],
                         "classes": len(block), "spearman": rho, "regret": regret,
                         "class_agreement_min": min(agreement),
                         "class_agreement_median": float(np.median(agreement))})
        for size in (s for s in SUBSET_SIZES if s <= len(block)):
            combos = list(itertools.combinations(range(len(block)), size))
            if len(combos) > 200:
                combos = [combos[i] for i in rng.choice(len(combos), 200,
                                                        replace=False)]
            regrets = [pick_quality(block.iloc[list(combo)].mean(axis=0), scores)[1]
                       for combo in combos for _, scores in targets.iterrows()]
            subsets.append({"recipe": recipe, "epoch": epoch, "classes": size,
                            "regret": float(np.mean(regrets))})
    return pd.DataFrame(rows), pd.DataFrame(subsets)


def metric_agreement(metrics: pd.DataFrame, csfs: list[str],
                     converged: pd.Series | None = None) -> pd.DataFrame:
    """E6: Kendall correlation between the CSF rankings under each pair of metrics.

    With `converged`, only the converged cells are used.
    """
    kept = metrics[metrics["csf"].isin(csfs) & (metrics["epoch"] >= 1)
                   & (metrics["held_out"] < 0) & metrics["set"].isin(EXTERNAL)]
    if converged is not None:
        kept = kept[is_converged(kept["run"], kept["epoch"], converged)]
    rows = []
    for (run, epoch, name), cell in kept.groupby(["run", "epoch", "set"]):
        scores = {metric: sign * cell.set_index("csf")[metric].reindex(csfs)
                  for metric, sign in HIGHER_IS_BETTER.items()}
        for first, second in itertools.combinations(scores, 2):
            rows.append({
                "set": name, "run": run, "epoch": epoch, "pair": f"{first}, {second}",
                "kendall": float(kendalltau(scores[first], scores[second]).statistic),
                "same_best": scores[first].idxmax() == scores[second].idxmax()})
    return pd.DataFrame(rows)


def seed_noise(points: pd.DataFrame) -> pd.DataFrame:
    """E7: spread over seeds of each AUROC and of the best-to-second gap."""
    rows = []
    cells = main_rows(points)
    for (recipe, name, epoch), block in cells.groupby(level=["recipe", "set",
                                                             "epoch"]):
        if name not in EXTERNAL or len(block) < 2:
            continue
        ordered = np.sort(block.to_numpy(), axis=1)
        gap = ordered[:, -1] - ordered[:, -2]
        rows.append({"recipe": recipe, "set": name, "epoch": epoch,
                     "seeds": len(block),
                     "auroc_sd": float(block.std(axis=0, ddof=1).mean()),
                     "gap_mean": float(gap.mean()),
                     "gap_sd": float(gap.std(ddof=1))})
    return pd.DataFrame(rows)


def out_of_run_r2(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  penalty: float = PENALTY) -> float:
    """R2 of a ridge regression where each run is predicted from the other runs.

    Inputs are standardized on the training runs. NaN if there are fewer than two
    runs or the target is constant.
    """
    if len(np.unique(groups)) < 2 or np.ptp(y) == 0:
        return float("nan")
    predicted = np.empty_like(y, dtype=np.float64)
    for group in np.unique(groups):
        test = groups == group
        mean, std = x[~test].mean(axis=0), x[~test].std(axis=0)
        std[std == 0] = 1.0
        fit, offset = (x[~test] - mean) / std, y[~test].mean()
        beta = np.linalg.solve(fit.T @ fit + penalty * np.eye(fit.shape[1]),
                               fit.T @ (y[~test] - offset))
        predicted[test] = (x[test] - mean) / std @ beta + offset
    return float(1 - ((y - predicted) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def dummies(values: pd.Series) -> np.ndarray:
    return pd.get_dummies(values, drop_first=True).to_numpy(dtype=np.float64)


def placement_wide(placement: pd.DataFrame) -> pd.DataFrame:
    """Penultimate-layer placement descriptors and the margin ratio, per cell."""
    kept = placement[placement["layer"].isin([PENULTIMATE, "logits"])
                     & (placement["epoch"] >= 1) & (placement["held_out"] < 0)]
    return kept.set_index(KEYS + ["set", "name"])["value"].unstack("name")


def link_two(points: pd.DataFrame, placement: pd.DataFrame, clock: pd.DataFrame,
             converged: pd.Series | None = None) -> pd.DataFrame:
    """E3: does placement predict the AUROC of each CSF beyond the clock?

    With `converged`, a fitted CSF uses only the converged cells, where it is
    inside its design regime. A no-fit CSF uses every cell.
    """
    features = placement_wide(placement)
    cells = features.join(main_rows(points), how="inner").reset_index()
    cells = cells.merge(clock[KEYS + CLOCK], on=KEYS)
    names = list(features.columns)
    cells = cells[np.isfinite(cells[names + CLOCK]).all(axis=1)]
    if cells.empty:
        return pd.DataFrame()
    base = np.hstack([dummies(cells["set"]), dummies(cells["recipe"])])
    designs = {"base": base,
               "clock": np.hstack([base, cells[CLOCK].to_numpy()]),
               "placement": np.hstack([base, cells[names].to_numpy()]),
               "both": np.hstack([base, cells[CLOCK + names].to_numpy()])}
    groups = cells["run"].to_numpy()
    everywhere = np.ones(len(cells), dtype=bool)
    late = (everywhere if converged is None else
            is_converged(cells["run"], cells["epoch"], converged))
    rows = []
    for csf in points.columns:
        keep = everywhere if csf in NO_FIT_CSFS else late
        share = (cells[csf].to_numpy()[keep] / 100).clip(1e-3, 1 - 1e-3)
        target = np.log(share / (1 - share))
        rows.append({"csf": csf, "group": CSF_GROUP.get(csf), "cells": int(keep.sum()),
                     **{name: out_of_run_r2(design[keep], target, groups[keep])
                        for name, design in designs.items()}})
    return pd.DataFrame(rows)


def geometry_wide(descriptors: pd.DataFrame) -> pd.DataFrame:
    """ID-only descriptors without the clock group, one column per descriptor."""
    kept = descriptors[(descriptors["group"] != "clock") & (descriptors["epoch"] >= 1)
                       & (descriptors["held_out"] < 0)].copy()
    kept["column"] = kept["group"] + ":" + kept["name"] + "@" + kept["layer"]
    table = kept.set_index(KEYS + ["column"])["value"].unstack("column")
    finite = np.isfinite(table).all(axis=0) & (table.std(axis=0) > 0)
    return table.loc[:, finite]


def link_one(descriptors: pd.DataFrame, placement: pd.DataFrame,
             clock: pd.DataFrame) -> pd.DataFrame:
    """E4: do the ID-only descriptors predict placement beyond the clock?"""
    geometry = geometry_wide(descriptors)
    targets = placement_wide(placement)
    rows = []
    for name in EXTERNAL:
        if name not in targets.index.get_level_values("set"):
            continue
        cells = geometry.join(targets.xs(name, level="set"),
                              how="inner").reset_index()
        cells = cells.merge(clock[KEYS + CLOCK], on=KEYS)
        base = dummies(cells["recipe"])
        designs = {
            "clock": np.hstack([base, cells[CLOCK].to_numpy()]),
            "geometry": np.hstack([base, cells[list(geometry.columns)].to_numpy()]),
            "both": np.hstack([base, cells[CLOCK + list(geometry.columns)]
                               .to_numpy()])}
        groups = cells["run"].to_numpy()
        for target in targets.columns:
            values = cells[target].to_numpy(dtype=np.float64)
            keep = np.isfinite(values)
            rows.append({"set": name, "placement": target, "cells": int(keep.sum()),
                         **{key: out_of_run_r2(design[keep], values[keep],
                                               groups[keep])
                            for key, design in designs.items()}})
    return pd.DataFrame(rows)


def block(frame: pd.DataFrame, digits: int = 3) -> str:
    if frame.empty:
        return "No data."
    return f"```\n{frame.round(digits).to_string(index=False)}\n```"


def expected_runs(found: set[str]) -> str:
    lines = []
    for stage in ("e_main", "e_heldout"):
        names = [spec.name for spec in STAGES[stage]]
        missing = [name for name in names if name not in found]
        lines.append(f"- `{stage}`: {len(names) - len(missing)} of {len(names)} runs"
                     + (f". Missing: {', '.join(missing)}" if missing else ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--precision", choices=sorted(PRECISIONS),
                        default=DEFAULT_PRECISION,
                        help="which results folder to read")
    parser.add_argument("--out", default=str(HERE / "outputs" / "stage_e_report.md"))
    args = parser.parse_args()
    root = Path(args.results_root)
    everything = read_runs(root, results_folder(args.precision))
    tables = stage_e_only(everything)
    if tables.metrics.empty:
        raise SystemExit(f"no Stage E results under {root}/runs")
    ignored = sorted(set(everything.metrics["run"]) - STAGE_E)
    recheck = stage_e_only(read_runs(root, results_folder(args.precision, True)))
    manifests = sorted((root / "runs").glob("*/manifest.json"))
    manifest = json.loads(manifests[0].read_text()) if manifests else None
    out_dir = Path(args.out).parent / "stage_e"
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_table, unstable = gates(tables, recheck, manifest)
    csfs, dropped = roster(tables, unstable)
    rosters = {"full": csfs, "no_fit": [csf for csf in csfs if csf in NO_FIT_CSFS]}
    points = wide(tables.metrics, csfs)
    clock = clock_table(tables.descriptors)
    converged = converged_checkpoints(clock)
    final = int(tables.metrics["epoch"].max())

    best = pd.concat([fixed_best(points[names], points, converged)
                      .assign(roster=name) for name, names in rosters.items()],
                     ignore_index=True)
    paths = trajectories(points, converged)
    regime = design_regime(points, converged)
    stand_ins = pd.concat([within_model_stand_ins(points[names]).assign(roster=name)
                           for name, names in rosters.items()], ignore_index=True)
    held_parts, subset_parts = [], []
    for name, names in rosters.items():
        held_part, subset_part = held_out_stand_in(points[names])
        held_parts.append(held_part.assign(roster=name))
        subset_parts.append(subset_part.assign(roster=name))
    held = pd.concat(held_parts, ignore_index=True)
    subsets = pd.concat(subset_parts, ignore_index=True)
    for frame in (stand_ins, held):
        if not frame.empty:
            frame["phase"] = phase_of(frame, converged)
    agreement = metric_agreement(tables.metrics, csfs, converged)
    noise = seed_noise(points)
    second = link_two(points, tables.placement, clock, converged)
    first = link_one(tables.descriptors, tables.placement, clock)

    recipes = clock[clock["held_out"] < 0][["run", "recipe"]].drop_duplicates()
    flags = converged.reset_index().merge(recipes, on="run")
    flags = flags[flags["epoch"] >= 1]
    converged_at = (convergence_epochs(converged).rename("first_converged_epoch")
                    .reset_index().merge(recipes, on="run"))
    converged_at["converged_checkpoints"] = converged_at["run"].map(
        flags.groupby("run")["converged"].sum())
    disturbed = disturbed_checkpoints(converged).merge(recipes, on="run")
    converged_at["disturbed_checkpoints"] = converged_at["run"].map(
        disturbed.groupby("run")["epoch"].apply(
            lambda epochs: " ".join(str(e) for e in epochs))).fillna("")
    descriptor_paths = (tables.descriptors[tables.descriptors["held_out"] < 0]
                        .groupby(["recipe", "group", "name", "layer", "epoch"])
                        ["value"].agg(["mean", "std", "count"]).reset_index())
    placement_paths = (tables.placement[tables.placement["held_out"] < 0]
                       .groupby(["recipe", "set", "layer", "name", "epoch"])["value"]
                       .agg(["mean", "std", "count"]).reset_index())
    for name, frame in (("auroc_trajectories", paths),
                        ("descriptor_trajectories", descriptor_paths),
                        ("placement_trajectories", placement_paths),
                        ("convergence", converged_at), ("design_regime", regime),
                        ("e1_fixed_best", best), ("e3_link_two", second),
                        ("e4_link_one", first), ("e5_stand_ins", stand_ins),
                        ("e5_held_out", held), ("e5_held_out_subsets", subsets),
                        ("e6_metric_agreement", agreement), ("e7_seed_noise", noise)):
        frame.to_csv(out_dir / f"{name}.csv", index=False)

    def summary(frame: pd.DataFrame, by: list[str], columns: list[str]) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.groupby(by)[columns].mean()
        out["cells"] = frame.groupby(by).size()
        return out.reset_index()

    shown = best[best["epoch"].isin({1, 10, 50, 100, final})] if not best.empty else best
    if not shown.empty:
        shown = shown.sort_values(["set", "epoch", "roster"])[
            ["set", "epoch", "roster", "runs", "converged_runs", "b1", "b1_regret",
             "random_regret", "distinct_winners", "cost_vs_full"]]
    top = (paths[paths["epoch"] == final].sort_values("mean", ascending=False)
           .groupby(["recipe", "set"]).head(5)
           .sort_values(["set", "recipe", "mean"], ascending=[True, True, False])
           [["set", "recipe", "csf", "group", "mean", "std"]])
    last_held = held[held["epoch"] == final] if not held.empty else held
    last_subsets = subsets[subsets["epoch"] == final] if not subsets.empty else subsets
    noise_last = noise[noise["epoch"] == final] if not noise.empty else noise
    threshold = MATERIALITY
    if not noise_last.empty and np.isfinite(noise_last["gap_sd"]).any():
        threshold = max(MATERIALITY, 2 * float(noise_last["gap_sd"].max()))
    models = ["base", "clock", "placement", "both"]
    link_two_summary = (second.groupby("group")[models].median().reset_index()
                        if not second.empty else second)
    regime_summary = summary(regime, ["group"],
                             ["unconverged", "converged", "excess"])
    if not regime_summary.empty:
        regime_summary = regime_summary.rename(columns={"cells": "csfs"})

    found = set(tables.metrics["run"])
    sections = [
        ("Runs", (expected_runs(found)
                  + (f"\n- Ignored runs of other stages: {', '.join(ignored)}"
                     if ignored else "")
                  + f"\n- Precision of the fitted CSFs: {args.precision}."
                  + f"\n- Last epoch in the results: {final}. "
                  "Epoch 0 is the random start. It is outside the roster rule and "
                  "the tables below.")),
        ("Gates", block(gate_table) + "\n\nLargest relative gap between our "
         "class-mean NC metrics and the paper's, at the penultimate layer: "
         f"{nc_cross_check(tables.descriptors):.2e}."),
        ("Rosters and convergence",
         f"- Full roster, {len(csfs)} CSFs: {', '.join(csfs)}.\n"
         f"- No-fit roster, {len(rosters['no_fit'])} CSFs: "
         f"{', '.join(rosters['no_fit'])}.\n"
         + ("- No CSF is removed by the roster rule.\n" if dropped.empty else
            "- Removed by the roster rule (number of failing cells, and the first "
            "one):\n\n" + block(dropped) + "\n")
         + "\nA checkpoint is converged if its validation accuracy is at most "
         f"{100 * CONVERGENCE_MARGIN:.0f} point below the final value of its run. On "
         "the other checkpoints, the fitted CSFs are outside their design regime, "
         "and the no-fit roster is the reference. A disturbed checkpoint is an "
         "unconverged one after the first converged one. Main runs:\n\n"
         + block(converged_at[["run", "first_converged_epoch",
                               "converged_checkpoints", "disturbed_checkpoints"]]
                 if not converged_at.empty else converged_at)
         + f"\n\nAlso in `{out_dir.name}/convergence.csv`."),
        ("E1: is there something to predict?",
         "For each external set and roster: the CSF that is best on average over the "
         "main runs (B1), its regret in AUROC points, the regret of a random CSF, how "
         "many different CSFs win, and the mean gap to the best CSF of the full "
         "roster (`cost_vs_full`). `converged_runs` is the share of the runs that "
         "have converged at that epoch.\n\n" + block(shown)),
        ("E2: trajectories",
         f"Full tables: `{out_dir.name}/auroc_trajectories.csv`, "
         "`descriptor_trajectories.csv`, and `placement_trajectories.csv`. "
         "The five best CSFs at the last epoch, mean and spread over seeds:\n\n"
         + block(top)),
        ("Design regime of the fitted CSFs",
         "Mean regret of each CSF group on the external sets, on the unconverged "
         "and on the converged checkpoints, and the difference (`excess`). A larger "
         "excess for the fitted groups than for `no_fit` means that the fitted CSFs "
         "lose more on unconverged checkpoints.\n\n" + block(regime_summary)
         + f"\n\nPer CSF: `{out_dir.name}/design_regime.csv`."),
        ("E3: placement to ranking",
         "Out-of-run R2 of the logit of each CSF's AUROC. All models read the set "
         "and the regime. A no-fit CSF uses every cell, and a fitted CSF uses only "
         "the converged cells. Median over the CSFs of each group:\n\n"
         + block(link_two_summary) + f"\n\nPer CSF: `{out_dir.name}/e3_link_two.csv`."),
        ("E4: ID-only descriptors to placement",
         "Out-of-run R2 of each placement descriptor of an external set, from the "
         "clock group, from the geometric ID-only descriptors, and from both:\n\n"
         + block(first)),
        ("E5: stand-ins",
         "Spearman correlation between the CSF ranking on the stand-in and on the "
         "external set of its type, and the regret of the CSF that the stand-in "
         "ranks first, per roster and phase:\n\n"
         + block(summary(stand_ins, ["stand_in", "roster", "phase"],
                         ["spearman", "regret"]))
         + "\n\nHeld-out classes at the last epoch (ranking averaged over the "
         "held-out models of the regime, tested on the main runs):\n\n"
         + block(summary(last_held, ["roster", "recipe"],
                         ["classes", "spearman", "regret", "class_agreement_min",
                          "class_agreement_median"]))
         + "\n\nRegret against the number of held-out classes at the last epoch:"
         "\n\n" + block(last_subsets)),
        ("E6: agreement between metrics",
         "Converged cells only, full roster.\n\n"
         + block(summary(agreement, ["set", "pair"], ["kendall", "same_best"]))),
        ("E7: seed noise",
         "Spread over seeds at the last epoch, in AUROC points:\n\n"
         + block(noise_last)
         + f"\n\nMateriality threshold for the freeze: {threshold:.2f} points "
         f"(the larger of {MATERIALITY:.1f} and twice the largest spread of the "
         "best-to-second gap)."),
    ]
    text = ("# Stage E: exploration report\n\nStage E makes no claim. Every number "
            "here is descriptive.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
