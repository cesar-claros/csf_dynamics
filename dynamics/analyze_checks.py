"""Four checks on the step 0 results that need no new training.

The novelty check of 2026-11-12 listed checks that make the pilot comparable with
prior work. These four read only results that already exist:

- Check 3: does each single score decline after convergence, or only the best
  score over the roster?
- Check 8: the regret of a fixed score in each run, for near and far sets.
- Check 11: the rank agreement of each stand-in with each true OOD set, per run.
- Check 12: what is lost in FPR@95 when the score is chosen by AUROC.

All four are post hoc and descriptive. Run from code/:

    python -m dynamics.analyze_checks --results-root dyn_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dynamics import analyze_e
from dynamics.analyze_e import block
from dynamics.analyze_step0 import DECLINE, EARLY, LATE, SEED_SHARE, SUITE
from dynamics.config import DEFAULT_PRECISION, PRECISIONS, results_folder
from dynamics.scores import CSF_GROUP

HERE = Path(__file__).resolve().parent
NEAR = ("imagewoof", "ssb_hard", "ninco", "imagenet_o")
FAR = ("textures", "inaturalist", "places", "sun", "openimage_o")
EXTERNAL = (*NEAR, *FAR)
STAND_INS = ("blend", "id_errors", "shuffle")
#: Single scores that prior work plots over training, plus the roster leaders.
KNOWN_SCORES = ("MSR", "MLS", "Energy", "GradNorm", "Maha", "MahaPP", "kNN", "ViM",
                "Residual", "PCA_RecError", "NCI", "CTM")
RESAMPLES = 2000


def distance(name: str) -> str:
    return "near" if name in NEAR else "far"


def score_declines(metrics: pd.DataFrame, early: int = EARLY, late: int = LATE
                   ) -> pd.DataFrame:
    """Check 3: per regime, set, and CSF, the mean AUROC at two epochs and the drop.

    `declines` uses the thresholds of step 0: a mean drop of at least 3.0 points
    in at least 80% of the seeds.
    """
    at = metrics[metrics["epoch"].isin([early, late])
                 & metrics["set"].isin(EXTERNAL)]
    wide = (at.pivot_table(index=["recipe", "set", "csf", "run"], columns="epoch",
                           values="auroc") * 100).reindex(columns=[early, late])
    wide["drop"] = wide[early] - wide[late]
    group = wide.groupby(["recipe", "set", "csf"])
    out = pd.DataFrame({"early": group[early].mean(), "late": group[late].mean(),
                        "drop": group["drop"].mean(),
                        "seeds_dropping": group["drop"].apply(
                            lambda d: int((d > 0).sum())),
                        "seeds": group.size()}).reset_index()
    out["declines"] = ((out["drop"] >= DECLINE)
                       & (out["seeds_dropping"] >= SEED_SHARE * out["seeds"]))
    return out


def fixed_score_regret(metrics: pd.DataFrame, epoch: int = LATE) -> pd.DataFrame:
    """Check 8: per run and set, the regret of two fixed choices at one epoch.

    `set_fixed` is the CSF with the best mean AUROC over the seeds for that set and
    regime. `regime_fixed` is one CSF for the whole regime: the best mean AUROC over
    all external sets and seeds.
    """
    at = metrics[(metrics["epoch"] == epoch) & metrics["set"].isin(EXTERNAL)]
    rows = []
    for recipe, part in at.groupby("recipe"):
        overall = part.groupby("csf")["auroc"].mean().idxmax()
        for name, cell in part.groupby("set"):
            wide = cell.pivot(index="run", columns="csf", values="auroc") * 100
            choice = wide.mean(axis=0).idxmax()
            best = wide.max(axis=1)
            for run in wide.index:
                rows.append({
                    "recipe": recipe, "set": name, "distance": distance(name),
                    "run": run, "winner": wide.loc[run].idxmax(),
                    "set_fixed": choice,
                    "set_fixed_regret": best[run] - wide.loc[run, choice],
                    "regime_fixed": overall,
                    "regime_fixed_regret": best[run] - wide.loc[run, overall]})
    return pd.DataFrame(rows)


def seed_spread(metrics: pd.DataFrame, epoch: int = LATE) -> pd.DataFrame:
    """Check 8: standard deviation over seeds of each score's AUROC, in points."""
    at = metrics[(metrics["epoch"] == epoch) & metrics["set"].isin(EXTERNAL)]
    spread = (at.groupby(["recipe", "set", "csf"])["auroc"].std(ddof=1) * 100)
    return spread.rename("sd").reset_index()


def run_interval(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of the mean, resampling runs."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(RESAMPLES, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def stand_in_agreement(metrics: pd.DataFrame, epoch: int = LATE) -> pd.DataFrame:
    """Check 11: per run, the rank agreement of each stand-in with each true set."""
    at = metrics[metrics["epoch"] == epoch]
    rows = []
    for (run, recipe), cell in at.groupby(["run", "recipe"]):
        wide = cell.pivot(index="csf", columns="set", values="auroc") * 100
        for stand_in in STAND_INS:
            if stand_in not in wide or wide[stand_in].isna().any():
                continue
            for target in EXTERNAL:
                if target not in wide:
                    continue
                rho, regret = analyze_e.pick_quality(wide[stand_in], wide[target])
                rows.append({"run": run, "recipe": recipe, "stand_in": stand_in,
                             "target": target, "distance": distance(target),
                             "spearman": rho, "regret": regret,
                             "random_regret": float((wide[target].max()
                                                     - wide[target]).mean())})
    return pd.DataFrame(rows)


def summarize_agreement(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean over runs with run-level intervals, per stand-in and target."""
    out = []
    for (stand_in, target), part in rows.groupby(["stand_in", "target"]):
        low, high = run_interval(part["spearman"].to_numpy())
        out.append({"stand_in": stand_in, "target": target,
                    "distance": distance(target), "runs": len(part),
                    "spearman": part["spearman"].mean(), "low": low, "high": high,
                    "regret": part["regret"].mean(),
                    "random_regret": part["random_regret"].mean()})
    return pd.DataFrame(out)


def metric_choice(metrics: pd.DataFrame, counts: dict[str, int],
                  epoch: int = LATE) -> pd.DataFrame:
    """Check 12: per run and set, the FPR@95 lost when AUROC chooses the score.

    `fpr_se` is the binomial standard error of FPR@95 for the chosen score, from
    the number of OOD images. It ignores the error of the threshold, so it is a
    lower bound of the sampling error.
    """
    at = metrics[(metrics["epoch"] == epoch) & metrics["set"].isin(EXTERNAL)]
    rows = []
    for (run, recipe, name), cell in at.groupby(["run", "recipe", "set"]):
        cell = cell.set_index("csf")
        by_auroc, by_fpr = cell["auroc"].idxmax(), cell["fpr95"].idxmin()
        rate = cell.loc[by_auroc, "fpr95"]
        size = counts.get(name, np.nan)
        rows.append({
            "run": run, "recipe": recipe, "set": name, "distance": distance(name),
            "same_best": by_auroc == by_fpr,
            "fpr_regret": 100 * (rate - cell["fpr95"].min()),
            "auroc_regret": 100 * (cell["auroc"].max() - cell.loc[by_fpr, "auroc"]),
            "fpr_se": 100 * float(np.sqrt(rate * (1 - rate) / size)),
            "spearman": float(spearmanr(cell["auroc"], -cell["fpr95"]).statistic)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--precision", choices=sorted(PRECISIONS),
                        default=DEFAULT_PRECISION)
    parser.add_argument("--out", default=str(HERE / "outputs" / "checks_report.md"))
    args = parser.parse_args()
    root = Path(args.results_root)
    tag = results_folder(args.precision, suite=SUITE)
    tables = analyze_e.stage_e_only(analyze_e.read_runs(root, tag))
    if tables.metrics.empty:
        raise SystemExit(f"no results of the suite {SUITE} under {root}/runs")
    csfs, _ = analyze_e.roster(tables, set())
    metrics = tables.metrics[(tables.metrics["held_out"] < 0)
                             & tables.metrics["csf"].isin(csfs)]
    first = min((root / "runs").glob(f"*/{tag}/epoch_*.json"))
    counts = json.loads(first.read_text())["counts"]
    out_dir = Path(args.out).parent / "checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    declines = score_declines(metrics)
    declines["group"] = declines["csf"].map(CSF_GROUP)
    regret = fixed_score_regret(metrics)
    spread = seed_spread(metrics)
    agreement = stand_in_agreement(metrics)
    agreement_summary = summarize_agreement(agreement)
    choice = metric_choice(metrics, counts)
    for name, frame in (("check3_score_declines", declines),
                        ("check8_fixed_score_regret", regret),
                        ("check8_seed_spread", spread),
                        ("check11_stand_in_agreement", agreement),
                        ("check11_summary", agreement_summary),
                        ("check12_metric_choice", choice)):
        frame.to_csv(out_dir / f"{name}.csv", index=False)

    order = {name: position for position, name in enumerate(EXTERNAL)}

    def by_set(frame: pd.DataFrame, column: str = "set") -> pd.DataFrame:
        return frame.assign(_o=frame[column].map(order)).sort_values(
            ["_o", *[c for c in ("recipe", "stand_in") if c in frame]]).drop(
            columns="_o")

    counted = (declines.groupby(["set", "recipe"])
               .agg(scores_declining=("declines", "sum"), scores=("csf", "size"),
                    median_drop=("drop", "median"), largest_rise=("drop", "min"))
               .reset_index())
    known = declines[(declines["recipe"] == "A")
                     & declines["csf"].isin(KNOWN_SCORES)]
    known_table = (known.pivot(index="csf", columns="set", values="drop")
                   .reindex(index=list(KNOWN_SCORES), columns=list(EXTERNAL)))
    by_group = (declines[declines["recipe"] == "A"]
                .assign(distance=lambda f: f["set"].map(distance))
                .groupby(["group", "distance"])["drop"].mean().unstack("distance"))

    regret_summary = (regret.groupby(["set", "recipe"])
                      .agg(set_fixed=("set_fixed", "first"),
                           mean=("set_fixed_regret", "mean"),
                           worst_run=("set_fixed_regret", "max"),
                           winners=("winner", "nunique"),
                           regime_fixed=("regime_fixed", "first"),
                           regime_mean=("regime_fixed_regret", "mean"),
                           regime_worst=("regime_fixed_regret", "max"))
                      .reset_index())
    regret_groups = (regret.groupby(["recipe", "distance"])
                     [["set_fixed_regret", "regime_fixed_regret"]]
                     .agg(["mean", "max"]))
    spread_summary = (spread.groupby(["set", "recipe"])["sd"]
                      .agg(["median", "max"]).reset_index())

    choice_summary = (choice.groupby(["set", "recipe"])
                      .agg(same_best=("same_best", "mean"),
                           fpr_regret=("fpr_regret", "mean"),
                           fpr_se=("fpr_se", "mean"),
                           spearman=("spearman", "mean"),
                           auroc_regret=("auroc_regret", "mean")).reset_index())
    beyond = float((choice["fpr_regret"] > 2 * choice["fpr_se"]).mean())

    sections = [
        ("Check 3: single scores against the best score",
         f"A score declines on a set if its mean AUROC falls by at least {DECLINE} "
         f"points from epoch {EARLY} to epoch {LATE} in at least "
         f"{SEED_SHARE:.0%} of the seeds. Number of the {len(csfs)} scores that "
         "decline, their median drop, and the largest rise (a negative drop):\n\n"
         + block(by_set(counted), 2)
         + "\n\nDrop of well-known single scores in regime A, in AUROC points:\n\n"
         + f"```\n{known_table.round(1).to_string()}\n```"
         + "\n\nMean drop in regime A by score group and by distance of the set:"
         + f"\n\n```\n{by_group.round(2).to_string()}\n```"),
        ("Check 8: the regret of a fixed score in each run",
         f"At epoch {LATE}. `set_fixed` is the best score for that set and regime "
         "on average over the seeds. `regime_fixed` is one score for all sets of "
         "the regime. Regret per run in AUROC points:\n\n"
         + block(by_set(regret_summary), 2)
         + "\n\nBy regime and distance (mean and worst run):\n\n"
         + f"```\n{regret_groups.round(2).to_string()}\n```"
         + "\n\nSpread over seeds of a score's AUROC (median and largest over the "
         "scores):\n\n" + block(by_set(spread_summary), 2)),
        ("Check 11: rank agreement of the stand-ins with the true sets",
         f"At epoch {LATE}, over the {len(csfs)} scores, per run. Mean over the "
         "runs, with a 95% bootstrap interval over runs. `regret` is the AUROC "
         "lost on the true set by the score that the stand-in ranks first, and "
         "`random_regret` is the loss of a random score:\n\n"
         + block(by_set(agreement_summary, "target").sort_values(
             ["stand_in", "distance", "target"]), 2)),
        ("Check 12: AUROC against FPR@95",
         f"At epoch {LATE}. `fpr_regret` is the FPR@95 lost, in points, when the "
         "score with the best AUROC is used. `fpr_se` is a lower bound of the "
         "sampling error of that FPR@95. `spearman` is the rank correlation of "
         f"the two metrics over the {len(csfs)} scores:\n\n"
         + block(by_set(choice_summary), 2)
         + f"\n\nShare of all cells where the FPR@95 loss is more than twice its "
         f"sampling error: {beyond:.0%}."),
    ]
    text = ("# Four checks without new training\n\nThese checks come from the "
            "novelty check of 2026-11-12. They are post hoc and descriptive. They "
            f"read the {metrics['run'].nunique()} main runs of step 0.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
