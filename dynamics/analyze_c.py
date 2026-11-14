"""Stage C analysis: the frozen selection rules on new recipes (amendment A5).

Applies the rules of `dynamics/frozen/stage_c_rules.json` to the Stage C runs,
without a new fit, and reads the result with the decision rule of the amendment.

Run from code/ after copying the run folders from the HPC:

    python -m dynamics.analyze_c --results-root dyn_results

Protocol: ../documentation/stage_c_selection_amendment_2026-11-14.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_e, rules
from dynamics import analyze_multimetric as mm
from dynamics.analyze_e import block
from dynamics.config import (
    DEFAULT_PRECISION,
    PRECISIONS,
    STAGES,
    parse_run,
    results_folder,
)

HERE = Path(__file__).resolve().parent
SUITE = "step0"
#: SHA-256 of the frozen file, recorded in the amendment before any Stage C run.
RULES_SHA256 = "48362fcaeeeab40c91b44bdc2165c92452c92f83e2b7e2f5ee411575d39f9233"
NEW_RECIPES = ("C1", "C2", "C3")
KNOWN_RECIPES = ("A", "B")
PRIMARY = "geometry"
B1, B2, STAND_IN, NEAREST = "B1", "B2", "stand_in", "nearest"
MATERIALITY = 1.0
GATE_K = 1.0
RECIPES_NEEDED = 2
RESAMPLES = 2000
#: State against clock: reference recipe and new recipe of the same kind.
PAIRS = (("A", "C3"), ("B", "C1"))
STATE_FROM = 10
STATE_MARGIN = 0.5
SEEDS_NEEDED = 2


def final_mask(meta: pd.DataFrame) -> np.ndarray:
    """True for the last checkpoint of each run."""
    epochs = pd.Series(meta.index.get_level_values("epoch"), index=meta.index)
    return (epochs.groupby(level="run").transform("max") == epochs).to_numpy()


def choices(target: dict, meta: pd.DataFrame, scores: pd.DataFrame,
            metrics: pd.DataFrame, mask: np.ndarray,
            blend: pd.DataFrame | None = None,
            reference: tuple[pd.DataFrame, pd.DataFrame] | None = None
            ) -> pd.DataFrame:
    """The choice and the regret of every rule on the cells of `mask`.

    Args:
        target: One target of the frozen file, with `rules` and `baselines`.
        blend: AUROC of each score on the class-blend set, by cell. With it, the
            stand-in rule is added.
        reference: Metrics and scores of the training cells. With it, the model
            with the nearest cells is added.

    Returns:
        One row per cell and rule. A rule with no choice for a cell has NaN regret.
    """
    cells, values = metrics[mask], scores[mask]
    best = values.max(axis=1)
    picked: dict[str, list] = {
        name: rule.choose(cells) for name, rule in target["rules"].items()}
    base = target["baselines"]
    picked[B1] = [base["b1"]["score"]] * len(cells)
    picked[B2] = [base["b2"][kind]["score"] for kind in meta.loc[mask, "family"]]
    if blend is not None:
        picked[STAND_IN] = list(blend.reindex(cells.index).idxmax(axis=1))
    if reference is not None:
        rule = target["rules"][PRIMARY]
        near = mm.nearest(rule.inputs(reference[0]),
                          mm.logit(reference[1][rule.csfs].to_numpy()),
                          rule.inputs(cells))
        picked[NEAREST] = [rule.csfs[int(row.argmax())] if np.isfinite(row).all()
                           else None for row in near]
    rows = []
    for name, chosen in picked.items():
        for (run, epoch), choice in zip(cells.index, chosen):
            valid = isinstance(choice, str)
            rows.append({"run": run, "epoch": epoch, "recipe": parse_run(run).recipe,
                         "rule": name, "choice": choice if valid else None,
                         "regret": (best[run, epoch] - values.loc[(run, epoch), choice]
                                    if valid else np.nan)})
    return pd.DataFrame(rows)


def run_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean regret per run (rows) and rule (columns), with the recipe as a column."""
    table = frame.pivot_table(index="run", columns="rule", values="regret")
    table.insert(0, "recipe", [parse_run(run).recipe for run in table.index])
    return table


def interval(difference: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Mean of the per-run differences, with a 95% bootstrap interval over runs."""
    difference = difference[np.isfinite(difference)]
    if len(difference) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = difference[rng.integers(0, len(difference),
                                    (RESAMPLES, len(difference)))].mean(axis=1)
    return (float(difference.mean()), float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def compare(table: pd.DataFrame, rule: str, baseline: str) -> dict:
    """Does a rule beat a baseline on the runs of `table`, by the three conditions?"""
    mean, low, high = interval((table[rule] - table[baseline]).to_numpy())
    by_recipe = table.groupby("recipe")[[rule, baseline]].mean()
    lower = int((by_recipe[rule] < by_recipe[baseline]).sum())
    return {"rule": rule, "baseline": baseline, "rule_regret": table[rule].mean(),
            "baseline_regret": table[baseline].mean(), "difference": mean,
            "low": low, "high": high, "recipes_lower": lower,
            "holds": bool(mean < 0 and high < 0 and lower >= RECIPES_NEEDED)}


def decide(table: pd.DataFrame, rule: str = PRIMARY, stand_in: bool = False
           ) -> tuple[str, pd.DataFrame]:
    """The decision rule of the amendment on the runs with new recipes."""
    new = table[table["recipe"].isin(NEW_RECIPES)]
    if new.empty or new[[rule, B1, B2]].isna().any().any():
        return "not decided: a run with a new recipe has no choice", pd.DataFrame()
    against = [compare(new, rule, B2), compare(new, rule, B1)]
    if stand_in:
        against.append(compare(new, STAND_IN, B2))
    details = pd.DataFrame(against)
    if new[B2].mean() < MATERIALITY:
        return ("M, nothing to select: the score of the kind of recipe is within "
                f"{MATERIALITY} point of the best"), details
    if against[0]["holds"]:
        return "S2, selection works: the rule beats the score of the kind", details
    if against[1]["holds"]:
        return "S1, the rule finds the kind: it beats one fixed score only", details
    if stand_in and against[2]["holds"]:
        return "T, the stand-in rule beats the score of the kind", details
    harmful = against[0]["difference"] > 0 and against[0]["low"] > 0
    return "N, no claim" + (" (the rule is harmful)" if harmful else ""), details


def gate_k(table: pd.DataFrame) -> tuple[bool, str]:
    known = table[table["recipe"].isin(KNOWN_RECIPES)]
    if known.empty:
        return False, "no fresh run of a known recipe"
    regret = known[PRIMARY].mean()
    return bool(regret <= GATE_K), (f"mean regret {regret:.2f} on {len(known)} fresh "
                                    f"runs of known recipes (limit {GATE_K})")


def state_against_clock(old: analyze_e.Tables, new: analyze_e.Tables,
                        sets: tuple[str, ...]) -> pd.DataFrame:
    """Per new run: error of the state prediction and of the clock prediction.

    The reference is the mean curve of the pilot recipe of the same kind: the far
    AUROC of each score and log10 NC1 at each checkpoint from `STATE_FROM`, on
    the part where collapse advances. A checkpoint of a new run is predicted at
    the same log10 NC1 (state) and at the same epoch (clock), if its NC1 is inside
    the range of the reference.
    """
    def curves(tables: analyze_e.Tables) -> tuple[pd.DataFrame, pd.Series]:
        kept = tables.metrics[tables.metrics["set"].isin(sets)
                              & (tables.metrics["epoch"] >= STATE_FROM)]
        auroc = (kept.groupby(["run", "epoch", "csf"])["auroc"].mean() * 100
                 ).unstack("csf")
        nc1 = tables.descriptors[(tables.descriptors["group"] == "nc_paper")
                                 & (tables.descriptors["name"] == "var_collapse")]
        state = np.log10(nc1.set_index(["run", "epoch"])["value"])
        return auroc, state.reindex(auroc.index)

    old_auroc, old_state = curves(old)
    new_auroc, new_state = curves(new)
    rows = []
    for reference, recipe in PAIRS:
        names = [run for run in old_auroc.index.get_level_values("run").unique()
                 if parse_run(run).recipe == reference]
        if not names:
            continue
        mean_auroc = old_auroc.loc[names].groupby(level="epoch").mean()
        mean_state = old_state.loc[names].groupby(level="epoch").mean()
        path = mean_state[mean_state.cummin() == mean_state]
        order = np.argsort(path.to_numpy())
        for run in new_auroc.index.get_level_values("run").unique():
            if parse_run(run).recipe != recipe:
                continue
            errors = []
            for epoch in new_auroc.loc[run].index:
                level = new_state[run, epoch]
                if (epoch not in mean_auroc.index or not np.isfinite(level)
                        or not path.min() <= level <= path.max()):
                    continue
                actual = new_auroc.loc[(run, epoch)]
                by_state = np.array([np.interp(
                    level, path.to_numpy()[order],
                    mean_auroc.loc[path.index, csf].to_numpy()[order])
                    for csf in actual.index])
                errors.append((np.abs(actual - by_state).mean(),
                               np.abs(actual - mean_auroc.loc[epoch]).mean()))
            if errors:
                state, clock = np.mean(errors, axis=0)
                rows.append({"reference": reference, "recipe": recipe, "run": run,
                             "checkpoints": len(errors), "state_error": state,
                             "clock_error": clock,
                             "state_wins": bool(clock - state >= STATE_MARGIN)})
    return pd.DataFrame(rows)


def leave_one_recipe_out(parts: list[tuple[pd.DataFrame, pd.DataFrame,
                                           pd.DataFrame]]) -> pd.DataFrame:
    """The rules fitted again on the other recipes, tested on final checkpoints."""
    meta, scores, metrics = (pd.concat(frames) for frames in zip(*parts))
    meta = meta.assign(
        family=[parse_run(run).recipe for run in meta.index.get_level_values("run")],
        eligible=final_mask(meta))
    return mm.evaluate(meta, scores, metrics, "family")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--rules", default=str(rules.RULES_PATH))
    parser.add_argument("--allow-other-rules", action="store_true",
                        help="accept a rules file with another hash (tests only)")
    parser.add_argument("--precision", choices=sorted(PRECISIONS),
                        default=DEFAULT_PRECISION)
    parser.add_argument("--out", default=str(HERE / "outputs" / "stage_c_report.md"))
    args = parser.parse_args()
    digest = rules.file_hash(Path(args.rules))
    if digest != RULES_SHA256 and not args.allow_other_rules:
        raise SystemExit(f"the rules file has hash {digest}, not the frozen one")
    frozen = rules.load(Path(args.rules))
    root = Path(args.results_root)
    wanted = {spec.name for spec in STAGES["c"]}
    everything = analyze_e.read_runs(root, results_folder(args.precision, suite=SUITE))
    new = analyze_e.Tables(**{
        name: frame if frame.empty else frame[frame["run"].isin(wanted)]
        .reset_index(drop=True) for name, frame in vars(everything).items()})
    if new.metrics.empty:
        raise SystemExit(f"no Stage C results under {root}/runs")
    old = analyze_e.read_runs(root, results_folder(args.precision,
                                                   suite=rules.TRAINING_SUITE))
    found = sorted(new.metrics["run"].unique())
    out_dir = Path(args.out).parent / "stage_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    blend = (new.metrics[new.metrics["set"] == "blend"]
             .pivot_table(index=["run", "epoch"], columns="csf", values="auroc"))

    sections = [("Runs and rules", (
        f"- Runs: {len(found)} of {len(wanted)}. Missing: "
        f"{', '.join(sorted(wanted - set(found))) or 'none'}.\n- Rules file hash: "
        f"`{digest}`" + ("" if digest == RULES_SHA256 else " (NOT the frozen file)")
        + f".\n- Training runs of the rules: {len(frozen['training_runs'])}."))]
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    outcomes: dict[tuple[str, ...], tuple[str, pd.DataFrame]] = {}
    for name, target in frozen["targets"].items():
        sets = tuple(target["sets"])
        meta, scores, metrics = mm.build_cells(new, sets)
        reference = None
        if not old.metrics.empty:
            _, old_scores, old_metrics = mm.build_cells(old, sets)
            reference = (old_metrics, old_scores)
        near = name == "near"
        for label, mask in (("final", final_mask(meta)),
                            ("eligible", meta["eligible"].to_numpy())):
            frame = choices(target, meta, scores, metrics, mask,
                            blend if near else None, reference)
            frame.to_csv(out_dir / f"choices_{name}_{label}.csv", index=False)
            tables[name, label] = run_table(frame)
            outcomes[name, label] = decide(tables[name, label], stand_in=near)
            outcomes[name, label, "clock"] = decide(tables[name, label], "clock")
        picks = (frame[frame["epoch"] == frame.groupby("run")["epoch"]
                       .transform("max")].pivot_table(
            index="run", columns="rule", values="choice", aggfunc="first"))
        table = tables[name, "final"]
        by_recipe = table.groupby("recipe").mean(numeric_only=True)
        outcome, details = outcomes[name, "final"]
        clock_outcome, _ = outcomes[name, "final", "clock"]
        credit = compare(table[table["recipe"].isin(NEW_RECIPES)], PRIMARY, "clock")
        body = (f"Sets: {', '.join(sets)}. Regret at the final checkpoint, in AUROC "
                "points:\n\n" + block(table.reset_index(), 2) + "\n\nBy recipe:\n\n"
                + block(by_recipe.reset_index(), 2) + "\n\nChoices:\n\n"
                + block(picks.reset_index()))
        if name == "far":
            passed, text = gate_k(table)
            body += f"\n\n**Gate K: {'pass' if passed else 'FAIL'}.** {text}."
            if not passed:
                outcome = ("rule not valid, because gate K failed (the decision "
                           f"rule alone gives: {outcome})")
        body += ("\n\nThe rule against the baselines, on the runs with new recipes:"
                 "\n\n" + block(details, 2) + f"\n\n**Outcome: {outcome}.**\n\n"
                 f"With the clock rule in place of the geometric rule: "
                 f"{clock_outcome}. The geometric rule against the clock rule: "
                 f"difference {credit['difference']:.2f} "
                 f"({credit['low']:.2f} to {credit['high']:.2f}).\n\nOn all "
                 f"eligible checkpoints: {outcomes[name, 'eligible'][0]}.")
        sections.append((f"Target: {name}" + (" (primary)" if name == "far" else
                                              " (secondary)"), body))
        if name == "far" and reference is not None:
            again = leave_one_recipe_out([(meta, scores, metrics),
                                          mm.build_cells(old, sets)])
            state = state_against_clock(old, new, sets)
            wins = state.groupby(["reference", "recipe"])["state_wins"].agg(
                ["sum", "count"]).reset_index() if not state.empty else state
            sections.append(("Secondary: one recipe left out", (
                "The rules are fitted again on the other recipes. This is not a "
                "test of the frozen rule:\n\n" + block(mm.summarize(again), 2))))
            sections.append(("Secondary: state against clock across recipes", (
                block(state, 2) + "\n\nThe state wins in a run if its error is at "
                f"least {STATE_MARGIN} point lower. A pair is supported with "
                f"{SEEDS_NEEDED} of 3 seeds:\n\n" + block(wins))))
    text = ("# Stage C: frozen selection rules on new recipes\n\nThe rules, the "
            "baselines, and the decision rule were fixed before these runs "
            "(amendment A5).\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
