"""Stage H: score selection with 100 classes (gate, freeze, and test).

Three steps, in this order:

    python -m dynamics.analyze_h gate --results-root dyn_results
    python -m dynamics.analyze_h freeze --results-root dyn_results
    python -m dynamics.analyze_h test --results-root dyn_results

`gate` reads phase 1 and says if one fixed score is enough. `freeze` writes the
rules and the shortlists of phase 1 to one file and prints its hash. `test`
applies that file to phase 2 and reads the result with the decision rules.

Protocol: ../documentation/imagenet100_selection_protocol_2026-11-14.md
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_c, analyze_c2, analyze_e, rules
from dynamics import analyze_multimetric as mm
from dynamics import analyze_shortlist as sl
from dynamics.analyze_e import block
from dynamics.analyze_selection import FAMILY
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import DEFAULT_PRECISION, STAGES, parse_run, results_folder

HERE = Path(__file__).resolve().parent
PHASE_1, PHASE_2 = "h1", "h2"
RULES_H_PATH = HERE / "frozen" / "stage_h_rules.json"
#: SHA-256 of the frozen file of phase 1. Set after `freeze`, before any phase 2 run.
RULES_H_SHA256 = ""
ROOM = 1.0
SIZES = (3, 5)
TEST_SIZE = 3
#: The fixed shortlist of the 10-class study (shortlist check of 2026-11-14).
TEN_CLASS_SHORTLIST = ["MahaPP", "PCA_RecError", "KPCA_RecError"]
RULE, RECIPE_RULE, TEN_CLASS = "all", "B3", "ten_class_rule"


def read_stage(root: Path, stages: tuple[str, ...]) -> analyze_e.Tables:
    """The results of the runs of some stages."""
    names = {spec.name for stage in stages for spec in STAGES[stage]}
    tables = analyze_e.read_runs(root, results_folder(DEFAULT_PRECISION,
                                                      suite=analyze_c.SUITE))
    return analyze_e.Tables(**{
        key: frame if frame.empty else frame[frame["run"].isin(names)]
        .reset_index(drop=True) for key, frame in vars(tables).items()})


def recipes_of(index: pd.Index) -> pd.Series:
    return pd.Series([parse_run(run).recipe for run in index], index=index)


def gate_r(scores: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """One fixed score with each recipe left out, on the final checkpoints.

    Returns:
        Per left-out recipe: the score with the best mean on the other recipes, and
        its mean regret on the runs of the left-out recipe. Also the mean of these
        regrets over the recipes.
    """
    recipe = recipes_of(scores.index)
    best = scores.max(axis=1)
    rows = []
    for name in sorted(recipe.unique()):
        inside = (recipe == name).to_numpy()
        chosen = str(scores[~inside].mean().idxmax())
        rows.append({"left_out": name, "fixed_score": chosen,
                     "regret": float((best[inside] - scores.loc[inside, chosen])
                                     .mean())})
    table = pd.DataFrame(rows)
    return table, float(table["regret"].mean())


def build(tables: analyze_e.Tables) -> dict:
    """The rules, the baselines, and the shortlists of phase 1, for the far target."""
    meta, scores, metrics = mm.build_cells(tables, FAR_SETS)
    columns = mm.input_columns(list(mm.transform(metrics).columns))
    final, _ = sl.final_scores(tables, FAR_SETS)
    kind = recipes_of(final.index).map(FAMILY)
    return {
        "stage": "H", "sets": list(FAR_SETS), "penalty": mm.PENALTY,
        "training_runs": sorted(final.index), "training_cells": len(meta),
        "rules": {name: asdict(rules.fit_ridge_rule(metrics, scores, columns[name]))
                  for name in rules.RIDGE_RULES},
        "b1": str(final.mean().idxmax()),
        "recipe_rule": {str(name): str(part.mean().idxmax())
                        for name, part in final.groupby(kind)},
        "shortlists": {str(k): sl.greedy(final, k) for k in SIZES}}


def load(path: Path) -> dict:
    import json

    frozen = json.loads(Path(path).read_text())
    frozen["rules"] = {name: rules.RidgeRule(**rule)
                       for name, rule in frozen["rules"].items()}
    return frozen


def regret_of(scores: pd.DataFrame, picks: list) -> pd.Series:
    """Per run: the best score minus the chosen score. NaN without a choice."""
    best = scores.max(axis=1)
    values = [best.iloc[i] - scores.iloc[i][pick] if isinstance(pick, str)
              else np.nan for i, pick in enumerate(picks)]
    return pd.Series(values, index=scores.index)


def top(rule: rules.RidgeRule, metrics: pd.DataFrame, k: int) -> list[list[str]]:
    order = np.argsort(-rule.predict(metrics), axis=1)[:, :k]
    return [list(np.array(rule.csfs)[row]) for row in order]


def test_tables(frozen: dict, ten: rules.RidgeRule, scores: pd.DataFrame,
                metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regret of every single choice and of every shortlist, per run."""
    kind = recipes_of(scores.index).map(FAMILY)
    single = pd.DataFrame({
        "recipe": recipes_of(scores.index),
        RULE: regret_of(scores, frozen["rules"][RULE].choose(metrics)),
        RECIPE_RULE: regret_of(scores, [frozen["recipe_rule"][k] for k in kind]),
        analyze_c.B1: regret_of(scores, [frozen["b1"]] * len(scores)),
        "geometry": regret_of(scores, frozen["rules"]["geometry"].choose(metrics)),
        "clock": regret_of(scores, frozen["rules"]["clock"].choose(metrics)),
        TEN_CLASS: regret_of(scores, ten.choose(metrics))})
    short = pd.DataFrame({
        "recipe": recipes_of(scores.index),
        "fixed": sl.shortlist_regret(scores, frozen["shortlists"][str(TEST_SIZE)]),
        "rule": sl.shortlist_regret(scores, top(frozen["rules"][RULE], metrics,
                                                TEST_SIZE)),
        "ten_class_fixed": sl.shortlist_regret(scores, TEN_CLASS_SHORTLIST),
        "random": sl.random_regret(scores, TEST_SIZE)})
    return single, short


def read_shortlists(short: pd.DataFrame) -> tuple[str, tuple[float, float, float]]:
    """The reading of question 3 at the test size."""
    useful = {name: bool(short[name].mean() <= ROOM) for name in ("fixed", "rule")}
    difference = analyze_c.interval((short["rule"] - short["fixed"]).to_numpy())
    adds = bool(difference[0] < 0 and difference[2] < 0)
    text = (f"the fixed shortlist is {'useful' if useful['fixed'] else 'not useful'}, "
            f"the shortlist of the rule is "
            f"{'useful' if useful['rule'] else 'not useful'}, and the metrics "
            f"{'add' if adds else 'do not add'} to the fixed shortlist")
    return text, difference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("gate", "freeze", "test"))
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--rules", default=str(RULES_H_PATH))
    parser.add_argument("--allow-other-rules", action="store_true",
                        help="accept a rules file with another hash (tests only)")
    parser.add_argument("--out", default=str(HERE / "outputs" / "stage_h_report.md"))
    args = parser.parse_args()
    root, path = Path(args.results_root), Path(args.rules)
    first = read_stage(root, (PHASE_1,))
    if first.metrics.empty:
        raise SystemExit(f"no phase 1 results under {root}/runs")
    final, _ = sl.final_scores(first, FAR_SETS)
    table, mean = gate_r(final)
    room = mean >= ROOM
    gate = ("Each recipe is left out once. `fixed_score` is the best score of the "
            "other recipes:\n\n" + block(table, 2) + f"\n\n**Gate R: "
            f"{'there is room to select' if room else 'NO ROOM: one fixed score is enough'}"
            f".** The mean regret over the recipes is {mean:.2f} (limit {ROOM}).")
    sections = [("Phase 1 runs", f"{len(final)} of {len(STAGES[PHASE_1])} runs."),
                ("Gate R", gate)]
    if args.step == "freeze":
        if path.exists():
            raise SystemExit(f"{path} exists. The rules are frozen once")
        if len(final) != len(STAGES[PHASE_1]):
            raise SystemExit("phase 1 is not complete")
        digest = rules.save(build(first), path)
        frozen = load(path)
        sections.append(("Frozen file", (
            f"- File: `{path}`\n- SHA-256: `{digest}`\n- B1: {frozen['b1']}. Recipe "
            f"rule: {frozen['recipe_rule']}.\n- Shortlists: {frozen['shortlists']}."
            f"\n- Training cells: {frozen['training_cells']}.")))
    if args.step == "test":
        digest = rules.file_hash(path)
        if digest != RULES_H_SHA256 and not args.allow_other_rules:
            raise SystemExit(f"the rules file has hash {digest}, not the frozen one")
        frozen = load(path)
        ten = rules.load(rules.RULES_PATH)["targets"]["far"]["rules"][RULE]
        second = read_stage(root, (PHASE_2,))
        if second.metrics.empty:
            raise SystemExit(f"no phase 2 results under {root}/runs")
        scores, metrics = sl.final_scores(second, FAR_SETS)
        single, short = test_tables(frozen, ten, scores, metrics)
        outcome, details = analyze_c2.decide(single)
        reading, difference = read_shortlists(short)
        every, every_metrics = sl.final_scores(read_stage(root, (PHASE_1, PHASE_2)),
                                               FAR_SETS)
        both, both_short = test_tables(frozen, ten, every, every_metrics)
        moves = (both_short["ten_class_fixed"].mean(), both[TEN_CLASS].mean())
        sections += [
            ("Question 2: single choice", (
                f"{len(scores)} of {len(STAGES[PHASE_2])} runs of phase 2. Regret at "
                "the final checkpoint, in AUROC points on the far sets:\n\n"
                + block(single.reset_index(), 2) + "\n\nBy recipe:\n\n"
                + block(single.groupby("recipe").mean().reset_index(), 2) + "\n\n"
                + block(details, 2) + f"\n\n**Outcome: {outcome}.**")),
            ("Question 3: shortlists of size 3", (
                block(short.reset_index(), 2) + "\n\nMeans: "
                + ", ".join(f"{name} {short[name].mean():.2f}"
                            for name in short.columns[1:])
                + f". The rule minus the fixed shortlist: {difference[0]:.2f} "
                f"({difference[1]:.2f} to {difference[2]:.2f}).\n\n"
                f"**Reading: {reading}.**")),
            ("Question 4: transfer from 10 classes", (
                f"On all {len(every)} runs of both phases. The 10-class shortlist "
                f"has a mean shortlist regret of {moves[0]:.2f}: "
                f"**{'it transfers' if moves[0] <= ROOM else 'it does not transfer'}"
                f"**. The 10-class rule has a mean regret of {moves[1]:.2f} for its "
                f"single choice: **{'it transfers' if moves[1] <= ROOM else 'it does not transfer'}**."))]
    text = ("# Stage H: score selection with 100 classes\n\nThe gate, the rules, "
            "and the decision rules were fixed before the runs.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
