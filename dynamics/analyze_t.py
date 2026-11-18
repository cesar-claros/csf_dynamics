"""Amendment A7: a selection rule from validation statistics (freeze and test).

    python -m dynamics.analyze_t freeze --results-root dyn_results
    python -m dynamics.analyze_t test --results-root dyn_results

`freeze` fits the rules on the 36 earlier 10-class runs, with the validation
statistics of the light pass, and writes them to one file. `test` applies that
file to the runs with the new recipes and reads the result.

Protocol: ../documentation/validation_statistics_amendment_2026-11-17.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from dynamics import analyze_c, analyze_e, analyze_h, rules
from dynamics import analyze_multimetric as mm
from dynamics import analyze_shortlist as sl
from dynamics import analyze_valonly as vo
from dynamics.analyze_e import block
from dynamics.analyze_selection import FAMILY
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import (
    DEFAULT_PRECISION,
    STAGES,
    parse_run,
    results_folder,
)
from dynamics.descriptors import VALID
from dynamics.evaluate import VALSTATS_SUITE

HERE = Path(__file__).resolve().parent
TRAIN_STAGE, TEST_STAGE = "a7_train", "t"
RULES_T_PATH = HERE / "frozen" / "stage_t_rules.json"
#: SHA-256 of the frozen file. Set after `freeze`, before any run of the test stage.
RULES_T_SHA256 = "a21233c622d62aa449f5525272d671402dce86f1752ed61324f21c433799546e"
WIDE, TRAINING, STORED, TIERS, ALL = (
    "wide_validation", "training", "stored_validation", "all_tiers", "all_metrics")
RULE_NAMES = (WIDE, TRAINING, STORED, TIERS, ALL)
B1, RECIPE_RULE = "one_score", "recipe_rule"
#: Statistics of each score on the validation set (tier 3).
SCORE_SETS = ("id_errors", "blend", "shuffle")
HOLD = 1.0
MIN_ACCURACY = 0.60
SHORT = 3


def read_valstats(root: Path, names: set[str]) -> pd.DataFrame:
    """The rows of the light validation pass, in the form of `Tables.descriptors`."""
    rows = []
    tag = results_folder(DEFAULT_PRECISION, suite=VALSTATS_SUITE)
    for name in sorted(names):
        spec = parse_run(name)
        for path in sorted((Path(root) / "runs" / name / tag).glob("epoch_*.json")):
            out = json.loads(path.read_text())
            rows += [{"run": name, "recipe": spec.recipe, "seed": spec.seed,
                      "held_out": -1, "epoch": out["epoch"], **row}
                     for row in out["descriptors"]]
    return pd.DataFrame(rows)


def earlier_tables(root: Path) -> analyze_e.Tables:
    """The 36 earlier 10-class runs, with the validation statistics of the pass."""
    first = rules.training_tables(root, with_stage_c=True)
    second = analyze_h.read_stage(root, ("c2",))
    tables = analyze_e.Tables(**{
        key: pd.concat([frame, other], ignore_index=True)
        for (key, frame), other in zip(vars(first).items(), vars(second).values())})
    done = set(tables.descriptors.loc[tables.descriptors["group"] == VALID, "run"])
    names = set(tables.descriptors["run"]) - done
    extra = read_valstats(root, names)
    return analyze_e.Tables(**{**vars(tables), "descriptors": pd.concat(
        [tables.descriptors, extra], ignore_index=True)})


def cells(tables: analyze_e.Tables
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cells from epoch 1, with the derived validation NC1 and the score statistics."""
    meta, scores, metrics = mm.build_cells(tables, FAR_SETS)
    metrics = vo.with_validation_nc1(metrics)
    for name in SCORE_SETS:
        part = tables.metrics[tables.metrics["set"] == name]
        if part.empty:
            continue
        wide = part.pivot_table(index=["run", "epoch"], columns="csf", values="auroc")
        metrics = metrics.join(wide.add_prefix(f"score/{name}/"))
    return meta, scores, metrics


def input_sets(columns: list[str]) -> dict[str, list[str]]:
    """The inputs of each frozen rule."""
    wide = [c for c in columns if c.startswith(f"{VALID}/")]
    per_score = [c for c in columns if c.startswith("score/")]
    derived = [c for c in columns if c.startswith("derived/")]
    original = [c for c in columns if c not in wide + per_score + derived]
    return {
        WIDE: wide,
        TRAINING: [c for c in original if vo.input_set(c) == "training"],
        STORED: [c for c in original + derived if vo.input_set(c) == "validation"],
        TIERS: wide + per_score,
        ALL: original}


def build(tables: analyze_e.Tables, names: tuple[str, ...] = RULE_NAMES,
          amendment: str = "A7") -> dict:
    """The frozen rules `names` and the baselines, fitted on `tables`."""
    meta, scores, metrics = cells(tables)
    usable = list(mm.transform(metrics).columns)
    sets = {name: columns for name, columns in input_sets(usable).items()
            if name in names}
    final = scores[analyze_c.final_mask(meta)].droplevel("epoch")
    final = final[[parse_run(run).recipe[0] != "D" for run in final.index]]
    kind = analyze_h.recipes_of(final.index).map(FAMILY)
    return {
        "amendment": amendment, "sets": list(FAR_SETS), "penalty": mm.PENALTY,
        "training_runs": sorted(set(meta.index.get_level_values("run"))),
        "training_cells": len(meta), "baseline_runs": len(final),
        "rules": {name: asdict(rules.fit_ridge_rule(metrics, scores, columns))
                  for name, columns in sets.items()},
        B1: str(final.mean().idxmax()),
        RECIPE_RULE: {str(name): str(part.mean().idxmax())
                      for name, part in final.groupby(kind)},
        "shortlists": {str(k): sl.greedy(final, k) for k in (SHORT, 5)}}


def load(path: Path) -> dict:
    frozen = json.loads(Path(path).read_text())
    frozen["rules"] = {name: rules.RidgeRule(**rule)
                       for name, rule in frozen["rules"].items()}
    return frozen


def compare(table: pd.DataFrame, rule: str, baseline: str, needed: int,
            or_equal: bool = False) -> dict:
    """Mean difference with its interval, and the recipes where the rule is lower."""
    mean, low, high = analyze_c.interval((table[rule] - table[baseline]).to_numpy())
    by_recipe = table.groupby("recipe")[[rule, baseline]].mean()
    lower = (by_recipe[rule] <= by_recipe[baseline] if or_equal
             else by_recipe[rule] < by_recipe[baseline])
    return {"rule": rule, "baseline": baseline, "rule_regret": table[rule].mean(),
            "baseline_regret": table[baseline].mean(), "difference": mean,
            "low": low, "high": high, "recipes_lower": int(lower.sum()),
            "holds": bool(mean < 0 and high < 0 and lower.sum() >= needed)}


def holds(table: pd.DataFrame, rule: str) -> bool:
    by_recipe = table.groupby("recipe")[rule].mean()
    return bool(table[rule].mean() <= HOLD and (by_recipe > HOLD).sum() < 2)


def decide(table: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    """The outcome of question 1, the reading of question 2, and the comparisons."""
    if table.empty or table[[WIDE, TRAINING, RECIPE_RULE]].isna().any().any():
        return "not decided: a run has no choice", "not decided", pd.DataFrame()
    needed = max(1, table["recipe"].nunique() - 1)
    against = [compare(table, WIDE, RECIPE_RULE, needed),
               compare(table, WIDE, TRAINING, needed, or_equal=True),
               compare(table, WIDE, B1, needed)]
    if not holds(table, WIDE):
        selection = "F, the rule is not within 1 point"
    elif against[0]["holds"]:
        selection = "H+, the rule holds and beats the recipe rule"
    else:
        selection = "H, the rule holds"
    hypothesis = ("supported: validation statistics choose better than training "
                  "statistics" if against[1]["holds"] else "not supported")
    return selection, hypothesis, pd.DataFrame(against)


def regrets(frozen: dict, scores: pd.DataFrame, metrics: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regret of every single choice and of the shortlists of size 3, per run."""
    recipe = analyze_h.recipes_of(scores.index)
    kind = recipe.map(FAMILY)
    single = pd.DataFrame({"recipe": recipe})
    for name in RULE_NAMES:
        single[name] = analyze_h.regret_of(scores, frozen["rules"][name].choose(metrics))
    single[RECIPE_RULE] = analyze_h.regret_of(
        scores, [frozen[RECIPE_RULE][k] for k in kind])
    single[B1] = analyze_h.regret_of(scores, [frozen[B1]] * len(scores))
    short = pd.DataFrame({
        "recipe": recipe,
        "fixed": sl.shortlist_regret(scores, frozen["shortlists"][str(SHORT)]),
        "rule": sl.shortlist_regret(scores, analyze_h.top(frozen["rules"][WIDE],
                                                          metrics, SHORT)),
        "random": sl.random_regret(scores, SHORT)})
    return single, short


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("freeze", "test"))
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--rules", default=str(RULES_T_PATH))
    parser.add_argument("--allow-other-rules", action="store_true",
                        help="accept a rules file with another hash (tests only)")
    parser.add_argument("--out", default=str(HERE / "outputs" / "stage_t_report.md"))
    args = parser.parse_args()
    root, path = Path(args.results_root), Path(args.rules)
    sections = []
    if args.step == "freeze":
        if path.exists():
            raise SystemExit(f"{path} exists. The rules are frozen once")
        tables = earlier_tables(root)
        wanted = {spec.name for spec in STAGES[TRAIN_STAGE]}
        ready = set(tables.descriptors.loc[tables.descriptors["group"] == VALID,
                                           "run"])
        if wanted - ready:
            raise SystemExit("runs without validation statistics: "
                             f"{sorted(wanted - ready)}")
        digest = rules.save(build(tables), path)
        frozen = load(path)
        sizes = {name: len(rule.columns) for name, rule in frozen["rules"].items()}
        sections.append(("Frozen file", (
            f"- File: `{path}`\n- SHA-256: `{digest}`\n- Training runs: "
            f"{len(frozen['training_runs'])}, cells: {frozen['training_cells']}.\n"
            f"- Inputs of each rule: {sizes}.\n- One score: {frozen[B1]}. Recipe "
            f"rule: {frozen[RECIPE_RULE]}.\n- Shortlists: {frozen['shortlists']}.")))
    else:
        digest = rules.file_hash(path)
        if digest != RULES_T_SHA256 and not args.allow_other_rules:
            raise SystemExit(f"the rules file has hash {digest}, not the frozen one")
        frozen = load(path)
        new = analyze_h.read_stage(root, (TEST_STAGE,))
        if new.metrics.empty:
            raise SystemExit(f"no results of stage {TEST_STAGE} under {root}/runs")
        meta, scores, metrics = cells(new)
        mask = analyze_c.final_mask(meta)
        scores, metrics = scores[mask].droplevel("epoch"), metrics[mask].droplevel(
            "epoch")
        accuracy = metrics["clock/val_accuracy"]
        recipe = analyze_h.recipes_of(scores.index)
        failed = sorted(set(recipe[accuracy < MIN_ACCURACY]))
        keep = ~recipe.isin(failed).to_numpy()
        single, short = regrets(frozen, scores[keep], metrics[keep])
        selection, hypothesis, details = decide(single)
        reading, difference = analyze_h.read_shortlists(short)
        gate = pd.DataFrame({"recipe": recipe, "val_accuracy": accuracy})
        sections += [
            ("Runs and gate T", (
                f"{len(scores)} of {len(STAGES[TEST_STAGE])} runs. A run passes with a "
                f"validation accuracy of at least {MIN_ACCURACY:.0%}:\n\n"
                + block(gate.reset_index(), 3) + "\n\nRecipes that leave the "
                f"decision rule: {', '.join(failed) or 'none'}.")),
            ("Question 1: selection", (
                "Regret at the final checkpoint, in AUROC points on the far sets:"
                "\n\n" + block(single.reset_index(), 2) + "\n\nBy recipe:\n\n"
                + block(single.groupby("recipe").mean().reset_index(), 2) + "\n\n"
                + block(details, 2) + f"\n\n**Outcome: {selection}.**")),
            ("Question 2: validation against training statistics", (
                f"**The hypothesis is {hypothesis}.** The comparison is the second "
                "row of the table above.")),
            ("Shortlists of size 3", (
                block(short.reset_index(), 2) + "\n\nMeans: "
                + ", ".join(f"{name} {short[name].mean():.2f}"
                            for name in short.columns[1:])
                + f". The rule minus the fixed shortlist: {difference[0]:.2f} "
                f"({difference[1]:.2f} to {difference[2]:.2f}).\n\n"
                f"**Reading: {reading}.**"))]
    text = ("# Amendment A7: a selection rule from validation statistics\n\nThe "
            "statistics, the rules, and the decision rule were fixed before the "
            "new runs.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
