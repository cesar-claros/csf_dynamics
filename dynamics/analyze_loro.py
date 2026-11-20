"""Exploratory check: do ID-only metrics help to select a score with 100 classes?

    python -m dynamics.analyze_loro --results-root dyn_results

Each recipe is left out once. A ridge rule is fitted on every checkpoint of the
other recipes, and it chooses a score at the final checkpoint of the runs of the
recipe that is left out. The check repeats this for each set of inputs: training
statistics, validation statistics, the statistics of each score on the validation
set, all metrics, and the clock. Choices with no metric have the same split: one
fixed score, a score for each kind of recipe, a score for recipes with and
without label smoothing, and the stand-in choice.

The check uses the 100-class runs, with the correct-only target and with all ID
images. Every analysis has seen these runs, and a result of this kind did not
hold in a planned test before, so it is a description.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dynamics import analyze_augrc as ag
from dynamics import analyze_c, analyze_h, rules
from dynamics import analyze_multimetric as mm
from dynamics import analyze_t as at
from dynamics import analyze_u as au
from dynamics import analyze_valonly as vo
from dynamics.analyze_e import block
from dynamics.config import RECIPES, parse_run

HERE = Path(__file__).resolve().parent
STAGES_100 = ("h1", "h2", "w", "y")
ONE, KIND, REGIME, STAND_IN = "one_score", "by_kind", "by_smoothing", "stand_in"
LIMIT = 1.0


def input_groups(columns: list[str]) -> dict[str, list[str]]:
    """The sets of inputs, by what a metric reads."""
    score = [c for c in columns if c.startswith("score/")]
    derived = [c for c in columns if c.startswith("derived/")]
    original = [c for c in columns
                if c not in score + derived and not c.startswith("valid/")]
    return {
        "training": [c for c in original if vo.input_set(c) == "training"],
        "validation": [c for c in original + derived
                       if vo.input_set(c) == "validation"],
        "score_statistics": score,
        "all_metrics": original,
        "clock": [c for c in original if c.startswith("clock/")]}


def leave_one_recipe_out(meta: pd.DataFrame, scores: pd.DataFrame,
                         metrics: pd.DataFrame) -> pd.DataFrame:
    """Regret of each choice at the final checkpoint, with its recipe left out."""
    groups = {name: columns for name, columns in input_groups(
        list(mm.transform(metrics).columns)).items() if columns}
    recipe = pd.Series([parse_run(run).recipe
                        for run in meta.index.get_level_values("run")],
                       index=meta.index)
    final = analyze_c.final_mask(meta)
    kind = recipe.map(lambda name: RECIPES[name].augment)
    regime = recipe.map(lambda name: RECIPES[name].label_smoothing > 0)
    parts = []
    for held in sorted(set(recipe)):
        train = (recipe != held).to_numpy()
        test = (recipe == held).to_numpy() & final
        test_scores = scores[test].droplevel("epoch")
        test_metrics = metrics[test].droplevel("epoch")
        rest = scores[train & final]
        out = pd.DataFrame({"recipe": held,
                            "smoothing": RECIPES[held].label_smoothing > 0},
                           index=test_scores.index)
        for name, columns in groups.items():
            rule = rules.fit_ridge_rule(metrics[train], scores[train], columns)
            out[name] = analyze_h.regret_of(test_scores, rule.choose(test_metrics))
        picks = {ONE: rest.mean().idxmax()}
        for label, flag in ((KIND, kind), (REGIME, regime)):
            same = (flag[train & final] == flag[test].iloc[0]).to_numpy()
            picks[label] = (rest[same] if same.any() else rest).mean().idxmax()
        for label, pick in picks.items():
            out[label] = analyze_h.regret_of(test_scores, [pick] * len(test_scores))
        out[STAND_IN] = analyze_h.regret_of(test_scores, au.stand_in_choice(
            test_metrics, list(scores.columns)))
        parts.append(out)
    return pd.concat(parts)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean regret and recipes above 1 point, for all recipes and by smoothing."""
    names = [c for c in table.columns if c not in ("recipe", "smoothing")]
    by_recipe = table.groupby("recipe")[names].mean()
    flag = table.groupby("recipe")["smoothing"].first()
    rows = []
    for label, part in (("all recipes", by_recipe),
                        ("no label smoothing", by_recipe[~flag]),
                        ("label smoothing", by_recipe[flag])):
        for name in names:
            rows.append({"recipes": label, "n": len(part), "choice": name,
                         "mean": part[name].mean(), "worst": part[name].max(),
                         "above_1_point": int((part[name] > LIMIT).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--stages", nargs="+", default=list(STAGES_100))
    parser.add_argument("--out", default=str(HERE / "outputs" / "loro_report.md"))
    args = parser.parse_args()
    root = Path(args.results_root)
    tables = analyze_h.read_stage(root, tuple(args.stages))
    n_id, n_unseen = ag.counts(root, sorted(set(tables.metrics["run"])))
    sections = []
    for target in (ag.CORRECT_ONLY, ag.ALL_ID):
        used = tables if target == ag.ALL_ID else ag.with_target(
            tables, n_id, n_unseen, target)
        meta, scores, metrics = at.cells(used)
        sizes = {name: len(columns) for name, columns in input_groups(
            list(mm.transform(metrics).columns)).items()}
        table = leave_one_recipe_out(meta, scores, metrics)
        sections.append((f"Target {target}", (
            f"{table.index.nunique()} runs of {table['recipe'].nunique()} recipes, "
            f"{len(meta)} checkpoints. Inputs of each rule: {sizes}.\n\n"
            + block(summary(table), 2) + "\n\nBy recipe:\n\n"
            + block(table.groupby("recipe").mean().reset_index(), 2))))
    text = ("# Exploratory check: ID-only metrics with one recipe left out\n\nPost "
            "hoc, on runs that every analysis has seen.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
