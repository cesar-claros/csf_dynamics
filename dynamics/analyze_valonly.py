"""A selection rule from validation statistics only (post hoc check of 2026-11-17).

The training features collapse, so their statistics can mislead a rule. This
check fits the ridge model of the frozen rules on three sets of inputs
(validation only, training only, and all) and tests each on the same runs.

    python -m dynamics.analyze_valonly --results-root dyn_results

Protocol: ../documentation/imagenet100_selection_protocol_2026-11-14.md, section
"Validation-only rule". This analysis is post hoc.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dynamics import analyze_h, rules
from dynamics import analyze_multimetric as mm
from dynamics import analyze_shortlist as sl
from dynamics.analyze_e import block
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import LAYERS

HERE = Path(__file__).resolve().parent
SETS = ("validation", "training", "all")
MIXED = ("val_over_train", "class_cosine_val", "radial_tail_val",
         "neighbor_radius_ratio")
OTHER = ("clock/log_epoch", "weights/")
VALIDATION = ("clock/val_accuracy", "clock/val_loss", "logits/", "derived/val_nc1")
NEW_10 = ("C1", "C2", "C3", "C4", "C5", "C6")


def input_set(column: str) -> str:
    """`validation`, `training`, `mixed`, or `other`, by what a metric reads."""
    name = column.split("@")[0]
    if any(part in name for part in MIXED):
        return "mixed"
    if name.startswith(OTHER):
        return "other"
    if name.startswith(VALIDATION) or name.endswith("_val"):
        return "validation"
    return "training"


def with_validation_nc1(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add the validation NC1 of each stage: the stored ratio times the training NC1."""
    out = metrics.copy()
    for layer in LAYERS:
        ratio = f"gap/var_collapse_val_over_train@{layer}"
        base = f"nc/var_collapse@{layer}"
        if ratio in out and base in out:
            out[f"derived/val_nc1@{layer}"] = out[ratio] * out[base]
    return out


def evaluate(train: tuple[pd.DataFrame, pd.DataFrame],
             test: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """Per test run and input set: regret of the single choice and of the top 3."""
    scores, metrics = train[0], with_validation_nc1(train[1])
    test_scores, test_metrics = test[0], with_validation_nc1(test[1])
    usable = list(mm.transform(metrics).columns)
    rows = []
    for name in SETS:
        columns = [c for c in usable if name == "all" or input_set(c) == name]
        rule = rules.fit_ridge_rule(metrics, scores, columns)
        single = analyze_h.regret_of(test_scores, rule.choose(test_metrics))
        short = sl.shortlist_regret(test_scores, analyze_h.top(rule, test_metrics, 3))
        for run in test_scores.index:
            rows.append({"run": run, "recipe": analyze_h.recipes_of([run]).iloc[0],
                         "inputs": name, "n_inputs": len(columns),
                         "single": single[run], "top3": short[run]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", default=str(HERE / "outputs" / "valonly_report.md"))
    args = parser.parse_args()
    root = Path(args.results_root)
    _, scores, metrics = mm.build_cells(rules.training_tables(root), FAR_SETS)
    new = analyze_h.read_stage(root, ("c", "c2"))
    test_scores, test_metrics = sl.final_scores(new, FAR_SETS)
    keep = analyze_h.recipes_of(test_scores.index).isin(NEW_10).to_numpy()
    ten = evaluate((scores, metrics), (test_scores[keep], test_metrics[keep]))
    _, scores, metrics = mm.build_cells(analyze_h.read_stage(root, ("h1",)), FAR_SETS)
    hundred = evaluate((scores, metrics),
                       sl.final_scores(analyze_h.read_stage(root, ("h2",)), FAR_SETS))
    sections = []
    for title, table in (("10 classes: 12 training runs, 18 test runs", ten),
                         ("100 classes: 10 training runs, 6 test runs", hundred)):
        means = table.groupby("inputs")[["n_inputs", "single", "top3"]].mean()
        by_recipe = table.pivot_table(index="recipe", columns="inputs",
                                      values="single")[list(SETS)]
        sections.append((title, (
            "Mean regret in AUROC points on the far sets, at the final checkpoint:"
            "\n\n" + block(means.loc[list(SETS)].reset_index(), 2)
            + "\n\nSingle choice by recipe:\n\n" + block(by_recipe.reset_index(), 2))))
    text = ("# A selection rule from validation statistics only\n\nPost hoc. The "
            "input sets and the reading were fixed before the computation.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
