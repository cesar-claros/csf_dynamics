"""Second test of the all-metrics rule on new recipes (amendment A6).

Applies the frozen rule `all` of amendment A5 to the runs of stage `c2`, without
a new fit, and reads the result with the decision rule of amendment A6.

Run from code/ after copying the run folders from the HPC:

    python -m dynamics.analyze_c2 --results-root dyn_results

Protocol: ../documentation/stage_c2_all_metrics_amendment_2026-11-14.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_c, analyze_e, rules
from dynamics import analyze_multimetric as mm
from dynamics.analyze_c import B1, B2, compare, final_mask, run_table
from dynamics.analyze_e import block
from dynamics.config import (
    DEFAULT_PRECISION,
    PRECISIONS,
    STAGES,
    parse_run,
    results_folder,
)

HERE = Path(__file__).resolve().parent
STAGE = "c2"
#: SHA-256 of the second rules file, recorded in amendment A6 before any run.
RULES_V2_SHA256 = "3529c358505b41229ec01ae73760213411ecf6552c87c87262744efa71abacce"
PRIMARY = "all"
SECOND = "all_27_runs"
B3 = "B3"
#: The recipe rule, fixed in amendment A6 before the runs.
B3_SCORES = {"plain": "PCA_RecError", "augmented": "MahaPP"}
HOLD_LIMIT = 1.0
RECIPES_OVER = 2


def with_rule(frame: pd.DataFrame, name: str, picks: list, index: pd.MultiIndex,
              scores: pd.DataFrame) -> pd.DataFrame:
    """Add the rows of one more rule to a table of `analyze_c.choices`."""
    best = scores.loc[index].max(axis=1)
    rows = [{"run": run, "epoch": epoch, "recipe": parse_run(run).recipe,
             "rule": name, "choice": pick if isinstance(pick, str) else None,
             "regret": (best[run, epoch] - scores.loc[(run, epoch), pick]
                        if isinstance(pick, str) else np.nan)}
            for (run, epoch), pick in zip(index, picks)]
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)


def holds(table: pd.DataFrame, rule: str) -> bool:
    """Within the limit on the mean, and over it in fewer than 2 recipes."""
    by_recipe = table.groupby("recipe")[rule].mean()
    return bool(table[rule].mean() <= HOLD_LIMIT
                and (by_recipe > HOLD_LIMIT).sum() < RECIPES_OVER)


def decide(table: pd.DataFrame, rule: str = PRIMARY) -> tuple[str, pd.DataFrame]:
    """The decision rule of amendment A6 on the runs of the stage."""
    if table.empty or table[[rule, B1, B3]].isna().any().any():
        return "not decided: a run has no choice", pd.DataFrame()
    details = pd.DataFrame([compare(table, rule, B3), compare(table, rule, B1)])
    if not holds(table, rule):
        return "F, the lead fails: the rule is not within 1 point", details
    if details.loc[0, "holds"]:
        return "H+, the metrics add to the recipe: the rule beats the recipe rule", details
    recipe_rule = ("the recipe rule holds too" if holds(table, B3)
                   else "the recipe rule does not hold")
    return f"H, the rule holds, and {recipe_rule}", details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--allow-other-rules", action="store_true",
                        help="accept rules files with other hashes (tests only)")
    parser.add_argument("--precision", choices=sorted(PRECISIONS),
                        default=DEFAULT_PRECISION)
    parser.add_argument("--out", default=str(HERE / "outputs" / "stage_c2_report.md"))
    args = parser.parse_args()
    hashes = (rules.file_hash(rules.RULES_PATH), rules.file_hash(rules.RULES_V2_PATH))
    if hashes != (analyze_c.RULES_SHA256, RULES_V2_SHA256) and not (
            args.allow_other_rules):
        raise SystemExit(f"a rules file is not the frozen one: {hashes}")
    first, second = rules.load(rules.RULES_PATH), rules.load(rules.RULES_V2_PATH)
    root = Path(args.results_root)
    wanted = {spec.name for spec in STAGES[STAGE]}
    step0 = analyze_e.read_runs(root, results_folder(args.precision,
                                                     suite=analyze_c.SUITE))

    def only(names: set[str]) -> analyze_e.Tables:
        return analyze_e.Tables(**{
            key: frame if frame.empty else frame[frame["run"].isin(names)]
            .reset_index(drop=True) for key, frame in vars(step0).items()})

    new = only(wanted)
    if new.metrics.empty:
        raise SystemExit(f"no results of stage {STAGE} under {root}/runs")
    earlier = rules.training_tables(root, args.precision, with_stage_c=True)
    found = sorted(new.metrics["run"].unique())
    out_dir = Path(args.out).parent / "stage_c2"
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = [("Runs and rules", (
        f"- Runs: {len(found)} of {len(wanted)}. Missing: "
        f"{', '.join(sorted(wanted - set(found))) or 'none'}.\n- Rules files: "
        f"`{hashes[0][:12]}` (the rule under test) and `{hashes[1][:12]}` (the "
        "second rule)" + ("" if hashes == (analyze_c.RULES_SHA256, RULES_V2_SHA256)
                          else " (NOT the frozen files)") + "."))]
    for name in first["targets"]:
        target = first["targets"][name]
        sets = tuple(target["sets"])
        meta, scores, metrics = mm.build_cells(new, sets)
        _, old_scores, old_metrics = mm.build_cells(earlier, sets)
        mask = final_mask(meta)
        frame = analyze_c.choices(target, meta, scores, metrics, mask, None,
                                  (old_metrics, old_scores))
        index = metrics[mask].index
        frame = with_rule(frame, B3, [B3_SCORES[kind] for kind
                                      in meta.loc[mask, "family"]], index, scores)
        frame = with_rule(frame, SECOND, second["targets"][name]["rules"][PRIMARY]
                          .choose(metrics[mask]), index, scores)
        frame.to_csv(out_dir / f"choices_{name}_final.csv", index=False)
        table = run_table(frame)
        picks = frame.pivot_table(index="run", columns="rule", values="choice",
                                  aggfunc="first")
        outcome, details = decide(table)
        other, _ = decide(table, SECOND)
        order = [PRIMARY, SECOND, B3, B1, B2, "geometry", "clock", analyze_c.NEAREST]
        body = (f"Sets: {', '.join(sets)}. Regret at the final checkpoint, in AUROC "
                "points:\n\n" + block(table[["recipe", *order]].reset_index(), 2)
                + "\n\nBy recipe:\n\n"
                + block(table.groupby("recipe")[order].mean().reset_index(), 2)
                + "\n\nChoices:\n\n" + block(picks[order].reset_index())
                + "\n\nThe rule against the recipe rule and against one score:\n\n"
                + block(details, 2) + f"\n\n**Outcome: {outcome}.**\n\nWith the "
                f"second rule (27 training runs): {other}.")
        sections.append((f"Target: {name}" + (" (primary)" if name == "far" else
                                              " (secondary)"), body))
        if name == "far":
            again = analyze_c.leave_one_recipe_out(
                [(meta, scores, metrics), mm.build_cells(earlier, sets)])
            sections.append(("Secondary: one recipe left out", (
                "The rules are fitted again on the other recipes. This is not a "
                "test of the frozen rule:\n\n" + block(mm.summarize(again), 2))))
    text = ("# Second test of the all-metrics rule on new recipes\n\nThe rule, the "
            "baselines, and the decision rule were fixed before these runs "
            "(amendment A6).\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
