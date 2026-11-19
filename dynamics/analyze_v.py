"""Amendments A9 and A10: selection with the correct-only target (freeze and test).

    python -m dynamics.analyze_v freeze --part 1 --results-root dyn_results
    python -m dynamics.analyze_v test --part 1 --results-root dyn_results

The target is the AUROC of the correct ID validation images against the unseen
images, as a mean over the far sets. Each part has its own frozen file:

- Part 1 (A9, 10 classes): the all-tiers rule on the recipes V1 to V6.
- Part 2 (A9, 100 classes): the all-metrics rule on the recipes W1 to W4.
- Part 3 (A10, 10 classes, no label smoothing): the all-tiers rule, fitted on the
  earlier runs without label smoothing, on the recipes X1 to X6.

`freeze` fits the rule and the baselines of a part on the earlier runs, where the
target comes from stored values. `test` reads the new runs, where the evaluation
stores the target directly.

Protocols: ../documentation/correct_only_selection_amendment_2026-11-18.md and
../documentation/no_label_smoothing_amendment_2026-11-18.md
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_augrc as ag
from dynamics import analyze_c, analyze_e, analyze_h, rules
from dynamics import analyze_shortlist as sl
from dynamics import analyze_t as at
from dynamics import analyze_u as au
from dynamics.analyze_e import block
from dynamics.analyze_selection import FAMILY
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import STAGES
from dynamics.descriptors import VALID

HERE = Path(__file__).resolve().parent
RULE, ONE, RECIPE_RULE, STAND_IN = "rule", at.B1, at.RECIPE_RULE, au.STAND_IN
#: Part 1: the same rule, fitted on the 36 runs of amendment A7 only (secondary).
SMALL = "fit_36_runs"
#: Part 3: the same rule, fitted on all earlier 10-class runs (secondary).
WIDE_FIT = "fit_all_runs"
HOLD = 1.0
MIN_ACCURACY = 0.60
#: Gate M: largest difference between the stored target and the derived one.
METRIC_LIMIT = 1e-6
SHORT = 3


@dataclass(frozen=True)
class Part:
    """One part of the amendment.

    `rule` is the name of the rule under test in the frozen file. `earlier` is a
    frozen rule of an earlier test, read as a secondary rule: its label, file,
    hash, and the name of the rule in that file.
    """

    amendment: str
    label: str
    train_stage: str
    test_stage: str
    path: Path
    sha256: str
    rule: str
    earlier: tuple[str, Path, str, str]


#: SHA-256 of the two frozen files. Set after `freeze`, before any new run.
RULES_V_SHA256 = "0c2e23a62678a3d34568b12a588f673eb721552f9819ef84676f4f19c1f960ba"
RULES_W_SHA256 = "254ddf23b0532a8260dff7830997f7c8b9d4f681006d81aea22b8e0162562b66"
RULES_X_SHA256 = "d64c050265cfb9d5a787d6e39acb9ad6d3d3b709a260305948e6b7d99859fed6"
RULES_V_PATH = HERE / "frozen" / "stage_v_rules.json"
PARTS = {
    1: Part("A9", "part 1, 10 classes", "a9_train", "v", RULES_V_PATH,
            RULES_V_SHA256, at.TIERS,
            ("a7_rule", at.RULES_T_PATH, at.RULES_T_SHA256, at.TIERS)),
    2: Part("A9", "part 2, 100 classes", "a9_train_h", "w",
            HERE / "frozen" / "stage_w_rules.json", RULES_W_SHA256, analyze_h.RULE,
            ("stage_h_rule", analyze_h.RULES_H_PATH, analyze_h.RULES_H_SHA256,
             analyze_h.RULE)),
    3: Part("A10", "10 classes, no label smoothing", "a10_train", "x",
            HERE / "frozen" / "stage_x_rules.json", RULES_X_SHA256, at.TIERS,
            ("a9_rule", RULES_V_PATH, RULES_V_SHA256, at.TIERS))}


def join(first: analyze_e.Tables, second: analyze_e.Tables) -> analyze_e.Tables:
    return analyze_e.Tables(**{
        key: pd.concat([frame, other], ignore_index=True)
        for (key, frame), other in zip(vars(first).items(), vars(second).values())})


def ten_class_tables(root: Path, with_v: bool = False) -> analyze_e.Tables:
    """The earlier 10-class runs: the 52 of amendment A9, and stage `v` if asked."""
    tables = join(au.training_tables(root),
                  analyze_h.read_stage(root, (au.TEST_STAGE,)))
    if with_v:
        tables = join(tables, analyze_h.read_stage(root, (PARTS[1].test_stage,)))
    return tables


def training_tables(number: int, root: Path) -> analyze_e.Tables:
    """The runs that the rule of a part is fitted on.

    Part 1 has 52 runs with 10 classes, part 2 has 16 runs with 100 classes, and
    part 3 has the 10-class runs without label smoothing.
    """
    if number == 2:
        return analyze_h.read_stage(root, (analyze_h.PHASE_1, analyze_h.PHASE_2))
    if number == 1:
        return ten_class_tables(root)
    keep = {spec.name for spec in STAGES[PARTS[number].train_stage]}
    tables = ten_class_tables(root, with_v=True)
    return analyze_e.Tables(**{
        key: frame if frame.empty else frame[frame["run"].isin(keep)]
        .reset_index(drop=True) for key, frame in vars(tables).items()})


def build(number: int, root: Path) -> dict:
    """The rule under test and the baselines of a part, for the correct-only target."""
    part = PARTS[number]
    tables = training_tables(number, root)
    n_id, n_unseen = ag.counts(root, sorted(set(tables.metrics["run"])))
    aimed = ag.with_target(tables, n_id, n_unseen, ag.CORRECT_ONLY)
    if number == 1:
        built = at.build(aimed, names=(part.rule,), amendment="A9, part 1")
        small = at.build(ag.with_target(at.earlier_tables(root), n_id, n_unseen,
                                        ag.CORRECT_ONLY), names=(part.rule,))
        built["rules"][SMALL] = small["rules"][part.rule]
    elif number == 3:
        built = at.build(aimed, names=(part.rule,), amendment="A10")
        wide = at.build(ag.with_target(ten_class_tables(root, with_v=True), n_id,
                                       n_unseen, ag.CORRECT_ONLY),
                        names=(part.rule,))
        built["rules"][WIDE_FIT] = wide["rules"][part.rule]
    else:
        found = analyze_h.build(aimed)
        built = {
            "amendment": "A9, part 2", "sets": found["sets"],
            "penalty": found["penalty"], "training_runs": found["training_runs"],
            "training_cells": found["training_cells"],
            "baseline_runs": len(found["training_runs"]),
            "rules": {part.rule: found["rules"][part.rule]}, ONE: found["b1"],
            RECIPE_RULE: found["recipe_rule"], "shortlists": found["shortlists"]}
    return {**built, "target": ag.CORRECT_ONLY,
            "images": {"id": n_id, "unseen": n_unseen}}


def metric_gap(tables: analyze_e.Tables, n_id: int, n_unseen: int
               ) -> tuple[float, int]:
    """Gate M: the stored correct-only AUROC against the one from the old values.

    Returns:
        The largest difference over the far-set rows from epoch 1 where both
        values are finite, and the number of rows that could not be compared.

    Raises:
        SystemExit: If the runs do not store the correct-only AUROC.
    """
    if "auroc_correct" not in tables.metrics:
        raise SystemExit("the runs do not store the correct-only values")
    derived = ag.with_target(tables, n_id, n_unseen, ag.CORRECT_ONLY).metrics
    rows = (derived["set"].isin(FAR_SETS) & (derived["epoch"] >= 1)).to_numpy()
    stored = tables.metrics.loc[rows, "auroc_correct"].to_numpy(dtype=np.float64)
    other = derived.loc[rows, "auroc"].to_numpy(dtype=np.float64)
    both = np.isfinite(stored) & np.isfinite(other)
    if not both.any():
        raise SystemExit("no row with a stored and a derived target")
    return float(np.abs(stored[both] - other[both]).max()), int((~both).sum())


def with_stored_target(tables: analyze_e.Tables) -> analyze_e.Tables:
    """The same tables, with the stored correct-only AUROC as the far `auroc`."""
    metrics = tables.metrics.copy()
    far = metrics["set"].isin(FAR_SETS).to_numpy()
    metrics.loc[far, "auroc"] = metrics.loc[far, "auroc_correct"]
    return analyze_e.Tables(**{**vars(tables), "metrics": metrics})


def choices(number: int, frozen: dict, earlier: dict, metrics: pd.DataFrame,
            csfs: list[str]) -> pd.DataFrame:
    """The score that each single choice takes, per run."""
    part = PARTS[number]
    kind = analyze_h.recipes_of(metrics.index).map(FAMILY)
    picks = pd.DataFrame({
        RULE: frozen["rules"][part.rule].choose(metrics),
        ONE: [frozen[ONE]] * len(metrics),
        RECIPE_RULE: [frozen[RECIPE_RULE][k] for k in kind],
        STAND_IN: au.stand_in_choice(metrics, csfs)}, index=metrics.index)
    for name, rule in frozen["rules"].items():        # secondary rules of the file
        if name != part.rule:
            picks[name] = rule.choose(metrics)
    picks[part.earlier[0]] = earlier["rules"][part.earlier[3]].choose(metrics)
    return picks


def regrets(number: int, frozen: dict, picks: pd.DataFrame, scores: pd.DataFrame,
            metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regret of every single choice and of the shortlists of size 3, per run."""
    recipe = analyze_h.recipes_of(scores.index)
    single = pd.DataFrame({"recipe": recipe})
    for name in picks.columns:
        single[name] = analyze_h.regret_of(scores, list(picks[name]))
    short = pd.DataFrame({
        "recipe": recipe,
        "fixed": sl.shortlist_regret(scores, frozen["shortlists"][str(SHORT)]),
        "rule": sl.shortlist_regret(scores, analyze_h.top(
            frozen["rules"][PARTS[number].rule], metrics, SHORT)),
        "random": sl.random_regret(scores, SHORT)})
    return single, short


def holds(table: pd.DataFrame, name: str) -> bool:
    """Within 1 point on the mean, and above it in fewer than half of the recipes."""
    by_recipe = table.groupby("recipe")[name].mean()
    return bool(table[name].mean() <= HOLD
                and 2 * (by_recipe > HOLD).sum() < len(by_recipe))


def decide(table: pd.DataFrame) -> dict:
    """The readings of questions 1 and 2 of amendment A9."""
    if table.empty or table[[RULE, ONE]].isna().any().any():
        return {"selection": "not decided: a run has no choice", "adds": "not decided",
                "enough": "not decided"}
    difference = table[RULE] - table[ONE]
    mean, low, high = analyze_c.interval(difference.to_numpy())
    supported = bool(mean < 0 and high < 0)
    by_recipe = difference.groupby(table["recipe"]).sum()
    share = float(by_recipe.min() / difference.sum()) if difference.sum() < 0 else 0.0
    adds = "not supported"
    if supported and share > 0.5:
        adds = (f"supported, and the rule avoids one bad case of the fixed score: "
                f"{by_recipe.idxmin()} gives {share:.0%} of the difference")
    elif supported:
        adds = "supported, and no recipe gives more than half of the difference"
    return {
        "selection": ("H, the rule holds" if holds(table, RULE)
                      else "F, the rule is not within 1 point"),
        "adds": adds,
        "enough": ("a fixed score is enough" if holds(table, ONE)
                   else "the fixed score does not hold"),
        "difference": (mean, low, high),
        "recipes_lower": int((table.groupby("recipe")[RULE].mean()
                              < table.groupby("recipe")[ONE].mean()).sum())}


def augrc_table(tables: analyze_e.Tables, picks: pd.DataFrame) -> pd.DataFrame:
    """Correct-only AUGRC of the rule's choice, the fixed score, and the best score.

    In points, as a mean over the far sets at the final checkpoint. Lower is
    better. Empty if the runs do not store the value.
    """
    if "augrc_correct" not in tables.metrics:
        return pd.DataFrame()
    part = tables.metrics[tables.metrics["set"].isin(FAR_SETS)]
    last = part.groupby("run")["epoch"].transform("max") == part["epoch"]
    wide = 100 * part[last].pivot_table(index="run", columns="csf",
                                        values="augrc_correct")
    rows = []
    for run in picks.index:
        row = {"run": run, "best": wide.loc[run].min()}
        for name in (RULE, ONE):
            pick = picks.loc[run, name]
            row[name] = wide.loc[run, pick] if isinstance(pick, str) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def check(path: Path, wanted: str, allow: bool) -> dict:
    digest = rules.file_hash(path)
    if digest != wanted and not allow:
        raise SystemExit(f"{path.name} has hash {digest}, not the frozen one")
    return at.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("freeze", "test"))
    parser.add_argument("--part", type=int, choices=sorted(PARTS), required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--rules", help="another rules file of the part (tests only)")
    parser.add_argument("--allow-other-rules", action="store_true",
                        help="accept rules files with another hash (tests only)")
    parser.add_argument("--stage", help="read another stage (dry runs only)")
    parser.add_argument("--derived-target", action="store_true",
                        help="take the target from the old stored values and skip "
                        "gate M (dry runs on earlier runs only)")
    parser.add_argument("--out")
    args = parser.parse_args()
    part = PARTS[args.part]
    root = Path(args.results_root)
    path = Path(args.rules) if args.rules else part.path
    out = Path(args.out) if args.out else (
        HERE / "outputs" / f"stage_{part.test_stage}_report.md")
    sections = []
    if args.step == "freeze":
        if path.exists():
            raise SystemExit(f"{path} exists. The rules are frozen once")
        tables = training_tables(args.part, root)
        wanted = {spec.name for spec in STAGES[part.train_stage]}
        ready = set(tables.metrics["run"])
        if args.part != 2:
            rows = tables.descriptors
            ready = set(rows.loc[rows["group"] == VALID, "run"])
        if wanted - ready:
            raise SystemExit(f"runs without results: {sorted(wanted - ready)}")
        digest = rules.save(build(args.part, root), path)
        frozen = at.load(path)
        sections.append((f"Frozen file of {part.label}", (
            f"- File: `{path}`\n- SHA-256: `{digest}`\n- Target: {frozen['target']}, "
            f"with {frozen['images']} images for each far set.\n- Training runs: "
            f"{len(frozen['training_runs'])}, cells: {frozen['training_cells']}, "
            f"baseline runs: {frozen['baseline_runs']}.\n- Rule under test: "
            f"`{part.rule}` with {len(frozen['rules'][part.rule].columns)} inputs. "
            f"Rules in the file: {sorted(frozen['rules'])}.\n- One score: "
            f"{frozen[ONE]}. Recipe rule: {frozen[RECIPE_RULE]}.\n- Shortlists: "
            f"{frozen['shortlists']}.")))
    else:
        frozen = check(path, part.sha256, args.allow_other_rules)
        earlier = check(part.earlier[1], part.earlier[2], args.allow_other_rules)
        stage = args.stage or part.test_stage
        new = analyze_h.read_stage(root, (stage,))
        if new.metrics.empty:
            raise SystemExit(f"no results of stage {stage} under {root}/runs")
        n_id, n_unseen = ag.counts(root, sorted(set(new.metrics["run"])))
        if frozen["images"] != {"id": n_id, "unseen": n_unseen}:
            raise SystemExit("the image counts are not those of the frozen file")
        if args.derived_target:
            aimed = ag.with_target(new, n_id, n_unseen, ag.CORRECT_ONLY)
            gate_m = "Dry run: the target comes from the old stored values."
        else:
            gap, skipped = metric_gap(new, n_id, n_unseen)
            if gap > METRIC_LIMIT:
                raise SystemExit(f"gate M fails: the stored and the derived target "
                                 f"differ by {gap:.2e}")
            aimed = with_stored_target(new)
            gate_m = (f"**Gate M passes.** Largest difference between the stored "
                      f"and the derived target: {gap:.2e} (limit {METRIC_LIMIT:.0e}"
                      f"). Rows that could not be compared: {skipped}.")
        scores, metrics = ag.final(aimed)
        old_scores, _ = ag.final(new)
        accuracy = metrics["clock/val_accuracy"]
        recipe = analyze_h.recipes_of(scores.index)
        failed = sorted(set(recipe[accuracy < MIN_ACCURACY]))
        keep = ~recipe.isin(failed).to_numpy()
        scores, metrics, old_scores = scores[keep], metrics[keep], old_scores[keep]
        picks = choices(args.part, frozen, earlier, metrics, list(scores.columns))
        single, short = regrets(args.part, frozen, picks, scores, metrics)
        old_single, _ = regrets(args.part, frozen, picks, old_scores, metrics)
        result = decide(single)
        reading, difference = analyze_h.read_shortlists(short)
        gate = pd.DataFrame({"recipe": analyze_h.recipes_of(accuracy.index),
                             "val_accuracy": accuracy})
        names = [c for c in single.columns if c != "recipe"]
        augrc = augrc_table(new, picks)
        mean, low, high = result.get("difference", (np.nan, np.nan, np.nan))
        sections += [
            ("Runs and gates", (
                f"{len(accuracy)} of {len(STAGES[stage])} runs. Gate T: a run passes "
                f"with a validation accuracy of at least {MIN_ACCURACY:.0%}:\n\n"
                + block(gate.reset_index(), 3) + "\n\nRecipes that leave the "
                f"decision rule: {', '.join(failed) or 'none'}.\n\n" + gate_m)),
            ("Question 1: selection", (
                f"Rule under test: `{part.rule}` of `{path.name}`. One score: "
                f"{frozen[ONE]}. Regret at the final checkpoint, in points of the "
                "correct-only AUROC on the far sets:\n\n"
                + block(single.reset_index(), 2) + "\n\nBy recipe:\n\n"
                + block(single.groupby("recipe").mean().reset_index(), 2)
                + "\n\nMeans: " + ", ".join(
                    f"{name} {single[name].mean():.2f}" for name in names)
                + "\n\nChoices:\n\n" + block(picks.reset_index())
                + f"\n\n**Outcome: {result['selection']}.**")),
            ("Question 2: the rule against the fixed score", (
                f"The rule minus the one fixed score: {mean:.2f} ({low:.2f} to "
                f"{high:.2f}). The rule is lower in "
                f"{result.get('recipes_lower', 0)} of {single['recipe'].nunique()} "
                f"recipes.\n\n**The claim that the rule adds is {result['adds']}.** "
                f"**Fixed score: {result['enough']}.**")),
            ("Shortlists of size 3", (
                block(short.reset_index(), 2) + "\n\nMeans: "
                + ", ".join(f"{name} {short[name].mean():.2f}"
                            for name in short.columns[1:])
                + f". The rule minus the fixed shortlist: {difference[0]:.2f} "
                f"({difference[1]:.2f} to {difference[2]:.2f}).\n\n"
                f"**Reading: {reading}.**")),
            ("Other values, no outcome", (
                "Regret of the same choices with the target of the earlier tests "
                "(all ID images), by recipe:\n\n"
                + block(old_single.groupby("recipe").mean().reset_index(), 2)
                + ("" if augrc.empty else
                   "\n\nCorrect-only AUGRC in points (lower is better) of the "
                   "choice of the rule, of the fixed score, and of the best "
                   "score:\n\n" + block(augrc, 2))))]
    text = (f"# Amendment {part.amendment}, {part.label}: selection with the "
            "correct-only target"
            "\n\nThe target, the rule, the baselines, and the decision rule were "
            "fixed before the new runs.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
