"""Amendment A8: the all-tiers rule on new recipes (second fit, and test).

    python -m dynamics.analyze_u freeze --results-root dyn_results
    python -m dynamics.analyze_u test --results-root dyn_results

The rule under test is the all-tiers rule of the file that amendment A7 froze. It
is not fitted again. `freeze` writes a second fit of the same rule, with the eight
runs of stage `t` added, as a secondary rule. `test` applies both files to the
runs with the new recipes and reads the result.

Protocol: ../documentation/all_tiers_test_amendment_2026-11-17.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dynamics import analyze_c, analyze_e, analyze_h, rules
from dynamics import analyze_shortlist as sl
from dynamics import analyze_t as at
from dynamics.analyze_e import block
from dynamics.analyze_selection import FAMILY
from dynamics.config import STAGES
from dynamics.descriptors import VALID

HERE = Path(__file__).resolve().parent
TRAIN_STAGE, TEST_STAGE = "a8_train", "u"
RULES_U_PATH = HERE / "frozen" / "stage_u_rules.json"
#: SHA-256 of the second fit. Set after `freeze`, before any run of the test stage.
RULES_U_SHA256 = "583a97fc0d74b2fb919979d5442506c3603b6d3af5d1ecaf81aa9a85d98cc686"
TIERS, WIDE, TRAINING = at.TIERS, at.WIDE, at.TRAINING
B1, RECIPE_RULE = at.B1, at.RECIPE_RULE
STAND_IN, SECOND = "stand_in", "second_fit"
SECOND_RECIPE, SECOND_B1 = "recipe_rule_second", "one_score_second"


def training_tables(root: Path) -> analyze_e.Tables:
    """The runs of the second fit: the 36 of amendment A7 and the 8 of stage `t`."""
    first = at.earlier_tables(root)
    second = analyze_h.read_stage(root, (at.TEST_STAGE,))
    return analyze_e.Tables(**{
        key: pd.concat([frame, other], ignore_index=True)
        for (key, frame), other in zip(vars(first).items(), vars(second).values())})


def stand_in_choice(metrics: pd.DataFrame, csfs: list[str]) -> list:
    """Per run: the score with the highest mean AUROC over the three stand-in sets.

    This choice has no fitted value. A score without a value on a set leaves the
    choice, and a run without any value has no choice.
    """
    total = pd.DataFrame(0.0, index=metrics.index, columns=csfs)
    for name in at.SCORE_SETS:
        total += metrics.reindex(
            columns=[f"score/{name}/{csf}" for csf in csfs]).to_numpy()
    return [row.idxmax() if row.notna().any() else None
            for _, row in total.iterrows()]


def regrets(first: dict, second: dict, scores: pd.DataFrame, metrics: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regret of every single choice and of the shortlists of size 3, per run.

    `first` is the file of amendment A7, with the rule under test. `second` is the
    second fit of this amendment.
    """
    recipe = analyze_h.recipes_of(scores.index)
    kind = recipe.map(FAMILY)
    single = pd.DataFrame({"recipe": recipe})
    for name in (TIERS, WIDE, TRAINING):
        single[name] = analyze_h.regret_of(scores, first["rules"][name].choose(metrics))
    single[RECIPE_RULE] = analyze_h.regret_of(
        scores, [first[RECIPE_RULE][k] for k in kind])
    single[B1] = analyze_h.regret_of(scores, [first[B1]] * len(scores))
    single[STAND_IN] = analyze_h.regret_of(
        scores, stand_in_choice(metrics, list(scores.columns)))
    single[SECOND] = analyze_h.regret_of(
        scores, second["rules"][TIERS].choose(metrics))
    single[SECOND_RECIPE] = analyze_h.regret_of(
        scores, [second[RECIPE_RULE][k] for k in kind])
    single[SECOND_B1] = analyze_h.regret_of(scores, [second[B1]] * len(scores))
    size = str(at.SHORT)
    short = pd.DataFrame({
        "recipe": recipe,
        "fixed": sl.shortlist_regret(scores, first["shortlists"][size]),
        "rule": sl.shortlist_regret(scores, analyze_h.top(
            first["rules"][TIERS], metrics, at.SHORT)),
        "fixed_second": sl.shortlist_regret(scores, second["shortlists"][size]),
        "rule_second": sl.shortlist_regret(scores, analyze_h.top(
            second["rules"][TIERS], metrics, at.SHORT)),
        "random": sl.random_regret(scores, at.SHORT)})
    return single, short


def decide(table: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    """The outcome of question 1, the reading of question 2, and the comparisons."""
    if table.empty or table[[TIERS, WIDE, RECIPE_RULE]].isna().any().any():
        return "not decided: a run has no choice", "not decided", pd.DataFrame()
    needed = max(1, table["recipe"].nunique() - 1)
    against = [at.compare(table, TIERS, RECIPE_RULE, needed),
               at.compare(table, TIERS, WIDE, needed, or_equal=True),
               at.compare(table, TIERS, B1, needed),
               at.compare(table, TIERS, STAND_IN, needed),
               at.compare(table, SECOND, TIERS, needed, or_equal=True)]
    if not at.holds(table, TIERS):
        selection = "F, the rule is not within 1 point"
    elif against[0]["holds"]:
        selection = "H+, the rule holds and beats the recipe rule"
    else:
        selection = "H, the rule holds"
    adds = ("supported: the statistics of each score add to the validation "
            "statistics" if against[1]["holds"] else "not supported")
    return selection, adds, pd.DataFrame(against)


def check(path: Path, wanted: str, allow: bool) -> dict:
    digest = rules.file_hash(path)
    if digest != wanted and not allow:
        raise SystemExit(f"{path.name} has hash {digest}, not the frozen one")
    return at.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("freeze", "test"))
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--rules", default=str(at.RULES_T_PATH),
                        help="the file of amendment A7, with the rule under test")
    parser.add_argument("--second-rules", default=str(RULES_U_PATH))
    parser.add_argument("--allow-other-rules", action="store_true",
                        help="accept rules files with another hash (tests only)")
    parser.add_argument("--out", default=str(HERE / "outputs" / "stage_u_report.md"))
    args = parser.parse_args()
    root, path = Path(args.results_root), Path(args.second_rules)
    first = check(Path(args.rules), at.RULES_T_SHA256, args.allow_other_rules)
    sections = []
    if args.step == "freeze":
        if path.exists():
            raise SystemExit(f"{path} exists. The rules are frozen once")
        tables = training_tables(root)
        wanted = {spec.name for spec in STAGES[TRAIN_STAGE]}
        ready = set(tables.descriptors.loc[tables.descriptors["group"] == VALID,
                                           "run"])
        if wanted - ready:
            raise SystemExit("runs without validation statistics: "
                             f"{sorted(wanted - ready)}")
        digest = rules.save(at.build(tables, names=(TIERS,), amendment="A8"), path)
        second = at.load(path)
        sections.append(("Second fit of the all-tiers rule", (
            f"- File: `{path}`\n- SHA-256: `{digest}`\n- Training runs: "
            f"{len(second['training_runs'])}, cells: {second['training_cells']}.\n"
            f"- Inputs: {len(second['rules'][TIERS].columns)}, against "
            f"{len(first['rules'][TIERS].columns)} of the rule under test.\n"
            f"- One score: {second[B1]}. Recipe rule: {second[RECIPE_RULE]}.\n"
            f"- Shortlists: {second['shortlists']}.")))
    else:
        second = check(path, RULES_U_SHA256, args.allow_other_rules)
        new = analyze_h.read_stage(root, (TEST_STAGE,))
        if new.metrics.empty:
            raise SystemExit(f"no results of stage {TEST_STAGE} under {root}/runs")
        meta, scores, metrics = at.cells(new)
        mask = analyze_c.final_mask(meta)
        scores, metrics = scores[mask].droplevel("epoch"), metrics[mask].droplevel(
            "epoch")
        accuracy = metrics["clock/val_accuracy"]
        recipe = analyze_h.recipes_of(scores.index)
        failed = sorted(set(recipe[accuracy < at.MIN_ACCURACY]))
        keep = ~recipe.isin(failed).to_numpy()
        single, short = regrets(first, second, scores[keep], metrics[keep])
        selection, adds, details = decide(single)
        reading, difference = analyze_h.read_shortlists(short)
        gate = pd.DataFrame({"recipe": recipe, "val_accuracy": accuracy})
        primary = ["recipe", TIERS, WIDE, TRAINING, RECIPE_RULE, B1, STAND_IN]
        secondary = ["recipe", SECOND, SECOND_RECIPE, SECOND_B1]
        within = ("within" if not single.empty and at.holds(single, SECOND)
                  else "not within")
        sections += [
            ("Runs and gate T", (
                f"{len(scores)} of {len(STAGES[TEST_STAGE])} runs. A run passes with a "
                f"validation accuracy of at least {at.MIN_ACCURACY:.0%}:\n\n"
                + block(gate.reset_index(), 3) + "\n\nRecipes that leave the "
                f"decision rule: {', '.join(failed) or 'none'}.")),
            ("Question 1: selection with the frozen all-tiers rule", (
                "Regret at the final checkpoint, in AUROC points on the far sets:"
                "\n\n" + block(single[primary].reset_index(), 2) + "\n\nBy recipe:"
                "\n\n" + block(single[primary].groupby("recipe").mean()
                               .reset_index(), 2) + "\n\n"
                + block(details, 2) + f"\n\n**Outcome: {selection}.**")),
            ("Question 2: the statistics of each score", (
                f"**The claim is {adds}.** The comparison is the second row of the "
                "table above.")),
            ("Shortlists of size 3", (
                block(short.reset_index(), 2) + "\n\nMeans: "
                + ", ".join(f"{name} {short[name].mean():.2f}"
                            for name in short.columns[1:])
                + f". The rule minus the fixed shortlist: {difference[0]:.2f} "
                f"({difference[1]:.2f} to {difference[2]:.2f}).\n\n"
                f"**Reading: {reading}.**")),
            ("The second fit (secondary, no outcome)", (
                "The same rule fitted again with the eight runs of stage `t`, and "
                "the baselines taken again with those runs:\n\n"
                + block(single[secondary].groupby("recipe").mean().reset_index(), 2)
                + "\n\nMeans: " + ", ".join(
                    f"{name} {single[name].mean():.2f}" for name in secondary[1:])
                + f". By the limits of question 1, the second fit is {within} 1 "
                "point. This is a description. It does not change the outcome."))]
    text = ("# Amendment A8: the all-tiers rule on new recipes\n\nThe rule under "
            "test, the second fit, and the decision rule were fixed before the new "
            "runs.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
