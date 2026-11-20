"""Amendment A11: the fixed score at high weight decay, with 100 classes.

    python -m dynamics.analyze_y freeze --results-root dyn_results
    python -m dynamics.analyze_y test --results-root dyn_results

The check has no rule. `freeze` takes the fixed choices from the final checkpoints
of the earlier 100-class runs without label smoothing and writes them to one file.
`test` reads the runs of the recipes Y1 to Y3 and says if the one fixed score of
the file stays within 1 point of the best score: S (safe), H (holds), or F.

The target is the AUROC of the correct ID validation images against the unseen
images, as a mean over the far sets, as in amendments A9 and A10.

Protocol: ../documentation/fixed_score_high_weight_decay_amendment_2026-11-19.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dynamics import analyze_augrc as ag
from dynamics import analyze_e, analyze_h, rules
from dynamics import analyze_shortlist as sl
from dynamics import analyze_t as at
from dynamics import analyze_u as au
from dynamics import analyze_v as av
from dynamics.analyze_e import block
from dynamics.analyze_selection import FAMILY
from dynamics.config import STAGES

HERE = Path(__file__).resolve().parent
TRAIN_STAGE, TEST_STAGE = "a11_train", "y"
CHOICES_PATH = HERE / "frozen" / "stage_y_choices.json"
#: SHA-256 of the frozen file. Set after `freeze`, before any run of the stage.
CHOICES_SHA256 = "9a1c474a0fa5af5ded742ca5b114fe93eb542b8008230aa52012c6bd27f6f187"
UNDER, RECIPE_RULE, STAND_IN = at.B1, at.RECIPE_RULE, au.STAND_IN
#: Other fixed scores, reported with no outcome.
FIXED = ("fDBD", "kNN")
#: The frozen rule of amendment A9 part 2, reported with no outcome: its label,
#: file, hash, and the name of the rule in that file.
SECONDARY = ("a9_rule", av.PARTS[2].path, av.PARTS[2].sha256, av.PARTS[2].rule)
HOLD = av.HOLD
SHORT = 3


def training_tables(root: Path) -> analyze_e.Tables:
    """The earlier 100-class runs without label smoothing."""
    keep = {spec.name for spec in STAGES[TRAIN_STAGE]}
    tables = analyze_h.read_stage(root, (analyze_h.PHASE_1, analyze_h.PHASE_2,
                                         av.PARTS[2].test_stage))
    return analyze_e.Tables(**{
        key: frame if frame.empty else frame[frame["run"].isin(keep)]
        .reset_index(drop=True) for key, frame in vars(tables).items()})


def earlier_scores(root: Path) -> tuple[pd.DataFrame, tuple[int, int]]:
    """Final scores of the earlier runs with the correct-only target, and the counts."""
    tables = training_tables(root)
    counts = ag.counts(root, sorted(set(tables.metrics["run"])))
    scores, _ = ag.final(ag.with_target(tables, *counts, ag.CORRECT_ONLY))
    return scores, counts


def build(root: Path) -> dict:
    """The fixed choices from the final checkpoints of the earlier runs."""
    final, (n_id, n_unseen) = earlier_scores(root)
    kind = analyze_h.recipes_of(final.index).map(FAMILY)
    return {
        "amendment": "A11", "target": ag.CORRECT_ONLY,
        "images": {"id": n_id, "unseen": n_unseen}, "runs": sorted(final.index),
        UNDER: str(final.mean().idxmax()),
        RECIPE_RULE: {str(name): str(part.mean().idxmax())
                      for name, part in final.groupby(kind)},
        "shortlists": {str(k): sl.greedy(final, k) for k in (SHORT, 5)}}


def decide(table: pd.DataFrame, name: str = UNDER) -> str:
    """S, H, or F for the regret of one choice, by the rule of amendment A11."""
    if table.empty or table[name].isna().any():
        return "not decided: a run has no value"
    by_recipe = table.groupby("recipe")[name].mean()
    if (by_recipe <= HOLD).all():
        return "S, safe: the fixed score is within 1 point in every recipe"
    if table[name].mean() <= HOLD and 2 * (by_recipe > HOLD).sum() < len(by_recipe):
        return "H, holds: the fixed score is within 1 point on the mean"
    return "F, the fixed score is not within 1 point"


def regrets(frozen: dict, rule: rules.RidgeRule, scores: pd.DataFrame,
            metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The choices, their regret, and the regret of the shortlists, per run."""
    recipe = analyze_h.recipes_of(scores.index)
    picks = pd.DataFrame({
        UNDER: [frozen[UNDER]] * len(scores),
        RECIPE_RULE: [frozen[RECIPE_RULE][kind] for kind in recipe.map(FAMILY)],
        **{name: [name] * len(scores) for name in FIXED},
        STAND_IN: au.stand_in_choice(metrics, list(scores.columns)),
        SECONDARY[0]: rule.choose(metrics)}, index=scores.index)
    single = pd.DataFrame({"recipe": recipe})
    for name in picks.columns:
        single[name] = analyze_h.regret_of(scores, list(picks[name]))
    short = pd.DataFrame({
        "recipe": recipe,
        "fixed": sl.shortlist_regret(scores, frozen["shortlists"][str(SHORT)]),
        "random": sl.random_regret(scores, SHORT)})
    return picks, single, short


def coverage(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Each score as a fixed choice: its regret on the new and the earlier recipes.

    `covers` is true if the score is within 1 point in every recipe of both sets.
    """
    parts = {}
    for label, scores in (("earlier", old), ("new", new)):
        recipe = analyze_h.recipes_of(scores.index)
        by_recipe = (scores.max(axis=1).to_numpy()[:, None] - scores).groupby(
            recipe).mean()
        parts[f"{label}_mean"], parts[f"{label}_worst"] = (by_recipe.mean(),
                                                           by_recipe.max())
    table = pd.DataFrame(parts)
    table["covers"] = (table["earlier_worst"] <= HOLD) & (table["new_worst"] <= HOLD)
    return table.sort_values("new_mean").reset_index(names="csf")


def load(path: Path, wanted: str, allow: bool) -> dict:
    digest = rules.file_hash(path)
    if digest != wanted and not allow:
        raise SystemExit(f"{path.name} has hash {digest}, not the frozen one")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("freeze", "test"))
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--choices", default=str(CHOICES_PATH))
    parser.add_argument("--allow-other-files", action="store_true",
                        help="accept frozen files with another hash (tests only)")
    parser.add_argument("--stage", help="read another stage (dry runs only)")
    parser.add_argument("--derived-target", action="store_true",
                        help="take the target from the old stored values and skip "
                        "gate M (dry runs on earlier runs only)")
    parser.add_argument("--out", default=str(HERE / "outputs" / "stage_y_report.md"))
    args = parser.parse_args()
    root, path = Path(args.results_root), Path(args.choices)
    sections = []
    if args.step == "freeze":
        if path.exists():
            raise SystemExit(f"{path} exists. The choices are frozen once")
        wanted = {spec.name for spec in STAGES[TRAIN_STAGE]}
        ready = set(training_tables(root).metrics["run"])
        if wanted - ready:
            raise SystemExit(f"runs without results: {sorted(wanted - ready)}")
        digest = rules.save(build(root), path)
        frozen = json.loads(path.read_text())
        sections.append(("Frozen file", (
            f"- File: `{path}`\n- SHA-256: `{digest}`\n- Target: {frozen['target']}, "
            f"with {frozen['images']} images for each far set.\n- Runs: "
            f"{len(frozen['runs'])}.\n- Score under test: {frozen[UNDER]}. Choice by "
            f"kind: {frozen[RECIPE_RULE]}.\n- Shortlists: {frozen['shortlists']}.")))
    else:
        frozen = load(path, CHOICES_SHA256, args.allow_other_files)
        second = at.load(SECONDARY[1])
        if rules.file_hash(SECONDARY[1]) != SECONDARY[2] and not args.allow_other_files:
            raise SystemExit(f"{SECONDARY[1].name} is not the frozen file")
        stage = args.stage or TEST_STAGE
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
            gap, skipped = av.metric_gap(new, n_id, n_unseen)
            if gap > av.METRIC_LIMIT:
                raise SystemExit(f"gate M fails: the stored and the derived target "
                                 f"differ by {gap:.2e}")
            aimed = av.with_stored_target(new)
            gate_m = (f"**Gate M passes.** Largest difference between the stored "
                      f"and the derived target: {gap:.2e} (limit "
                      f"{av.METRIC_LIMIT:.0e}). Rows that could not be compared: "
                      f"{skipped}.")
        scores, metrics = ag.final(aimed)
        old_scores, _ = ag.final(new)
        accuracy = metrics["clock/val_accuracy"]
        recipe = analyze_h.recipes_of(scores.index)
        failed = sorted(set(recipe[accuracy < av.MIN_ACCURACY]))
        keep = ~recipe.isin(failed).to_numpy()
        scores, metrics, old_scores = scores[keep], metrics[keep], old_scores[keep]
        rule = second["rules"][SECONDARY[3]]
        picks, single, short = regrets(frozen, rule, scores, metrics)
        _, old_single, _ = regrets(frozen, rule, old_scores, metrics)
        outcome = decide(single)
        earlier, _ = earlier_scores(root)
        best = pd.DataFrame({"recipe": analyze_h.recipes_of(scores.index),
                             "best": scores.idxmax(axis=1),
                             "best_auroc": scores.max(axis=1)})
        gate = pd.DataFrame({"recipe": analyze_h.recipes_of(accuracy.index),
                             "val_accuracy": accuracy})
        names = [c for c in single.columns if c != "recipe"]
        sections += [
            ("Runs and gates", (
                f"{len(accuracy)} of {len(STAGES[stage])} runs. Gate T: a run passes "
                f"with a validation accuracy of at least {av.MIN_ACCURACY:.0%}:\n\n"
                + block(gate.reset_index(), 3) + "\n\nRecipes that leave the "
                f"decision rule: {', '.join(failed) or 'none'}.\n\n" + gate_m)),
            (f"The fixed score under test: {frozen[UNDER]}", (
                "Regret at the final checkpoint, in points of the correct-only "
                "AUROC on the far sets. The other columns have no outcome:\n\n"
                + block(single.reset_index(), 2) + "\n\nBy recipe:\n\n"
                + block(single.groupby("recipe").mean().reset_index(), 2)
                + "\n\nMeans: " + ", ".join(
                    f"{name} {single[name].mean():.2f}" for name in names)
                + "\n\nThe best score of each run:\n\n" + block(best.reset_index(), 2)
                + f"\n\n**Outcome: {outcome}.**")),
            ("Every score as a fixed choice", (
                f"On the {earlier.index.map(lambda r: r[:2]).nunique()} earlier "
                "recipes without label smoothing and on the new recipes. `covers` "
                "is true if the score is within 1 point in every recipe of both "
                "sets:\n\n" + block(coverage(scores, earlier), 2))),
            ("Other values, no outcome", (
                "The fixed shortlist of size 3 "
                f"({', '.join(frozen['shortlists'][str(SHORT)])}) and random "
                "subsets, by recipe:\n\n"
                + block(short.groupby("recipe").mean().reset_index(), 2)
                + "\n\nThe choices of the stand-in and of the frozen rule:\n\n"
                + block(picks[[STAND_IN, SECONDARY[0]]].reset_index())
                + "\n\nRegret of the same choices with all ID images as the "
                "target, by recipe:\n\n"
                + block(old_single.groupby("recipe").mean().reset_index(), 2)))]
    text = ("# Amendment A11: the fixed score at high weight decay, with 100 classes"
            "\n\nThe score under test and the decision rule were fixed before the "
            "new runs.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
