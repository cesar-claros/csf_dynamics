"""Amendment A14: the frozen rule on recipes drawn at random and on new ID classes.

    python -m dynamics.analyze_p --part 1 --results-root dyn_results
    python -m dynamics.analyze_p --part 2 --results-root dyn_results
    python -m dynamics.analyze_p --together --results-root dyn_results

Nothing is fitted. The rule under test and the fixed score come from the frozen
file of amendment A12 part 1. Part 1 is stage `p`: Imagenette with 48 recipes
drawn at random. Part 2 is stage `q`: four new sets of 10 ImageNet classes with 48
other recipes. Each recipe has one run, and the recipe is the unit.

The outcome follows the decision rule of amendment A12. This analysis adds gate D
(the frozen recipes and class sets, and the settings and classes of every run), a
sign-flip test over the recipes, and the planned secondary readings: by cell of
kind and label smoothing, by class set, the best fixed score in hindsight, the
recipes where the rule loses, and the inputs outside the range of the rule.
`--together` reads the two parts as one set of recipes, with no outcome letter.

Protocol: ../documentation/random_recipes_new_classes_amendment_2026-11-22.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_c, analyze_e, analyze_h, rules
from dynamics import analyze_r as ar
from dynamics import analyze_t as at
from dynamics import analyze_z as az
from dynamics.analyze_e import block
from dynamics.config import (
    A14_CLASS_SETS_PATH,
    A14_RECIPES_PATH,
    A14_TRAIN_IMAGES,
    BATCH_SIZE,
    IMAGE_SIZE,
    RECIPES,
    STAGES,
    class_set_of,
    expected_classes,
    parse_run,
)

HERE = Path(__file__).resolve().parent
#: Label and stage of each part.
PARTS = {1: ("part 1, Imagenette with recipes drawn at random", "p"),
         2: ("part 2, new ID classes with recipes drawn at random", "q")}
#: The frozen file of amendment A12 part 1, with the rule and the fixed score.
FROZEN = az.PARTS[1]
#: SHA-256 of the frozen recipes and class sets, public before any run.
RECIPES_SHA256 = "31b15f38100299d811f64dba2244356d728160e30ad4c29bf5f67164571b1c52"
CLASS_SETS_SHA256 = "6dfbcc791986cb7981604410df1ea7b4637a9af8d2dea6ff54b85ddd6adb3acf"
#: A part is not decided if more than this share of its recipes fails gate T.
FAILED_SHARE = 0.25
#: Random sign vectors of the sign-flip test, and its limit.
FLIPS = 100_000
MARGIN = 0.05
#: An input is outside the range of the rule beyond this many standard deviations.
BEYOND = 3.0
#: The rule loses in a recipe if it is this many points above the fixed score.
LOSS = 1.0
SETTINGS = ("augment", "weight_decay", "label_smoothing", "lr", "epochs", "arch")
HINDSIGHT = "hindsight"


@dataclass(frozen=True)
class Far:
    """The far target of one stage, for the recipes that pass gate T."""

    picks: pd.DataFrame
    single: pd.DataFrame
    scores: pd.DataFrame
    metrics: pd.DataFrame
    accuracy: pd.Series
    failed: list[str]


def frozen_inputs() -> list[str]:
    """Problems with the frozen recipes and class sets. Empty if there are none."""
    problems = []
    for path, wanted in ((A14_RECIPES_PATH, RECIPES_SHA256),
                         (A14_CLASS_SETS_PATH, CLASS_SETS_SHA256)):
        if rules.file_hash(path) != wanted:
            problems.append(f"{path.name} does not have the recorded hash")
    return problems


def gate_d(root: Path, runs: list[str]) -> pd.DataFrame:
    """Gate D: runs whose `meta.json` is not that of their recipe and class set."""
    rows = []
    for run in runs:
        meta = json.loads((Path(root) / "runs" / run / "meta.json").read_text())
        recipe = parse_run(run).recipe
        wrong = [name for name in SETTINGS
                 if meta.get(name, "resnet18" if name == "arch" else None)
                 != getattr(RECIPES[recipe], name)]
        if (meta.get("limit") is not None or meta.get("image_size") != IMAGE_SIZE
                or meta.get("batch_size") != BATCH_SIZE):
            wrong.append("smoke settings")
        classes = expected_classes(recipe)
        if classes is not None:
            if tuple(meta.get("classes", ())) != classes:
                wrong.append("classes")
            if meta.get("train_images") != A14_TRAIN_IMAGES:
                wrong.append("train_images")
        if wrong:
            rows.append({"run": run, "problem": ", ".join(wrong)})
    return pd.DataFrame(rows, columns=["run", "problem"])


def sign_flip(difference: np.ndarray, draws: int = FLIPS, seed: int = 0) -> float:
    """One-sided p of a mean difference below 0, with random signs of the units."""
    values = np.asarray(difference, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(draws, len(values)))
    means = (signs * values).mean(axis=1)
    return float((1 + (means <= values.mean() + 1e-12).sum()) / (1 + draws))


def far(frozen: dict, new: analyze_e.Tables) -> Far:
    """The choices and their regret on the far sets, as in `az.test_sections`."""
    meta, _, metrics = at.cells(new)
    final = analyze_c.final_mask(meta)
    last = metrics[final].droplevel("epoch")
    accuracy = last["clock/val_accuracy"]
    recipe = analyze_h.recipes_of(last.index)
    failed = sorted(set(recipe[accuracy < az.MIN_ACCURACY]))
    keep = ~recipe.isin(failed).to_numpy()
    sets = az.GROUPS[az.PRIMARY]
    scores = az.group_scores(az.stored(new, sets), sets, meta.index)[final].droplevel(
        "epoch")[keep]
    picks, single, _ = az.regrets(frozen["groups"][az.PRIMARY], scores, last[keep])
    return Far(picks, single, scores, last[keep], accuracy, failed)


def cell_of(recipe: str) -> str:
    """The cell of a recipe: its kind, and if it has label smoothing."""
    settings = RECIPES[recipe]
    return (("augmented" if settings.augment else "plain") + ", "
            + ("with" if settings.label_smoothing > 0 else "no") + " smoothing")


def by_group(recipes: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """Mean regret of the rule and of the fixed score for each group of recipes."""
    table = recipes[[az.RULE, az.ONE]].groupby(group.loc[recipes.index]).agg(
        ["mean", "size"])
    out = pd.DataFrame({"recipes": table[(az.RULE, "size")],
                        "rule": table[(az.RULE, "mean")],
                        "fixed": table[(az.ONE, "mean")]})
    out["difference"] = out["rule"] - out["fixed"]
    return out.rename_axis("group").reset_index()


def hindsight(found: Far) -> tuple[str, dict]:
    """The fixed score with the lowest mean regret of these runs, and the rule
    against it. This baseline uses the OOD results of the test runs."""
    regret = found.scores.rsub(found.scores.max(axis=1), axis=0)
    by = regret.groupby(found.single["recipe"]).mean()
    name = str(by.mean().idxmin())
    recipes = az.by_recipe(found.single).assign(**{HINDSIGHT: by[name]})
    return name, az.adds(recipes, az.RULE, HINDSIGHT)


def settings_table(found: Far) -> pd.DataFrame:
    """One row for each recipe: its settings, accuracy, choice, and regret."""
    recipes = az.by_recipe(found.single)
    runs = found.single["recipe"]
    rows = []
    for name in recipes.index:
        inside = (runs == name).to_numpy()
        settings = RECIPES[name]
        rows.append({
            "recipe": name, "kind": "augmented" if settings.augment else "plain",
            "smoothing": settings.label_smoothing,
            "weight_decay": f"{settings.weight_decay:g}", "lr": f"{settings.lr:g}",
            "accuracy": float(found.accuracy[found.single.index[inside]].mean()),
            "choice": "/".join(dict.fromkeys(found.picks.loc[inside, az.RULE])),
            "rule": recipes.loc[name, az.RULE], "fixed": recipes.loc[name, az.ONE]})
    return pd.DataFrame(rows).round({"accuracy": 3, "rule": 2, "fixed": 2})


def beyond(frozen: dict, found: Far) -> str:
    """The runs with an input outside the range of the rule, and their regret."""
    rule = frozen["groups"][az.PRIMARY]["rules"][az.RULE]
    count = (np.abs(rule.inputs(found.metrics)) > BEYOND).sum(axis=1)
    outside = count > 0
    text = (f"{int(outside.sum())} of {len(count)} runs have an input beyond "
            f"{BEYOND:g} standard deviations of the runs that the rule was fitted "
            f"on (mean number of such inputs for a run: {count.mean():.2f} of "
            f"{len(rule.columns)}).")
    for label, rows in (("these runs", outside), ("the other runs", ~outside)):
        if rows.any():
            text += (f" In {label} the rule has a mean regret of "
                     f"{found.single.loc[rows, az.RULE].mean():.2f} and the fixed "
                     f"score has {found.single.loc[rows, az.ONE].mean():.2f}.")
    return text


def margin(reading: dict, p: float) -> str:
    """The sentence of the sign-flip test for one reading of `az.adds`."""
    text = (f"Sign-flip test over the recipes ({FLIPS:,} random sign vectors, seed "
            f"0): one-sided p = {p:.4f}.")
    if reading["supported"] and not p < MARGIN:
        text += (f" The interval lies below 0 and this p is not below {MARGIN:g}: "
                 "**the gain is at the margin.**")
    elif reading["supported"]:
        text += " It agrees with the interval."
    return text


def planned(frozen: dict, found: Far) -> list[tuple[str, str]]:
    """The planned checks and secondary readings of one part, on the far sets."""
    recipes = az.by_recipe(found.single)
    reading = az.adds(recipes, az.RULE)
    difference = (recipes[az.RULE] - recipes[az.ONE]).to_numpy()
    sections = [("Planned checks on the far sets", (
        f"- The fixed score holds by the limits of question 2: "
        f"{'yes' if az.holds(recipes, az.ONE) else 'no'} (mean regret "
        f"{recipes[az.ONE].mean():.2f}, above {az.HOLD:g} point in "
        f"{int((recipes[az.ONE] > az.HOLD).sum())} of {len(recipes)} recipes; the "
        f"rule: {recipes[az.RULE].mean():.2f}, in "
        f"{int((recipes[az.RULE] > az.HOLD).sum())} of {len(recipes)}).\n- "
        + margin(reading, sign_flip(difference))))]
    cells = pd.Series({name: cell_of(name) for name in recipes.index})
    sections.append(("By cell of kind and label smoothing (secondary)",
                     block(by_group(recipes, cells), 2)))
    sets = pd.Series({name: class_set_of(name) for name in recipes.index})
    if sets.notna().all():
        sections.append(("By class set (secondary)", block(by_group(
            recipes, "set " + sets.astype(int).astype(str)), 2)))
    name, against = hindsight(found)
    sections.append(("The rule against the best fixed score in hindsight (secondary)", (
        f"The fixed score with the lowest mean regret of these runs is {name}. It "
        "is known only with the OOD results of the test runs. The rule has "
        f"{against['regret']:.2f} and this score has {against['fixed_regret']:.2f}; "
        f"difference {against['difference']:.2f} ({against['low']:.2f} to "
        f"{against['high']:.2f}); {against['reading']}.")))
    table = settings_table(found)
    lost = table[table["rule"] - table["fixed"] > LOSS]
    sections.append(("Where the rule loses (secondary)", (
        f"Recipes where the rule is more than {LOSS:g} point above the fixed score: "
        f"{len(lost)} of {len(table)}." + ("" if lost.empty else "\n\n"
                                           + block(lost)))))
    sections.append(("Inputs outside the range of the rule (secondary)",
                     beyond(frozen, found)))
    sections.append(("Recipes", "Settings, validation accuracy, the choice of the "
                     "rule, and the regret of the rule and of the fixed score:\n\n"
                     + block(table)))
    return sections


def read_part(stage: str, root: Path, frozen: dict, dry: bool,
              count_missing: bool) -> tuple[list[tuple[str, str]], Far | None,
                                            analyze_e.Tables, dict[str, int], str]:
    """Gates D, S, and T of one stage.

    Returns:
        The sections of the gates, the far target (None if a gate stops the part),
        the tables, the image counts, and the reason of the stop ("" if none).
    """
    new = analyze_h.read_stage(root, (stage,))
    if new.metrics.empty:
        raise SystemExit(f"no results of stage {stage} under {root}/runs")
    runs = sorted(set(new.metrics["run"]))
    missing = sorted({spec.name for spec in STAGES[stage]} - set(runs))
    if missing and not count_missing:
        raise SystemExit(f"stage {stage} is not complete; runs with no results: "
                         f"{', '.join(missing)}")
    counts = az.set_counts(root, runs)
    if frozen["images"] != counts:
        raise SystemExit("the image counts are not those of the frozen file")
    problems = [] if dry else frozen_inputs()
    wrong = gate_d(root, runs)
    failures = ar.score_failures(new)
    passes = not problems and wrong.empty
    text = (f"Recipes: `{A14_RECIPES_PATH.name}`. Class sets: "
            f"`{A14_CLASS_SETS_PATH.name}`. {len(runs)} of {len(STAGES[stage])} runs "
            "have results." + (f" Runs with no results, counted as failures of "
                               f"gate T: {', '.join(missing)}." if missing else "")
            + "\n\n" + ("**Gate D passes.** " + (
                "Dry run: the hashes of the two files are not checked. Every run "
                "has the settings of its recipe." if dry else
                "The frozen files have the recorded hashes, and every run has the "
                "settings of its recipe and the classes of its ID set.") if passes else
                        "**Gate D fails.**" + "".join(f"\n- {p}" for p in problems)
                        + ("" if wrong.empty else "\n\n" + block(wrong)))
            + "\n\n" + ("**Gate S passes.** Every score has a finite value at the "
                        "final checkpoint of every run." if failures.empty else
                        "**Gate S fails.** Final checkpoints with a score that has "
                        "an error or a non-finite value:\n\n" + block(failures)))
    sections = [("Frozen files, gate D, and gate S", text)]
    if not passes:
        return sections, None, new, counts, "gate D fails"
    if not failures.empty:
        return sections, None, new, counts, "gate S fails"
    found = far(frozen, new)
    recipes = {spec.recipe for spec in STAGES[stage]}
    out = len(found.failed) + len({parse_run(run).recipe for run in missing})
    if out > FAILED_SHARE * len(recipes):
        gate = pd.DataFrame({"recipe": analyze_h.recipes_of(found.accuracy.index),
                             "val_accuracy": found.accuracy})
        sections.append(("Gate T", (
            f"{out} of {len(recipes)} recipes fail gate T, more than "
            f"{FAILED_SHARE:.0%}. **The part is not decided.**\n\n"
            + block(gate.reset_index(), 3))))
        return sections, None, new, counts, "gate T: too many recipes fail"
    return sections, found, new, counts, ""


def together(parts: dict[int, Far]) -> list[tuple[str, str]]:
    """The two parts as one set of recipes, on the far sets. No outcome letter."""
    recipes = pd.concat([az.by_recipe(found.single) for found in parts.values()])
    reading = az.adds(recipes, az.RULE)
    difference = (recipes[az.RULE] - recipes[az.ONE]).to_numpy()
    part = pd.Series({name: f"part {number}" for number, found in parts.items()
                      for name in az.by_recipe(found.single).index})
    cells = pd.Series({name: cell_of(name) for name in recipes.index})
    return [
        ("Both parts together (secondary, no outcome letter)", (
            f"{len(recipes)} recipes. The rule has {reading['regret']:.2f} and the "
            f"fixed score has {reading['fixed_regret']:.2f}; difference "
            f"{reading['difference']:.2f} ({reading['low']:.2f} to "
            f"{reading['high']:.2f}); {reading['reading']}. The rule is lower in "
            f"{reading['lower']} recipes and higher in {reading['higher']}.\n\n- The "
            f"rule holds: {'yes' if az.holds(recipes, az.RULE) else 'no'}. The fixed "
            f"score holds: {'yes' if az.holds(recipes, az.ONE) else 'no'}.\n- "
            + margin(reading, sign_flip(difference)))),
        ("By part", block(by_group(recipes, part), 2)),
        ("By cell of kind and label smoothing", block(by_group(recipes, cells), 2))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--part", type=int, choices=sorted(PARTS))
    parser.add_argument("--together", action="store_true",
                        help="read the two parts as one set of recipes")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--rules", help="another rules file (tests only)")
    parser.add_argument("--allow-other-rules", action="store_true",
                        help="accept a rules file with another hash (tests only)")
    parser.add_argument("--stage", help="read another stage (dry runs only)")
    parser.add_argument("--count-missing-as-failed", action="store_true",
                        help="a run with no results (its training failed) counts "
                             "as a failure of gate T")
    parser.add_argument("--out")
    args = parser.parse_args()
    if bool(args.part) == args.together:
        parser.error("give --part or --together")
    root = Path(args.results_root)
    path = Path(args.rules) if args.rules else FROZEN.path
    digest = rules.file_hash(path)
    if digest != FROZEN.sha256 and not args.allow_other_rules:
        raise SystemExit(f"{path.name} has hash {digest}, not the frozen one")
    frozen = az.load(path)
    head = ("The rule under test and the fixed score come from the file of "
            f"amendment A12 part 1 (`{path.name}`), with no new fit. Each recipe is "
            "one unit.\n")
    if args.together:
        parts = {}
        for number, (_, stage) in PARTS.items():
            _, found, _, _, stop = read_part(stage, root, frozen, False,
                                             args.count_missing_as_failed)
            if found is None:
                raise SystemExit(f"part {number} is not decided: {stop}")
            parts[number] = found
        out = Path(args.out) if args.out else HERE / "outputs" / "stage_pq_report.md"
        title, sections = "both parts together", together(parts)
    else:
        label, stage = PARTS[args.part]
        dry = bool(args.stage)
        stage = args.stage or stage
        out = Path(args.out) if args.out else (
            HERE / "outputs" / f"stage_{stage}_report.md")
        sections, found, new, counts, stop = read_part(
            stage, root, frozen, dry, args.count_missing_as_failed)
        title = label + (f" (dry run on stage {stage})" if dry else "")
        if found is not None:
            sections += az.test_sections(frozen, new, counts, stage)
            sections += planned(frozen, found)
    text = (f"# Amendment A14, {title}: the frozen rule on recipes drawn at random\n\n"
            + head + "".join(f"\n## {name}\n\n{body}\n" for name, body in sections))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    if not args.together and found is None:
        raise SystemExit(f"{stop}: the part is not decided")


if __name__ == "__main__":
    main()
