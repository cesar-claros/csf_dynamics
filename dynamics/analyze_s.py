"""Amendment A15: a rule fitted on several ID sets against a choice from the recipe.

    python -m dynamics.analyze_s read --results-root dyn_results
    python -m dynamics.analyze_s freeze --results-root dyn_results
    python -m dynamics.analyze_s test --results-root dyn_results

`freeze` fits every choice on the earlier ResNet-18 runs with 10 classes: the 76
runs of amendment A12 part 1 and the stages `z`, `p`, and `q`. The rule under test
is one ridge rule from the 69 score statistics, fitted on the final checkpoints of
the runs with 200 epochs. The baseline under test is the choice by cell: the best
score for each combination of kind and label smoothing, from the same final
checkpoints. The file is frozen: `freeze` does not write it a second time.

`test` reads stage `s`: 32 new class sets with 4 recipes each. The class set is
the unit. `read` is a check of the fit: it leaves one class set of amendment A14
out at a time, and it must give the numbers of the exploratory read of the plan.

Protocol: ../documentation/several_id_sets_rule_amendment_2026-11-23.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_c, analyze_e, analyze_h, rules
from dynamics import analyze_loro as lo
from dynamics import analyze_multimetric as mm
from dynamics import analyze_p as ap
from dynamics import analyze_r as ar
from dynamics import analyze_shortlist as sl
from dynamics import analyze_t as at
from dynamics import analyze_v as av
from dynamics import analyze_z as az
from dynamics.analyze_e import block
from dynamics.config import (
    A15_CLASS_SETS_PATH,
    A15_PER_SET,
    A15_RECIPES_PATH,
    EPOCHS,
    RECIPES,
    STAGES,
    a15_set_of,
    class_set_of,
    parse_run,
)

HERE = Path(__file__).resolve().parent
RULES_S_PATH = HERE / "frozen" / "stage_s_rules.json"
#: SHA-256 of the frozen files. The rules file was set after `freeze`, and the
#: class sets after their draw, both before any new run.
RULES_S_SHA256 = "f4eee53e1698cdcdf06ec43459db39e7b0f80463f70db0d7838e8e25fbebe115"
RECIPES_SHA256 = "77768b6711b327f88fa02fea360f79a53ad2e296db410218542a91a082c7ffdd"
CLASS_SETS_SHA256 = "6b4420a5ba8ea2c2b3c6ba7009a10c395011de4565fd3359b6b711d3c03f598f"
#: Stages of the fit, after the 76 runs of amendment A12 part 1.
TRAIN_STAGES = ("z", "p", "q")
TEST_STAGE = "s"
#: The choices. RULE and BY_CELL are under test.
RULE, BY_CELL, ONE = "statistics", "by_cell", "one_score"
TWO, RULE_CELLS, TWO_CELLS = "two_rules", "statistics_cells", "two_rules_cells"
BOTH, OLD = "statistics_and_recipe", "a12_rule"
SECONDARY = (TWO, RULE_CELLS, TWO_CELLS, BOTH, OLD)
CHOICES = (RULE, BY_CELL, ONE, *SECONDARY)
WITH, WITHOUT = az.WITH, az.WITHOUT
#: A class set needs this many recipes that pass gate T.
MIN_RECIPES = 3
#: The test is not decided if more than this share of the class sets leaves.
LEFT_SHARE = 0.25
SHORT = 3
#: A class set where the rule is this many points above the choice by cell.
LOSS = 1.0


def recipes_of(index: pd.Index) -> np.ndarray:
    runs = index.get_level_values("run") if index.nlevels > 1 else index
    return np.array([parse_run(run).recipe for run in runs])


def cell_of(recipe: str) -> str:
    """The cell of a recipe: its kind, and if it has label smoothing."""
    settings = RECIPES[recipe]
    return (("augmented" if settings.augment else "plain") + "/"
            + (WITH if settings.label_smoothing > 0 else WITHOUT))


CELLS = tuple(f"{kind}/{flag}" for kind in ("plain", "augmented")
              for flag in (WITHOUT, WITH))


def set_of(recipe: str) -> int:
    """Number of the ID set of a recipe: 0 is Imagenette. The class sets of
    amendment A14 are 1 to 4, and those of amendment A15 are 101 to 132."""
    number = a15_set_of(recipe)
    if number is not None:
        return 100 + number
    return class_set_of(recipe) or 0


def recipe_inputs(index: pd.Index) -> pd.DataFrame:
    """The recipe settings of each row as inputs, all above 0 for the log scale."""
    recipes = [RECIPES[name] for name in recipes_of(index)]
    return pd.DataFrame({
        "recipe/augment": [1.0 + r.augment for r in recipes],
        "recipe/smoothing_on": [1.0 + (r.label_smoothing > 0) for r in recipes],
        "recipe/smoothing": [1.0 + r.label_smoothing for r in recipes],
        "recipe/weight_decay": [max(r.weight_decay, 1e-6) for r in recipes],
        "recipe/lr": [r.lr for r in recipes]}, index=index)


def statistics_columns(metrics: pd.DataFrame) -> list[str]:
    return lo.input_groups(list(mm.transform(metrics).columns))[az.RULE]


def with_recipe(metrics: pd.DataFrame) -> pd.DataFrame:
    score = [c for c in metrics.columns if c.startswith("score/")]
    return pd.concat([metrics[score], recipe_inputs(metrics.index)], axis=1)


def fit_choices(metrics: pd.DataFrame, scores: pd.DataFrame,
                final: np.ndarray) -> dict:
    """Every choice of one target, from training cells.

    Args:
        metrics: Raw inputs of every cell (run and checkpoint).
        scores: Mean AUROC of each score on the sets of the target, per cell.
        final: True for the final checkpoint of a run with the usual epochs.
    """
    smooth = np.array([RECIPES[name].label_smoothing > 0
                       for name in recipes_of(metrics.index)])
    cells = np.array([cell_of(name) for name in recipes_of(metrics.index)])
    wide = with_recipe(metrics)
    found = {
        RULE: rules.fit_ridge_rule(metrics[final], scores[final],
                                   statistics_columns(metrics[final])),
        RULE_CELLS: rules.fit_ridge_rule(metrics, scores, statistics_columns(metrics)),
        BOTH: rules.fit_ridge_rule(wide[final], scores[final], [
            *statistics_columns(metrics[final]), *recipe_inputs(metrics.index)])}
    for flag, label in ((True, WITH), (False, WITHOUT)):
        for name, rows in ((TWO, final & (smooth == flag)),
                           (TWO_CELLS, smooth == flag)):
            found[f"{name}/{label}"] = rules.fit_ridge_rule(
                metrics[rows], scores[rows], statistics_columns(metrics[rows]))
    last = scores[final].droplevel("epoch")
    return {"rules": {name: asdict(rule) for name, rule in found.items()},
            BY_CELL: {cell: str(last[cells[final] == cell].mean().idxmax())
                      for cell in CELLS},
            ONE: str(last.mean().idxmax()), "shortlist": sl.greedy(last, SHORT),
            "final_rows": int(final.sum())}


def choose(group: dict, metrics: pd.DataFrame,
           old: rules.RidgeRule | None = None) -> pd.DataFrame:
    """The choice of every fitted object for each run. `metrics` has one row per
    run, at its final checkpoint."""
    found = {name: rule if isinstance(rule, rules.RidgeRule)
             else rules.RidgeRule(**rule) for name, rule in group["rules"].items()}
    names = recipes_of(metrics.index)
    smooth = np.array([RECIPES[name].label_smoothing > 0 for name in names])
    picks = pd.DataFrame({
        RULE: found[RULE].choose(metrics),
        BY_CELL: [group[BY_CELL][cell_of(name)] for name in names],
        ONE: [group[ONE]] * len(metrics),
        RULE_CELLS: found[RULE_CELLS].choose(metrics),
        BOTH: found[BOTH].choose(with_recipe(metrics))}, index=metrics.index)
    for name in (TWO, TWO_CELLS):
        both = [found[f"{name}/{WITH}"].choose(metrics),
                found[f"{name}/{WITHOUT}"].choose(metrics)]
        picks[name] = [both[0][i] if on else both[1][i] for i, on in enumerate(smooth)]
    if old is not None:
        picks[OLD] = old.choose(metrics)
    return picks


def regret_table(picks: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Regret of every choice per run, with the recipe and the ID set of the run."""
    names = recipes_of(scores.index)
    single = pd.DataFrame({"recipe": names, "set": [set_of(name) for name in names]},
                          index=scores.index)
    for name in picks.columns:
        single[name] = analyze_h.regret_of(scores, list(picks[name]))
    return single


def by_set(single: pd.DataFrame) -> pd.DataFrame:
    """The regret of each choice with the class set as the unit."""
    names = [c for c in single.columns if c not in ("recipe", "set")]
    return single.groupby("set")[names].mean()


def holds(units: pd.DataFrame, name: str) -> bool:
    """Within 1 point on the mean, and above it in fewer than half of the units."""
    return bool(units[name].mean() <= az.HOLD
                and 2 * (units[name] > az.HOLD).sum() < len(units))


def adds(units: pd.DataFrame, name: str, other: str) -> dict:
    """One choice against another, over the units: the difference, its interval,
    and the sign-flip test."""
    difference = (units[name] - units[other]).to_numpy()
    mean, low, high = analyze_c.interval(difference)
    return {"choice": name, "against": other, "regret": units[name].mean(),
            "other_regret": units[other].mean(), "difference": mean, "low": low,
            "high": high, "lower": int((difference < -1e-9).sum()),
            "higher": int((difference > 1e-9).sum()),
            "supported": bool(mean < 0 and high < 0), "p": ap.sign_flip(difference)}


def outcome(units: pd.DataFrame) -> str:
    """A, H, or F for the rule under test against the choice by cell."""
    if units.empty or units[[RULE, BY_CELL]].isna().any().any():
        return "not decided: a class set has no choice"
    gain = adds(units, RULE, BY_CELL)["supported"]
    if not holds(units, RULE):
        return ("F, the rule is not within 1 point"
                + (", and it is below the choice by cell" if gain else ""))
    return ("A, the statistics add to the recipe and the rule holds" if gain
            else "H, the rule holds with no shown gain over the recipe")


def sentence(reading: dict) -> str:
    return (f"{reading['regret']:.2f} against {reading['other_regret']:.2f}, "
            f"difference {reading['difference']:.2f} ({reading['low']:.2f} to "
            f"{reading['high']:.2f}), lower in {reading['lower']} units and higher "
            f"in {reading['higher']}, sign-flip p {reading['p']:.4f}; "
            + ("supported" if reading["supported"] else "not supported"))


def stability(single: pd.DataFrame, name: str = RULE, other: str = BY_CELL) -> str:
    """How much the difference changes between the class sets."""
    difference = single[name] - single[other]
    groups = difference.groupby(single["set"])
    means, sizes = groups.mean(), groups.size()
    inside = float(groups.var(ddof=1).mean())
    between = float(means.var(ddof=1) - inside / sizes.mean()) if len(means) > 1 else 0.0
    return (f"The difference of the {len(means)} class sets goes from "
            f"{means.min():.2f} to {means.max():.2f}, with a standard deviation of "
            f"{means.std(ddof=1):.2f}. The recipes inside a class set have a "
            f"standard deviation of {inside ** 0.5:.2f}. The part between class "
            f"sets that the recipes do not explain has a standard deviation of "
            f"{max(between, 0.0) ** 0.5:.2f}. The rule is more than {LOSS:g} point "
            f"above the choice by cell in {int((means > LOSS).sum())} class sets, "
            f"and below it in {int((means < 0).sum())}.")


def training(root: Path) -> tuple[analyze_e.Tables, dict[str, int]]:
    """The runs of the fit, and their image counts."""
    tables = av.join(az.training_tables(1, root),
                     analyze_h.read_stage(root, TRAIN_STAGES))
    return tables, az.set_counts(root, sorted(set(tables.metrics["run"])))


def usual_final(meta: pd.DataFrame) -> np.ndarray:
    """True for the final checkpoint of a run with the usual number of epochs."""
    return analyze_c.final_mask(meta) & np.array([
        RECIPES[name].epochs == EPOCHS for name in recipes_of(meta.index)])


def build(root: Path) -> dict:
    """The content of the frozen file."""
    tables, counts = training(root)
    meta, _, metrics = at.cells(tables)
    final = usual_final(meta)
    groups = {}
    for name, sets in az.GROUPS.items():
        scores = az.group_scores(az.aimed(tables, counts, sets), sets, meta.index)
        if not np.isfinite(scores.to_numpy()).all():
            raise SystemExit(f"a score has no value on the {name} sets")
        groups[name] = {"sets": list(sets), **fit_choices(metrics, scores, final)}
    runs = meta.index[final].get_level_values("run")
    return {"amendment": "A15", "target": "correct_only", "images": counts,
            "penalty": mm.PENALTY, "training_runs": sorted(runs),
            "training_cells": len(meta),
            "runs_by_id_set": {str(number): int(amount) for number, amount in sorted(
                pd.Series([set_of(parse_run(run).recipe) for run in runs])
                .value_counts().items())},
            "groups": groups}


def load(path: Path) -> dict:
    frozen = json.loads(Path(path).read_text())
    for group in frozen["groups"].values():
        group["rules"] = {name: rules.RidgeRule(**rule)
                          for name, rule in group["rules"].items()}
    return frozen


def read(root: Path) -> pd.DataFrame:
    """The exploratory read of the plan: one class set of amendment A14 left out.

    Returns:
        The regret of every choice on the far sets, per run of the left-out set.
    """
    tables, counts = training(root)
    meta, _, metrics = at.cells(tables)
    final = usual_final(meta)
    sets = az.GROUPS[az.PRIMARY]
    scores = az.group_scores(az.aimed(tables, counts, sets), sets, meta.index)
    number = np.array([set_of(name) for name in recipes_of(meta.index)])
    frames = []
    for left in sorted(set(number[number > 0])):
        inside = number != left
        group = fit_choices(metrics[inside], scores[inside], final[inside])
        rows = final & (number == left)
        last, target = metrics[rows].droplevel("epoch"), scores[rows].droplevel("epoch")
        frames.append(regret_table(choose(group, last), target))
    return pd.concat(frames)


def gate_t(single: pd.DataFrame, accuracy: pd.Series, sets_planned: int
           ) -> tuple[pd.DataFrame, list[str], list[int], bool]:
    """The runs that stay, the recipes and class sets that leave, and if the test
    is decided."""
    failed = sorted(single.loc[accuracy[single.index] < az.MIN_ACCURACY, "recipe"])
    kept = single[~single["recipe"].isin(failed)]
    sizes = kept.groupby("set").size()
    small = sorted(set(single["set"]) - set(sizes[sizes >= MIN_RECIPES].index))
    kept = kept[~kept["set"].isin(small)]
    left = sets_planned - kept["set"].nunique()
    return kept, failed, small, left <= LEFT_SHARE * sets_planned


def frozen_inputs() -> list[str]:
    """Problems with the frozen recipes and class sets. Empty if there are none."""
    problems = []
    for path, wanted in ((A15_RECIPES_PATH, RECIPES_SHA256),
                         (A15_CLASS_SETS_PATH, CLASS_SETS_SHA256)):
        if not path.exists() or rules.file_hash(path) != wanted:
            problems.append(f"{path.name} does not have the recorded hash")
    return problems


def target_tables(frozen: dict, old: dict, new: analyze_e.Tables, name: str
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """The choices, their regret, the scores, and the accuracy, for one target."""
    meta, _, metrics = at.cells(new)
    final = analyze_c.final_mask(meta)
    last = metrics[final].droplevel("epoch")
    sets = az.GROUPS[name]
    scores = az.group_scores(az.stored(new, sets), sets, meta.index)[final].droplevel(
        "epoch")
    picks = choose(frozen["groups"][name], last, old["groups"][name]["rules"][az.RULE])
    return picks, regret_table(picks, scores), scores, last["clock/val_accuracy"]


def readings(units: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Every choice against the choice by cell and against the fixed score."""
    rows = []
    for name in (RULE, *SECONDARY):
        for other in (BY_CELL, ONE):
            rows.append(adds(units, name, other))
    rows.append(adds(units, BY_CELL, ONE))
    return pd.DataFrame(rows).assign(unit=unit)


def test_sections(frozen: dict, old: dict, new: analyze_e.Tables,
                  counts: dict[str, int], stage: str) -> list[tuple[str, str]]:
    """Gate T, gate M, the outcome, and the planned secondary readings."""
    gap, skipped = az.metric_gap(new, counts)
    if gap > az.METRIC_LIMIT:
        raise SystemExit(f"gate M fails: the stored and the derived target differ "
                         f"by {gap:.2e}")
    planned = len({set_of(spec.recipe) for spec in STAGES[stage]})
    sections = []
    for name in az.GROUPS:
        picks, single, scores, accuracy = target_tables(frozen, old, new, name)
        kept, failed, small, decided = gate_t(single, accuracy, planned)
        units = by_set(kept)
        if name == az.PRIMARY:
            gate = pd.DataFrame({"recipe": single["recipe"], "set": single["set"],
                                 "val_accuracy": accuracy[single.index]})
            sections.append(("Runs, gate T, and gate M", (
                f"{len(single)} of {len(STAGES[stage])} runs, in "
                f"{single['set'].nunique()} class sets. Gate T: a run passes with a "
                f"validation accuracy of at least {az.MIN_ACCURACY:.0%} (lowest: "
                f"{accuracy.min():.3f}). Recipes that leave: "
                f"{', '.join(failed) or 'none'}. Class sets that leave, with fewer "
                f"than {MIN_RECIPES} recipes: "
                f"{', '.join(str(s) for s in small) or 'none'}.\n\n"
                + block(gate.groupby("set")["val_accuracy"].agg(["size", "mean",
                                                                 "min"])
                        .reset_index(), 3)
                + f"\n\n**Gate M passes** on the sets of every target. Largest "
                f"difference between the stored and the derived target: {gap:.2e} "
                f"(limit {az.METRIC_LIMIT:.0e}). Rows that could not be compared: "
                f"{skipped}.")))
            if not decided:
                sections.append(("Outcome", (
                    "More than a quarter of the class sets leave. **The test is "
                    "not decided.**")))
                return sections
        main, fixed = adds(units, RULE, BY_CELL), adds(units, RULE, ONE)
        cell = adds(units, BY_CELL, ONE)
        title = (f"Target {name}: the outcome" if name == az.PRIMARY
                 else f"Target {name} (secondary, no outcome letter)")
        group = frozen["groups"][name]
        text = (
            f"Sets: {', '.join(az.GROUPS[name])}. Choice by cell: {group[BY_CELL]}. "
            f"One fixed score: {group[ONE]}. Regret at the final checkpoint, in "
            "points of the correct-only AUROC, with the class set as the unit:\n\n"
            + block(units.reset_index(), 2) + "\n\nMeans: " + ", ".join(
                f"{choice} {units[choice].mean():.2f}" for choice in units)
            + f"\n\n- **Question 1, the statistics add to the recipe** (rule "
            f"against choice by cell): {sentence(main)}."
            f"\n- **Question 2, the rule holds:** "
            f"{'yes' if holds(units, RULE) else 'no'} (mean {units[RULE].mean():.2f}"
            f", above {az.HOLD:g} point in {int((units[RULE] > az.HOLD).sum())} of "
            f"{len(units)} class sets)."
            f"\n- The choice by cell holds: "
            f"{'yes' if holds(units, BY_CELL) else 'no'} (mean "
            f"{units[BY_CELL].mean():.2f}, above {az.HOLD:g} point in "
            f"{int((units[BY_CELL] > az.HOLD).sum())} of {len(units)})."
            f"\n- The rule against the fixed score: {sentence(fixed)}."
            f"\n- The choice by cell against the fixed score: {sentence(cell)}.")
        if name == az.PRIMARY:
            if main["supported"] and not main["p"] < ap.MARGIN:
                text += (f"\n- The interval lies below 0 and the sign-flip p is not "
                         f"below {ap.MARGIN:g}: **the gain is at the margin.**")
            text += f"\n\n**Outcome: {outcome(units)}.**"
        sections.append((title, text))
        if name != az.PRIMARY:
            continue
        sections.append(("Stability between class sets (secondary)",
                         stability(kept)))
        sections.append(("Every choice, with the class set as the unit (secondary)",
                         block(readings(units, "class set").drop(columns="unit"), 2)))
        recipes = kept.set_index("recipe").drop(columns="set")
        sections.append(("The recipe as the unit (secondary)", (
            f"{len(recipes)} recipes, for comparison with amendment A14:\n\n"
            + block(readings(recipes, "recipe").drop(columns="unit"), 2))))
        cells = pd.Series([cell_of(r) for r in kept["recipe"]], index=kept.index)
        sections.append(("By cell of kind and label smoothing (secondary)", block(
            kept.groupby(cells)[list(CHOICES)].mean().rename_axis("cell")
            .reset_index(), 2)))
        rule = frozen["groups"][name]["rules"][RULE]
        inside = scores.loc[kept.index]
        metrics = at.cells(new)[2]
        last = metrics[analyze_c.final_mask(at.cells(new)[0])].droplevel(
            "epoch").loc[kept.index]
        short = pd.DataFrame({
            "fixed": sl.shortlist_regret(inside, group["shortlist"]),
            "rule": sl.shortlist_regret(inside, analyze_h.top(rule, last, SHORT)),
            "random": sl.random_regret(inside, SHORT)})
        reading, difference = analyze_h.read_shortlists(short)
        augrc = az.augrc_regret(new, az.GROUPS[name], picks.loc[kept.index].rename(
            columns={RULE: az.RULE, BY_CELL: az.ONE}), inside)
        sections.append(("Shortlists and AUGRC (secondary)", (
            f"- Shortlists of size {SHORT}: fixed {short['fixed'].mean():.2f}, rule "
            f"{short['rule'].mean():.2f}, random {short['random'].mean():.2f}; rule "
            f"minus fixed {difference[0]:.2f} ({difference[1]:.2f} to "
            f"{difference[2]:.2f}), with the recipe as the unit. "
            f"{reading.capitalize()}.\n- Regret in AUGRC points: rule "
            f"{augrc[az.RULE].mean():.3f}, choice by cell {augrc[az.ONE].mean():.3f}. "
            f"Runs where the best score by AUGRC is not the best by AUROC: "
            f"{int(augrc['other_order'].sum())} of {len(augrc)}.")))
        table = kept[["recipe", "set", RULE, BY_CELL, ONE]].assign(
            cell=cells, accuracy=accuracy[kept.index],
            choice=picks.loc[kept.index, RULE], by_cell=picks.loc[kept.index, BY_CELL])
        sections.append(("Runs", block(table.round(
            {"accuracy": 3, RULE: 2, BY_CELL: 2, ONE: 2}))))
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("read", "freeze", "test"))
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--rules", default=str(RULES_S_PATH))
    parser.add_argument("--allow-other-rules", action="store_true",
                        help="accept a rules file with another hash (tests only)")
    parser.add_argument("--stage", help="read another stage (dry runs only)")
    parser.add_argument("--count-missing-as-failed", action="store_true",
                        help="a run with no results (its training failed) counts "
                             "as a failure of gate T")
    parser.add_argument("--out")
    args = parser.parse_args()
    root, path = Path(args.results_root), Path(args.rules)
    if args.step == "read":
        single = read(root)
        units = by_set(single)
        text = ("# Amendment A15, check of the fit: one class set of amendment A14 "
                "left out\n\nEvery choice is fitted on the other runs. Far sets, "
                "final checkpoint, mean regret of the left-out class set:\n\n"
                + block(units.reset_index(), 2) + "\n\nMeans over the runs: "
                + ", ".join(f"{name} {single[name].mean():.2f}" for name in units)
                + "\n\nThe rule against the choice by cell, with the recipe as the "
                "unit: " + sentence(adds(single, RULE, BY_CELL)) + ".\n")
        out = Path(args.out) if args.out else HERE / "outputs" / "stage_s_read.md"
    elif args.step == "freeze":
        if path.exists():
            raise SystemExit(f"{path.name} exists: the rules are frozen once")
        content = build(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=1) + "\n")
        far = content["groups"][az.PRIMARY]
        text = (f"# Amendment A15: the frozen choices\n\nFile `{path.name}`, SHA-256 "
                f"`{rules.file_hash(path)}`.\n\n- Runs of the fit, at their final "
                f"checkpoint: {far['final_rows']}. By ID set (0 is Imagenette): "
                f"{content['runs_by_id_set']}. Cells of the fits on all "
                f"checkpoints: {content['training_cells']}.\n- Far sets: the choice "
                f"by cell is {far[BY_CELL]}, the fixed score is {far[ONE]}, and the "
                f"fixed shortlist is {far['shortlist']}.\n- Inputs of the rule "
                f"under test: {len(far['rules'][RULE]['columns'])}.\n")
        out = Path(args.out) if args.out else HERE / "outputs" / "stage_s_freeze.md"
    else:
        dry = bool(args.stage)
        stage = args.stage or TEST_STAGE
        digest = rules.file_hash(path)
        if digest != RULES_S_SHA256 and not args.allow_other_rules:
            raise SystemExit(f"{path.name} has hash {digest}, not the frozen one")
        if rules.file_hash(az.PARTS[1].path) != az.PARTS[1].sha256:
            raise SystemExit("the file of amendment A12 does not have its hash")
        frozen, old = load(path), az.load(az.PARTS[1].path)
        new = analyze_h.read_stage(root, (stage,))
        if new.metrics.empty:
            raise SystemExit(f"no results of stage {stage} under {root}/runs")
        runs = sorted(set(new.metrics["run"]))
        missing = sorted({spec.name for spec in STAGES[stage]} - set(runs))
        if missing and not args.count_missing_as_failed:
            raise SystemExit(f"stage {stage} is not complete; runs with no results: "
                             f"{', '.join(missing)}")
        counts = az.set_counts(root, runs)
        if frozen["images"] != counts:
            raise SystemExit("the image counts are not those of the frozen file")
        problems = [] if dry else frozen_inputs()
        wrong = ap.gate_d(root, runs)
        failures = ar.score_failures(new)
        passes = not problems and wrong.empty
        sections = [("Frozen files, gate D, and gate S", (
            f"Rules: `{path.name}`. Recipes: `{A15_RECIPES_PATH.name}`. Class sets: "
            f"`{A15_CLASS_SETS_PATH.name}`. {len(runs)} of {len(STAGES[stage])} runs "
            "have results." + (f" Runs with no results: {', '.join(missing)}."
                               if missing else "")
            + "\n\n" + ("**Gate D passes.** " + (
                "Dry run: the hashes of the recipes and class sets are not checked."
                if dry else "The frozen files have the recorded hashes, and every "
                "run has the settings of its recipe and the classes of its ID set.")
                if passes else "**Gate D fails.**"
                + "".join(f"\n- {p}" for p in problems)
                + ("" if wrong.empty else "\n\n" + block(wrong)))
            + "\n\n" + ("**Gate S passes.** Every score has a finite value at the "
                        "final checkpoint of every run." if failures.empty else
                        "**Gate S fails.** Final checkpoints with a score that has "
                        "an error or a non-finite value:\n\n" + block(failures))))]
        stop = ("" if passes else "gate D fails") or (
            "" if failures.empty else "gate S fails")
        if not stop:
            sections += test_sections(frozen, old, new, counts, stage)
            if "not decided" in sections[-1][1]:
                stop = "gate T: too many class sets leave"
        text = ("# Amendment A15" + (f" (dry run on stage {stage})" if dry else "")
                + ": a rule fitted on several ID sets against a choice from the "
                "recipe\n\nThe rule under test and the choice by cell come from "
                f"`{path.name}`. Each class set is one unit, with "
                f"{A15_PER_SET} recipes.\n"
                + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
        out = Path(args.out) if args.out else (
            HERE / "outputs" / f"stage_{stage}_report.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(text)
        if stop:
            raise SystemExit(f"{stop}: the test is not decided")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
