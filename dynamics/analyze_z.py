"""Amendment A12: a rule from the statistics of each score (freeze and test, two parts).

    python -m dynamics.analyze_z freeze --part 1 --results-root dyn_results
    python -m dynamics.analyze_z test --part 1 --results-root dyn_results

The inputs of the rule under test are, for each score, its AUROC on three sets
that are made from the validation images: ID errors, class blends, and shuffled
patches. Part 1 has 10 classes (recipes Z1 to Z8) and part 2 has 100 classes
(recipes K1 to K4). Each part has its own frozen file.

The target is the AUROC of the correct ID validation images against the unseen
images. The outcome uses the far sets. The near sets and textures are planned
secondary targets, each with its own rules. The unit of every interval is the
recipe.

Protocol: ../documentation/score_statistics_rule_amendment_2026-11-20.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_augrc as ag
from dynamics import analyze_c, analyze_e, analyze_h, rules
from dynamics import analyze_loro as lo
from dynamics import analyze_multimetric as mm
from dynamics import analyze_shortlist as sl
from dynamics import analyze_t as at
from dynamics import analyze_u as au
from dynamics import analyze_v as av
from dynamics.analyze_e import block
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import (
    DEFAULT_PRECISION,
    RECIPES,
    STAGES,
    parse_run,
    results_folder,
)

HERE = Path(__file__).resolve().parent
#: The targets. The first one gives the outcome of a part.
GROUPS = {"far": FAR_SETS, "near": ("ssb_hard", "ninco", "imagewoof"),
          "textures": ("textures",)}
PRIMARY = "far"
RULE, ONE, STAND_IN, TWO = "score_statistics", "one_score", au.STAND_IN, "two_rules"
#: Rules that are frozen with the rule under test: two for question 3, two controls.
QUESTION_3 = ("training", "validation")
CONTROLS = ("all_metrics", "clock")
WITH, WITHOUT = "smoothing", "no_smoothing"
HOLD = 1.0
MIN_ACCURACY = 0.60
METRIC_LIMIT = 1e-6
SHORT = 3
#: SHA-256 of the two frozen files. Set after `freeze`, before any new run.
RULES_Z_SHA256 = "57582220c5461d7e02e7db9fd56837056c161414b5985fe8baaf397bdcef6d1c"
RULES_K_SHA256 = "9005791fe17efbc6b98aaea19537f33b3f936baf98a5d65742ff8f636cf947a3"


@dataclass(frozen=True)
class Part:
    label: str
    train_stage: str
    test_stage: str
    path: Path
    sha256: str


PARTS = {
    1: Part("part 1, 10 classes", "a12_train", "z",
            HERE / "frozen" / "stage_z_rules.json", RULES_Z_SHA256),
    2: Part("part 2, 100 classes", "a12_train_h", "k",
            HERE / "frozen" / "stage_k_rules.json", RULES_K_SHA256)}


def every_set() -> tuple[str, ...]:
    return tuple(name for sets in GROUPS.values() for name in sets)


def set_counts(root: Path, runs: list[str]) -> dict[str, int]:
    """ID images and unseen images of each set. They must be equal in every run."""
    found: dict[str, set[int]] = {}
    tag = results_folder(DEFAULT_PRECISION, suite=analyze_c.SUITE)
    for run in runs:
        path = max((Path(root) / "runs" / run / tag).glob("epoch_*.json"))
        stored = json.loads(path.read_text())["counts"]
        for name in ("val", *every_set()):
            found.setdefault(name, set()).add(int(stored[name]))
    if any(len(values) != 1 for values in found.values()):
        raise SystemExit(f"the runs do not have equal image counts: {found}")
    return {name: values.pop() for name, values in found.items()}


def aimed(tables: analyze_e.Tables, counts: dict[str, int],
          sets: tuple[str, ...]) -> analyze_e.Tables:
    """The same tables, with the correct-only AUROC as the `auroc` of `sets`.

    The value comes from the stored AUGRC and the stored ID-errors AUROC, as in
    `analyze_augrc`, with the number of images of each set.
    """
    metrics = tables.metrics.copy()
    errors = metrics[metrics["set"] == "id_errors"].groupby(
        ["run", "epoch", "csf"])["auroc"].mean()
    n_id = counts["val"]
    for name in sets:
        rows = (metrics["set"] == name).to_numpy()
        part = metrics[rows]
        correct = ag.correct_images(tables, part, n_id)
        keys = pd.MultiIndex.from_frame(part[["run", "epoch", "csf"]])
        with np.errstate(divide="ignore", invalid="ignore"):
            failure = ag.failure_auroc(part["augrc"].to_numpy(), correct, n_id,
                                       counts[name])
            metrics.loc[rows, "auroc"] = ag.correct_only_auroc(
                failure, errors.reindex(keys).to_numpy(), n_id - correct,
                counts[name])
    return analyze_e.Tables(**{**vars(tables), "metrics": metrics})


def stored(tables: analyze_e.Tables, sets: tuple[str, ...]) -> analyze_e.Tables:
    """The same tables, with the stored correct-only AUROC as the `auroc` of `sets`."""
    metrics = tables.metrics.copy()
    rows = metrics["set"].isin(sets).to_numpy()
    metrics.loc[rows, "auroc"] = metrics.loc[rows, "auroc_correct"]
    return analyze_e.Tables(**{**vars(tables), "metrics": metrics})


def metric_gap(tables: analyze_e.Tables, counts: dict[str, int]) -> tuple[float, int]:
    """Gate M on every set of the targets: the stored target against the derived one."""
    if "auroc_correct" not in tables.metrics:
        raise SystemExit("the runs do not store the correct-only values")
    sets = every_set()
    derived = aimed(tables, counts, sets).metrics
    rows = (derived["set"].isin(sets) & (derived["epoch"] >= 1)).to_numpy()
    kept = tables.metrics.loc[rows, "auroc_correct"].to_numpy(dtype=np.float64)
    other = derived.loc[rows, "auroc"].to_numpy(dtype=np.float64)
    both = np.isfinite(kept) & np.isfinite(other)
    if not both.any():
        raise SystemExit("no row with a stored and a derived target")
    return float(np.abs(kept[both] - other[both]).max()), int((~both).sum())


def group_scores(tables: analyze_e.Tables, sets: tuple[str, ...],
                 index: pd.Index) -> pd.DataFrame:
    """Mean AUROC of each score over `sets`, in points, for the cells of `index`."""
    return mm.build_cells(tables, sets)[1].loc[index]


def smoothing_of(index: pd.Index) -> np.ndarray:
    """True for the rows of a run whose recipe has label smoothing."""
    runs = index.get_level_values("run") if index.nlevels > 1 else index
    return np.array([RECIPES[parse_run(run).recipe].label_smoothing > 0
                     for run in runs])


def fit_rules(scores: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, dict]:
    """Every rule of one target, fitted on the cells of the earlier runs."""
    groups = lo.input_groups(list(mm.transform(metrics).columns))
    found = {name: rules.fit_ridge_rule(metrics, scores, groups[name])
             for name in (RULE, *QUESTION_3, *CONTROLS)}
    flag = smoothing_of(metrics.index)
    for kind, rows in ((WITH, flag), (WITHOUT, ~flag)):
        columns = lo.input_groups(list(mm.transform(metrics[rows]).columns))[RULE]
        found[f"{TWO}/{kind}"] = rules.fit_ridge_rule(metrics[rows], scores[rows],
                                                      columns)
    return {name: asdict(rule) for name, rule in found.items()}


def training_tables(number: int, root: Path) -> analyze_e.Tables:
    """The earlier runs of a part: 76 with 10 classes, or 30 with 100 classes.

    The first 12 runs with 10 classes come from their reference-check evaluation,
    as in the earlier amendments, because it has every metric of the later runs.
    """
    if number == 2:
        return analyze_h.read_stage(root, (PARTS[number].train_stage,))
    return av.join(av.ten_class_tables(root, with_v=True),
                   analyze_h.read_stage(root, (av.PARTS[3].test_stage,)))


def build(number: int, root: Path) -> dict:
    """The rules and the fixed choices of a part, for each target."""
    tables = training_tables(number, root)
    counts = set_counts(root, sorted(set(tables.metrics["run"])))
    meta, _, metrics = at.cells(tables)
    final = analyze_c.final_mask(meta)
    usual = max(RECIPES[spec.recipe].epochs for spec in STAGES[PARTS[number].test_stage])
    kept = final & np.array([RECIPES[parse_run(run).recipe].epochs == usual
                             for run in meta.index.get_level_values("run")])
    groups = {}
    for name, sets in GROUPS.items():
        scores = group_scores(aimed(tables, counts, sets), sets, meta.index)
        if not np.isfinite(scores.to_numpy()).all():
            raise SystemExit(f"a score has no value on the {name} sets")
        last = scores[kept].droplevel("epoch")
        groups[name] = {
            "sets": list(sets), "rules": fit_rules(scores, metrics),
            ONE: str(last.mean().idxmax()), "baseline_runs": len(last),
            "shortlists": {str(k): sl.greedy(last, k) for k in (SHORT, 5)}}
    return {"amendment": "A12", "part": number, "target": ag.CORRECT_ONLY,
            "images": counts, "penalty": mm.PENALTY,
            "training_runs": sorted(set(meta.index.get_level_values("run"))),
            "training_cells": len(meta), "groups": groups}


def load(path: Path) -> dict:
    frozen = json.loads(Path(path).read_text())
    for group in frozen["groups"].values():
        group["rules"] = {name: rules.RidgeRule(**rule)
                          for name, rule in group["rules"].items()}
    return frozen


def regrets(group: dict, scores: pd.DataFrame, metrics: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The choices, their regret, and the regret of the shortlists, per run."""
    found = group["rules"]
    flag = smoothing_of(scores.index)
    both = [found[f"{TWO}/{WITH}"].choose(metrics),
            found[f"{TWO}/{WITHOUT}"].choose(metrics)]
    picks = pd.DataFrame({
        RULE: found[RULE].choose(metrics),
        TWO: [both[0][i] if on else both[1][i] for i, on in enumerate(flag)],
        **{name: found[name].choose(metrics) for name in (*QUESTION_3, *CONTROLS)},
        ONE: [group[ONE]] * len(scores),
        STAND_IN: au.stand_in_choice(metrics, list(scores.columns))},
        index=scores.index)
    recipe = analyze_h.recipes_of(scores.index)
    single = pd.DataFrame({"recipe": recipe, WITH: flag})
    for name in picks.columns:
        single[name] = analyze_h.regret_of(scores, list(picks[name]))
    short = pd.DataFrame({
        "recipe": recipe,
        "fixed": sl.shortlist_regret(scores, group["shortlists"][str(SHORT)]),
        "rule": sl.shortlist_regret(scores, analyze_h.top(found[RULE], metrics,
                                                          SHORT)),
        "random": sl.random_regret(scores, SHORT)})
    return picks, single, short


def by_recipe(single: pd.DataFrame) -> pd.DataFrame:
    """The regret of each choice with the recipe as the unit: the mean of its runs."""
    names = [c for c in single.columns if c not in ("recipe", WITH)]
    return single.groupby("recipe")[names].mean()


def holds(recipes: pd.DataFrame, name: str) -> bool:
    """Within 1 point on the mean, and above it in fewer than half of the recipes."""
    return bool(recipes[name].mean() <= HOLD
                and 2 * (recipes[name] > HOLD).sum() < len(recipes))


def adds(recipes: pd.DataFrame, name: str, other: str = ONE) -> dict:
    """Question 1 for one choice: its difference from another, over the recipes."""
    difference = recipes[name] - recipes[other]
    mean, low, high = analyze_c.interval(difference.to_numpy())
    supported = bool(mean < 0 and high < 0)
    share = (float(difference.min() / difference.sum())
             if difference.sum() < 0 else 0.0)
    reading = "not supported"
    if supported and share > 0.5:
        reading = (f"supported, and the choice avoids one bad case of the fixed "
                   f"score: {difference.idxmin()} gives {share:.0%} of the difference")
    elif supported:
        reading = "supported, and no recipe gives more than half of the difference"
    return {"choice": name, "regret": recipes[name].mean(),
            "fixed_regret": recipes[other].mean(), "difference": mean, "low": low,
            "high": high, "lower": int((difference < 0).sum()),
            "higher": int((difference > 0).sum()), "supported": supported,
            "reading": reading}


def outcome(recipes: pd.DataFrame) -> str:
    """A, H, or F for the rule under test, by the decision rule of amendment A12."""
    if recipes.empty or recipes[[RULE, ONE]].isna().any().any():
        return "not decided: a run has no choice"
    gain = adds(recipes, RULE)["supported"]
    if not holds(recipes, RULE):
        return ("F, the rule is not within 1 point"
                + (", and it is below the fixed score" if gain else ""))
    return ("A, the rule adds to the fixed score" if gain
            else "H, the rule holds with no shown gain")


def augrc_regret(tables: analyze_e.Tables, sets: tuple[str, ...],
                 picks: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Regret of the rule and of the fixed score in AUGRC points, per run.

    `other_order` marks a run where the best score by the mean AUGRC is not the
    best score by the mean AUROC.
    """
    part = tables.metrics[tables.metrics["set"].isin(sets)]
    last = part.groupby("run")["epoch"].transform("max") == part["epoch"]
    wide = 100 * part[last].pivot_table(index="run", columns="csf",
                                        values="augrc_correct")
    rows = []
    for run in picks.index:
        row = {"run": run, "best_augrc": wide.loc[run].min(),
               "other_order": wide.loc[run].idxmin() != scores.loc[run].idxmax()}
        for name in (RULE, ONE):
            pick = picks.loc[run, name]
            row[name] = (wide.loc[run, pick] - wide.loc[run].min()
                         if isinstance(pick, str) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("freeze", "test"))
    parser.add_argument("--part", type=int, choices=sorted(PARTS), required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--rules", help="another rules file of the part (tests only)")
    parser.add_argument("--allow-other-rules", action="store_true",
                        help="accept a rules file with another hash (tests only)")
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
        wanted = {spec.name for spec in STAGES[part.train_stage]}
        ready = set(training_tables(args.part, root).metrics["run"])
        if wanted - ready:
            raise SystemExit(f"runs without results: {sorted(wanted - ready)}")
        digest = rules.save(build(args.part, root), path)
        frozen = load(path)
        lines = [(f"- File: `{path}`\n- SHA-256: `{digest}`\n- Target: "
                  f"{frozen['target']}. Images: {frozen['images']}.\n- Training "
                  f"runs: {len(frozen['training_runs'])}, cells: "
                  f"{frozen['training_cells']}.")]
        for name, group in frozen["groups"].items():
            sizes = {rule: len(found.columns) for rule, found in group["rules"].items()}
            lines.append(f"- **{name}** ({', '.join(group['sets'])}): one score "
                         f"{group[ONE]}, shortlists {group['shortlists']}, baseline "
                         f"runs {group['baseline_runs']}, inputs {sizes}.")
        sections.append((f"Frozen file of {part.label}", "\n".join(lines)))
    else:
        digest = rules.file_hash(path)
        if digest != part.sha256 and not args.allow_other_rules:
            raise SystemExit(f"{path.name} has hash {digest}, not the frozen one")
        frozen = load(path)
        stage = args.stage or part.test_stage
        new = analyze_h.read_stage(root, (stage,))
        if new.metrics.empty:
            raise SystemExit(f"no results of stage {stage} under {root}/runs")
        counts = set_counts(root, sorted(set(new.metrics["run"])))
        if frozen["images"] != counts:
            raise SystemExit("the image counts are not those of the frozen file")
        if args.derived_target:
            gate_m = "Dry run: the target comes from the old stored values."
        else:
            gap, skipped = metric_gap(new, counts)
            if gap > METRIC_LIMIT:
                raise SystemExit(f"gate M fails: the stored and the derived target "
                                 f"differ by {gap:.2e}")
            gate_m = (f"**Gate M passes** on the sets of every target. Largest "
                      f"difference between the stored and the derived target: "
                      f"{gap:.2e} (limit {METRIC_LIMIT:.0e}). Rows that could not be "
                      f"compared: {skipped}.")
        meta, _, metrics = at.cells(new)
        final = analyze_c.final_mask(meta)
        last_metrics = metrics[final].droplevel("epoch")
        accuracy = last_metrics["clock/val_accuracy"]
        recipe = analyze_h.recipes_of(last_metrics.index)
        failed = sorted(set(recipe[accuracy < MIN_ACCURACY]))
        keep = ~recipe.isin(failed).to_numpy()
        gate = pd.DataFrame({"recipe": recipe, "val_accuracy": accuracy})
        sections.append(("Runs and gates", (
            f"{len(accuracy)} of {len(STAGES[stage])} runs. Gate T: a run passes "
            f"with a validation accuracy of at least {MIN_ACCURACY:.0%}:\n\n"
            + block(gate.reset_index(), 3) + "\n\nRecipes that leave the decision "
            f"rule: {', '.join(failed) or 'none'}.\n\n" + gate_m)))
        for name, sets in GROUPS.items():
            used = aimed(new, counts, sets) if args.derived_target else stored(new, sets)
            scores = group_scores(used, sets, meta.index)[final].droplevel("epoch")
            scores, kept_metrics = scores[keep], last_metrics[keep]
            group = frozen["groups"][name]
            picks, single, short = regrets(group, scores, kept_metrics)
            recipes = by_recipe(single)
            readings = pd.DataFrame([adds(recipes, choice) for choice in
                                     (RULE, TWO, *QUESTION_3, *CONTROLS, STAND_IN)])
            kinds = single.drop(columns="recipe").groupby(WITH).mean()
            kinds.index = [WITH if on else WITHOUT for on in kinds.index]
            reading, difference = analyze_h.read_shortlists(short)
            title = (f"Target {name}: the outcome" if name == PRIMARY
                     else f"Target {name} (secondary, no outcome letter)")
            text = (
                f"Sets: {', '.join(sets)}. One score: {group[ONE]}. Regret at the "
                "final checkpoint, in points of the correct-only AUROC, with the "
                "recipe as the unit:\n\n" + block(recipes.reset_index(), 2)
                + "\n\nMeans: " + ", ".join(
                    f"{choice} {recipes[choice].mean():.2f}" for choice in recipes)
                + "\n\nBy the use of label smoothing:\n\n"
                + block(kinds.reset_index(names="recipes"), 2)
                + "\n\nEach choice against the one fixed score, with the interval "
                "over the recipes:\n\n" + block(readings, 2)
                + "\n\nChoices:\n\n" + block(picks[[RULE, TWO, ONE, STAND_IN]]
                                             .reset_index())
                + f"\n\n- Question 1, the rule adds: {readings.loc[0, 'reading']}."
                f"\n- Question 2, the rule holds: "
                f"{'yes' if holds(recipes, RULE) else 'no'}.\n- Question 3, "
                f"training statistics: {readings.loc[2, 'reading']}. Validation "
                f"statistics: {readings.loc[3, 'reading']}.\n- Two rules by label "
                f"smoothing (secondary): {readings.loc[1, 'reading']}.\n- Shortlists "
                f"of size 3: fixed {short['fixed'].mean():.2f}, rule "
                f"{short['rule'].mean():.2f}, random {short['random'].mean():.2f}; "
                f"rule minus fixed {difference[0]:.2f} ({difference[1]:.2f} to "
                f"{difference[2]:.2f}). {reading.capitalize()}.")
            if "augrc_correct" in new.metrics:
                augrc = augrc_regret(new, sets, picks, scores)
                text += (
                    f"\n- Regret in AUGRC points: rule {augrc[RULE].mean():.3f}, one "
                    f"score {augrc[ONE].mean():.3f}. Runs where the best score by "
                    f"AUGRC is not the best by AUROC: {int(augrc['other_order'].sum())}"
                    f" of {len(augrc)}.")
            if name == PRIMARY:
                text += f"\n\n**Outcome: {outcome(recipes)}.**"
            sections.append((title, text))
    text = (f"# Amendment A12, {part.label}: a rule from the statistics of each score"
            "\n\nThe inputs, the rules, the targets, and the decision rule were "
            "fixed before the new runs.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
