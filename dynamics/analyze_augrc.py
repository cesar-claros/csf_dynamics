"""Exploratory check: the selection results with other conventions for the ID errors.

    python -m dynamics.analyze_augrc --results-root dyn_results

The planned tests used the AUROC of all ID images against the unseen images. This
check reads the same runs with two other targets, both taken from stored values:

- `correct_only`: the AUROC of the correct ID images against the unseen images.
  The misclassified ID images are left out. This is the convention of the paper
  code (`compute_metrics` in `src/csf_pipeline.py`) and of the proposed mode of
  FD-Shifts. With it, AUROC and AUGRC give the scores the same order.
- `failure`: the AUROC of the correct ID images against the misclassified ID
  images and all unseen images. The stored AUGRC is a linear function of it.

All regrets are in AUROC points of the target, so the limit of 1 point keeps its
meaning. The check uses runs that every analysis has seen, so its result is a
description and not a test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_c, analyze_e, analyze_h, rules
from dynamics import analyze_shortlist as sl
from dynamics import analyze_t as at
from dynamics import analyze_u as au
from dynamics.analyze_e import block
from dynamics.analyze_selection import FAMILY
from dynamics.analyze_step0 import FAR_SETS

HERE = Path(__file__).resolve().parent
TESTS = ("t", "u")
NAMES = (at.TIERS, at.WIDE, at.TRAINING)
ALL_ID, CORRECT_ONLY, FAILURE = "all_id", "correct_only", "failure"
DERIVED = (CORRECT_ONLY, FAILURE)
#: The three earlier planned tests: label, frozen file, rules, stage, new recipes.
#: Amendment A6 tested the all-metrics rule of the first file. Its second rule,
#: fitted on 27 runs, is in the second file.
EARLIER = (("A5", rules.RULES_PATH, ("geometry", "all"), "c", ("C1", "C2", "C3")),
           ("A6", rules.RULES_PATH, ("all",), "c2", ("C4", "C5", "C6")),
           ("A6, second rule", rules.RULES_V2_PATH, ("all",), "c2",
            ("C4", "C5", "C6")),
           ("Stage H", analyze_h.RULES_H_PATH, ("all",), "h2", ("H5", "H6", "H7")))


def failure_auroc(augrc: np.ndarray, correct: np.ndarray, n_id: int, n_unseen: int
                  ) -> np.ndarray:
    """The failure AUROC that gives `augrc`, for `correct` correct ID images.

    With n images, f failures, and c correct images, the area of
    `metrics.risk_coverage` is ((1 - AUROC) c f + f (f + 1) / 2) / n^2. Ties count
    as one half in both values.
    """
    n = n_id + n_unseen
    wrong = n - correct
    return 1 - (augrc * n * n - wrong * (wrong + 1) / 2) / (correct * wrong)


def correct_only_auroc(failure: np.ndarray, errors: np.ndarray, wrong: np.ndarray,
                       n_unseen: int) -> np.ndarray:
    """AUROC of the correct ID images against the unseen images.

    The failure AUROC is the mean over two groups of failures: `wrong`
    misclassified ID images, with the AUROC `errors`, and the unseen images.
    Without a misclassified image, the two values are equal.
    """
    part = np.where(wrong > 0, wrong * np.nan_to_num(errors), 0.0)
    return (failure * (wrong + n_unseen) - part) / n_unseen


def counts(root: Path, runs: list[str]) -> tuple[int, int]:
    """ID and unseen images of each far set. They must be equal in every run."""
    found = set()
    for run in runs:
        # A run can have more than one evaluation. One with the far sets is read.
        for folder in sorted((Path(root) / "runs" / run).glob("results_f64*")):
            path = next(iter(sorted(folder.glob("epoch_*.json"))), None)
            stored = {} if path is None else json.loads(path.read_text()).get(
                "counts", {})
            if all(name in stored for name in FAR_SETS):
                found |= {(stored["val"], stored[name]) for name in FAR_SETS}
                break
        else:
            raise SystemExit(f"{run} has no evaluation with the far sets")
    if len(found) != 1:
        raise SystemExit(f"the runs do not have equal image counts: {sorted(found)}")
    return found.pop()


def correct_images(tables: analyze_e.Tables, frame: pd.DataFrame, n_id: int
                   ) -> np.ndarray:
    """Number of correct ID images of the checkpoint of each row of `frame`."""
    rows = tables.descriptors
    accuracy = rows[(rows["group"] == "clock") & (rows["name"] == "val_accuracy")
                    ].groupby(["run", "epoch"])["value"].mean()
    keys = pd.MultiIndex.from_frame(frame[["run", "epoch"]])
    return np.rint(accuracy.reindex(keys).to_numpy() * n_id)


def with_target(tables: analyze_e.Tables, n_id: int, n_unseen: int,
                target: str = FAILURE) -> analyze_e.Tables:
    """The same tables, with `target` as the `auroc` of the far sets."""
    metrics = tables.metrics.copy()
    far = metrics["set"].isin(FAR_SETS).to_numpy()
    part = metrics[far]
    correct = correct_images(tables, part, n_id)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = failure_auroc(part["augrc"].to_numpy(), correct, n_id, n_unseen)
        if target == CORRECT_ONLY:
            errors = metrics[metrics["set"] == "id_errors"].groupby(
                ["run", "epoch", "csf"])["auroc"].mean()
            keys = pd.MultiIndex.from_frame(part[["run", "epoch", "csf"]])
            value = correct_only_auroc(value, errors.reindex(keys).to_numpy(),
                                       n_id - correct, n_unseen)
    metrics.loc[far, "auroc"] = value
    return analyze_e.Tables(**{**vars(tables), "metrics": metrics})


def conversion_error(tables: analyze_e.Tables, n_id: int) -> float:
    """Largest error of the conversion on the ID-errors set, where both are stored."""
    part = tables.metrics[(tables.metrics["set"] == "id_errors")
                          & (tables.metrics["epoch"] >= 1)]
    derived = failure_auroc(part["augrc"].to_numpy(),
                            correct_images(tables, part, n_id), n_id, 0)
    return float(np.nanmax(np.abs(derived - part["auroc"].to_numpy())))


def final(tables: analyze_e.Tables) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta, scores, metrics = at.cells(tables)
    mask = analyze_c.final_mask(meta)
    return scores[mask].droplevel("epoch"), metrics[mask].droplevel("epoch")


def as_rules(built: dict) -> dict:
    return {**built, "rules": {name: rules.RidgeRule(**rule)
                               for name, rule in built["rules"].items()}}


def regrets(chooser: dict, scores: pd.DataFrame, metrics: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regret of the single choices and of the shortlists of size 3, per run."""
    recipe = analyze_h.recipes_of(scores.index)
    kind = recipe.map(FAMILY)
    single = pd.DataFrame({"recipe": recipe})
    for name in NAMES:
        single[name] = analyze_h.regret_of(
            scores, chooser["rules"][name].choose(metrics))
    single[at.RECIPE_RULE] = analyze_h.regret_of(
        scores, [chooser[at.RECIPE_RULE][k] for k in kind])
    single[at.B1] = analyze_h.regret_of(scores, [chooser[at.B1]] * len(scores))
    single[au.STAND_IN] = analyze_h.regret_of(
        scores, au.stand_in_choice(metrics, list(scores.columns)))
    short = pd.DataFrame({
        "recipe": recipe,
        "fixed": sl.shortlist_regret(scores, chooser["shortlists"][str(at.SHORT)]),
        "rule": sl.shortlist_regret(scores, analyze_h.top(
            chooser["rules"][at.TIERS], metrics, at.SHORT)),
        "random": sl.random_regret(scores, at.SHORT)})
    return single, short


def summary(single: pd.DataFrame) -> pd.DataFrame:
    """Per test: the mean regret of each choice, and if it holds by the limits."""
    names = [c for c in single.columns if c not in ("recipe", "test")]
    rows = []
    for test, part in single.groupby("test"):
        for name in names:
            by_recipe = part.groupby("recipe")[name].mean()
            rows.append({"test": test, "choice": name, "mean": part[name].mean(),
                         "worst_recipe": by_recipe.max(),
                         "recipes_above_1": int((by_recipe > at.HOLD).sum()),
                         "holds": at.holds(part, name)})
    return pd.DataFrame(rows)


def earlier_tests(root: Path) -> pd.DataFrame:
    """The frozen choices of the three earlier tests, with each target.

    One row per test and choice: the mean regret on the new recipes and the regret
    of the worst recipe, in AUROC points of each target.
    """
    rows = []
    for label, path, names, stage, recipes in EARLIER:
        if label == "Stage H":
            frozen = analyze_h.load(path)
            one, by_kind = frozen["b1"], frozen["recipe_rule"]
        else:
            frozen = rules.load(path)["targets"]["far"]
            base = frozen["baselines"]
            one = base["b1"]["score"]
            by_kind = {kind: entry["score"] for kind, entry in base["b2"].items()}
        tables = analyze_h.read_stage(root, (stage,))
        tables = analyze_e.Tables(**{
            key: frame if frame.empty else frame[frame["recipe"].isin(recipes)]
            .reset_index(drop=True) for key, frame in vars(tables).items()})
        n_id, n_unseen = counts(root, sorted(set(tables.metrics["run"])))
        found: dict[str, dict[str, pd.Series]] = {}
        for target in (ALL_ID, *DERIVED):
            used = tables if target == ALL_ID else with_target(
                tables, n_id, n_unseen, target)
            scores, metrics = sl.final_scores(used, FAR_SETS)
            recipe = analyze_h.recipes_of(scores.index)
            picks = {f"rule {name}": frozen["rules"][name].choose(metrics)
                     for name in names}
            picks[at.B1] = [one] * len(scores)
            picks[at.RECIPE_RULE] = [by_kind[kind] for kind in recipe.map(FAMILY)]
            for choice, chosen in picks.items():
                regret = analyze_h.regret_of(scores, chosen)
                found.setdefault(choice, {})[target] = regret.groupby(recipe).mean()
        for choice, each in found.items():
            rows.append({"test": label, "choice": choice,
                         **{f"{target}_{name}": getattr(each[target], name)()
                            for target in (ALL_ID, *DERIVED)
                            for name in ("mean", "max")}})
    return pd.DataFrame(rows)


def stage_h_fitted(root: Path, target: str) -> pd.DataFrame:
    """Stage H with choices fitted for `target` on phase 1, read on phase 2."""
    first = analyze_h.read_stage(root, (analyze_h.PHASE_1,))
    second = analyze_h.read_stage(root, (analyze_h.PHASE_2,))
    n_id, n_unseen = counts(root, sorted(set(first.metrics["run"])
                                         | set(second.metrics["run"])))
    fitted = as_rules(analyze_h.build(with_target(first, n_id, n_unseen, target)))
    scores, metrics = sl.final_scores(with_target(second, n_id, n_unseen, target),
                                      FAR_SETS)
    recipe = analyze_h.recipes_of(scores.index)
    size = str(analyze_h.TEST_SIZE)
    rule = fitted["rules"][analyze_h.RULE]
    found = {
        "rule all": analyze_h.regret_of(scores, rule.choose(metrics)),
        f"one score ({fitted['b1']})": analyze_h.regret_of(
            scores, [fitted["b1"]] * len(scores)),
        "recipe rule": analyze_h.regret_of(
            scores, [fitted["recipe_rule"][k] for k in recipe.map(FAMILY)]),
        f"fixed shortlist ({', '.join(fitted['shortlists'][size])})":
            sl.shortlist_regret(scores, fitted["shortlists"][size]),
        "top 3 of the rule": sl.shortlist_regret(scores, analyze_h.top(
            rule, metrics, analyze_h.TEST_SIZE)),
        "random 3": sl.random_regret(scores, analyze_h.TEST_SIZE)}
    one = next(values for name, values in found.items() if name.startswith("one"))
    rows = []
    for name, values in found.items():
        mean, low, high = analyze_c.interval((values - one).to_numpy())
        rows.append({"choice": name, "mean": values.mean(),
                     "worst_recipe": values.groupby(recipe).mean().max(),
                     "minus_one_score": mean, "low": low, "high": high})
    return pd.DataFrame(rows)


def agreement(new: analyze_e.Tables, targets: dict) -> pd.DataFrame:
    """Rank agreement of the scores between the targets, by recipe (Spearman)."""
    base = targets[ALL_ID][0]
    recipe = analyze_h.recipes_of(base.index)
    rows = {f"{ALL_ID} with {name}": base.corrwith(targets[name][0], axis=1,
                                                   method="spearman")
            for name in DERIVED}
    rows[f"{CORRECT_ONLY} with {FAILURE}"] = targets[CORRECT_ONLY][0].corrwith(
        targets[FAILURE][0], axis=1, method="spearman")
    table = pd.DataFrame(rows).groupby(recipe).mean()
    table.loc["mean"] = table.mean()
    return table.reset_index(names="recipe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", default=str(HERE / "outputs" / "augrc_report.md"))
    args = parser.parse_args()
    root = Path(args.results_root)
    if rules.file_hash(at.RULES_T_PATH) != at.RULES_T_SHA256:
        raise SystemExit("the rules file of amendment A7 is not the frozen one")
    frozen = at.load(at.RULES_T_PATH)
    earlier = at.earlier_tables(root)
    new = analyze_h.read_stage(root, TESTS)
    runs = sorted(set(earlier.metrics["run"]) | set(new.metrics["run"]))
    n_id, n_unseen = counts(root, runs)
    error = max(conversion_error(earlier, n_id), conversion_error(new, n_id))
    targets = {ALL_ID: final(new), **{
        name: final(with_target(new, n_id, n_unseen, name)) for name in DERIVED}}
    test = analyze_h.recipes_of(targets[ALL_ID][0].index).str[0]
    accuracy = targets[ALL_ID][1]["clock/val_accuracy"]
    wrong = (1 - accuracy) * n_id
    mix = pd.DataFrame({
        "recipe": analyze_h.recipes_of(accuracy.index),
        "id_errors_of_id": 1 - accuracy,
        "id_errors_of_failures": wrong / (wrong + n_unseen),
        **{f"best_{name}": targets[name][0].idxmax(axis=1) for name in targets}})
    sections = [
        ("The targets", (
            f"- Images of each far set: {n_id} ID and {n_unseen} unseen, in every "
            "run.\n- Largest error of the conversion from AUGRC to failure AUROC, "
            f"on the ID-errors set where both are stored: {error:.2e}.\n- Share of "
            "the misclassified ID images, and the best score with each target:\n\n"
            + block(mix.reset_index(), 3) + "\n\nRank agreement of the scores "
            "between the targets:\n\n" + block(agreement(new, targets), 2)))]
    cases = [("Frozen choices, target all_id (the planned tests)", frozen, ALL_ID)]
    for name in DERIVED:
        built = at.build(with_target(earlier, n_id, n_unseen, name), names=NAMES,
                         amendment="check of the target")       # in memory only
        cases += [(f"Frozen choices, target {name}", frozen, name),
                  (f"Choices fitted for {name}, target {name}", as_rules(built),
                   name)]
    for label, chooser, name in cases:
        single, short = regrets(chooser, *targets[name])
        single, short = single.assign(test=test), short.assign(test=test)
        note = ("Fitted, in memory: one score " + f"{chooser[at.B1]}, recipe rule "
                f"{chooser[at.RECIPE_RULE]}, shortlist "
                f"{chooser['shortlists'][str(at.SHORT)]}.\n\n"
                if "fitted" in label else "")
        sections.append((label, (
            note + "Regret by recipe, in points of the target:\n\n"
            + block(single.drop(columns="test").groupby("recipe").mean()
                    .reset_index(), 2) + "\n\n" + block(summary(single), 2)
            + "\n\nShortlists of size 3, by test:\n\n"
            + block(short.drop(columns="recipe").groupby("test").mean()
                    .reset_index(), 2))))
    for name in DERIVED:
        scores = targets[name][0]
        fixed = (scores.max(axis=1).to_numpy()[:, None] - scores).groupby(
            analyze_h.recipes_of(scores.index)).mean().T
        fixed["mean"], fixed["worst"] = fixed.mean(axis=1), fixed.max(axis=1)
        sections.append((f"Each fixed score, target {name}", (
            "Regret by recipe, with the mean and the worst recipe:\n\n"
            + block(fixed.sort_values("worst").reset_index(), 2))))
    sections.append(("The frozen choices of the three earlier tests", (
        "Mean regret over the new recipes of each test and the regret of the "
        "worst recipe, with each target:\n\n" + block(earlier_tests(root), 2))))
    for name in DERIVED:
        sections.append((f"Stage H, choices fitted for {name} on phase 1", (
            "Read on the 6 runs of phase 2:\n\n"
            + block(stage_h_fitted(root, name), 2))))
    text = ("# Exploratory check: the convention for the ID errors in the target\n\n"
            "Post hoc, on runs that every analysis has seen. No file is frozen.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
