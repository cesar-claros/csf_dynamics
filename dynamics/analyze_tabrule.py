"""Learner swap for the selection rule: tables for another learner, and the read.

The selection rule is a ridge regression from the statistics of the scores to the
logit AUROC of each score. This script writes the tables of every fit of four reads
that the ridge rule already has, so that another learner can make the predictions
with the same inputs, target, and fit rows. It then turns the predictions of that
learner into choices and compares them with the choices of the ridge rule.

    python -m dynamics.analyze_tabrule export --results-root dyn_results
    python -m dynamics.analyze_tabrule report --results-root dyn_results \
        --predictions rule_predictions.npz

The learner itself is not here. The fit runs in the repository of the probe
(`csf_dynamics_tabpfn`, `dynamics/fit_tables.py`), which reads only the tables.

Protocol: ../documentation/tabpfn_rule_check_2026-11-27.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_c, analyze_h, rules, zoo_timm_list
from dynamics import analyze_i as ai
from dynamics import analyze_j as aj
from dynamics import analyze_multimetric as mm
from dynamics import analyze_s as an
from dynamics import analyze_t as at
from dynamics import analyze_z as az
from dynamics.analyze_e import block
from dynamics.config import STAGES
from dynamics.scores import ALL_CSFS

HERE = Path(__file__).resolve().parent
TABLES = HERE / "outputs" / "rule_tables.npz"
#: The stored choice of the ridge rule, and the fixed score of the same read.
RIDGE, FIXED = "ridge_rule", "fixed_score"
#: The learner of the verdict, and the bed that gives the letter.
PRIMARY_LEARNER, PRIMARY_BED = "tabpfn_v3.5", "P"
#: The ridge regression through the tables. It must repeat the stored choices.
CHECK_LEARNER = "ridge"


@dataclass
class Fold:
    """One fit: the transformed inputs and the target of its rows, and the test rows."""

    name: str
    x_fit: np.ndarray
    y_fit: np.ndarray
    weights: np.ndarray
    x_test: np.ndarray
    test_ids: list[str]


@dataclass
class Bed:
    """One read of the ridge rule: its fits, its test units, and its stored choices."""

    name: str
    title: str
    unit: str
    csfs: list[str]
    #: Per test model: the AUROC of each score on the target, in points.
    scores: pd.DataFrame
    #: Per test model that enters the read: its unit.
    units: pd.Series
    #: Per test model: the stored choice of the ridge rule and of the fixed score.
    stored: pd.DataFrame
    folds: list[Fold] = field(default_factory=list)


def fold_of(name: str, rule: rules.RidgeRule, fit_metrics: pd.DataFrame,
            fit_scores: pd.DataFrame, weights: np.ndarray,
            test_metrics: pd.DataFrame) -> Fold:
    """The tables of one fit, with the transform of its ridge rule.

    Raises:
        ValueError: If a table has a value that is not finite.
    """
    fold = Fold(name, rule.inputs(fit_metrics),
                mm.logit(fit_scores[rule.csfs].to_numpy(dtype=np.float64)),
                np.asarray(weights, dtype=np.float64), rule.inputs(test_metrics),
                [str(item) for item in test_metrics.index])
    for table in (fold.x_fit, fold.y_fit, fold.weights, fold.x_test):
        if not np.isfinite(table).all():
            raise ValueError(f"fit {name}: a table has a value that is not finite")
    return fold


def ridge_predictions(fold: Fold, penalty: float = mm.PENALTY) -> np.ndarray:
    """The predictions of the weighted ridge regression of the rule, from tables."""
    w = fold.weights * len(fold.weights) / fold.weights.sum()
    offset = (w[:, None] * fold.y_fit).sum(axis=0) / w.sum()
    x = fold.x_fit
    beta = np.linalg.solve(x.T @ (w[:, None] * x) + penalty * np.eye(x.shape[1]),
                           x.T @ (w[:, None] * (fold.y_fit - offset)))
    return fold.x_test @ beta + offset


def picks_of(predictions: np.ndarray, csfs: list[str]) -> list[str]:
    """The score with the largest prediction, for each row."""
    return [csfs[int(row.argmax())] for row in predictions]


def a16_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inputs, far targets, and the list of the 100 models of amendment A16."""
    tables, listed, _, _ = aj.fit_tables(root)
    csfs = list(ALL_CSFS)
    metrics = ai.inputs_of(tables, csfs).loc[listed.index]
    scores = ai.targets_of(tables, ai.GROUPS[ai.PRIMARY], csfs).loc[listed.index]
    return metrics, scores, listed


def bed_a16(root: Path) -> Bed:
    """S1: amendment A16, each lineage left out of the fit once."""
    metrics, scores, listed = a16_tables(root)
    picks, _ = ai.leave_out(metrics, scores, listed)
    stored = pd.DataFrame({RIDGE: picks[ai.RULE], FIXED: picks[ai.ONE]})
    bed = Bed("S1", "A16: the 100 torchvision models, each lineage left out",
              "lineage", list(scores.columns), scores, listed["lineage"], stored)
    for lineage in sorted(listed["lineage"].unique()):
        test = (listed["lineage"] == lineage).to_numpy()
        weights = ai.balanced(listed.loc[~test, "lineage"])
        rule = ai.fit_weighted(metrics[~test], scores[~test],
                               ai.usable(metrics[~test]), weights)
        bed.folds.append(fold_of(lineage, rule, metrics[~test], scores[~test],
                                 weights, metrics[test]))
    return bed


def timm_tables(root: Path, group: dict
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The timm models that pass the gates of amendment A17.

    Returns:
        Their inputs, their far targets, their rows of the list, and the frozen
        choices for each of them.
    """
    listed = pd.DataFrame(zoo_timm_list.load()).set_index("id")
    tables, facts = ai.read(Path(root), aj.RESULTS)
    facts = facts.set_index("model")
    csfs = list(ALL_CSFS)
    present = [name for name in listed.index if name in facts.index]
    every = ai.inputs_of(tables, csfs).reindex(present)
    usable = pd.Series(np.isfinite(every.to_numpy()).all(axis=1), index=present)
    left = set(aj.gates(listed, facts, usable)["model"])
    kept = [name for name in present if name not in left]
    metrics = every.loc[kept]
    scores = ai.targets_of(ai.keep(tables, kept), tuple(group["sets"]),
                           csfs).loc[kept]
    chosen = listed.loc[kept]
    picks = aj.choose(group, metrics, facts.loc[kept, "val_accuracy"],
                      aj.cells_of(chosen))
    return metrics, scores, chosen, picks


def beds_a17(root: Path) -> tuple[Bed, Bed]:
    """P and S2: the new lineages of amendment A17.

    P uses the rule that is frozen on the 100 models of amendment A16. S2 fits a
    rule on the other new lineages, with each lineage left out once.
    """
    group = aj.load()["groups"][aj.PRIMARY]
    metrics, scores, chosen, picks = timm_tables(root, group)
    new = (chosen["group"] == "N").to_numpy()
    metrics, scores, chosen, picks = (metrics[new], scores[new], chosen[new],
                                      picks[new])
    lineage = chosen["lineage"]
    fit_metrics, fit_scores, fit_listed = a16_tables(root)
    frozen = Bed("P", "A17: the rule frozen on the 100 torchvision models, on the "
                 "new lineages", "lineage", list(scores.columns), scores, lineage,
                 pd.DataFrame({RIDGE: picks[aj.RULE], FIXED: picks[aj.ONE]}))
    frozen.folds.append(fold_of("frozen", group["rule"], fit_metrics, fit_scores,
                                ai.balanced(fit_listed["lineage"]), metrics))
    inside = Bed("S2", "A17: the new lineages, each one left out of a fit on the "
                 "others", "lineage", list(scores.columns), scores, lineage,
                 pd.DataFrame({RIDGE: None, FIXED: picks[aj.ONE]},
                              index=scores.index, dtype=object))
    for name in sorted(lineage.unique()):
        test = (lineage == name).to_numpy()
        weights = ai.balanced(lineage[~test])
        rule = ai.fit_weighted(metrics[~test], scores[~test],
                               ai.usable(metrics[~test]), weights)
        inside.stored.loc[scores.index[test], RIDGE] = rule.choose(metrics[test])
        inside.folds.append(fold_of(name, rule, metrics[~test], scores[~test],
                                    weights, metrics[test]))
    return frozen, inside


def bed_a15(root: Path) -> Bed:
    """S3: amendment A15, the rule frozen on 186 runs, on new class sets."""
    frozen, old = an.load(an.RULES_S_PATH), az.load(az.PARTS[1].path)
    rule = frozen["groups"][az.PRIMARY]["rules"][an.RULE]
    sets = az.GROUPS[az.PRIMARY]
    tables, counts = an.training(root)
    meta, _, fit_metrics = at.cells(tables)
    final = an.usual_final(meta)
    fit_scores = az.group_scores(az.aimed(tables, counts, sets), sets, meta.index)
    new = analyze_h.read_stage(root, (an.TEST_STAGE,))
    picks, single, scores, accuracy = an.target_tables(frozen, old, new, az.PRIMARY)
    cells, _, metrics = at.cells(new)
    last = metrics[analyze_c.final_mask(cells)].droplevel("epoch")
    if list(last.index) != list(scores.index):
        raise SystemExit("the runs of amendment A15 are not in one order")
    planned = len({an.set_of(spec.recipe) for spec in STAGES[an.TEST_STAGE]})
    kept = an.gate_t(single, accuracy, planned)[0]
    stored = pd.DataFrame({RIDGE: list(picks[an.RULE]), FIXED: list(picks[an.ONE])},
                          index=scores.index)
    bed = Bed("S3", "A15: the rule frozen on 186 runs with 10 classes, on new "
              "class sets", "class set", list(scores.columns), scores,
              kept["set"].astype(str), stored)
    bed.folds.append(fold_of("frozen", rule, fit_metrics[final], fit_scores[final],
                             np.ones(int(final.sum())), last))
    return bed


def all_beds(root: Path) -> list[Bed]:
    frozen, inside = beds_a17(root)
    return [frozen, bed_a16(root), inside, bed_a15(root)]


def arrays_of(beds: list[Bed]) -> dict[str, np.ndarray]:
    """The arrays of the tables file, with the names of the folds in `manifest`."""
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, dict] = {}
    for bed in beds:
        manifest[bed.name] = {"title": bed.title, "csfs": bed.csfs, "folds": [
            {"name": fold.name, "test_ids": fold.test_ids} for fold in bed.folds]}
        for number, fold in enumerate(bed.folds):
            for part in ("x_fit", "y_fit", "weights", "x_test"):
                arrays[f"{bed.name}.{number:03d}.{part}"] = getattr(fold, part)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode())
    for key in sorted(arrays):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(arrays[key], dtype=np.float64).tobytes())
    arrays["manifest"] = np.array(json.dumps(manifest))
    arrays["content_sha256"] = np.array(digest.hexdigest())
    return arrays


def regret_units(bed: Bed, picks: dict[str, list]) -> pd.DataFrame:
    """The mean regret of each choice in each unit of the bed."""
    single = pd.DataFrame({name: analyze_h.regret_of(bed.scores, list(chosen))
                           for name, chosen in picks.items()}, index=bed.scores.index)
    kept = single.loc[bed.units.index]
    return kept.groupby(bed.units).mean()


def path_gate(bed: Bed) -> tuple[int, int]:
    """Test models whose ridge choice from the tables differs from the stored one."""
    wrong = total = 0
    for fold in bed.folds:
        found = picks_of(ridge_predictions(fold), bed.csfs)
        wanted = list(bed.stored.loc[fold.test_ids, RIDGE])
        wrong += sum(a != b for a, b in zip(found, wanted, strict=True))
        total += len(found)
    return wrong, total


def export(root: Path, out: Path) -> str:
    """Write the tables, and return the text of the path gate."""
    beds = all_beds(root)
    lines, failed = [], False
    for bed in beds:
        wrong, total = path_gate(bed)
        failed = failed or wrong > 0
        units = regret_units(bed, {RIDGE: list(bed.stored[RIDGE]),
                                   FIXED: list(bed.stored[FIXED])})
        rows = [fold.x_fit.shape[0] for fold in bed.folds]
        lines.append(
            f"- **{bed.name}** ({bed.title}): {len(bed.folds)} fits with "
            f"{min(rows)} to {max(rows)} rows and {bed.folds[0].x_fit.shape[1]} "
            f"inputs, {total} test models in {len(units)} units ({bed.unit}). "
            f"Stored ridge rule {units[RIDGE].mean():.2f}, fixed score "
            f"{units[FIXED].mean():.2f}. Path gate: {total - wrong} of {total} "
            f"choices agree{'' if wrong == 0 else ' - FAIL'}.")
    arrays = arrays_of(beds)
    text = ("# Tables of the rule for another learner\n\n" + "\n".join(lines)
            + f"\n\nContent hash of the tables: `{arrays['content_sha256']}`.\n")
    if failed:
        raise SystemExit(text + "\nThe path gate fails. No file is written.")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)  # type: ignore[arg-type]
    return text + f"\nFile: `{out}`.\n"


def letter(units: pd.DataFrame, learner: str) -> str:
    """R, B, W, or S for one learner against the stored ridge rule."""
    reading = an.adds(units, learner, RIDGE)
    if an.holds(units, learner):
        return "R, the rule with this learner holds"
    if reading["supported"]:
        return "B, it does not hold, and it is below the ridge rule"
    if reading["low"] > 0:
        return "W, the ridge rule is below it"
    return "S, it is not shown different from the ridge rule"


def bed_section(bed: Bed, predictions: dict[str, np.ndarray], learners: list[str]
                ) -> tuple[str, pd.DataFrame, list[str]]:
    """The table of one bed, its unit regrets, and the learners with all fits."""
    picks: dict[str, list] = {RIDGE: list(bed.stored[RIDGE]),
                              FIXED: list(bed.stored[FIXED])}
    done = []
    for learner in learners:
        chosen = pd.Series(index=bed.scores.index, dtype=object)
        for number, fold in enumerate(bed.folds):
            table = predictions.get(f"{learner}.{bed.name}.{number:03d}")
            if table is None or not np.isfinite(table).all():
                break
            chosen.loc[fold.test_ids] = picks_of(table, bed.csfs)
        else:
            picks[learner] = list(chosen)
            done.append(learner)
    units = regret_units(bed, picks)
    rows = []
    for name, chosen_by in picks.items():
        row = {"choice": name, "regret": units[name].mean(),
               "above_1": int((units[name] > az.HOLD).sum()), "units": len(units),
               "holds": an.holds(units, name)}
        if name not in (RIDGE, FIXED):
            reading = an.adds(units, name, RIDGE)
            same = np.mean([a == b for a, b in zip(chosen_by, picks[RIDGE],
                                                   strict=True)])
            row |= {"minus_ridge": reading["difference"], "low": reading["low"],
                    "high": reading["high"], "p": reading["p"],
                    "same_choice": float(same)}
        rows.append(row)
    return block(pd.DataFrame(rows), 2), units, done


def report(root: Path, path: Path) -> str:
    """The read of the predictions of another learner."""
    beds = all_beds(root)
    wanted = str(arrays_of(beds)["content_sha256"])
    saved = np.load(path, allow_pickle=False)
    if str(saved["content_sha256"]) != wanted:
        raise SystemExit("the predictions come from other tables: the content hash "
                         "differs from the one of the stored results")
    predictions = {key: saved[key] for key in saved.files
                   if key not in ("content_sha256", "learners")}
    learners = [str(name) for name in saved["learners"]]
    sections, verdict, against = [], "", []
    for bed in beds:
        if path_gate(bed)[0]:
            raise SystemExit(f"the path gate fails on bed {bed.name}")
        table, units, done = bed_section(bed, predictions, learners)
        text = (f"Unit: {bed.unit}. Regret in points of the correct-only far AUROC, "
                f"means over the units. `minus_ridge` is the paired difference "
                f"with the stored ridge rule, with its 95% interval:\n\n{table}")
        missing = [name for name in learners if name not in done]
        if missing:
            text += ("\n\nLearners with a fit that is missing or not finite, left "
                     "out: " + ", ".join(missing) + ".")
        if CHECK_LEARNER in done and (units[CHECK_LEARNER] - units[RIDGE]
                                      ).abs().max() > 1e-9:
            text += "\n\n**The ridge regression of the fit step differs from the " \
                    "stored rule. Do not read this bed.**"
        if PRIMARY_LEARNER in done:
            found = letter(units, PRIMARY_LEARNER)
            if bed.name == PRIMARY_BED:
                verdict = found
                text += f"\n\n**Letter of {PRIMARY_LEARNER}: {found}.**"
            else:
                text += f"\n\n{PRIMARY_LEARNER} against the ridge rule: {found}."
                if found.startswith("W"):
                    against.append(bed.name)
        sections.append((f"Bed {bed.name}: {bed.title}", text))
    if not verdict:
        overall = "The primary learner has no complete result on the primary bed."
    elif verdict[0] in "RB" and not against:
        overall = (f"{verdict}. No secondary bed is against it. **By the plan, this "
                   "is a lead for a prospective test.**")
    elif verdict[0] in "RB":
        overall = (f"{verdict}. But the ridge rule is below it on "
                   f"{', '.join(against)}. **By the plan, a mixed result.**")
    else:
        overall = f"{verdict}. **By the plan, no further run.**"
    return ("# TabPFN as the selection rule: a learner swap\n\nThe learner, the "
            "beds, and the rules were fixed before these predictions. Every bed was "
            "read before for the ridge rule.\n\n## Outcome\n\n" + overall + "\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("export", "report"))
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--tables", default=str(TABLES))
    parser.add_argument("--predictions", help="the file of the fit step")
    parser.add_argument("--out")
    args = parser.parse_args()
    root = Path(args.results_root)
    if args.step == "export":
        text = export(root, Path(args.tables))
        out = Path(args.out) if args.out else HERE / "outputs" / "rule_tables.md"
    else:
        if not args.predictions:
            parser.error("report needs --predictions")
        text = report(root, Path(args.predictions))
        out = Path(args.out) if args.out else HERE / "outputs" / "tabrule_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
