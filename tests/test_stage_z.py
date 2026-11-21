import json

import numpy as np
import pandas as pd
import pytest
from test_stage_t import BEST, fake

from dynamics import analyze_augrc, analyze_e, analyze_z, rules
from dynamics.analyze_selection import FAMILY
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import RECIPES, STAGES, RunSpec, results_folder
from dynamics.metrics import error_metrics, set_metrics
from dynamics.scores import ALL_CSFS

N_ID, N_UNSEEN = 100, 100
SETS = analyze_z.every_set()


def test_recipes_and_stages_of_amendment_a12():
    assert len(STAGES["z"]) == 16 and len(STAGES["k"]) == 8
    assert len(STAGES["a12_train"]) == 76 and len(STAGES["a12_train_h"]) == 30
    names = {spec.name for spec in STAGES["z"] + STAGES["k"]}
    assert not names & {spec.name for spec in STAGES["a12_train"]
                        + STAGES["a12_train_h"]}
    z = [RECIPES[f"Z{i}"] for i in range(1, 9)]
    k = [RECIPES[f"K{i}"] for i in range(1, 5)]
    # Half of the recipes of each part have label smoothing.
    assert [recipe.label_smoothing for recipe in z] == [0.1] * 4 + [0.0] * 4
    assert [recipe.label_smoothing for recipe in k] == [0.1, 0.1, 0.0, 0.0]
    assert all(recipe.epochs == 200 for recipe in z)
    assert all(recipe.epochs == 100 for recipe in k)
    assert (z[0].augment, z[0].lr) == (False, 0.1) and (z[1].augment, z[1].lr) == (
        True, 0.1)
    assert (z[2].augment, z[2].weight_decay, z[2].lr) == (False, 5e-5, 0.01)
    assert (z[3].augment, z[3].weight_decay) == (True, 5e-3)
    assert (z[4].weight_decay, z[5].weight_decay, z[6].lr, z[7].weight_decay) == (
        2e-3, 2e-3, 0.02, 0.0)
    assert (k[0].augment, k[0].weight_decay) == (False, 5e-5)
    assert (k[1].augment, k[1].weight_decay) == (True, 5e-3)
    assert (k[2].augment, k[2].lr) == (True, 0.01)
    assert (k[3].augment, k[3].weight_decay, k[3].lr) == (False, 5e-3, 0.01)
    # No earlier recipe has the settings of a new one.
    for name in [f"Z{i}" for i in range(1, 9)] + [f"K{i}" for i in range(1, 5)]:
        assert all(RECIPES[name] != recipe for other, recipe in RECIPES.items()
                   if other != name)
        assert name in FAMILY
    assert analyze_z.GROUPS[analyze_z.PRIMARY] == FAR_SETS


def test_the_target_of_every_set_is_exact_with_its_own_count():
    rng = np.random.default_rng(0)
    counts = {"val": 40, **{name: 30 + 7 * i for i, name in enumerate(SETS)}}
    key = {"run": "Z1_seed0", "recipe": "Z1", "seed": 0, "held_out": -1}
    rows, descriptors = [], []
    for epoch in (100, 200):
        correct = rng.random(40) < 0.8
        for csf in ("kNN", "MSR"):
            scores = np.round(rng.normal(1.0, 1.0, 40), 1)        # with ties
            for name in SETS:
                unseen = np.round(rng.normal(0.0, 1.0, counts[name]), 1)
                rows.append({**key, "epoch": epoch, "set": name, "csf": csf,
                             **set_metrics(scores, correct, unseen)})
            rows.append({**key, "epoch": epoch, "set": "id_errors", "csf": csf,
                         **error_metrics(scores, correct)})
        descriptors.append({**key, "epoch": epoch, "group": "clock",
                            "name": "val_accuracy", "layer": "",
                            "value": float(correct.mean())})
    tables = analyze_e.Tables(metrics=pd.DataFrame(rows),
                              descriptors=pd.DataFrame(descriptors),
                              placement=pd.DataFrame(), checks=pd.DataFrame())
    gap, skipped = analyze_z.metric_gap(tables, counts)
    assert gap < 1e-12 and skipped == 0
    kept = analyze_z.stored(tables, SETS).metrics
    inside = kept["set"].isin(SETS)
    assert (kept.loc[inside, "auroc"] == kept.loc[inside, "auroc_correct"]).all()
    assert (kept.loc[~inside, "auroc"] == tables.metrics.loc[~inside, "auroc"]).all()
    # A wrong count of one set gives another value, so the count matters.
    other = {**counts, "textures": counts["textures"] + 5}
    assert analyze_z.metric_gap(tables, other)[0] > 1e-4


def recipes(rule: list[float], one: list[float]) -> pd.DataFrame:
    return pd.DataFrame({analyze_z.RULE: rule, analyze_z.ONE: one},
                        index=[f"Z{i + 1}" for i in range(len(rule))])


def test_decision_rule_of_amendment_a12_has_the_recipe_as_the_unit():
    good = [0.2, 0.4, 0.1, 0.5, 0.3, 0.2, 0.6, 0.1]
    worse = [value + 1.0 for value in good]
    assert analyze_z.outcome(recipes(good, worse)).startswith("A,")
    reading = analyze_z.adds(recipes(good, worse), analyze_z.RULE)
    assert reading["supported"] and reading["lower"] == 8 and reading["high"] < 0
    assert "no recipe gives more than half" in reading["reading"]
    # The same regret: the rule holds, with no shown gain.
    assert analyze_z.outcome(recipes(good, good)).startswith("H,")
    # One recipe gives most of the difference.
    one_case = [value + (9.0 if i == 0 else 0.05) for i, value in enumerate(good)]
    assert "avoids one bad case" in analyze_z.adds(
        recipes(good, one_case), analyze_z.RULE)["reading"]
    # Above 1 point in half of the recipes: F, with the gain stated if there is one.
    half = [1.5, 1.5, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0]
    assert analyze_z.outcome(recipes(half, half)) == "F, the rule is not within 1 point"
    assert analyze_z.outcome(recipes(half, [value + 2 for value in half])) == (
        "F, the rule is not within 1 point, and it is below the fixed score")
    # Four recipes, as in part 2: one recipe above 1 point is inside the limits.
    assert analyze_z.holds(recipes([1.5, 0.1, 0.1, 0.1], [0.0] * 4), analyze_z.RULE)
    assert not analyze_z.holds(recipes([1.5, 1.5, 0.1, 0.1], [0.0] * 4),
                               analyze_z.RULE)
    missing = recipes(good, worse)
    missing.iloc[0, 0] = np.nan
    assert analyze_z.outcome(missing).startswith("not decided")


def test_by_recipe_takes_the_mean_of_the_runs():
    single = pd.DataFrame({"recipe": ["Z1", "Z1", "Z5", "Z5"],
                           analyze_z.WITH: [True, True, False, False],
                           analyze_z.RULE: [0.0, 2.0, 1.0, 3.0]})
    out = analyze_z.by_recipe(single)
    assert out[analyze_z.RULE].to_dict() == {"Z1": 1.0, "Z5": 2.0}
    assert list(analyze_z.smoothing_of(pd.Index(["Z1_seed0", "Z5_seed1"]))) == [
        True, False]


def evaluation(spec, epoch, columns, valid, kept=False, shift=0.0) -> dict:
    """A fabricated evaluation with every set of the targets and consistent values."""
    best = BEST[FAMILY[spec.recipe]]
    out = fake(spec.name, epoch, columns, best, valid)
    out["metrics"] += [{"set": name, "csf": csf, "auroc": 0.9 if csf == best else 0.8}
                       for name in SETS if name not in FAR_SETS for csf in ALL_CSFS]
    correct, wrong, errors = 80, N_ID - 80, 0.7
    failures, n = wrong + N_UNSEEN, N_ID + N_UNSEEN
    for row in out["metrics"]:
        if row["set"] == "id_errors":
            row["auroc"] = errors
        elif row["set"] in SETS:
            only = row["auroc"]
            failure = (wrong * errors + N_UNSEEN * only) / failures
            row["augrc"] = ((1 - failure) * correct * failures
                            + failures * (failures + 1) / 2) / (n * n)
            if kept:
                row["auroc_correct"] = only + shift
                row["augrc_correct"] = (1 - only) * 0.25 + 0.2
    out["counts"] = {"val": N_ID, **{name: N_UNSEEN for name in SETS}}
    return out


def world(tmp_path, monkeypatch):
    report = tmp_path / "report.md"

    def write(spec, tag, columns, valid, kept=False, shift=0.0):
        folder = tmp_path / "runs" / spec.name / tag
        folder.mkdir(parents=True, exist_ok=True)
        for epoch in (100, 200):
            out = evaluation(spec, epoch, columns, valid, kept, shift)
            (folder / f"epoch_{epoch:03d}.json").write_text(json.dumps(out))

    def run(step: str, part: int, path, *extra: str) -> str:
        monkeypatch.setattr("sys.argv", [
            "analyze_z", step, "--part", str(part), "--results-root", str(tmp_path),
            "--rules", str(path), "--out", str(report), *extra])
        analyze_z.main()
        return report.read_text()

    return write, run


def check_part(run, write, part, path, new, columns, valid):
    text = run("freeze", part, path)
    frozen = analyze_z.load(path)
    assert rules.file_hash(path) in text
    assert frozen["amendment"] == "A12" and frozen["part"] == part
    assert frozen["target"] == analyze_augrc.CORRECT_ONLY
    assert frozen["images"] == {"val": N_ID, **{name: N_UNSEEN for name in SETS}}
    assert list(frozen["groups"]) == list(analyze_z.GROUPS)
    wanted = {analyze_z.RULE, *analyze_z.QUESTION_3, *analyze_z.CONTROLS,
              f"{analyze_z.TWO}/{analyze_z.WITH}",
              f"{analyze_z.TWO}/{analyze_z.WITHOUT}"}
    assert all(set(group["rules"]) == wanted for group in frozen["groups"].values())
    assert all(column.startswith("score/")
               for column in frozen["groups"]["far"]["rules"][analyze_z.RULE].columns)
    with pytest.raises(SystemExit, match="frozen once"):
        run("freeze", part, path)
    with pytest.raises(SystemExit, match="not the frozen one"):
        run("test", part, path)
    with pytest.raises(SystemExit, match="no results of stage"):
        run("test", part, path, "--allow-other-rules")
    for spec in new:
        write(spec, results_folder(suite="step0"), columns, valid)
    with pytest.raises(SystemExit, match="do not store"):
        run("test", part, path, "--allow-other-rules")
    assert "Dry run" in run("test", part, path, "--allow-other-rules",
                            "--derived-target")
    for spec in new:
        write(spec, results_folder(suite="step0"), columns, valid, kept=True,
              shift=0.01)
    with pytest.raises(SystemExit, match="gate M fails"):
        run("test", part, path, "--allow-other-rules")
    for spec in new:
        write(spec, results_folder(suite="step0"), columns, valid, kept=True)
    text = run("test", part, path, "--allow-other-rules")
    assert "2 of 2 runs" in text and "**Gate M passes**" in text
    assert "## Target far: the outcome" in text and "**Outcome:" in text
    assert "## Target near (secondary" in text and "## Target textures" in text
    assert "Two rules by label smoothing" in text and "Regret in AUGRC points" in text


def part_1_world(tmp_path, monkeypatch):
    """Fabricated earlier 10-class runs, for the fit of part 1.

    Returns:
        `write`, `run`, the columns of a new evaluation, and the earlier runs.
    """
    base = rules.load()["targets"]["far"]["rules"]["all"].columns
    wide = [f"valid/{name}@stage4" for name in ("cdnv", "trace_ratio", "span_share")]
    old = [RunSpec("A", 0), RunSpec("B", 0)]
    late = [RunSpec("C5", 0), RunSpec("C6", 0)]          # C5 has label smoothing
    t, u = [RunSpec("T1", 0), RunSpec("T2", 0)], [RunSpec("U1", 0), RunSpec("U2", 0)]
    v, x = [RunSpec("V1", 0), RunSpec("V3", 0)], [RunSpec("X1", 0), RunSpec("X2", 0)]
    every = old + late + t + u + v + x
    for name, runs in (("e_main", old), ("c", []), ("c2", late),
                       ("a7_train", old + late), ("t", t),
                       ("a8_train", old + late + t), ("u", u),
                       ("a9_train", old + late + t + u), ("v", v), ("x", x),
                       ("a12_train", every)):
        monkeypatch.setitem(STAGES, name, runs)
    write, run = world(tmp_path, monkeypatch)
    for spec in old:              # the first runs have two evaluations
        write(spec, results_folder(suite="refcheck"), base, False)
        write(spec, results_folder(suite="step0"), base, False)
    for spec in late:
        write(spec, results_folder(suite="step0"), base, False)
    for spec in old + late:       # the light pass stores only the validation rows
        folder = tmp_path / "runs" / spec.name / results_folder(suite="valstats")
        folder.mkdir(parents=True)
        for epoch in (100, 200):
            rows = [row for row in fake(spec.name, epoch, wide, "kNN", True)[
                "descriptors"] if row["group"] == "valid"]
            (folder / f"epoch_{epoch:03d}.json").write_text(json.dumps(
                {"run": spec.name, "epoch": epoch, "descriptors": rows}))
    for spec in t + u + v + x:
        write(spec, results_folder(suite="step0"), base + wide, True)
    return write, run, base + wide, every


def test_part_1_freeze_and_test_on_fabricated_runs(tmp_path, monkeypatch):
    write, run, columns, every = part_1_world(tmp_path, monkeypatch)
    new = [RunSpec("Z1", 0), RunSpec("Z5", 0)]
    monkeypatch.setitem(STAGES, "z", new)
    path = tmp_path / "z.json"
    check_part(run, write, 1, path, new, columns, True)
    frozen = analyze_z.load(path)
    assert frozen["training_runs"] == sorted(spec.name for spec in every)
    assert frozen["groups"]["far"]["baseline_runs"] == 12


def test_part_2_freeze_and_test_on_fabricated_runs(tmp_path, monkeypatch):
    base = rules.load()["targets"]["far"]["rules"]["all"].columns
    every = [RunSpec(name, 0) for name in ("H0", "H1", "H5", "H7", "W3", "W4",
                                           "Y1", "Y2")]
    new = [RunSpec("K1", 0), RunSpec("K3", 0)]
    monkeypatch.setitem(STAGES, "a12_train_h", every)
    monkeypatch.setitem(STAGES, "k", new)
    write, run = world(tmp_path, monkeypatch)
    for spec in every:
        write(spec, results_folder(suite="step0"), base, False)
    path = tmp_path / "k.json"
    check_part(run, write, 2, path, new, base, False)
    frozen = analyze_z.load(path)
    assert len(frozen["training_runs"]) == 8
    assert frozen["groups"]["far"]["baseline_runs"] == 8


def test_the_committed_a12_rules_are_the_frozen_ones():
    wanted = {analyze_z.RULE: 69, "training": 59, "validation": 20, "clock": 5,
              f"{analyze_z.TWO}/{analyze_z.WITH}": 69,
              f"{analyze_z.TWO}/{analyze_z.WITHOUT}": 69}
    for number, runs, cells, baseline, every, images in (
            (1, 76, 1702, 74, 106, 3925), (2, 30, 240, 30, 105, 5000)):
        part = analyze_z.PARTS[number]
        assert rules.file_hash(part.path) == part.sha256 != ""
        frozen = analyze_z.load(part.path)
        assert frozen["amendment"] == "A12" and frozen["part"] == number
        assert frozen["target"] == analyze_augrc.CORRECT_ONLY
        assert frozen["images"]["val"] == images
        assert set(frozen["training_runs"]) == {
            spec.name for spec in STAGES[part.train_stage]}
        assert (len(frozen["training_runs"]), frozen["training_cells"]) == (runs, cells)
        assert list(frozen["groups"]) == ["far", "near", "textures"]
        for group in frozen["groups"].values():
            sizes = {name: len(rule.columns) for name, rule in group["rules"].items()}
            assert sizes == {**wanted, "all_metrics": every}
            assert group["baseline_runs"] == baseline
    assert analyze_z.load(analyze_z.PARTS[1].path)["groups"]["far"][
        analyze_z.ONE] == "kNN"
    assert analyze_z.load(analyze_z.PARTS[2].path)["groups"]["far"][
        analyze_z.ONE] == "fDBD"

