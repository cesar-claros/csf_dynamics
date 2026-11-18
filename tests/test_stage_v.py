import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_stage_t import BEST, fake

from dynamics import analyze_augrc, analyze_e, analyze_t, analyze_v, rules
from dynamics.analyze_selection import FAMILY
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import RECIPES, STAGES, RunSpec, results_folder
from dynamics.metrics import error_metrics, set_metrics

N_ID, N_UNSEEN = 100, 100


def test_recipes_and_stages_of_amendment_a9():
    assert len(STAGES["v"]) == 12 and len(STAGES["w"]) == 8
    assert len(STAGES["a9_train"]) == 52 and len(STAGES["a9_train_h"]) == 16
    names = {spec.name for spec in STAGES["v"] + STAGES["w"]}
    assert not names & {spec.name for spec in STAGES["a9_train"]
                        + STAGES["a9_train_h"]}
    v = [RECIPES[f"V{i}"] for i in range(1, 7)]
    assert (v[0].augment, v[0].weight_decay, v[0].label_smoothing) == (False, 5e-3, 0.1)
    assert (v[1].augment, v[1].lr, v[1].label_smoothing) == (True, 0.01, 0.1)
    assert (v[2].augment, v[2].lr, v[2].weight_decay) == (False, 0.01, 5e-5)
    assert (v[3].augment, v[3].lr) == (False, 0.1)
    assert (v[4].augment, v[4].label_smoothing) == (True, 0.3)
    assert (v[5].augment, v[5].lr) == (True, 0.1)
    assert all(recipe.epochs == 200 for recipe in v)
    # Part 2 has the forms of V1, V2, V4, and V5 with 100 epochs.
    for name, source in (("W1", "V1"), ("W2", "V2"), ("W3", "V4"), ("W4", "V5")):
        assert RECIPES[name] == replace(RECIPES[source], epochs=100)
        assert FAMILY[name] == FAMILY[source]
    assert [FAMILY[f"V{i}"] for i in range(1, 7)] == [
        "plain", "augmented", "plain", "plain", "augmented", "augmented"]
    # No earlier recipe has the settings of a new one.
    for name in [f"V{i}" for i in range(1, 7)] + [f"W{i}" for i in range(1, 5)]:
        assert all(RECIPES[name] != recipe for other, recipe in RECIPES.items()
                   if other != name)


def table(rule: list[float], one: list[float], recipes: int = 6) -> pd.DataFrame:
    names = [f"V{i + 1}" for i in range(recipes) for _ in range(2)]
    return pd.DataFrame({"recipe": names, analyze_v.RULE: rule, analyze_v.ONE: one})


def test_limits_of_question_1_use_half_of_the_recipes():
    # Six recipes: above 1 point in two is inside the limits, in three it is not.
    two = [1.5, 1.5, 1.5, 1.5] + [0.0] * 8                       # mean 0.5
    three = [1.2] * 6 + [0.0] * 6                                # mean 0.6
    assert analyze_v.holds(table(two, two), analyze_v.RULE)
    assert not analyze_v.holds(table(three, three), analyze_v.RULE)
    # Four recipes: the limit of the earlier tests, fewer than 2.
    one_over = [1.5, 1.5] + [0.0] * 6
    two_over = [1.5] * 4 + [0.0] * 4
    assert analyze_v.holds(table(one_over, one_over, 4), analyze_v.RULE)
    assert not analyze_v.holds(table(two_over, two_over, 4), analyze_v.RULE)
    # The mean alone can fail.
    assert not analyze_v.holds(table([9.0, 9.0] + [0.0] * 10, [0.0] * 12),
                               analyze_v.RULE)


def test_decision_rule_of_amendment_a9():
    good = [0.2, 0.4, 0.1, 0.5, 0.3, 0.2, 0.6, 0.1, 0.2, 0.3, 0.4, 0.1]
    bad = [value + 2.0 for value in good]
    out = analyze_v.decide(table(good, bad))
    assert out["selection"] == "H, the rule holds"
    assert out["adds"].startswith("supported, and no recipe")
    assert out["enough"] == "the fixed score does not hold"
    assert out["recipes_lower"] == 6 and out["difference"][2] < 0
    # One recipe gives most of the difference: the rule avoids one bad case.
    one_case = [value + (8.0 if i < 2 else 0.05) for i, value in enumerate(good)]
    out = analyze_v.decide(table(good, one_case))
    assert "avoids one bad case" in out["adds"] and "V1" in out["adds"]
    # The same regret: not supported, and a fixed score is enough.
    out = analyze_v.decide(table(good, good))
    assert out["adds"] == "not supported" and out["enough"] == "a fixed score is enough"
    out = analyze_v.decide(table(bad, good))
    assert out["selection"].startswith("F,") and out["adds"] == "not supported"
    missing = table(good, bad)
    missing.loc[0, analyze_v.RULE] = np.nan
    assert analyze_v.decide(missing)["selection"].startswith("not decided")


def consistent_tables(stored: bool = True) -> analyze_e.Tables:
    """Two checkpoints of one run, with metric rows from real scores."""
    rng = np.random.default_rng(0)
    rows, descriptors = [], []
    key = {"run": "V1_seed0", "recipe": "V1", "seed": 0, "held_out": -1}
    for epoch in (100, 200):
        correct = rng.random(40) < 0.8
        for csf in ("kNN", "MSR"):
            scores = rng.normal(1.0, 1.0, 40)
            for name in FAR_SETS:
                row = set_metrics(scores, correct, rng.normal(0.0, 1.0, 50))
                if not stored:
                    row = {k: v for k, v in row.items() if not k.endswith("_correct")}
                rows.append({**key, "epoch": epoch, "set": name, "csf": csf, **row})
            rows.append({**key, "epoch": epoch, "set": "id_errors", "csf": csf,
                         **error_metrics(scores, correct)})
        descriptors.append({**key, "epoch": epoch, "group": "clock",
                            "name": "val_accuracy", "layer": "",
                            "value": float(correct.mean())})
    return analyze_e.Tables(metrics=pd.DataFrame(rows),
                            descriptors=pd.DataFrame(descriptors),
                            placement=pd.DataFrame(), checks=pd.DataFrame())


def test_gate_m_compares_the_stored_target_with_the_derived_one():
    tables = consistent_tables()
    gap, skipped = analyze_v.metric_gap(tables, 40, 50)
    assert gap < 1e-12 and skipped == 0
    aimed = analyze_v.with_stored_target(tables).metrics
    far = aimed["set"].isin(FAR_SETS)
    assert (aimed.loc[far, "auroc"] == aimed.loc[far, "auroc_correct"]).all()
    assert (aimed.loc[~far, "auroc"] == tables.metrics.loc[~far, "auroc"]).all()
    wrong = tables.metrics.copy()
    wrong.loc[wrong.index[0], "auroc_correct"] += 0.01
    changed = analyze_e.Tables(**{**vars(tables), "metrics": wrong})
    assert analyze_v.metric_gap(changed, 40, 50)[0] == pytest.approx(0.01)
    with pytest.raises(SystemExit, match="do not store"):
        analyze_v.metric_gap(consistent_tables(stored=False), 40, 50)


def with_values(out: dict, stored: bool, shift: float = 0.0) -> dict:
    """Give a fabricated evaluation the stored values that the target needs.

    The correct-only AUROC of a far set is the fabricated `auroc`. The AUGRC is
    the one of that value, with 80 correct ID images of 100.
    """
    correct, wrong, errors = 80, N_ID - 80, 0.7
    n = N_ID + N_UNSEEN
    for row in out["metrics"]:
        if row["set"] == "id_errors":
            row["auroc"] = errors
        elif row["set"] in FAR_SETS:
            only = row["auroc"]
            failure = (wrong * errors + N_UNSEEN * only) / (wrong + N_UNSEEN)
            failures = wrong + N_UNSEEN
            row["augrc"] = ((1 - failure) * correct * failures
                            + failures * (failures + 1) / 2) / (n * n)
            if stored:
                row["auroc_correct"] = only + shift
                row["augrc_correct"] = 0.3
    out["counts"] = {"val": N_ID, **{name: N_UNSEEN for name in FAR_SETS}}
    return out


def world(tmp_path, monkeypatch):
    """Helpers to write fabricated runs and to call the two analysis modules."""
    report = tmp_path / "report.md"

    def write(spec, tag, columns, valid, stored=False, shift=0.0):
        folder = tmp_path / "runs" / spec.name / tag
        folder.mkdir(parents=True, exist_ok=True)
        for epoch in (100, 200):
            out = with_values(fake(spec.name, epoch, columns,
                                   BEST[FAMILY[spec.recipe]], valid), stored, shift)
            (folder / f"epoch_{epoch:03d}.json").write_text(json.dumps(out))

    def run(module, step: str, *extra: str) -> str:
        monkeypatch.setattr("sys.argv", [
            "analyze", step, "--results-root", str(tmp_path), "--out", str(report),
            *extra])
        module.main()
        return report.read_text()

    return write, run


def test_part_1_freeze_and_test_on_fabricated_runs(tmp_path, monkeypatch):
    base = rules.load()["targets"]["far"]["rules"]["all"].columns
    wide = [f"valid/{name}@stage4" for name in ("cdnv", "trace_ratio", "span_share")]
    old = [RunSpec("A", 0), RunSpec("B", 0)]
    late = [RunSpec("C4", 0), RunSpec("C5", 0)]
    seen = [RunSpec("T1", 0), RunSpec("T2", 0), RunSpec("U1", 0), RunSpec("U2", 0)]
    new = [RunSpec("V1", 0), RunSpec("V2", 0)]
    for name, runs in (("e_main", old), ("c", []), ("c2", late),
                       ("a7_train", old + late), ("t", seen[:2]),
                       ("a8_train", old + late + seen[:2]), ("u", seen[2:]),
                       ("a9_train", old + late + seen), ("v", new)):
        monkeypatch.setitem(STAGES, name, runs)
    write, run = world(tmp_path, monkeypatch)
    for spec in old:
        write(spec, results_folder(suite="refcheck"), base, False)
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
    for spec in seen:
        write(spec, results_folder(suite="step0"), base + wide, True)
    first, path = tmp_path / "first.json", tmp_path / "v.json"
    run(analyze_t, "freeze", "--rules", str(first))
    part = replace(analyze_v.PARTS[1], earlier=(
        "a7_rule", first, "", analyze_t.TIERS))
    monkeypatch.setitem(analyze_v.PARTS, 1, part)
    common = ("--part", "1", "--rules", str(path))

    text = run(analyze_v, "freeze", *common)
    frozen = analyze_t.load(path)
    assert rules.file_hash(path) in text
    assert frozen["target"] == analyze_augrc.CORRECT_ONLY
    assert frozen["images"] == {"id": N_ID, "unseen": N_UNSEEN}
    assert len(frozen["training_runs"]) == 8 and frozen["baseline_runs"] == 8
    assert sorted(frozen["rules"]) == sorted([analyze_t.TIERS, analyze_v.SMALL])
    assert frozen[analyze_v.RECIPE_RULE] == BEST
    with pytest.raises(SystemExit, match="frozen once"):
        run(analyze_v, "freeze", *common)
    with pytest.raises(SystemExit, match="not the frozen one"):
        run(analyze_v, "test", *common)
    allow = (*common, "--allow-other-rules")
    with pytest.raises(SystemExit, match="no results of stage v"):
        run(analyze_v, "test", *allow)
    for spec in new:
        write(spec, results_folder(suite="step0"), base + wide, True)
    with pytest.raises(SystemExit, match="do not store"):
        run(analyze_v, "test", *allow)
    assert "Dry run" in run(analyze_v, "test", *allow, "--derived-target")
    for spec in new:
        write(spec, results_folder(suite="step0"), base + wide, True, stored=True,
              shift=0.01)
    with pytest.raises(SystemExit, match="gate M fails"):
        run(analyze_v, "test", *allow)
    for spec in new:
        write(spec, results_folder(suite="step0"), base + wide, True, stored=True)
    text = run(analyze_v, "test", *allow)
    assert "2 of 2 runs" in text and "leave the decision rule: none" in text
    assert "**Gate M passes.**" in text and "**Outcome:" in text
    assert "**The claim that the rule adds is" in text
    assert "Correct-only AUGRC" in text and "all ID images" in text


def test_part_2_freeze_and_test_on_fabricated_runs(tmp_path, monkeypatch):
    base = rules.load()["targets"]["far"]["rules"]["all"].columns
    first = [RunSpec("H0", 0), RunSpec("H1", 0)]
    second = [RunSpec("H5", 0), RunSpec("H6", 0)]
    new = [RunSpec("W1", 0), RunSpec("W2", 0)]
    for name, runs in (("h1", first), ("h2", second), ("a9_train_h", first + second),
                       ("w", new)):
        monkeypatch.setitem(STAGES, name, runs)
    write, run = world(tmp_path, monkeypatch)
    for spec in first + second:
        write(spec, results_folder(suite="step0"), base, False)
    path = tmp_path / "w.json"
    # The frozen file of the part also stands in for the earlier Stage H file.
    part = replace(analyze_v.PARTS[2], earlier=("stage_h_rule", path, "", "all"))
    monkeypatch.setitem(analyze_v.PARTS, 2, part)
    common = ("--part", "2", "--rules", str(path))
    run(analyze_v, "freeze", *common)
    frozen = analyze_t.load(path)
    assert list(frozen["rules"]) == ["all"] and len(frozen["training_runs"]) == 4
    assert frozen["baseline_runs"] == 4 and frozen[analyze_v.RECIPE_RULE] == BEST
    for spec in new:
        write(spec, results_folder(suite="step0"), base, False, stored=True)
    text = run(analyze_v, "test", *common, "--allow-other-rules")
    assert "2 of 2 runs" in text and "**Gate M passes.**" in text
    assert "**Outcome:" in text and "stage_h_rule" in text


def test_the_committed_a9_rules_are_the_frozen_ones():
    for number, runs, cells, baseline, inputs in ((1, 52, 1174, 50, 137),
                                                  (2, 16, 128, 16, 105)):
        part = analyze_v.PARTS[number]
        assert rules.file_hash(part.path) == part.sha256 != ""
        frozen = analyze_t.load(part.path)
        assert frozen["target"] == analyze_augrc.CORRECT_ONLY
        assert set(frozen["training_runs"]) == {
            spec.name for spec in STAGES[part.train_stage]}
        assert (len(frozen["training_runs"]), frozen["training_cells"],
                frozen["baseline_runs"]) == (runs, cells, baseline)
        assert len(frozen["rules"][part.rule].columns) == inputs
        # The earlier frozen file of the secondary rule is the recorded one.
        assert rules.file_hash(part.earlier[1]) == part.earlier[2]
    first = analyze_t.load(analyze_v.PARTS[1].path)
    assert sorted(first["rules"]) == sorted([analyze_t.TIERS, analyze_v.SMALL])
    assert first["images"] == {"id": 3925, "unseen": 5000}
    assert first[analyze_v.ONE] == "kNN"
    second = analyze_t.load(analyze_v.PARTS[2].path)
    assert list(second["rules"]) == ["all"]
    assert second["images"] == {"id": 5000, "unseen": 5000}
    assert second[analyze_v.ONE] == "fDBD"

