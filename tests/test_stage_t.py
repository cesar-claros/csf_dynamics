import json
import zlib

import numpy as np
import pandas as pd
import pytest
import torch

from dynamics import analyze_t, rules
from dynamics.analyze_selection import FAMILY
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import RECIPES, STAGES, RunSpec, results_folder
from dynamics.descriptors import (
    calibration_error,
    neighbor_agreement,
    validation_statistics,
)
from dynamics.scores import ALL_CSFS

BEST = {"plain": "PCA_RecError", "augmented": "MahaPP"}


def test_validation_statistics_have_68_finite_rows(layers):
    rows = validation_statistics(layers["val"], layers["w"], layers["n_classes"])
    assert len(rows) == 4 * 14 + 12 and {row["group"] for row in rows} == {"valid"}
    names = {(row["name"], row["layer"]) for row in rows}
    assert len(names) == len(rows)
    assert {("cdnv", "stage1"), ("neighbor_agreement", "stage4"),
            ("calibration_error", "logits"), ("head_distance", "logits")} <= names
    values = {(row["name"], row["layer"]): row["value"] for row in rows}
    val = layers["val"]
    accuracy = float((val["logits"].argmax(1) == val["labels"]).double().mean())
    # Without a misclassified image, the two error statistics have no value.
    skip = {"decay_rate"} | ({"max_softmax_errors", "error_auroc"}
                             if accuracy == 1 else set())
    finite = [v for (name, _), v in values.items() if name not in skip]
    assert np.isfinite(finite).all()
    assert values["accuracy", "logits"] == pytest.approx(accuracy)
    assert 0 <= values["neighbor_agreement", "stage4"] <= 1


def test_neighbor_agreement_and_calibration_error():
    gen = torch.Generator().manual_seed(0)
    centers = torch.tensor([[10.0, 0, 0], [0, 10.0, 0]])
    labels = torch.arange(40) % 2
    tight = centers[labels] + 0.1 * torch.randn(40, 3, generator=gen)
    assert neighbor_agreement(tight, labels) == 1.0
    mixed = torch.randn(400, 3, generator=gen)
    assert neighbor_agreement(mixed, torch.arange(400) % 2) == pytest.approx(0.5, abs=0.06)
    confidence = np.array([0.95] * 100)
    assert calibration_error(confidence, np.array([True] * 95 + [False] * 5)) == (
        pytest.approx(0.0, abs=1e-9))
    assert calibration_error(confidence, np.array([True] * 50 + [False] * 50)) == (
        pytest.approx(0.45))


def test_input_sets_split_the_columns():
    columns = ["valid/cdnv@stage4", "valid/accuracy@logits", "score/blend/kNN",
               "derived/val_nc1@stage4", "spread/cdnv_val@stage4",
               "nc_paper/var_collapse@stage4", "spread/val_over_train@stage4",
               "clock/log_epoch", "clock/val_loss"]
    sets = analyze_t.input_sets(columns)
    assert sets[analyze_t.WIDE] == columns[:2]
    assert sets[analyze_t.TIERS] == columns[:3]
    assert sets[analyze_t.TRAINING] == ["nc_paper/var_collapse@stage4"]
    assert set(sets[analyze_t.STORED]) == {"spread/cdnv_val@stage4", "clock/val_loss",
                                           "derived/val_nc1@stage4"}
    assert sets[analyze_t.ALL] == columns[4:]


def table(wide: list[float], training: list[float], recipe_rule: list[float]
          ) -> pd.DataFrame:
    recipes = [name for name in ("T1", "T2", "T3", "T4") for _ in range(2)]
    return pd.DataFrame({"recipe": recipes, analyze_t.WIDE: wide,
                         analyze_t.TRAINING: training,
                         analyze_t.RECIPE_RULE: recipe_rule,
                         analyze_t.B1: [5.0] * 8})


def test_decision_rule_of_amendment_a7():
    good = [0.2, 0.4, 0.1, 0.5, 0.3, 0.2, 0.6, 0.1]
    bad = [3.0, 2.5, 3.5, 4.0, 3.0, 3.6, 0.7, 0.2]
    selection, hypothesis, _ = analyze_t.decide(table(good, bad, good))
    assert selection == "H, the rule holds" and hypothesis.startswith("supported")
    selection, hypothesis, details = analyze_t.decide(table(good, good, bad))
    assert selection.startswith("H+,") and hypothesis == "not supported"
    assert details.loc[0, "recipes_lower"] == 4
    selection, hypothesis, _ = analyze_t.decide(table(bad, [v + 2 for v in bad], good))
    assert selection.startswith("F,") and hypothesis.startswith("supported")
    two_over = [1.5, 1.5, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0]          # mean 0.75
    assert analyze_t.decide(table(two_over, bad, good))[0].startswith("F,")
    missing = table(good, bad, good)
    missing.loc[0, analyze_t.WIDE] = np.nan
    assert analyze_t.decide(missing)[0].startswith("not decided")


def test_recipes_and_stages_of_amendment_a7():
    assert len(STAGES["a7_train"]) == 36 and len(STAGES["t"]) == 8
    assert (RECIPES["T1"].augment, RECIPES["T1"].lr) == (True, 0.01)
    assert (RECIPES["T2"].label_smoothing, RECIPES["T2"].weight_decay) == (0.1, 5e-5)
    assert (RECIPES["T3"].augment, RECIPES["T3"].weight_decay) == (True, 5e-3)
    assert (RECIPES["T4"].augment, RECIPES["T4"].label_smoothing) == (False, 0.2)
    assert [FAMILY[name] for name in ("T1", "T2", "T3", "T4")] == [
        "augmented", "plain", "augmented", "plain"]


def fake(run: str, epoch: int, columns: list[str], best: str, valid: bool) -> dict:
    rng = np.random.default_rng(zlib.crc32(f"{run}-{epoch}".encode()))
    descriptors = []
    for column in columns:
        name, _, layer = column.partition("@")
        group, _, short = name.partition("/")
        if short == "log_epoch" or (group == "valid" and not valid):
            continue
        value = {"val_accuracy": 0.8, "train_accuracy": 1.0}.get(
            short, float(rng.uniform(0.1, 1.0)))
        descriptors.append({"group": group, "name": short, "layer": layer,
                            "value": value})
    metrics = [{"set": name, "csf": csf, "auroc": 0.9 if csf == best else 0.8}
               for name in (*FAR_SETS, "id_errors", "blend", "shuffle")
               for csf in ALL_CSFS]
    return {"run": run, "epoch": epoch, "metrics": metrics, "descriptors": descriptors,
            "placement": [], "layer_metrics": [], "checks": {}, "errors": {},
            "nonfinite": {}}


def test_freeze_and_test_on_fabricated_runs(tmp_path, monkeypatch):
    base = rules.load()["targets"]["far"]["rules"]["all"].columns
    wide = [f"valid/{name}@stage4" for name in ("cdnv", "trace_ratio", "span_share")]
    old = [RunSpec("A", 0), RunSpec("B", 0)]
    late = [RunSpec("C4", 0), RunSpec("C5", 0)]
    new = [RunSpec("T1", 0), RunSpec("T2", 0)]
    monkeypatch.setitem(STAGES, "e_main", old)
    monkeypatch.setitem(STAGES, "c", [])
    monkeypatch.setitem(STAGES, "c2", late)
    monkeypatch.setitem(STAGES, "a7_train", old + late)
    monkeypatch.setitem(STAGES, "t", new)

    def write(spec, tag, columns, valid):
        folder = tmp_path / "runs" / spec.name / tag
        folder.mkdir(parents=True, exist_ok=True)
        for epoch in (100, 200):
            out = fake(spec.name, epoch, columns, BEST[FAMILY[spec.recipe]], valid)
            (folder / f"epoch_{epoch:03d}.json").write_text(json.dumps(out))

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
    frozen_path, report = tmp_path / "stage_t_rules.json", tmp_path / "report.md"

    def run(step: str, *extra: str) -> str:
        monkeypatch.setattr("sys.argv", [
            "analyze_t", step, "--results-root", str(tmp_path), "--rules",
            str(frozen_path), "--out", str(report), *extra])
        analyze_t.main()
        return report.read_text()

    text = run("freeze")
    frozen = analyze_t.load(frozen_path)
    assert rules.file_hash(frozen_path) in text
    assert len(frozen["training_runs"]) == 4 and frozen["baseline_runs"] == 4
    assert sorted(frozen["rules"][analyze_t.WIDE].columns) == sorted(wide)
    # Only the two scores that differ between the kinds have a value that varies.
    assert len(frozen["rules"][analyze_t.TIERS].columns) == 3 + 3 * 2
    assert frozen[analyze_t.RECIPE_RULE] == BEST
    with pytest.raises(SystemExit, match="frozen once"):
        run("freeze")
    for spec in new:
        write(spec, results_folder(suite="step0"), base + wide, True)
    with pytest.raises(SystemExit, match="not the frozen one"):
        run("test")
    text = run("test", "--allow-other-rules")
    assert "2 of 2 runs" in text and "leave the decision rule: none" in text
    assert "**Outcome:" in text and "**The hypothesis is" in text
    assert "the fixed shortlist is useful" in text
