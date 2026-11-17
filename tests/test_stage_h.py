import json
import zlib

import numpy as np
import pandas as pd
import pytest

from dynamics import analyze_h, rules
from dynamics.analyze_selection import FAMILY
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import RECIPES, STAGES, checkpoint_epochs, results_folder
from dynamics.scores import ALL_CSFS

BEST = {"plain": "PCA_RecError", "augmented": "MahaPP"}


def result(run: str, epoch: int, columns: list[str], best: str) -> dict:
    """One result file with every metric that the rules read, at random values."""
    rng = np.random.default_rng(zlib.crc32(f"{run}-{epoch}".encode()))
    descriptors = []
    for column in columns:
        name, _, layer = column.partition("@")
        group, _, short = name.partition("/")
        if short != "log_epoch":
            value = {"val_accuracy": 0.8, "train_accuracy": 1.0}.get(
                short, float(rng.uniform(0.1, 1.0)))
            descriptors.append({"group": group, "name": short, "layer": layer,
                                "value": value})
    metrics = [{"set": name, "csf": csf, "auroc": 0.9 if csf == best else 0.8,
                "fpr95": 0.5} for name in FAR_SETS for csf in ALL_CSFS]
    return {"run": run, "epoch": epoch, "metrics": metrics, "descriptors": descriptors,
            "placement": [], "layer_metrics": [], "checks": {}, "errors": {},
            "nonfinite": {}}


def write_stage(root, stage: str, columns: list[str]) -> None:
    tag = results_folder(suite="step0")
    for spec in STAGES[stage]:
        folder = root / "runs" / spec.name / tag
        folder.mkdir(parents=True)
        for epoch in (50, 100):
            (folder / f"epoch_{epoch:03d}.json").write_text(json.dumps(
                result(spec.name, epoch, columns, BEST[FAMILY[spec.recipe]])))


def test_stage_h_recipes_and_stages():
    assert [spec.name for spec in STAGES["h_first"]] == ["H0_seed0"]
    assert len(STAGES["h1"]) == 10 and len(STAGES["h2"]) == 6
    assert {spec.recipe for spec in STAGES["h2"]} == {"H5", "H6", "H7"}
    pairs = {"H0": "A", "H1": "B", "H2": "C1", "H3": "C2", "H4": "C3", "H5": "C4",
             "H6": "C5", "H7": "C6"}
    for new, old in pairs.items():
        a, b = RECIPES[new], RECIPES[old]
        assert (a.augment, a.weight_decay, a.label_smoothing, a.lr) == (
            b.augment, b.weight_decay, b.label_smoothing, b.lr)
        assert a.epochs == 100 and FAMILY[new] == FAMILY[old]
    assert {10, 20, 30, 40, 50, 60, 80, 100} <= set(checkpoint_epochs(100))


def test_gate_r_reads_the_room_to_select():
    index = [f"H{r}_seed{s}" for r in (0, 1, 2) for s in (0, 1)]
    same = pd.DataFrame({"a": [90.0] * 6, "b": [80.0] * 6}, index=index)
    table, mean = analyze_h.gate_r(same)
    assert mean == 0 and set(table["fixed_score"]) == {"a"}
    split = same.copy()
    split.loc[["H0_seed0", "H0_seed1"], ["a", "b"]] = [70.0, 95.0]
    table, mean = analyze_h.gate_r(split)
    assert table.set_index("left_out").loc["H0", "regret"] == 25.0
    # With H1 or H2 left out, the other recipes prefer `b`, which loses 10 there.
    assert mean == pytest.approx((25 + 10 + 10) / 3)


def test_gate_freeze_and_test_on_fabricated_runs(tmp_path, monkeypatch):
    columns = rules.load()["targets"]["far"]["rules"]["all"].columns
    write_stage(tmp_path, "h1", columns)
    frozen_path = tmp_path / "stage_h_rules.json"
    report = tmp_path / "report.md"

    def run(step: str, *extra: str) -> str:
        monkeypatch.setattr("sys.argv", [
            "analyze_h", step, "--results-root", str(tmp_path), "--rules",
            str(frozen_path), "--out", str(report), *extra])
        analyze_h.main()
        return report.read_text()

    assert "**Gate R: there is room to select.**" in run("gate")
    assert not frozen_path.exists()
    text = run("freeze")
    frozen = analyze_h.load(frozen_path)
    assert frozen["b1"] == "MahaPP" and frozen["recipe_rule"] == BEST
    assert set(BEST.values()) <= set(frozen["shortlists"]["3"])
    assert len(frozen["training_runs"]) == 10 and frozen["training_cells"] == 20
    assert rules.file_hash(frozen_path) in text
    with pytest.raises(SystemExit, match="frozen once"):
        run("freeze")
    write_stage(tmp_path, "h2", columns)
    with pytest.raises(SystemExit, match="not the frozen one"):
        run("test")
    text = run("test", "--allow-other-rules")
    assert "6 of 6 runs of phase 2" in text
    assert "**Outcome: F, the lead fails" in text, "random metrics give no rule"
    assert "the fixed shortlist is useful" in text
    assert "mean shortlist regret of 0.00: **it transfers**" in text


def test_the_committed_stage_h_rules_are_the_frozen_ones():
    assert rules.file_hash(analyze_h.RULES_H_PATH) == analyze_h.RULES_H_SHA256
    frozen = analyze_h.load(analyze_h.RULES_H_PATH)
    assert len(frozen["training_runs"]) == 10 and frozen["training_cells"] == 80
    assert frozen["b1"] == "NCI" and len(frozen["shortlists"]["3"]) == 3
