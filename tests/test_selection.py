import json

import numpy as np
import pandas as pd
import pytest

from dynamics import analyze_selection as sel
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import PENULTIMATE, results_folder
from dynamics.scores import ALL_CSFS


def frame(target: list[float], accuracy: list[float] | None = None,
          train: list[float] | None = None, index: list[float] | None = None
          ) -> pd.DataFrame:
    epochs = [10, 20, 30, 40, 50]
    out = pd.DataFrame({
        "val_accuracy": accuracy or [0.70, 0.80, 0.805, 0.80, 0.80],
        "val_loss": [1.0, 0.7, 0.6, 0.65, 0.7],
        "train_accuracy": train or [0.8, 0.99, 1.0, 1.0, 1.0],
        "nc1": index or [0.5, 0.3, 0.2, 0.1, 0.05],
        sel.ENVELOPE: target}, index=epochs)
    out["eligible"] = out["val_accuracy"] >= out["val_accuracy"].max() - 0.01
    return out


def test_fixed_rules_choose_among_the_eligible_checkpoints():
    table = frame([95, 80, 85, 82, 70])
    assert table["eligible"].tolist() == [False, True, True, True, True]
    assert sel.fixed_choices(table) == {"final": 50, "val_acc": 30, "val_loss": 30,
                                        "trivial": 30}
    # Epoch 10 has the best target, but its accuracy is too low to count.
    assert sel.regret(table, 30, sel.ENVELOPE) == 0
    assert sel.regret(table, 50, sel.ENVELOPE) == 15
    never = frame([95, 80, 85, 82, 70], train=[0.8, 0.9, 0.95, 0.97, 0.98])
    assert sel.fixed_choices(never)["trivial"] == 50


def test_threshold_rules_follow_the_direction_of_the_index():
    table = frame([95, 80, 85, 82, 70])
    assert sel.threshold_choice(table, "nc1", 0.25, -1) == 30
    assert sel.threshold_choice(table, "nc1", 0.9, -1) == 20, "epoch 10 is not eligible"
    assert sel.threshold_choice(table, "nc1", 0.01, -1) == 50, "never passed: the last"
    assert sel.threshold_choice(table, sel.CLOCK, 35, +1) == 40


def test_calibration_takes_the_threshold_with_the_lowest_regret():
    runs = [frame([0, 80, 85, 90, 70]), frame([0, 82, 84, 91, 60])]
    assert sel.calibrate(runs, "nc1", -1, sel.ENVELOPE) == 0.1
    assert sel.calibrate(runs, sel.CLOCK, +1, sel.ENVELOPE) == 40
    assert np.isnan(sel.calibrate([], "nc1", -1, sel.ENVELOPE))


def runs_where_the_index_decides() -> dict[str, pd.DataFrame]:
    """The target peaks where the index is 0.2, at a different epoch in each run."""
    courses = {"a": [0.5, 0.2, 0.1, 0.05, 0.02], "b": [0.6, 0.5, 0.2, 0.1, 0.05],
               "c": [0.7, 0.6, 0.5, 0.2, 0.1], "d": [0.8, 0.6, 0.4, 0.3, 0.2]}
    return {run: frame([90 - 100 * abs(v - 0.2) for v in course], index=course,
                       accuracy=[0.8] * 5, train=[1.0] * 5)
            for run, course in courses.items()}


def test_an_index_that_decides_the_target_beats_trivial_and_clock():
    frames = runs_where_the_index_decides()
    results = sel.evaluate(frames, dict.fromkeys(frames, "plain"))
    mean = results[(results["target"] == sel.ENVELOPE)
                   & (results["scheme"] != "across")].groupby("rule")["regret"].mean()
    assert mean["nc1"] == pytest.approx(0.0)
    assert mean["trivial"] > 20 and mean[sel.CLOCK] > 5
    assert results[results["scheme"] == "across"].empty, "there is one family"
    table, decision = sel.verdict(results)
    assert table["useful"].all() and decision.startswith("an index is useful")


def test_no_index_is_useful_when_the_trivial_rule_is_the_best():
    frames = {run: frame([70, 80, 90 - shift, 85, 80])
              for run, shift in zip("abcd", (0, 1, 2, 3))}
    family = {"a": "plain", "b": "plain", "c": "augmented", "d": "augmented"}
    results = sel.evaluate(frames, family)
    table, decision = sel.verdict(results)
    assert not table["useful"].any() and decision.endswith("stop the selection aim")
    across = results[(results["scheme"] == "across") & (results["rule"] == "nc1")]
    assert len(across) == 4 * len(sel.TARGETS[:1]), "only the envelope is a target here"
    summary = sel.summarize(results, "within")
    assert summary["rule"].tolist() == ["final", "val_acc", "val_loss", "trivial",
                                        "clock", "nc1"]
    assert summary.set_index("rule").loc["trivial", sel.ENVELOPE] == 0


def test_report_from_result_files(tmp_path, monkeypatch):
    tag = results_folder(suite="refcheck")
    for run, peak in (("A_seed0", 30), ("A_seed1", 30), ("B_seed0", 50),
                      ("B_seed1", 50)):
        folder = tmp_path / "runs" / run / tag
        folder.mkdir(parents=True)
        for epoch in (10, 20, 30, 40, 50):
            value = 0.9 - 0.002 * abs(epoch - peak)
            out = {"run": run, "epoch": epoch, "layer_metrics": [], "placement": [],
                   "metrics": [{"set": name, "csf": csf, "auroc": value,
                                "fpr95": 0.5} for name in FAR_SETS
                               for csf in ALL_CSFS],
                   "descriptors": [
                       {"group": "clock", "name": "train_accuracy", "layer": "",
                        "value": 1.0 if epoch >= 30 else 0.9},
                       {"group": "clock", "name": "val_accuracy", "layer": "",
                        "value": 0.8},
                       {"group": "clock", "name": "val_loss", "layer": "",
                        "value": 0.7},
                       {"group": "nc_paper", "name": "var_collapse",
                        "layer": PENULTIMATE, "value": 1 / epoch}],
                   "checks": {}, "errors": {}, "nonfinite": {}}
            (folder / f"epoch_{epoch:03d}.json").write_text(json.dumps(out))
    report = tmp_path / "selection_report.md"
    monkeypatch.setattr("sys.argv", ["analyze_selection", "--results-root",
                                     str(tmp_path), "--out", str(report)])
    sel.main()
    text = report.read_text()
    assert "**Result: no index meets both conditions: stop the selection aim.**" in text
    regrets = pd.read_csv(tmp_path / "selection" / "regrets.csv")
    plain = regrets[(regrets["family"] == "plain") & (regrets["target"] == "kNN")]
    assert plain[plain["rule"] == "trivial"]["regret"].max() == pytest.approx(0.0)
    assert plain[plain["rule"] == "final"]["regret"].min() == pytest.approx(4.0)
    assert set(regrets["rule"]) == {"final", "val_acc", "val_loss", "trivial",
                                    "clock", "nc1"}
