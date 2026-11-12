import json
import shutil

import numpy as np
import pandas as pd
import pytest

from dynamics import analyze_d, interpolation
from dynamics.analyze_step0 import FAR_SETS, SUITE
from dynamics.config import PENULTIMATE, RunSpec, results_folder
from dynamics.scores import ALL_CSFS

EPOCHS = (0, 20, 40, 200, 340, 500)
#: Late loss of the early-strong scores, and NC1 at the end, per arm.
LOSS = {"D1": 0.05, "D2": 0.05, "D3": 0.09, "D4": 0.01, "D5": 0.05}
COLLAPSE = {"D1": 0.02, "D2": 0.05, "D3": 0.005, "D4": 0.08, "D5": 0.02}


def test_decide_q1():
    assert analyze_d.decide_q1(3, 3).startswith("T")
    assert analyze_d.decide_q1(4, 4).startswith("T")
    assert analyze_d.decide_q1(1, 0).startswith("A")
    assert analyze_d.decide_q1(3, 1).startswith("M")
    assert analyze_d.decide_q1(2, 4).startswith("M")


def contrasts(**rows) -> pd.DataFrame:
    return pd.DataFrame([{"arm": arm, "gate_m": gate, "predicted_sign": sign,
                          "within_equivalence": inside}
                         for arm, (gate, sign, inside) in rows.items()])


def test_decide_q2():
    both = contrasts(D3=(True, True, False), D4=(True, True, False))
    assert analyze_d.decide_q2(both).startswith("G, geometry")
    one = contrasts(D3=(True, True, False), D4=(False, False, False))
    assert analyze_d.decide_q2(one).endswith("(one arm)")
    clock = contrasts(D3=(True, False, True), D4=(True, False, True))
    assert analyze_d.decide_q2(clock).startswith("C, clock")
    split = contrasts(D3=(True, True, False), D4=(True, False, True))
    assert analyze_d.decide_q2(split) == "inconclusive"
    none = contrasts(D3=(False, True, False), D4=(False, True, False))
    assert analyze_d.decide_q2(none).startswith("inconclusive: no arm")


def test_direction_and_reading_of_the_conventions():
    assert analyze_d.direction(np.array([-1.0, -2, -1, -3, 0.5])) == "falls"
    assert analyze_d.direction(np.array([-1.0, -2, 1, 1, -3])) == "no clear change"
    assert analyze_d.direction(np.array([1.0, 2, 1, 3, 2])) == "rises"
    assert "no conflict" in analyze_d.read_conventions("falls", "falls")
    assert "contradicts" in analyze_d.read_conventions("falls", "rises")
    assert "does not hold" in analyze_d.read_conventions("rises", "falls")
    assert "no data" in analyze_d.read_conventions("no data", "no data")
    assert analyze_d.direction(np.array([], dtype=object)) == "no data"


def test_a_failed_gate_leaves_the_question_open():
    assert analyze_d.gated("T, time", []) == "T, time"
    assert analyze_d.gated("T, time", ["I", "P"]) == (
        "not decided, because gate I and gate P failed")


def test_spike_epochs_and_q4():
    losses = dict.fromkeys(range(1, 61), 0.001)
    losses[30], losses[31], losses[5] = 0.5, 0.2, 0.9
    assert analyze_d.spike_epochs(losses, after=20) == [30, 31]
    assert analyze_d.spike_epochs(losses, after=40) == []
    assert analyze_d.decide_q4({"D1": (5, 5), "D5": (0, 5)}).startswith(
        "precision artefact")
    assert analyze_d.decide_q4({"D1": (5, 5), "D5": (4, 5)}).startswith(
        "not a precision artefact")
    assert analyze_d.decide_q4({"D1": (2, 5), "D5": (0, 5)}) == "unclear"
    assert analyze_d.decide_q4({"D1": (5, 5)}).startswith("not decided")


def test_reference_epochs_replace_a_disturbed_checkpoint():
    epochs = [0, 20, 40, 320, 340, 360]
    clock = pd.DataFrame({
        "run": ["r"] * 6, "recipe": ["D1"] * 6, "epoch": epochs,
        "train_accuracy": [0.1, 0.9, 1.0, 1.0, 0.97, 1.0],
        "val_accuracy": [0.1, 0.6, 0.8, 0.8, 0.7, 0.8]})
    converged = analyze_d.analyze_e.converged_checkpoints(clock)
    refs = analyze_d.reference_epochs(clock, converged).iloc[0]
    assert refs["t0"] == 40 and refs["start"] == 40
    assert refs["end"] == 320, "epoch 340 is disturbed, so the neighbor is used"
    never = clock.assign(train_accuracy=0.9)
    assert np.isnan(analyze_d.reference_epochs(never, converged).iloc[0]["end"])
    assert np.isnan(analyze_d.nearest(np.array([0, 100]), 340, 10))


def test_difference_interval():
    rng = np.random.default_rng(0)
    first, second = 5 + rng.normal(0, 0.1, 5), 2 + rng.normal(0, 0.1, 5)
    mean, low, high = analyze_d.difference_interval(first, second)
    assert low < mean < high and 2.5 < low and high < 3.5
    assert all(np.isnan(analyze_d.difference_interval(first[:1], second)))


def fake_epoch(spec: RunSpec, epoch: int, rng: np.random.Generator) -> dict:
    late = max(0.0, (epoch - 40) / 300)
    progress = min(late, 1.0)
    metrics = []
    for name in FAR_SETS:
        for csf in ALL_CSFS:
            if csf in analyze_d.EARLY_STRONG:
                value = 0.85 - LOSS[spec.recipe] * progress
            elif csf in analyze_d.LATE_GAINING:
                value = 0.50 + 0.25 * progress
            else:
                value = 0.70
            value += 0.002 * rng.standard_normal()
            metrics.append({"set": name, "csf": csf, "auroc": value,
                            "fpr95": 1 - value, "aurc": 0.1, "augrc": 0.1})
    collapse = 0.2 * (COLLAPSE[spec.recipe] / 0.2) ** progress
    collapse *= float(np.exp(0.05 * rng.standard_normal()))
    descriptors = [
        {"group": "clock", "name": "train_accuracy", "layer": "",
         "value": 1.0 if epoch >= 40 else 0.5},
        {"group": "clock", "name": "val_accuracy", "layer": "", "value": 0.8},
        {"group": "nc_paper", "name": "var_collapse", "layer": PENULTIMATE,
         "value": collapse}]
    geometry = {"outside_share_gap": 0.25 - 0.17 * progress,
                "outside_share_gap_raw": 0.20 - 0.10 * progress,
                "orthodev": 0.5 - 0.2 * progress, "rowspace_share": 0.6,
                "rowspace_share_gap": -0.1}
    placement = [{"set": name, "layer": PENULTIMATE, "name": key,
                  "value": value + 0.002 * rng.standard_normal()}
                 for name in FAR_SETS for key, value in geometry.items()]
    return {"run": spec.name, "epoch": epoch, "metrics": metrics,
            "layer_metrics": [], "layer_errors": {}, "descriptors": descriptors,
            "placement": placement, "checks": {"g4_max_gap": 0.0, "head_gap": 0.0},
            "errors": {}, "nonfinite": dict.fromkeys(ALL_CSFS, 0)}


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("stage_d")
    rng = np.random.default_rng(0)
    tag = results_folder(suite=SUITE)
    for spec in [RunSpec(arm, seed) for arm in analyze_d.ARMS for seed in range(5)]:
        folder = root / "runs" / spec.name
        (folder / tag).mkdir(parents=True)
        for epoch in EPOCHS:
            (folder / tag / f"epoch_{epoch:03d}.json").write_text(
                json.dumps(fake_epoch(spec, epoch, rng)))
        losses = dict.fromkeys(range(1, 501), 0.001)
        if spec.recipe != "D5":
            losses[100] = 1.0
        lines = ["epoch,lr,loss,accuracy,seconds"] + [
            f"{epoch},0.01,{loss},1.0,2.7" for epoch, loss in losses.items()]
        (folder / "train_log.csv").write_text("\n".join(lines))
    return root


def test_report_applies_the_gates_and_the_rules(results, tmp_path, monkeypatch):
    out = tmp_path / "stage_d_report.md"
    monkeypatch.setattr("sys.argv", ["analyze_d", "--results-root", str(results),
                                     "--out", str(out)])
    analyze_d.main()
    text = out.read_text()
    gates = {line.split("  ")[0].strip(): line for line in text.splitlines()
             if line.strip().startswith("Gate ")}
    assert all(" pass " in line for line in gates.values()), gates
    assert "5 of 5 seeds" in text
    assert "envelope drops on 4 of 4 far sets, 4 of 4 early-strong scores drop" in text
    assert "**Q1: T, time" in text and "not decided" not in text
    assert "**Q2: G, geometry: collapse is supported as the driver.**" in text
    assert text.count("no conflict of fact") == 2
    assert "**Q4: precision artefact" in text
    refs = pd.read_csv(tmp_path / "stage_d" / "references.csv")
    assert (refs["t0"] == 40).all() and (refs["end"] == 340).all()
    q2 = pd.read_csv(tmp_path / "stage_d" / "q2_contrasts.csv").set_index("arm")
    assert q2.loc["D3", "drop_gap"] == pytest.approx(4.0, abs=0.5)
    assert q2.loc["D4", "drop_gap"] == pytest.approx(-4.0, abs=0.5)
    assert q2["gate_m"].all() and q2.loc["D3", "log_nc1_shift"] < 0
    course = pd.read_csv(tmp_path / "stage_d" / "time_course.csv")
    early = course[(course["recipe"] == "D1") & (course["csf"] == "early_strong")]
    early = early.set_index("tau")["value"]
    assert early[0] - early[300] == pytest.approx(5.0, abs=0.5)
    assert (course["seeds"] == 5).all() and course["tau"].min() == -40
    fits = pd.read_csv(tmp_path / "stage_d" / "tail_regressions.csv")
    assert fits.empty, "4 checkpoints after t0 are too few for a fit"


def test_tail_regression_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    epochs = np.arange(40, 501, 20)
    log_nc1 = -1 - 0.002 * (epochs - 40) + 0.2 * rng.standard_normal(len(epochs))
    value = 80 - 1.5 * (epochs - 40) / 100 + 4.0 * log_nc1
    scores = pd.DataFrame({"run": "r", "recipe": "D1", "epoch": epochs,
                           "csf": "Y", "value": value})
    nc1 = pd.DataFrame({"run": "r", "recipe": "D1", "epoch": epochs,
                        "value": log_nc1})
    refs = pd.DataFrame({"run": ["r"], "recipe": ["D1"], "t0": [40.0],
                         "start": [40.0], "end": [340.0]})
    fit = analyze_d.tail_regressions(scores, nc1, refs).iloc[0]
    assert fit["per_100_epochs"] == pytest.approx(-1.5)
    assert fit["per_log_nc1"] == pytest.approx(4.0)
    assert -1 < fit["tau_nc1_corr"] < -0.5
    no_t0 = refs.assign(t0=np.nan)
    assert analyze_d.tail_regressions(scores, nc1, no_t0).empty
    assert analyze_d.time_course(scores, no_t0).empty


def test_first_runs_before_interpolation_give_no_verdict(results, tmp_path,
                                                        monkeypatch):
    """The first two runs, cut before t0: every question stays open, with no crash."""
    tag = results_folder(suite=SUITE)
    for run in ("D1_seed0", "D2_seed0"):
        target = tmp_path / "first" / "runs" / run / tag
        target.mkdir(parents=True)
        for source in sorted((results / "runs" / run / tag).glob("epoch_*.json")):
            if json.loads(source.read_text())["epoch"] < 40:
                shutil.copy(source, target / source.name)
    out = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", ["analyze_d", "--results-root",
                                     str(tmp_path / "first"), "--out", str(out)])
    analyze_d.main()
    text = out.read_text()
    assert "**Q1: not decided, because gate I and gate P failed.**" in text
    assert "**Q2: not decided, because gate P failed.**" in text
    assert text.count("no data at the two reference points") == 2
    assert "does not hold" not in text
    assert "**Q4: not decided: an arm is missing.**" in text


def test_interpolation_listing(results, tmp_path, monkeypatch, capsys):
    folder = results / "runs" / "D2_seed0" / results_folder(suite=SUITE)
    accuracies = interpolation.train_accuracies(folder)
    assert list(accuracies) == list(EPOCHS)
    assert interpolation.first_interpolation(accuracies) == 40
    assert interpolation.first_interpolation({0: 0.1, 20: 0.9999}) is None
    monkeypatch.setattr("sys.argv", ["interpolation", "--out-root", str(results),
                                     "--runs", "D2_seed0", "missing_run"])
    interpolation.main()
    shown = capsys.readouterr().out
    assert "D2_seed0: t0 = 40" in shown and "missing_run: no result files" in shown
    assert "epoch 200  train accuracy 1.00000" in shown
