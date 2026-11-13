import json

import numpy as np
import pandas as pd
import pytest
import torch

from dynamics import analyze_refcheck as check
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import PENULTIMATE, results_folder
from dynamics.evaluate import (
    REFERENCES,
    evaluate_features,
    matched_subset,
    reference_split,
)
from dynamics.metrics import auroc
from dynamics.model import build_model, weight_norms
from dynamics.scores import ALL_CSFS, NO_FIT_CSFS, all_scores


def test_reference_split_is_stratified_and_fixed():
    labels = torch.tensor([0] * 7 + [1] * 4 + [2] * 10)
    fit, test = reference_split(labels, 3)
    assert sorted([*fit, *test]) == list(range(21)) and not set(fit) & set(test)
    assert np.bincount(labels.numpy()[fit]).tolist() == [3, 2, 5]
    again = reference_split(labels, 3)
    assert (fit == again[0]).all() and (test == again[1]).all()
    assert not (fit == reference_split(labels, 3, seed=1)[0]).all()


def test_matched_subset_has_the_requested_class_counts():
    labels = torch.arange(60) % 3
    rows = matched_subset(labels, np.array([5, 0, 7]))
    assert np.bincount(labels.numpy()[rows], minlength=3).tolist() == [5, 0, 7]
    assert len(set(rows)) == 12 and (np.diff(rows) > 0).all()
    with pytest.raises(ValueError, match="class 1"):
        matched_subset(labels, np.array([1, 21, 1]))


def test_reference_check_rows_match_an_independent_fit(layers):
    features = {name: layers[name] for name in ("train", "val", "far")}
    w, b, n_classes = layers["w"], layers["b"], layers["n_classes"]
    out = evaluate_features(features, w, b, n_classes, reference_check=True)
    rows = pd.DataFrame(out["reference_metrics"])
    assert set(rows["reference"]) == set(REFERENCES) and set(rows["set"]) == {"far"}
    assert out["checks"]["reference_nofit_gap"] == 0.0
    assert all(key.split(":")[0] in REFERENCES[1:] for key in out["reference_errors"])
    val, far = layers["val"], layers["far"]
    fit, test = reference_split(val["labels"], n_classes)
    assert out["checks"]["reference_fit"] == len(fit) == 98
    assert out["checks"]["reference_test"] == len(test) == 102
    assert len(rows[rows["reference"] == "train"]) == len(ALL_CSFS)
    evaluation = torch.cat([val[PENULTIMATE][test], far[PENULTIMATE]])
    logits = torch.cat([val["logits"][test], far["logits"]])
    sources = {"train": (layers["train"], slice(None)), "heldout": (val, fit)}
    for reference, (source, keep) in sources.items():
        scores, _ = all_scores(source[PENULTIMATE][keep], source["labels"][keep],
                               source["logits"][keep], w, b, evaluation, logits,
                               n_classes)
        for csf in ("kNN", "NCI", "MSR"):
            expected = auroc(scores[csf][:len(test)], scores[csf][len(test):])
            found = rows[(rows["reference"] == reference) & (rows["csf"] == csf)]
            assert found["auroc"].item() == pytest.approx(expected, abs=1e-12)
    no_fit = rows[rows["csf"].isin(NO_FIT_CSFS)]
    assert (no_fit.groupby("csf")["auroc"].nunique() == 1).all()


def test_weight_norms_add_up():
    model = build_model(10)
    norms = weight_norms(model)
    blocks = [norms[f"conv_norm_{name}"] for name in
              ("stem", "stage1", "stage2", "stage3", "stage4")]
    assert norms["conv_norm"] == pytest.approx(sum(v ** 2 for v in blocks) ** 0.5)
    assert norms["fc_norm"] == pytest.approx(float(model.fc.weight.norm()), rel=1e-6)
    assert norms["bn_scale_norm"] == pytest.approx(4800 ** 0.5)   # every scale is 1


def test_reading_rule():
    assert check.reading(5.0, 5.0, 1.0) == "reference set"
    assert check.reading(5.0, 5.0, 2.5) == "reference set"
    assert check.reading(5.0, 5.0, 4.5) == "features"
    assert check.reading(5.0, 5.0, 3.0) == "mixed"
    assert check.reading(1.0, 1.5, 0.0) == "no decline to explain"
    assert check.reading(8.0, 5.0, 1.0) == ("reference set (the subset changes the "
                                            "decline)")
    assert check.reading(5.0, float("nan"), 1.0) == "no data"


def frames(reset: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clock and log10 NC1 of two runs: `a` interpolates at 40, `b` never does."""
    epochs = [20, 40, 60, 80, 100, 120, 140]
    state = ([-0.3, -0.6, -0.9, -1.2, -0.7, -1.0, -1.4] if reset
             else [-0.3, -0.6, -0.9, -1.2, -1.5, -1.8, -2.1])
    clock = pd.DataFrame({
        "run": ["a"] * 7 + ["b"] * 7, "recipe": ["D1"] * 7 + ["D2"] * 7,
        "epoch": epochs * 2, "train_accuracy": [0.9, 1, 1, 1, 0.99, 1, 1] + [0.98] * 7,
        "val_accuracy": [0.8] * 14})
    nc1 = pd.DataFrame({"run": ["a"] * 7 + ["b"] * 7,
                        "recipe": ["D1"] * 7 + ["D2"] * 7, "epoch": epochs * 2,
                        "value": state + [-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]})
    return clock, nc1


def test_reference_points_stop_before_the_first_reset():
    points = check.reference_points(*frames()).set_index("run")
    assert points.loc["a", ["start", "end", "final", "reset"]].tolist() == [
        40, 80, 140, 100]
    assert points.loc["b", ["start", "end", "final"]].tolist() == [60, 140, 140]
    assert np.isnan(points.loc["b", "reset"]) and not points.loc["b", "interpolates"]
    smooth = check.reference_points(*frames(reset=False)).set_index("run")
    assert smooth.loc["a", "end"] == 140 and np.isnan(smooth.loc["a", "reset"])


def test_declines_and_readings():
    points = check.reference_points(*frames())
    rows = []
    for run in ("a", "b"):
        for epoch in (40, 60, 80, 140):
            for reference, slope in (("train", 0.1), ("train_small", 0.1),
                                     ("heldout", 0.02)):
                for name in FAR_SETS:
                    for csf in ("kNN", "NCI", "Maha"):
                        loss = slope * (epoch - 40) if csf != "Maha" else 0.0
                        rows.append({"run": run, "recipe": "D1", "epoch": epoch,
                                     "reference": reference, "set": name,
                                     "csf": csf, "auroc": 90 - loss})
    table = check.declines(pd.DataFrame(rows), points)
    knn = table[(table["run"] == "a") & (table["csf"] == "kNN")].set_index(
        "reference")
    assert knn.loc["train", "decline"] == pytest.approx(4.0)
    assert knn.loc["heldout", "decline_final"] == pytest.approx(2.0)
    read = check.readings(table).set_index(["run", "csf"])
    assert set(read.index.get_level_values("csf")) == {"kNN", "NCI"}
    assert read.loc[("a", "kNN"), "reading"] == "reference set"
    assert read.loc[("b", "NCI"), "reading"] == "reference set"
    assert check.declines(pd.DataFrame(), points).empty


def test_index_errors_prefer_the_index_that_resets():
    clock, nc1 = frames()
    point = check.reference_points(clock, nc1).set_index("run", drop=False).loc["a"]
    state = nc1[nc1["run"] == "a"].set_index("epoch")["value"]
    metrics = pd.DataFrame([
        {"run": "a", "epoch": epoch, "set": name, "csf": csf,
         "auroc": 0.8 + 0.1 * value + 0.01 * position}
        for epoch, value in state.items() for name in FAR_SETS
        for position, csf in enumerate(("kNN", "NCI"))])
    indices = pd.DataFrame({"follows": state.to_numpy() * 2,
                            "counts_up": np.arange(7.0)},
                           index=pd.MultiIndex.from_product([["a"], state.index]))
    errors, clock_error, after = check.index_errors(metrics, indices, point, state)
    assert after == [100, 120]
    found = errors.set_index("index")["error"]
    assert found["follows"] == pytest.approx(0.0, abs=1e-9)
    assert found["counts_up"] == pytest.approx(clock_error) and clock_error > 2
    none = check.index_errors(metrics, indices, point.copy().replace(100, np.nan),
                              state)
    assert none[2] == [] and none[0].empty


def fake_result(run: str, epoch: int, level: float) -> dict:
    """One result file: the scores fall with the collapse level, less under `heldout`."""
    loss = 10 * (-level - 0.6)
    metrics = [{"set": name, "csf": csf, "auroc": (90 - loss) / 100, "fpr95": 0.5,
                "aurc": 0.1, "augrc": 0.1} for name in FAR_SETS for csf in ALL_CSFS]
    reference = [{"reference": name, "set": far, "csf": csf,
                  "auroc": (90 - loss * share) / 100, "fpr95": 0.5}
                 for name, share in zip(REFERENCES, (1.0, 1.0, 0.2))
                 for far in FAR_SETS for csf in ALL_CSFS]
    descriptors = [
        {"group": "clock", "name": "train_accuracy", "layer": "",
         "value": 1.0 if epoch >= 40 else 0.9},
        {"group": "clock", "name": "val_accuracy", "layer": "", "value": 0.8},
        {"group": "nc_paper", "name": "var_collapse", "layer": PENULTIMATE,
         "value": 10 ** level},
        {"group": "spread", "name": "cdnv_train", "layer": PENULTIMATE,
         "value": 10 ** level / 3},
        {"group": "weights", "name": "conv_norm", "layer": "", "value": 50 - epoch / 10}]
    return {"run": run, "epoch": epoch, "metrics": metrics, "layer_metrics": [],
            "layer_errors": {}, "descriptors": descriptors, "placement": [],
            "reference_metrics": reference, "reference_errors": {},
            "checks": {"g4_max_gap": 0.0, "head_gap": 0.0, "reference_nofit_gap": 0.0,
                       "reference_fit": 1962.0, "reference_test": 1963.0},
            "errors": {}, "nonfinite": dict.fromkeys(ALL_CSFS, 0)}


def test_report_on_fabricated_results(tmp_path, monkeypatch):
    _, nc1 = frames()
    folder = tmp_path / "runs" / "D1_seed0" / results_folder(suite=check.SUITE)
    folder.mkdir(parents=True)
    for _, row in nc1[nc1["run"] == "a"].iterrows():
        (folder / f"epoch_{int(row['epoch']):03d}.json").write_text(json.dumps(
            fake_result("D1_seed0", int(row["epoch"]), row["value"])))
    out = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", ["analyze_refcheck", "--results-root",
                                     str(tmp_path), "--out", str(out)])
    check.main()
    text = out.read_text()
    assert "No earlier evaluation of these runs" in text
    read = pd.read_csv(tmp_path / "refcheck" / "readings.csv").set_index("csf")
    assert read.loc["kNN", "train"] == pytest.approx(6.0)
    assert read.loc["kNN", "heldout"] == pytest.approx(1.2)
    assert (read["reading"] == "reference set").all()
    errors = pd.read_csv(tmp_path / "refcheck" / "index_errors_D1_seed0.csv")
    errors = errors.set_index("index")["error"]
    assert errors["nc_paper/var_collapse"] == pytest.approx(0.0, abs=1e-6)
    assert errors["spread/cdnv_train"] == pytest.approx(0.0, abs=1e-6)
    assert errors["weights/conv_norm"] > 1.0
    assert "Checkpoints after the reset: [100, 120]" in text
