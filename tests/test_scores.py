import numpy as np
import torch

from dynamics.config import PENULTIMATE
from dynamics.evaluate import evaluate_features, independent_logit_scores
from dynamics.paper_api import LOGIT_ONLY, NC_METRICS, PAPER_CSFS
from dynamics.scores import (
    ALL_CSFS,
    CSF_GROUP,
    CSF_GROUPS,
    NO_FIT_CSFS,
    all_scores,
    extra_scores,
)


def score(layers):
    train = layers["train"]
    evaluation = torch.cat([layers["val"][PENULTIMATE], layers["far"][PENULTIMATE]])
    logits = torch.cat([layers["val"]["logits"], layers["far"]["logits"]])
    return all_scores(train[PENULTIMATE], train["labels"], train["logits"],
                      layers["w"], layers["b"], evaluation, logits,
                      layers["n_classes"]), logits


def test_roster_has_the_21_paper_csfs_and_two_additions():
    assert len(PAPER_CSFS) == 21 and len(ALL_CSFS) == 23
    assert len(set(ALL_CSFS)) == 23


def test_groups_split_the_roster_into_no_fit_and_fitted():
    assert sorted(CSF_GROUP) == sorted(ALL_CSFS)
    assert sum(len(names) for names in CSF_GROUPS.values()) == len(ALL_CSFS)
    assert len(NO_FIT_CSFS) == 10 and "GradNorm" in NO_FIT_CSFS
    assert {len(names) for key, names in CSF_GROUPS.items() if key != "no_fit"} == {
        4, 5}
    assert CSF_GROUP["KPCA_RecError"] == "fit_relative"
    assert CSF_GROUP["Maha"] == "fit_plain" and CSF_GROUP["NeCo"] == "fit_absolute"


def test_all_scores_are_finite_and_repeatable(layers):
    (first, errors), logits = score(layers)
    (second, _), _ = score(layers)
    assert errors == {} and set(first) == set(ALL_CSFS)
    for name in ALL_CSFS:
        assert first[name].shape == (len(logits),)
        assert np.isfinite(first[name]).all(), name
        assert np.array_equal(first[name], second[name]), name


def test_logit_scores_match_an_independent_computation(layers):
    (scores, _), logits = score(layers)
    independent = independent_logit_scores(logits)
    assert np.allclose(scores["MSR"], independent["MSR"])
    assert np.allclose(scores["MLS"], independent["MLS"])
    order = np.argsort(scores["Energy"])
    assert np.array_equal(order, np.argsort(independent["Energy"]))


def test_extra_scores_orientation():
    train = torch.tensor([[1.0, 0.0], [0.9, 0.1]] * 10)
    evaluation = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    out = extra_scores(train, evaluation)
    assert out["kNN"][0] > out["kNN"][1]
    assert np.allclose(out["FeatureNorm"], [2.0, 3.0])


def test_evaluate_features_outputs(layers):
    features = {name: layers[name] for name in ("train", "val", "far")}
    out = evaluate_features(features, layers["w"], layers["b"], layers["n_classes"])
    assert len(out["metrics"]) == len(ALL_CSFS) * 2
    assert {row["set"] for row in out["metrics"]} == {"far", "id_errors"}
    assert out["checks"]["g4_max_gap"] < 1e-12
    assert out["checks"]["head_gap"] < 1e-4
    assert out["errors"] == {} and not any(out["nonfinite"].values())
    assert out["counts"] == {"val": 200, "far": 200, "train": 600}
    groups = {row["group"] for row in out["descriptors"]}
    assert groups == {"nc_paper", "clock", "nc", "gap", "spectrum", "norm_angle",
                      "region", "logits"}
    paper = [row for row in out["descriptors"] if row["group"] == "nc_paper"]
    assert [row["name"] for row in paper] == list(NC_METRICS)
    far = {row["csf"]: row["auroc"] for row in out["metrics"] if row["set"] == "far"}
    assert far["Maha"] > 0.9 and far["kNN"] > 0.9


def test_float64_fits_agree_with_float32_and_restore_the_default(layers):
    features = {name: layers[name] for name in ("train", "val", "far")}
    args = (features, layers["w"], layers["b"], layers["n_classes"])
    single = evaluate_features(*args, precision="float32")
    double = evaluate_features(*args)
    assert torch.get_default_dtype() == torch.float32
    assert single["precision"] == "float32" and double["precision"] == "float64"
    assert double["errors"] == {} and not any(double["nonfinite"].values())

    def aurocs(out):
        return {(row["set"], row["csf"]): row["auroc"] for row in out["metrics"]}

    # The fixture has 48 dimensions, fewer than NeCo's 100 components. So the NeCo
    # ratio is 1 up to rounding, and its ranking is rounding noise in any precision.
    first, second = aurocs(single), aurocs(double)
    far = [key for key in first if key[0] == "far"]
    assert max(abs(first[key] - second[key]) for key in far
               if key[1] != "NeCo") < 0.02
    for csf in LOGIT_ONLY:
        assert first[("far", csf)] == second[("far", csf)]
