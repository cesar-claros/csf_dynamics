import numpy as np
import pandas as pd
import pytest

from dynamics import analyze_augrc, analyze_e
from dynamics.analyze_step0 import FAR_SETS
from dynamics.metrics import auroc, risk_coverage


@pytest.mark.parametrize("ties", [False, True])
def test_stored_values_give_the_two_other_targets(ties):
    rng = np.random.default_rng(0)
    n_id, n_unseen = 400, 500
    id_scores = rng.normal(1.0, 1.0, n_id)
    unseen = rng.normal(0.0, 1.0, n_unseen)
    if ties:
        id_scores, unseen = np.round(id_scores), np.round(unseen)
    correct = rng.random(n_id) < 0.8
    residual = np.r_[1 - correct.astype(float), np.ones(n_unseen)]
    # The stored values: AUGRC of the mix, and the AUROC of the ID errors.
    _, augrc = risk_coverage(np.r_[id_scores, unseen], residual)
    errors = auroc(id_scores[correct], id_scores[~correct])
    failure = analyze_augrc.failure_auroc(np.array([augrc]), np.array(
        [correct.sum()]), n_id, n_unseen)
    assert failure[0] == pytest.approx(auroc(
        id_scores[correct], np.r_[id_scores[~correct], unseen]), abs=1e-12)
    only = analyze_augrc.correct_only_auroc(
        failure, np.array([errors]), np.array([(~correct).sum()]), n_unseen)[0]
    assert only == pytest.approx(auroc(id_scores[correct], unseen), abs=1e-12)
    # The three conventions give three values.
    assert abs(failure[0] - auroc(id_scores, unseen)) > 1e-3
    assert abs(only - auroc(id_scores, unseen)) > 1e-3
    assert abs(only - failure[0]) > 1e-3


def test_correct_only_auroc_without_a_misclassified_image():
    failure = np.array([0.9, 0.8])
    only = analyze_augrc.correct_only_auroc(failure, np.array([np.nan, 0.7]),
                                            np.array([0.0, 0.0]), 500)
    assert np.allclose(only, failure)


def test_with_target_changes_the_far_sets_only():
    key = {"run": "A_seed0", "recipe": "A", "seed": 0, "held_out": -1, "epoch": 200}
    metrics = pd.DataFrame([
        {**key, "set": name, "csf": "kNN", "auroc": 0.9, "augrc": 0.3}
        for name in (*FAR_SETS, "blend", "id_errors")])
    descriptors = pd.DataFrame([{**key, "group": "clock", "name": "val_accuracy",
                                 "layer": "", "value": 0.8}])
    tables = analyze_e.Tables(metrics=metrics, descriptors=descriptors,
                              placement=pd.DataFrame(), checks=pd.DataFrame())
    failure = analyze_augrc.failure_auroc(np.array([0.3]), np.array([80.0]), 100,
                                          100)
    only = analyze_augrc.correct_only_auroc(failure, np.array([0.9]),
                                            np.array([20.0]), 100)
    for target, expected in ((analyze_augrc.FAILURE, failure[0]),
                             (analyze_augrc.CORRECT_ONLY, only[0])):
        out = analyze_augrc.with_target(tables, 100, 100, target).metrics
        far = out["set"].isin(FAR_SETS)
        assert np.allclose(out.loc[far, "auroc"], expected) and expected != 0.9
        assert (out.loc[~far, "auroc"] == 0.9).all()
    assert (tables.metrics["auroc"] == 0.9).all()       # the input is not changed
