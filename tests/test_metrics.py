import itertools

import numpy as np
import pytest

from dynamics.metrics import (
    auroc,
    error_metrics,
    fpr_at_tpr,
    risk_coverage,
    set_metrics,
)


def test_auroc_limits_and_ties():
    assert auroc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auroc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert auroc(np.ones(5), np.ones(7)) == 0.5


def test_auroc_matches_pair_count():
    rng = np.random.default_rng(0)
    positive, negative = rng.integers(0, 6, 40) * 1.0, rng.integers(0, 6, 30) * 1.0
    pairs = [(p > n) + 0.5 * (p == n) for p in positive for n in negative]
    assert auroc(positive, negative) == pytest.approx(np.mean(pairs))


def test_metrics_are_nan_for_non_finite_or_empty_scores():
    assert np.isnan(auroc(np.array([1.0, np.nan]), np.array([0.0])))
    assert np.isnan(auroc(np.array([]), np.array([0.0])))
    assert np.isnan(fpr_at_tpr(np.array([np.inf]), np.array([0.0])))
    assert all(np.isnan(risk_coverage(np.array([1.0, np.nan]), np.array([0, 1]))))


def test_fpr_at_95_percent_tpr():
    positive = np.arange(100.0)
    # The threshold keeps 95 of the 100 positives: scores 5 and above.
    assert fpr_at_tpr(positive, np.array([4.0, 5.0, 50.0, -1.0])) == 0.5
    assert fpr_at_tpr(positive, np.array([200.0])) == 1.0


def test_risk_coverage_for_a_perfect_ranking():
    n = 400
    confidence = np.r_[np.ones(n) * 2, np.ones(n)] + np.linspace(0, 0.1, 2 * n)
    residual = np.r_[np.zeros(n), np.ones(n)]
    aurc, augrc = risk_coverage(confidence, residual)
    accepted = np.arange(1, 2 * n + 1)
    errors = np.clip(accepted - n, 0, None)
    assert aurc == pytest.approx((errors / accepted).mean())
    assert augrc == pytest.approx((errors / (2 * n)).mean())


def test_risk_coverage_ties_equal_the_mean_over_all_orders():
    confidence = np.array([3.0, 2.0, 2.0, 2.0, 1.0])
    residual = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    values = []
    for order in itertools.permutations([1, 2, 3]):
        index = [0, *order, 4]
        jitter = confidence.copy()
        jitter[list(order)] += np.array([0.3, 0.2, 0.1])
        cumulative = np.cumsum(residual[index][np.argsort(-jitter[index],
                                                          kind="stable")])
        values.append(((cumulative / np.arange(1, 6)).mean(),
                       (cumulative / 5).mean()))
    aurc, augrc = risk_coverage(confidence, residual)
    assert aurc == pytest.approx(np.mean([v[0] for v in values]))
    assert augrc == pytest.approx(np.mean([v[1] for v in values]))


def test_set_metrics_counts_unseen_inputs_as_failures():
    id_scores, unseen = np.array([5.0, 4.0, 3.0]), np.array([2.0, 1.0])
    out = set_metrics(id_scores, np.array([True, True, True]), unseen)
    assert out["auroc"] == 1.0 and out["fpr95"] == 0.0
    cumulative = np.array([0, 0, 0, 1, 2])
    assert out["augrc"] == pytest.approx((cumulative / 5).mean())
    assert out["aurc"] == pytest.approx((cumulative / np.arange(1, 6)).mean())


def test_error_metrics_separate_correct_from_misclassified():
    scores = np.array([4.0, 3.0, 2.0, 1.0])
    out = error_metrics(scores, np.array([True, True, False, False]))
    assert out["auroc"] == 1.0
    assert np.isnan(error_metrics(scores, np.ones(4, dtype=bool))["auroc"])
