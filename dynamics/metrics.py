"""Performance metrics of a CSF. A higher score means more confidence."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _finite(*arrays: np.ndarray) -> bool:
    return all(len(a) > 0 and np.isfinite(a).all() for a in arrays)


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a positive score is above a negative score; ties count half."""
    if not _finite(positive, negative):
        return float("nan")
    ranks = rankdata(np.concatenate([positive, negative]))
    n_pos, n_neg = len(positive), len(negative)
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def fpr_at_tpr(positive: np.ndarray, negative: np.ndarray, tpr: float = 0.95
               ) -> float:
    """Share of negatives at or above the threshold that keeps `tpr` of positives."""
    if not _finite(positive, negative):
        return float("nan")
    ordered = np.sort(positive)
    threshold = ordered[int(np.floor((1 - tpr) * len(ordered)))]
    return float((negative >= threshold).mean())


def risk_coverage(confidence: np.ndarray, residual: np.ndarray
                  ) -> tuple[float, float]:
    """AURC and AUGRC of a selective classifier.

    Inputs are accepted in the order of decreasing confidence. At coverage k/n, the
    selective risk is the error share among the k accepted inputs, and the
    generalized risk is the error count among them divided by n. Each area is the
    mean over k = 1..n. Tied confidences share their errors equally, which is the
    expectation over a random order of the ties.

    Returns:
        (AURC, AUGRC), or NaN values if a confidence is not finite.
    """
    if not _finite(confidence):
        return float("nan"), float("nan")
    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    errors = residual[order].astype(np.float64)
    n = len(errors)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_confidence) != 0) + 1]
    sizes = np.diff(np.r_[starts, n])
    shared = np.repeat(np.add.reduceat(errors, starts) / sizes, sizes)
    cumulative = np.cumsum(shared)
    accepted = np.arange(1, n + 1)
    return float((cumulative / accepted).mean()), float((cumulative / n).mean())


def set_metrics(id_scores: np.ndarray, id_correct: np.ndarray,
                unseen_scores: np.ndarray) -> dict[str, float]:
    """Metrics of one CSF on ID validation images against one unseen set.

    AUROC and FPR@95 measure the separation of all ID images from the unseen
    images. AURC and AUGRC follow the paper's convention: misclassified ID images
    and all unseen images are failures.
    """
    confidence = np.concatenate([id_scores, unseen_scores])
    residual = np.concatenate([1 - id_correct.astype(np.float64),
                               np.ones(len(unseen_scores))])
    aurc, augrc = risk_coverage(confidence, residual)
    return {"auroc": auroc(id_scores, unseen_scores),
            "fpr95": fpr_at_tpr(id_scores, unseen_scores),
            "aurc": aurc, "augrc": augrc}


def error_metrics(id_scores: np.ndarray, id_correct: np.ndarray) -> dict[str, float]:
    """Metrics of one CSF for the ID-errors stand-in: correct against misclassified."""
    correct = id_correct.astype(bool)
    aurc, augrc = risk_coverage(id_scores, 1 - id_correct.astype(np.float64))
    return {"auroc": auroc(id_scores[correct], id_scores[~correct]),
            "fpr95": fpr_at_tpr(id_scores[correct], id_scores[~correct]),
            "aurc": aurc, "augrc": augrc}
