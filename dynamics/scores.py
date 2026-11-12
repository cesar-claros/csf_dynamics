"""All CSFs of the pilot: the paper's 21 and two additions with new code."""
from __future__ import annotations

import numpy as np
import torch

from dynamics.config import NEIGHBORS
from dynamics.paper_api import LOGIT_ONLY, PAPER_CSFS, paper_scores

EXTRA_CSFS = ("kNN", "FeatureNorm")
ALL_CSFS = (*PAPER_CSFS, *EXTRA_CSFS)

#: Protocol amendment A1. A CSF of the group `no_fit` reads only the test input and
#: the head. Every other CSF fits statistics of the training features:
#: `fit_plain` has no hyperparameter, `fit_relative` has hyperparameters on
#: normalized features or as shares, and `fit_absolute` has hyperparameters that are
#: fixed component counts or act on the raw feature scale.
CSF_GROUPS: dict[str, tuple[str, ...]] = {
    "no_fit": (*LOGIT_ONLY, "GradNorm", "FeatureNorm"),
    "fit_plain": ("CTM", "fDBD", "pNML", "Maha", "MahaPP"),
    "fit_relative": ("kNN", "NNGuide", "PCA_RecError", "KPCA_RecError"),
    "fit_absolute": ("ViM", "Residual", "NeCo", "NCI"),
}
CSF_GROUP = {csf: group for group, names in CSF_GROUPS.items() for csf in names}
NO_FIT_CSFS = CSF_GROUPS["no_fit"]


def knn_distance(reference: torch.Tensor, queries: torch.Tensor,
                 k: int = NEIGHBORS, chunk: int = 4096,
                 device: torch.device | None = None) -> np.ndarray:
    """Euclidean distance from each query to its k-th nearest reference row.

    Runs in float64. In float32, the squared-distance expansion loses the small
    distances of collapsed features.

    Raises:
        ValueError: If the reference has fewer than k rows.
    """
    if len(reference) < k:
        raise ValueError(f"{len(reference)} reference rows, k = {k}")
    ref = reference.double().to(device or "cpu")
    ref_sq = ref.pow(2).sum(dim=1)
    out = []
    for start in range(0, len(queries), chunk):
        part = queries[start:start + chunk].double().to(ref.device)
        squared = part.pow(2).sum(dim=1, keepdim=True) + ref_sq - 2 * part @ ref.T
        kth = torch.kthvalue(squared.clamp_min(0), k, dim=1).values
        out.append(kth.sqrt().cpu())
    if not out:
        return np.empty(0)
    return torch.cat(out).numpy()


def unit_rows(features: torch.Tensor) -> torch.Tensor:
    features = features.double()
    return features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)


def extra_scores(train: torch.Tensor, evaluation: torch.Tensor,
                 device: torch.device | None = None) -> dict[str, np.ndarray]:
    """kNN on normalized features (negative distance) and the feature norm."""
    return {
        "kNN": -knn_distance(unit_rows(train), unit_rows(evaluation), device=device),
        "FeatureNorm": evaluation.double().norm(dim=1).numpy(),
    }


def all_scores(train: torch.Tensor, labels: torch.Tensor, logits_train: torch.Tensor,
               w: torch.Tensor, b: torch.Tensor, evaluation: torch.Tensor,
               logits: torch.Tensor, n_classes: int,
               device: torch.device | None = None, precision: str = "float64"
               ) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Scores of every CSF on the rows of `evaluation`. Higher means more confident.

    `precision` is the precision of the paper's feature CSFs (see `paper_scores`).

    Returns:
        Scores per CSF, and the error text of each paper CSF that failed.
    """
    scores, errors = paper_scores(train, labels, logits_train, w, b, evaluation,
                                  logits, n_classes, precision)
    scores.update(extra_scores(train, evaluation, device))
    return scores, errors
