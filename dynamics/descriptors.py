"""ID-only descriptors and placement descriptors of one checkpoint.

Definitions: ../documentation/imagenette_dynamics_protocol_2026-11-10.md.
All computations run in float64.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dynamics.config import LAYERS, PENULTIMATE
from dynamics.scores import knn_distance

CLASS_MEAN_METRICS = ("var_collapse", "equinorm_uc", "equiangular_uc",
                      "max_equiangular_uc")
PLACEMENT = ("norm_ratio", "outside_share_gap", "class_cosine_gap",
             "centroid_offset", "neighbor_ratio")


@dataclass
class Geometry:
    """Class geometry of the training features of one layer."""

    global_mean: torch.Tensor
    means: torch.Tensor
    centered: torch.Tensor
    basis: torch.Tensor

    @property
    def radius(self) -> float:
        """Mean distance of a class mean from the global mean."""
        return float(self.centered.norm(dim=1).mean())


def class_geometry(features: torch.Tensor, labels: torch.Tensor,
                   n_classes: int) -> Geometry:
    """Global mean, class means, and an orthonormal basis of their centered span."""
    x = features.double()
    means = torch.stack([x[labels == c].mean(dim=0) for c in range(n_classes)])
    global_mean = x.mean(dim=0)
    centered = means - global_mean
    left, singular, _ = torch.linalg.svd(centered.T, full_matrices=False)
    rank = int((singular > 1e-10 * singular[0]).sum())
    return Geometry(global_mean, means, centered, left[:, :rank])


def class_mean_metrics(features: torch.Tensor, labels: torch.Tensor,
                       n_classes: int) -> dict[str, float]:
    """The 4 NC metrics that need no head, with the formulas of the paper code.

    var_collapse is trace(Sigma_W pinv(Sigma_B)) / K, with Sigma_W averaged over the
    samples and Sigma_B averaged over the classes.
    """
    x = features.double()
    geometry = class_geometry(x, labels, n_classes)
    centered = geometry.centered
    within = x - geometry.means[labels]
    sigma_w = within.T @ within / len(x)
    sigma_b = centered.T @ centered / n_classes
    collapse = torch.trace(sigma_w @ torch.linalg.pinv(
        sigma_b, rtol=1e-6, hermitian=True)) / n_classes
    norms = centered.norm(dim=1)
    unit = centered / norms[:, None]
    off_diagonal = (unit @ unit.T)[~torch.eye(n_classes, dtype=torch.bool)]
    return {
        "var_collapse": float(collapse),
        "equinorm_uc": float(norms.std() / norms.mean()),
        "equiangular_uc": float(off_diagonal.std()),
        "max_equiangular_uc": float((off_diagonal + 1 / (n_classes - 1)).abs()
                                    .mean()),
    }


def spectrum(features: torch.Tensor) -> dict[str, float]:
    """Effective rank and decay rate of the eigenvalues of the total covariance.

    The decay rate is the slope of log eigenvalue against log rank, ranks 3 to 30.
    """
    x = features.double()
    z = x - x.mean(dim=0)
    eigenvalues = (torch.linalg.eigvalsh(z.T @ z / len(z)).flip(0).clamp_min(0)
                   .numpy())
    share = eigenvalues[eigenvalues > 0] / eigenvalues.sum()
    window = eigenvalues[2:30]
    decay = float("nan")
    if len(window) == 28 and (window > 0).all():
        decay = float(np.polyfit(np.log(np.arange(3, 31)), np.log(window), 1)[0])
    return {"effective_rank": float(np.exp(-(share * np.log(share)).sum())),
            "decay_rate": decay}


def cosine_to(z: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Cosine between each row of z and the matching row of directions."""
    return (z * directions).sum(dim=1) / (
        z.norm(dim=1) * directions.norm(dim=1)).clamp_min(1e-12)


def id_descriptors(train: dict, val: dict, n_classes: int,
                   device: torch.device | None = None) -> list[dict]:
    """The ID-only descriptors that this module computes, as long-format rows.

    The caller adds the paper's 8 NC metrics and the clock group.
    """
    rows: list[dict] = []

    def add(group: str, name: str, layer: str,
            value: float | np.floating | torch.Tensor) -> None:
        rows.append({"group": group, "name": name, "layer": layer,
                     "value": float(value)})

    labels, val_labels = train["labels"], val["labels"]
    for layer in LAYERS:
        x, v = train[layer].double(), val[layer].double()
        geometry = class_geometry(x, labels, n_classes)
        own = class_mean_metrics(x, labels, n_classes)
        for name, value in own.items():
            add("nc", name, layer, value)
        on_val = class_mean_metrics(v, val_labels, n_classes)["var_collapse"]
        add("gap", "var_collapse_val_over_train", layer,
            on_val / own["var_collapse"] if own["var_collapse"] > 0 else np.nan)
        for name, value in spectrum(x).items():
            add("spectrum", name, layer, value)
        z = x - geometry.global_mean
        inside = (z @ geometry.basis).pow(2).sum()
        add("spectrum", "span_share", layer, inside / z.pow(2).sum())
        norms = z.norm(dim=1).numpy()
        add("norm_angle", "norm_cv", layer, norms.std() / norms.mean())
        add("norm_angle", "class_cosine_train", layer,
            np.median(cosine_to(z, geometry.centered[labels]).numpy()))
        add("norm_angle", "class_cosine_val", layer, np.median(cosine_to(
            v - geometry.global_mean, geometry.centered[val_labels]).numpy()))

    x, v = train[PENULTIMATE].double(), val[PENULTIMATE].double()
    geometry = class_geometry(x, labels, n_classes)
    between = torch.cdist(geometry.means, geometry.means)
    between = between[~torch.eye(n_classes, dtype=torch.bool)].numpy()
    add("region", "neighbor_radius_ratio", PENULTIMATE,
        np.median(knn_distance(x, v, device=device)) / np.median(between))
    for name, features, targets in (("train", x, labels), ("val", v, val_labels)):
        radial = (features - geometry.means[targets]).norm(dim=1).numpy()
        add("region", f"radial_tail_{name}", PENULTIMATE,
            np.quantile(radial, 0.95) / np.median(radial))
    logits = val["logits"].double()
    top = logits.topk(2, dim=1).values
    add("logits", "margin_val", "logits", np.median((top[:, 0] - top[:, 1]).numpy()))
    add("logits", "max_softmax_val", "logits",
        torch.softmax(logits, dim=1).max(dim=1).values.mean())
    return rows


@dataclass
class RowStats:
    """Per-image quantities of one set at one layer, before any summary."""

    z: torch.Tensor
    norm: np.ndarray
    outside: np.ndarray
    cosine: np.ndarray
    neighbor: np.ndarray


def row_stats(features: torch.Tensor, geometry: Geometry,
              train_features: torch.Tensor,
              device: torch.device | None = None) -> RowStats:
    z = features.double() - geometry.global_mean
    norm = z.norm(dim=1).clamp_min(1e-12)
    inside = (z @ geometry.basis).pow(2).sum(dim=1)
    unit_means = geometry.centered / geometry.centered.norm(dim=1, keepdim=True)
    cosine = ((z / norm[:, None]) @ unit_means.T).max(dim=1).values
    return RowStats(z, norm.numpy(), (1 - inside / norm.pow(2)).numpy(),
                    cosine.numpy(),
                    knn_distance(train_features, features, device=device))


def summarize(stats: RowStats, mask: np.ndarray | None = None) -> dict[str, float]:
    """Summary of a set, or of the rows that a boolean mask keeps."""
    keep = np.ones(len(stats.norm), dtype=bool) if mask is None else mask
    if not keep.any():
        return dict.fromkeys(("norm", "outside", "cosine", "centroid", "neighbor"),
                             float("nan"))
    return {
        "norm": float(np.median(stats.norm[keep])),
        "outside": float(stats.outside[keep].mean()),
        "cosine": float(np.median(stats.cosine[keep])),
        "centroid": float(stats.z[torch.as_tensor(keep)].mean(dim=0).norm()),
        "neighbor": float(np.median(stats.neighbor[keep])),
    }


def compare(unseen: dict[str, float], reference: dict[str, float],
            radius: float) -> dict[str, float]:
    """The five feature-space placement descriptors of one set."""
    return {
        "norm_ratio": unseen["norm"] / reference["norm"],
        "outside_share_gap": unseen["outside"] - reference["outside"],
        "class_cosine_gap": unseen["cosine"] - reference["cosine"],
        "centroid_offset": (unseen["centroid"] - reference["centroid"]) / radius,
        "neighbor_ratio": unseen["neighbor"] / reference["neighbor"],
    }


def margin(logits: torch.Tensor) -> np.ndarray:
    top = logits.double().topk(2, dim=1).values
    return (top[:, 0] - top[:, 1]).numpy()


def placement_descriptors(train: dict, val: dict, unseen: dict[str, dict],
                          correct: np.ndarray, n_classes: int,
                          device: torch.device | None = None) -> list[dict]:
    """Placement of each unseen set relative to the ID validation set.

    The set `id_errors` compares the misclassified validation images with the
    correctly classified ones.
    """
    rows: list[dict] = []

    def add(name: str, layer: str, values: dict[str, float]) -> None:
        rows.extend({"set": name, "layer": layer, "name": key, "value": float(value)}
                    for key, value in values.items())

    for layer in LAYERS:
        geometry = class_geometry(train[layer], train["labels"], n_classes)
        val_stats = row_stats(val[layer], geometry, train[layer], device)
        reference = summarize(val_stats)
        for name, features in unseen.items():
            stats = row_stats(features[layer], geometry, train[layer], device)
            add(name, layer, compare(summarize(stats), reference, geometry.radius))
        add("id_errors", layer, compare(summarize(val_stats, ~correct),
                                        summarize(val_stats, correct),
                                        geometry.radius))
    val_margin = margin(val["logits"])
    for name, features in unseen.items():
        add(name, "logits", {"margin_ratio": float(
            np.median(margin(features["logits"])) / np.median(val_margin))})
    if (~correct).any() and correct.any():
        add("id_errors", "logits", {"margin_ratio": float(
            np.median(val_margin[~correct]) / np.median(val_margin[correct]))})
    return rows
