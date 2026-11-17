"""ID-only descriptors and placement descriptors of one checkpoint.

Definitions: ../documentation/imagenette_dynamics_protocol_2026-11-10.md.
All computations run in float64.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dynamics.config import LAYERS, NEIGHBORS, PENULTIMATE
from dynamics.metrics import auroc
from dynamics.scores import knn_distance

CLASS_MEAN_METRICS = ("var_collapse", "equinorm_uc", "equiangular_uc",
                      "max_equiangular_uc")
PLACEMENT = ("norm_ratio", "outside_share_gap", "class_cosine_gap",
             "centroid_offset", "neighbor_ratio")
#: Collapse metrics without a matrix inversion (reference check of 2026-11-12).
SPREAD = ("cdnv", "trace_ratio", "within_inside_share")
REFERENCE_SPREAD = ("val_over_train", "val_over_train_inside",
                    "val_over_train_outside")
#: Amendment A7: statistics of the validation set alone. 14 for each stage, and 12
#: of the logits and the head.
VALID = "valid"
CALIBRATION_BINS = 15
#: Stage D, question Q3: the same geometry under other conventions.
CONVENTIONS = ("outside_share_gap_raw", "orthodev", "rowspace_share",
               "rowspace_share_gap")


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


def spread_metrics(features: torch.Tensor, labels: torch.Tensor,
                   n_classes: int) -> dict[str, float]:
    """Collapse metrics without a matrix inversion, about the class means of a set.

    cdnv is the class-distance normalized variance of Galanti et al.: the mean over
    the class pairs of (V_c + V_d) / (2 ||mu_c - mu_d||^2), where V_c is the mean
    squared distance of the rows of a class to its mean. trace_ratio is
    trace(Sigma_W) / trace(Sigma_B), with the averages of `class_mean_metrics`.
    within_inside_share is the share of the within-class scatter that lies inside
    the centered span of the class means.
    """
    x = features.double()
    geometry = class_geometry(x, labels, n_classes)
    within = x - geometry.means[labels]
    squared = within.pow(2).sum(dim=1)
    per_class = torch.stack([squared[labels == c].mean() for c in range(n_classes)])
    distances = torch.cdist(geometry.means, geometry.means).pow(2)
    pairs = ~torch.eye(n_classes, dtype=torch.bool)
    cdnv = ((per_class[:, None] + per_class[None, :]) / (2 * distances))[pairs].mean()
    total = squared.mean()
    between = geometry.centered.pow(2).sum() / n_classes
    inside = (within @ geometry.basis).pow(2).sum(dim=1).mean()
    return {"cdnv": float(cdnv), "trace_ratio": float(total / between),
            "within_inside_share": float(inside / total)}


def reference_spread(train: torch.Tensor, labels: torch.Tensor, val: torch.Tensor,
                     val_labels: torch.Tensor, geometry: Geometry
                     ) -> dict[str, float]:
    """Spread of the validation rows about the training class means, over the
    spread of the training rows.

    The three values are the total mean squared distance, its part inside the
    class-mean span, and its part outside the span. A value above 1 means that the
    training rows sit closer to their class means than the validation rows do.
    """
    def parts(features: torch.Tensor, targets: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        within = features.double() - geometry.means[targets]
        total = within.pow(2).sum(dim=1).mean()
        inside = (within @ geometry.basis).pow(2).sum(dim=1).mean()
        return total, inside, total - inside

    held_out, reference = parts(val, val_labels), parts(train, labels)
    return {name: float(a / b)
            for name, a, b in zip(REFERENCE_SPREAD, held_out, reference)}


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
        for part, block, targets in (("train", x, labels), ("val", v, val_labels)):
            for name, value in spread_metrics(block, targets, n_classes).items():
                add("spread", f"{name}_{part}", layer, value)
        for name, value in reference_spread(x, labels, v, val_labels,
                                            geometry).items():
            add("spread", name, layer, value)
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


def span_basis(vectors: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis, as columns, of the span of the rows of `vectors`."""
    _, singular, right = torch.linalg.svd(vectors.double(), full_matrices=False)
    rank = int((singular > 1e-10 * singular[0]).sum())
    return right[:rank].T


def inside_share(features: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Per row, the share of the squared norm that lies inside a subspace."""
    x = features.double()
    return (x @ basis).pow(2).sum(dim=1) / x.pow(2).sum(dim=1).clamp_min(1e-24)


def convention_descriptors(train: dict, val: dict, unseen: dict[str, dict],
                           correct: np.ndarray, w: torch.Tensor, n_classes: int
                           ) -> list[dict]:
    """Geometry of each unseen set under three conventions without centering.

    All quantities use the penultimate features as the network gives them:

    - `outside_share_gap_raw`: mean share of the energy outside the span of the
      plain class means, on the unseen set minus on the ID validation set. This is
      our off-span gap without the centering at the training mean.
    - `orthodev`: mean over classes of the absolute cosine between the plain class
      mean and the mean of the set (the fifth collapse property of NECO).
    - `rowspace_share`: mean share of the energy inside the row space of the
      classifier (as in GEODE), and `rowspace_share_gap`: the same minus the value
      on the ID validation set.

    The set `val` holds the reference values of the ID validation set. The set
    `id_errors` compares the misclassified validation images with the correct ones.
    """
    x = train[PENULTIMATE].double()
    means = torch.stack([x[train["labels"] == c].mean(dim=0)
                         for c in range(n_classes)])
    mean_span, row_space = span_basis(means), span_basis(w)
    unit_means = means / means.norm(dim=1, keepdim=True)

    def summary(features: torch.Tensor) -> dict[str, float]:
        if len(features) == 0:
            return dict.fromkeys(("outside", "rowspace", "orthodev"), float("nan"))
        center = features.double().mean(dim=0)
        cosines = unit_means @ (center / center.norm().clamp_min(1e-12))
        return {"outside": float((1 - inside_share(features, mean_span)).mean()),
                "rowspace": float(inside_share(features, row_space).mean()),
                "orthodev": float(cosines.abs().mean())}

    rows: list[dict] = []

    def add(name: str, values: dict[str, float]) -> None:
        rows.extend({"set": name, "layer": PENULTIMATE, "name": key,
                     "value": float(value)} for key, value in values.items())

    def against(found: dict[str, float], reference: dict[str, float]) -> dict:
        return {"outside_share_gap_raw": found["outside"] - reference["outside"],
                "orthodev": found["orthodev"],
                "rowspace_share": found["rowspace"],
                "rowspace_share_gap": found["rowspace"] - reference["rowspace"]}

    features = val[PENULTIMATE]
    reference = summary(features)
    add("val", {"orthodev": reference["orthodev"],
                "rowspace_share": reference["rowspace"]})
    for name, values in unseen.items():
        add(name, against(summary(values[PENULTIMATE]), reference))
    mask = torch.as_tensor(correct)
    add("id_errors", against(summary(features[~mask]), summary(features[mask])))
    return rows


def neighbor_agreement(features: torch.Tensor, labels: torch.Tensor,
                       k: int = NEIGHBORS, chunk: int = 2048,
                       device: torch.device | None = None) -> float:
    """Mean share of the k nearest other rows that have the same label.

    Distances are between the unit-normalized rows, as in the kNN score.
    """
    unit = features.double()
    unit = (unit / unit.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(device or "cpu")
    k = min(k, len(unit) - 1)
    shares = []
    for start in range(0, len(unit), chunk):
        part = unit[start:start + chunk]
        distance = torch.cdist(part, unit)
        rows = torch.arange(len(part), device=unit.device)
        distance[rows, rows + start] = float("inf")
        nearest = distance.topk(k, dim=1, largest=False).indices.cpu()
        same = labels[nearest] == labels[start:start + len(part), None]
        shares.append(same.double().mean(dim=1))
    return float(torch.cat(shares).mean())


def calibration_error(confidence: np.ndarray, correct: np.ndarray,
                      bins: int = CALIBRATION_BINS) -> float:
    """Expected calibration error with bins of equal width."""
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    error = 0.0
    for b in range(bins):
        inside = index == b
        if inside.any():
            error += inside.mean() * abs(correct[inside].mean()
                                         - confidence[inside].mean())
    return float(error)


def validation_statistics(val: dict, w: torch.Tensor, n_classes: int,
                          device: torch.device | None = None) -> list[dict]:
    """Statistics of the validation set alone, as long-format rows (amendment A7).

    They read the validation features, logits, and labels, and the head `w`. No
    training image and no unseen image enters. The class means and the centre of
    each stage are those of the validation set.
    """
    labels = val["labels"]
    rows: list[dict] = []

    def add(name: str, layer: str, value: float | np.floating | torch.Tensor
            ) -> None:
        rows.append({"group": VALID, "name": name, "layer": layer,
                     "value": float(value)})

    for layer in LAYERS:
        v = val[layer].double()
        geometry = class_geometry(v, labels, n_classes)
        for name, value in {**class_mean_metrics(v, labels, n_classes),
                            **spread_metrics(v, labels, n_classes),
                            **spectrum(v)}.items():
            add(name, layer, value)
        z = v - geometry.global_mean
        add("span_share", layer, (z @ geometry.basis).pow(2).sum() / z.pow(2).sum())
        norms = z.norm(dim=1).numpy()
        add("norm_cv", layer, norms.std() / norms.mean())
        add("class_cosine", layer,
            np.median(cosine_to(z, geometry.centered[labels]).numpy()))
        radial = (v - geometry.means[labels]).norm(dim=1).numpy()
        add("radial_tail", layer, np.quantile(radial, 0.95) / np.median(radial))
        add("neighbor_agreement", layer, neighbor_agreement(v, labels, device=device))

    logits = val["logits"].double()
    probability = torch.softmax(logits, dim=1)
    confidence = probability.max(dim=1).values.numpy()
    correct = (logits.argmax(dim=1) == labels).numpy()
    top = logits.topk(2, dim=1).values
    layer = "logits"
    add("accuracy", layer, correct.mean())
    add("loss", layer, torch.nn.functional.cross_entropy(logits, labels))
    add("margin", layer, np.median((top[:, 0] - top[:, 1]).numpy()))
    add("max_softmax", layer, confidence.mean())
    add("entropy", layer, -(probability * probability.clamp_min(1e-300).log())
        .sum(dim=1).mean())
    add("max_logit", layer, top[:, 0].mean())
    add("energy", layer, torch.logsumexp(logits, dim=1).mean())
    add("logit_norm", layer, logits.norm(dim=1).mean())
    add("calibration_error", layer, calibration_error(confidence, correct))
    add("max_softmax_errors", layer,
        confidence[~correct].mean() if (~correct).any() else np.nan)
    add("error_auroc", layer, auroc(confidence[correct], confidence[~correct])
        if correct.any() and (~correct).any() else np.nan)
    means = class_geometry(val[PENULTIMATE].double(), labels, n_classes).centered
    head = w.double()
    add("head_distance", layer, (head / head.norm() - means / means.norm()).pow(2).sum())
    return rows
