import numpy as np
import pytest
import torch

from dynamics.config import LAYERS, PENULTIMATE
from dynamics.descriptors import (
    CLASS_MEAN_METRICS,
    CONVENTIONS,
    PLACEMENT,
    class_geometry,
    class_mean_metrics,
    compare,
    convention_descriptors,
    id_descriptors,
    inside_share,
    placement_descriptors,
    reference_spread,
    row_stats,
    span_basis,
    spectrum,
    spread_metrics,
    summarize,
)
from dynamics.paper_api import paper_nc
from dynamics.scores import knn_distance


def test_class_mean_metrics_equal_the_paper_class(layers):
    train = layers["train"]
    ours = class_mean_metrics(train[PENULTIMATE], train["labels"],
                              layers["n_classes"])
    paper = paper_nc(train[PENULTIMATE], train["labels"], layers["w"], layers["b"],
                     layers["n_classes"])
    for name in CLASS_MEAN_METRICS:
        assert ours[name] == pytest.approx(paper[name], rel=1e-4)


def test_class_mean_metrics_of_a_collapsed_simplex():
    k = 5
    simplex = torch.eye(k, dtype=torch.float64) - 1 / k
    labels = torch.arange(200) % k
    metrics = class_mean_metrics(simplex[labels] + 3.0, labels, k)
    for name in CLASS_MEAN_METRICS:
        assert metrics[name] == pytest.approx(0.0, abs=1e-9)


def test_spectrum_effective_rank_and_decay():
    gen = torch.Generator().manual_seed(0)
    noise = torch.randn(20000, 40, generator=gen, dtype=torch.float64)
    assert spectrum(noise)["effective_rank"] == pytest.approx(40, rel=0.02)
    line = torch.randn(500, 1, generator=gen, dtype=torch.float64) * torch.ones(1, 40)
    assert spectrum(line)["effective_rank"] == pytest.approx(1.0, abs=1e-6)
    scale = torch.arange(1, 41, dtype=torch.float64) ** -0.75
    assert spectrum(noise * scale)["decay_rate"] == pytest.approx(-1.5, abs=0.1)
    assert np.isnan(spectrum(noise[:, :10])["decay_rate"])


def test_knn_distance_matches_brute_force():
    gen = torch.Generator().manual_seed(1)
    reference = torch.randn(300, 12, generator=gen)
    queries = torch.randn(50, 12, generator=gen)
    brute = torch.cdist(queries.double(), reference.double()).sort(dim=1).values
    assert np.allclose(knn_distance(reference, queries, k=10, chunk=7),
                       brute[:, 9].numpy())
    with pytest.raises(ValueError):
        knn_distance(reference[:5], queries, k=10)


def test_placement_of_the_reference_set_is_neutral(layers):
    train, val = layers["train"], layers["val"]
    geometry = class_geometry(train[PENULTIMATE], train["labels"],
                              layers["n_classes"])
    stats = row_stats(val[PENULTIMATE], geometry, train[PENULTIMATE])
    same = compare(summarize(stats), summarize(stats), geometry.radius)
    assert same["norm_ratio"] == same["neighbor_ratio"] == 1.0
    assert same["outside_share_gap"] == same["class_cosine_gap"] == 0.0
    assert same["centroid_offset"] == 0.0
    empty = summarize(stats, np.zeros(len(stats.norm), dtype=bool))
    assert all(np.isnan(value) for value in empty.values())


def test_norm_ratio_follows_a_scaling_about_the_global_mean(layers):
    train, val = layers["train"], layers["val"]
    geometry = class_geometry(train[PENULTIMATE], train["labels"],
                              layers["n_classes"])
    center = geometry.global_mean.float()
    far = center + 2.5 * (val[PENULTIMATE] - center)
    reference = summarize(row_stats(val[PENULTIMATE], geometry, train[PENULTIMATE]))
    moved = compare(summarize(row_stats(far, geometry, train[PENULTIMATE])),
                    reference, geometry.radius)
    assert moved["norm_ratio"] == pytest.approx(2.5, rel=1e-5)
    assert moved["outside_share_gap"] == pytest.approx(0.0, abs=1e-6)
    assert moved["class_cosine_gap"] == pytest.approx(0.0, abs=1e-6)


def test_id_descriptors_cover_every_layer(layers):
    rows = id_descriptors(layers["train"], layers["val"], layers["n_classes"])
    names = {(row["group"], row["name"], row["layer"]) for row in rows}
    assert len(names) == len(rows)
    for layer in LAYERS:
        for name in CLASS_MEAN_METRICS:
            assert ("nc", name, layer) in names
        assert ("spectrum", "effective_rank", layer) in names
        assert ("norm_angle", "class_cosine_val", layer) in names
    assert ("region", "neighbor_radius_ratio", PENULTIMATE) in names
    assert ("logits", "margin_val", "logits") in names
    for layer in LAYERS:
        for name in ("cdnv_train", "cdnv_val", "trace_ratio_train",
                     "within_inside_share_val", "val_over_train_outside"):
            assert ("spread", name, layer) in names
    assert len(rows) == 4 * 20 + 3 + 2
    finite = [row for row in rows if row["name"] != "decay_rate"]
    assert all(np.isfinite(row["value"]) for row in finite)


def test_placement_descriptors_rows(layers):
    val = layers["val"]
    correct = (val["logits"].argmax(dim=1) == val["labels"]).numpy()
    rows = placement_descriptors(layers["train"], val, {"far": layers["far"]},
                                 correct, layers["n_classes"])
    keys = {(row["set"], row["layer"], row["name"]) for row in rows}
    for layer in LAYERS:
        for name in PLACEMENT:
            assert ("far", layer, name) in keys
            assert ("id_errors", layer, name) in keys
    assert ("far", "logits", "margin_ratio") in keys
    far = {row["name"]: row["value"] for row in rows
           if row["set"] == "far" and row["layer"] == PENULTIMATE}
    assert far["neighbor_ratio"] > 1 and far["class_cosine_gap"] < 0


def test_span_basis_and_inside_share():
    vectors = torch.tensor([[1.0, 0, 0], [2.0, 0, 0], [0, 3.0, 0]])
    basis = span_basis(vectors)
    assert basis.shape == (3, 2)
    points = torch.tensor([[1.0, 1.0, 0], [0, 0, 5.0], [3.0, 0, 4.0]])
    assert inside_share(points, basis).tolist() == pytest.approx([1.0, 0.0, 0.36])


def test_convention_descriptors_on_a_known_geometry():
    labels = torch.arange(40) % 2
    means = torch.tensor([[2.0, 0, 0, 0], [0, 2.0, 0, 0]])
    train = {PENULTIMATE: means[labels], "labels": labels}
    val = {PENULTIMATE: means[labels[:10]], "labels": labels[:10]}
    outside = {PENULTIMATE: torch.tensor([[0, 0, 1.0, 0], [0, 0, 0, 1.0]]).repeat(5, 1)}
    inside = {PENULTIMATE: torch.tensor([[1.0, 1.0, 0, 0]]).repeat(6, 1)}
    rows = convention_descriptors(train, val, {"outside": outside, "inside": inside},
                                  np.ones(10, dtype=bool), means, 2)
    found = {(row["set"], row["name"]): row["value"] for row in rows}
    assert {name for _, name in found} == set(CONVENTIONS)
    assert found[("outside", "orthodev")] == pytest.approx(0.0, abs=1e-12)
    assert found[("outside", "outside_share_gap_raw")] == pytest.approx(1.0)
    assert found[("outside", "rowspace_share")] == pytest.approx(0.0, abs=1e-12)
    assert found[("outside", "rowspace_share_gap")] == pytest.approx(-1.0)
    assert found[("inside", "orthodev")] == pytest.approx(2 ** -0.5)
    assert found[("inside", "outside_share_gap_raw")] == pytest.approx(0.0, abs=1e-12)
    assert found[("val", "rowspace_share")] == pytest.approx(1.0)
    assert np.isnan(found[("id_errors", "orthodev")])


def two_classes() -> tuple[torch.Tensor, torch.Tensor]:
    """Two classes on the first axis; each has spread 1 on it and spread 3 off it."""
    offsets = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0], [0, 3.0, 0], [0, -3.0, 0]])
    means = torch.tensor([[-2.0, 0, 0], [2.0, 0, 0]])
    rows = torch.cat([means[0] + offsets, means[1] + offsets])
    return rows, torch.tensor([0] * 4 + [1] * 4)


def test_spread_metrics_on_a_known_geometry():
    rows, labels = two_classes()
    found = spread_metrics(rows, labels, 2)
    # V_c = (1 + 1 + 9 + 9) / 4 = 5, and the squared distance of the means is 16.
    assert found["cdnv"] == pytest.approx(10 / 32)
    assert found["trace_ratio"] == pytest.approx(5 / 4)
    assert found["within_inside_share"] == pytest.approx(0.5 / 5)


def test_reference_spread_separates_inside_from_outside():
    rows, labels = two_classes()
    geometry = class_geometry(rows, labels, 2)
    same = reference_spread(rows, labels, rows, labels, geometry)
    assert all(value == pytest.approx(1.0) for value in same.values())
    wider = rows.clone()
    wider[:, 1] *= 2          # only the part outside the class-mean span grows
    found = reference_spread(rows, labels, wider, labels, geometry)
    assert found["val_over_train_inside"] == pytest.approx(1.0)
    assert found["val_over_train_outside"] == pytest.approx(4.0)
    assert found["val_over_train"] == pytest.approx((0.5 + 18) / 5)
