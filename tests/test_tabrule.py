import numpy as np
import pandas as pd
import pytest

from dynamics import analyze_i as ai
from dynamics import analyze_tabrule as tr
from dynamics import rules

CSFS = ["one", "two", "three"]


def made(seed: int = 0, rows: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inputs and scores in points, with a link between them."""
    rng = np.random.default_rng(seed)
    metrics = pd.DataFrame({
        "score/a": rng.uniform(0.2, 0.9, rows), "score/b": rng.uniform(0.2, 0.9, rows),
        "shift": rng.normal(0, 1, rows)},
        index=[f"m{seed}_{i}" for i in range(rows)])
    scores = pd.DataFrame({
        "one": 60 + 30 * metrics["score/a"], "two": 60 + 30 * metrics["score/b"],
        "three": 75 + 2 * metrics["shift"]}, index=metrics.index
    ) + rng.normal(0, 1, (rows, 3))
    return metrics, scores


def test_ridge_from_the_tables_is_the_rule():
    metrics, scores = made()
    new, _ = made(seed=1, rows=15)
    plain = rules.fit_ridge_rule(metrics, scores, list(metrics.columns))
    fold = tr.fold_of("plain", plain, metrics, scores, np.ones(len(metrics)), new)
    assert tr.ridge_predictions(fold) == pytest.approx(plain.predict(new))
    assert tr.picks_of(tr.ridge_predictions(fold), CSFS) == plain.choose(new)
    weights = np.random.default_rng(2).uniform(0.2, 3.0, len(metrics))
    weighted = ai.fit_weighted(metrics, scores, list(metrics.columns), weights)
    fold = tr.fold_of("weighted", weighted, metrics, scores, weights, new)
    assert tr.ridge_predictions(fold) == pytest.approx(weighted.predict(new))
    assert fold.test_ids == list(new.index) and fold.x_fit.shape == (40, 3)
    bad = new.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        tr.fold_of("bad", plain, metrics, scores, np.ones(len(metrics)), bad)


def bed(seed: int = 0) -> tr.Bed:
    """A bed with two fits, four units, and scores where `one` is always best."""
    metrics, scores = made(seed, rows=24)
    scores["one"] = 90.0
    rule = rules.fit_ridge_rule(metrics, scores, list(metrics.columns))
    units = pd.Series([f"u{i % 4}" for i in range(24)], index=scores.index)
    stored = pd.DataFrame({tr.RIDGE: ["two"] * 24, tr.FIXED: ["three"] * 24},
                          index=scores.index)
    out = tr.Bed("P", "a made bed", "lineage", CSFS, scores, units, stored)
    for name, rows in (("first", slice(0, 12)), ("second", slice(12, 24))):
        out.folds.append(tr.fold_of(name, rule, metrics, scores, np.ones(24),
                                    metrics.iloc[rows]))
    return out


def test_the_tables_have_a_stable_content_hash():
    first, again = tr.arrays_of([bed()]), tr.arrays_of([bed()])
    assert str(first["content_sha256"]) == str(again["content_sha256"])
    assert set(first) == {"manifest", "content_sha256"} | {
        f"P.{n:03d}.{part}" for n in range(2)
        for part in ("x_fit", "y_fit", "weights", "x_test")}
    other = bed()
    other.folds[1].x_test[0, 0] += 1e-9
    assert str(tr.arrays_of([other])["content_sha256"]) != str(
        first["content_sha256"])


def always(column: int, rows: int = 12) -> np.ndarray:
    table = np.zeros((rows, 3))
    table[:, column] = 1.0
    return table


def test_a_bed_reads_each_learner_and_leaves_out_an_incomplete_one():
    made_bed = bed()
    predictions = {
        "tabpfn_v3.5.P.000": always(0), "tabpfn_v3.5.P.001": always(0),
        "forest.P.000": always(1), "forest.P.001": always(1),
        "tabpfn_v2.5.P.000": always(0)}
    table, units, done = tr.bed_section(made_bed, predictions,
                                        ["tabpfn_v3.5", "tabpfn_v2.5", "forest"])
    assert done == ["tabpfn_v3.5", "forest"]
    assert (units["tabpfn_v3.5"] == 0).all(), "it picks the best score in each unit"
    assert units["forest"].to_numpy() == pytest.approx(units[tr.RIDGE].to_numpy())
    assert tr.letter(units, "tabpfn_v3.5").startswith("R")
    assert tr.letter(units, "forest").startswith("S")
    assert "same_choice" in table and len(units) == 4
    predictions["tabpfn_v3.5.P.001"] = np.full((12, 3), np.nan)
    assert tr.bed_section(made_bed, predictions, ["tabpfn_v3.5"])[2] == []


def test_the_four_letters():
    base = np.array([1.5, 2.5, 1.8, 2.2, 1.6, 2.4, 2.0, 1.9])
    units = pd.DataFrame({
        tr.RIDGE: base, "holds": base - 1.4, "better": base - 0.4 + 0.01 * np.arange(8),
        "worse": base + 0.6 + 0.01 * np.arange(8),
        "same": base + np.array([0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2])})
    assert tr.letter(units, "holds").startswith("R")
    assert tr.letter(units, "better").startswith("B")
    assert tr.letter(units, "worse").startswith("W")
    assert tr.letter(units, "same").startswith("S")
