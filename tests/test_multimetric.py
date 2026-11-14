import numpy as np
import pandas as pd
import pytest

from dynamics import analyze_multimetric as mm


def test_transform_uses_a_log_scale_for_positive_columns():
    metrics = pd.DataFrame({"positive": [1.0, 10.0, 100.0], "signed": [-1.0, 0.0, 2.0],
                            "constant": [3.0, 3.0, 3.0],
                            "broken": [1.0, np.nan, 2.0]})
    out = mm.transform(metrics)
    assert list(out.columns) == ["positive", "signed"]
    assert out["positive"].tolist() == [0.0, 1.0, 2.0]
    assert out["signed"].tolist() == [-1.0, 0.0, 2.0]


def test_input_columns_split_clock_geometry_and_weights():
    names = ["clock/log_epoch", "clock/train_loss", mm.NC1, "nc_paper/self_duality@x",
             "spread/cdnv_train@stage3", "weights/conv_norm"]
    columns = mm.input_columns(names)
    assert columns["clock"] == names[:2] and columns["nc1"] == [mm.NC1]
    assert columns["nc8"] == names[2:4]
    assert columns["geometry"] == names[2:5] and columns["all"] == names


def test_ridge_recovers_a_linear_map():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = x @ np.array([[1.0, -2.0], [0.5, 0.0], [0.0, 3.0]]) + 4
    predicted = mm.ridge(x[:150], y[:150], x[150:], penalty=1e-9)
    assert predicted == pytest.approx(y[150:], abs=1e-6)


def cells(signal: str, interaction: bool = False, seeds: int = 3
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two families and two stages, with four scores.

    Score `a` is best early and `b` late. Score `c` is a good compromise in family
    p, and `d` in family q. The metric `stage` rises with the epoch, and `kind` is
    the family. With `signal` equal to `none` the metrics are noise, with `family`
    only `kind` is informative, and with `stage` both are. With `interaction`, the
    scores `a` and `b` swap their roles in family q, so the stage effect differs
    between the families.
    """
    rng = np.random.default_rng(1)
    rows, index = [], []
    for family in "pq":
        for seed in range(seeds):
            for epoch in range(1, 21):
                late = epoch > 10
                if interaction and family == "q":
                    late = not late
                value = {"a": 70.0 if late else 90.0, "b": 90.0 if late else 70.0,
                         "c": 86.0 if family == "p" else 60.0,
                         "d": 86.0 if family == "q" else 60.0}
                kind = float(family == "q") if signal != "none" else rng.normal()
                stage = float(epoch) if signal == "stage" else rng.normal()
                rows.append({**value, "kind": kind + 0.01 * rng.normal(),
                             "stage": stage + 0.01 * rng.normal(), "family": family})
                index.append((f"{family}{seed}", epoch))
    table = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index,
                                                               names=["run", "epoch"]))
    meta = table[["family"]].assign(eligible=True)
    metrics = table[["kind", "stage"]].rename(columns={
        "kind": "geo/kind", "stage": "geo/stage"})
    metrics["clock/noise"] = rng.normal(size=len(table))
    return meta, table[list("abcd")], metrics


def run_case(signal: str, **options) -> tuple[pd.Series, str]:
    method = options.pop("method", "ridge")
    results = mm.evaluate(*cells(signal, **options), "run", penalty=0.1,
                          method=method)
    return results.groupby("input")["regret"].mean(), mm.verdict(results)[1]


def test_noise_metrics_do_not_beat_a_fixed_score():
    mean, decision = run_case("none")
    assert mean[mm.FIXED] == pytest.approx(10.0)      # `a` or `b`: wrong in one stage
    assert mean[mm.FAMILY_FIXED] == pytest.approx(4.0)   # the compromise of the family
    assert decision.startswith("R1 fails")


def test_a_family_metric_recognises_the_recipe_and_nothing_more():
    mean, decision = run_case("family")
    assert mean["geometry"] == pytest.approx(4.0)
    assert decision.startswith("R1 holds and R2 fails")


def test_a_stage_metric_also_reads_the_training_stage():
    mean, decision = run_case("stage")
    assert mean["geometry"] < 2.0 < mean[mm.FAMILY_FIXED]   # a line fits a step roughly
    assert decision == ("R1 and R2 hold: the metrics also read the training stage. "
                        "The geometry adds to the clock group")


def test_only_the_nearest_model_follows_a_stage_effect_that_differs_by_family():
    linear, _ = run_case("stage", interaction=True)
    local, decision = run_case("stage", interaction=True, method="nearest")
    assert linear["geometry"] >= 4.0, "an additive model cannot represent it"
    assert local["geometry"] < 1.0 and decision.startswith("R1 and R2 hold")


def test_one_family_left_out_has_no_family_reference():
    results = mm.evaluate(*cells("stage"), "family", penalty=0.1)
    assert mm.FAMILY_FIXED not in set(results["input"])
    assert set(results["run"]) == {f"{f}{s}" for f in "pq" for s in range(3)}
    table = mm.summarize(results)
    assert list(table.columns) == ["input", "all", "p", "q"]
    assert mm.verdict(results)[1].startswith("not decided")
