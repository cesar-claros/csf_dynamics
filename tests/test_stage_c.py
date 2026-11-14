import json

import numpy as np
import pandas as pd
import pytest

from dynamics import analyze_c, rules
from dynamics import analyze_multimetric as mm
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import PENULTIMATE, STAGES, results_folder
from dynamics.scores import ALL_CSFS


def training_cells(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product([["A_seed0", "B_seed0"], range(1, 31)],
                                       names=["run", "epoch"])
    metrics = pd.DataFrame({"geo/positive": rng.uniform(0.01, 5, len(index)),
                            "geo/signed": rng.normal(size=len(index)),
                            "clock/log_epoch": np.log(index.get_level_values(1))},
                           index=index)
    signal = np.log10(metrics["geo/positive"]) + metrics["geo/signed"]
    scores = pd.DataFrame({"a": 70 + 8 * signal, "b": 70 - 8 * signal,
                           "c": 72.0 + rng.normal(0, 0.1, len(index))}, index=index)
    return metrics, scores.clip(5, 95)


def test_a_frozen_rule_equals_the_ridge_of_the_multi_metric_test():
    metrics, scores = training_cells()
    columns = ["geo/positive", "geo/signed"]
    rule = rules.fit_ridge_rule(metrics, scores, columns)
    assert rule.log == [True, False] and rule.floor[1] is None
    expected = mm.ridge(mm.transform(metrics)[columns].to_numpy(),
                        mm.logit(scores.to_numpy()),
                        mm.transform(metrics)[columns].to_numpy())
    assert rule.predict(metrics) == pytest.approx(expected, abs=1e-9)
    assert set(rule.choose(metrics)) <= {"a", "b", "c"}


def test_a_rule_handles_inputs_outside_the_training_range():
    metrics, scores = training_cells()
    rule = rules.fit_ridge_rule(metrics, scores, ["geo/positive", "geo/signed"])
    new = pd.DataFrame({"geo/positive": [-1.0, 0.5, np.nan],
                        "geo/signed": [0.0, 0.0, 0.0]})
    floor = pd.DataFrame({"geo/positive": [rule.floor[0]], "geo/signed": [0.0]})
    chosen = rule.choose(new)
    assert chosen[0] == rule.choose(floor)[0], "a value below 0 is set to the floor"
    assert chosen[1] in ("a", "b", "c") and chosen[2] is None
    assert rule.choose(new.drop(columns="geo/signed")) == [None, None, None]


def test_rules_survive_a_round_trip_and_the_hash_is_stable(tmp_path):
    metrics, scores = training_cells()
    rule = rules.fit_ridge_rule(metrics, scores, list(metrics.columns))
    content = {"targets": {"far": {"rules": {"geometry": rules.asdict(rule)},
                                   "baselines": {}}}}
    first = rules.save(content, tmp_path / "rules.json")
    assert rules.save(content, tmp_path / "again.json") == first
    loaded = rules.load(tmp_path / "rules.json")["targets"]["far"]["rules"]["geometry"]
    assert loaded.predict(metrics) == pytest.approx(rule.predict(metrics))


def test_the_committed_rules_file_is_the_frozen_one():
    assert rules.file_hash(rules.RULES_PATH) == analyze_c.RULES_SHA256
    frozen = rules.load()
    assert len(frozen["training_runs"]) == 12 and frozen["penalty"] == 10
    far = frozen["targets"]["far"]
    assert far["sets"] == list(FAR_SETS)
    assert [len(far["rules"][name].columns) for name in rules.RIDGE_RULES] == [
        93, 5, 106]
    assert far["baselines"]["b1"]["score"] == "MahaPP"
    assert {kind: value["score"] for kind, value in far["baselines"]["b2"].items()
            } == {"plain": "PCA_RecError", "augmented": "ViM"}
    assert frozen["targets"]["near"]["baselines"]["b1"]["score"] == "kNN"


def table(geometry: list[float], b2: list[float], b1: list[float] | None = None
          ) -> pd.DataFrame:
    recipes = ["C1"] * 3 + ["C2"] * 3 + ["C3"] * 3
    return pd.DataFrame({"recipe": recipes, "geometry": geometry, "clock": b2,
                         "B1": b1 or [5.0] * 9, "B2": b2, "stand_in": b2},
                        index=[f"{r}_seed{i % 3}" for i, r in enumerate(recipes)])


def test_decision_rule_of_the_amendment():
    small = [0.2, 0.4, 0.1, 0.5, 0.3, 0.2, 0.6, 0.1, 0.3]
    assert analyze_c.decide(table(small, small))[0].startswith("M,")
    large = [3.0, 2.5, 3.5, 4.0, 3.0, 3.6, 2.0, 2.4, 2.8]
    works = [0.5, 0.2, 0.6, 0.4, 0.3, 0.5, 2.6, 2.4, 2.9]   # lower in C1 and C2 only
    outcome, details = analyze_c.decide(table(works, large))
    assert outcome.startswith("S2,") and details.loc[0, "recipes_lower"] == 2
    one_recipe = [0.0, 0.0, 0.0, 4.1, 3.1, 3.7, 2.1, 2.5, 2.9]   # lower in C1 only
    assert analyze_c.decide(table(one_recipe, large, [1.0] * 9))[0] == "N, no claim"
    kind_only = [3.1, 2.4, 3.6, 4.1, 2.9, 3.7, 2.1, 2.3, 2.9]
    assert analyze_c.decide(table(kind_only, large, [9.0] * 9))[0].startswith("S1,")
    worse = [value + 2 for value in large]
    assert analyze_c.decide(table(worse, large, [1.0] * 9))[0] == (
        "N, no claim (the rule is harmful)")
    missing = table(works, large)
    missing.loc["C2_seed1", "geometry"] = np.nan
    assert analyze_c.decide(missing)[0].startswith("not decided")


def test_gate_k_and_the_interval():
    known = pd.DataFrame({"recipe": ["A", "A", "B"], "geometry": [0.2, 1.4, 0.3]})
    assert analyze_c.gate_k(known)[0]
    assert not analyze_c.gate_k(known.assign(geometry=[2.0, 1.4, 0.3]))[0]
    assert not analyze_c.gate_k(table([1.0] * 9, [1.0] * 9))[0]
    mean, low, high = analyze_c.interval(np.array([-1.0, -1.2, -0.8, -1.1, -0.9]))
    assert low < mean < high < 0
    assert all(np.isnan(analyze_c.interval(np.array([1.0]))))


def result(run: str, epoch: int, rule_columns: list[str], best: str) -> dict:
    """One result file whose metrics exist for every column of the frozen rules."""
    rng = np.random.default_rng(abs(hash((run, epoch))) % 2 ** 32)
    descriptors = []
    for column in rule_columns:
        name, _, layer = column.partition("@")
        group, _, short = name.partition("/")
        if short == "log_epoch":
            continue
        descriptors.append({"group": group, "name": short, "layer": layer,
                            "value": float(rng.uniform(0.1, 1.0))})
    for row in descriptors:
        if (row["group"], row["name"]) == ("clock", "val_accuracy"):
            row["value"] = 0.8
        if (row["group"], row["name"]) == ("clock", "train_accuracy"):
            row["value"] = 1.0
    metrics = [{"set": name, "csf": csf, "auroc": 0.9 if csf == best else 0.8,
                "fpr95": 0.5} for name in (*FAR_SETS, "ssb_hard", "ninco",
                                           "imagewoof", "blend") for csf in ALL_CSFS]
    return {"run": run, "epoch": epoch, "metrics": metrics, "descriptors": descriptors,
            "placement": [], "layer_metrics": [], "checks": {}, "errors": {},
            "nonfinite": {}}


def test_report_on_fabricated_stage_c_results(tmp_path, monkeypatch):
    frozen = rules.load()
    columns = frozen["targets"]["far"]["rules"]["all"].columns
    assert f"nc_paper/var_collapse@{PENULTIMATE}" in columns
    tag = results_folder(suite=analyze_c.SUITE)
    for spec in STAGES["c"]:
        folder = tmp_path / "runs" / spec.name / tag
        folder.mkdir(parents=True)
        best = "ViM" if spec.recipe in ("B", "C1", "C2") else "PCA_RecError"
        for epoch in (100, 200):
            (folder / f"epoch_{epoch:03d}.json").write_text(json.dumps(
                result(spec.name, epoch, columns, best)))
    out = tmp_path / "stage_c_report.md"
    monkeypatch.setattr("sys.argv", ["analyze_c", "--results-root", str(tmp_path),
                                     "--out", str(out)])
    analyze_c.main()
    text = out.read_text()
    assert "Runs: 15 of 15. Missing: none." in text and "NOT the frozen" not in text
    # The metrics of these files are random, so the rule fails on known recipes.
    assert "**Gate K: FAIL.**" in text
    assert ("**Outcome: rule not valid, because gate K failed (the decision rule "
            "alone gives: M, nothing to select") in text
    assert "**Outcome: T, the stand-in rule beats the score of the kind.**" in text
    choices = pd.read_csv(tmp_path / "stage_c" / "choices_far_final.csv")
    assert set(choices["epoch"]) == {200} and len(choices) == 15 * 5
    b2 = choices[choices["rule"] == "B2"]
    assert (b2["regret"] == 0).all(), "the kind decides the best score here"
    assert b2.set_index("run").loc["C3_seed0", "choice"] == "PCA_RecError"
    assert choices[choices["rule"] == "B1"]["regret"].max() == pytest.approx(10.0)
    with pytest.raises(SystemExit, match="not the frozen one"):
        other = tmp_path / "other.json"
        other.write_text(rules.RULES_PATH.read_text() + " ")
        monkeypatch.setattr("sys.argv", ["analyze_c", "--results-root", str(tmp_path),
                                         "--rules", str(other)])
        analyze_c.main()
