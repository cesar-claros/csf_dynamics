import json

import numpy as np
import pandas as pd
import pytest
from test_stage_t import BEST, fake

from dynamics import analyze_t, analyze_u, rules
from dynamics.analyze_selection import FAMILY
from dynamics.config import RECIPES, STAGES, RunSpec, results_folder

NEW = ("U1", "U2", "U3", "U4")


def test_recipes_and_stages_of_amendment_a8():
    assert len(STAGES["u"]) == 8 and len(STAGES["a8_train"]) == 44
    assert {spec.recipe for spec in STAGES["u"]} == set(NEW)
    assert not ({spec.name for spec in STAGES["u"]}
                & {spec.name for spec in STAGES["a8_train"]})
    u1, u2, u3, u4 = (RECIPES[name] for name in NEW)
    assert (u1.augment, u1.lr, u1.label_smoothing) == (False, 0.01, 0.1)
    assert (u2.augment, u2.lr, u2.weight_decay) == (True, 0.01, 5e-5)
    assert (u3.augment, u3.weight_decay) == (False, 5e-3)
    assert (u4.augment, u4.label_smoothing) == (True, 0.2)
    assert [FAMILY[name] for name in NEW] == ["plain", "augmented", "plain",
                                              "augmented"]
    # No earlier recipe has the settings of a new one.
    for name in NEW:
        assert all(RECIPES[name] != recipe for other, recipe in RECIPES.items()
                   if other != name)


def table(tiers: list[float], wide: list[float], recipe_rule: list[float],
          second: list[float] | None = None) -> pd.DataFrame:
    recipes = [name for name in NEW for _ in range(2)]
    return pd.DataFrame({"recipe": recipes, analyze_u.TIERS: tiers,
                         analyze_u.WIDE: wide, analyze_u.RECIPE_RULE: recipe_rule,
                         analyze_u.B1: [5.0] * 8, analyze_u.STAND_IN: [3.0] * 8,
                         analyze_u.SECOND: second or tiers})


def test_decision_rule_of_amendment_a8():
    good = [0.2, 0.4, 0.1, 0.5, 0.3, 0.2, 0.6, 0.1]
    bad = [3.0, 2.5, 3.5, 4.0, 3.0, 3.6, 0.7, 0.2]
    selection, adds, details = analyze_u.decide(table(good, bad, good))
    assert selection == "H, the rule holds" and adds.startswith("supported")
    assert list(details["rule"]) == [analyze_u.TIERS] * 4 + [analyze_u.SECOND]
    selection, adds, details = analyze_u.decide(table(good, good, bad))
    assert selection.startswith("H+,") and adds == "not supported"
    assert details.loc[0, "recipes_lower"] == 4
    selection, adds, _ = analyze_u.decide(table(bad, [v + 2 for v in bad], good))
    assert selection.startswith("F,") and adds.startswith("supported")
    two_over = [1.5, 1.5, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0]          # mean 0.75
    assert analyze_u.decide(table(two_over, bad, good))[0].startswith("F,")
    # The second fit does not change the outcome.
    assert analyze_u.decide(table(bad, bad, good, second=good))[0].startswith("F,")
    # A run without a stand-in choice does not block the outcome.
    partial = table(good, bad, good)
    partial.loc[0, analyze_u.STAND_IN] = np.nan
    assert analyze_u.decide(partial)[0] == "H, the rule holds"
    missing = table(good, bad, good)
    missing.loc[0, analyze_u.TIERS] = np.nan
    assert analyze_u.decide(missing)[0].startswith("not decided")


def test_stand_in_choice_is_the_best_mean_over_the_three_sets():
    csfs = ["kNN", "MSR", "Maha"]
    values = {"first": {"kNN": (0.9, 0.8, 0.7), "MSR": (0.6, 0.9, 0.6),
                        "Maha": (0.75, 0.75, 0.75)},
              # Maha has no value on one set, so it leaves the choice.
              "second": {"kNN": (0.5, 0.5, 0.5), "MSR": (0.6, 0.6, 0.6),
                         "Maha": (0.99, np.nan, 0.99)},
              "third": {csf: (np.nan, np.nan, np.nan) for csf in csfs}}
    metrics = pd.DataFrame({
        f"score/{name}/{csf}": [values[run][csf][i] for run in values]
        for i, name in enumerate(analyze_t.SCORE_SETS) for csf in csfs},
        index=list(values))
    assert analyze_u.stand_in_choice(metrics, csfs) == ["kNN", "MSR", None]
    # A score without a column on a set leaves the choice too.
    assert analyze_u.stand_in_choice(
        metrics.drop(columns="score/blend/kNN"), csfs)[0] == "Maha"


def test_freeze_and_test_on_fabricated_runs(tmp_path, monkeypatch):
    base = rules.load()["targets"]["far"]["rules"]["all"].columns
    wide = [f"valid/{name}@stage4" for name in ("cdnv", "trace_ratio", "span_share")]
    old = [RunSpec("A", 0), RunSpec("B", 0)]
    late = [RunSpec("C4", 0), RunSpec("C5", 0)]
    seen = [RunSpec("T1", 0), RunSpec("T2", 0)]
    new = [RunSpec("U1", 0), RunSpec("U2", 0)]
    monkeypatch.setitem(STAGES, "e_main", old)
    monkeypatch.setitem(STAGES, "c", [])
    monkeypatch.setitem(STAGES, "c2", late)
    monkeypatch.setitem(STAGES, "a7_train", old + late)
    monkeypatch.setitem(STAGES, "t", seen)
    monkeypatch.setitem(STAGES, "a8_train", old + late + seen)
    monkeypatch.setitem(STAGES, "u", new)

    def write(spec, tag, columns, valid):
        folder = tmp_path / "runs" / spec.name / tag
        folder.mkdir(parents=True, exist_ok=True)
        for epoch in (100, 200):
            out = fake(spec.name, epoch, columns, BEST[FAMILY[spec.recipe]], valid)
            (folder / f"epoch_{epoch:03d}.json").write_text(json.dumps(out))

    for spec in old:
        write(spec, results_folder(suite="refcheck"), base, False)
    for spec in late:
        write(spec, results_folder(suite="step0"), base, False)
    for spec in old + late:       # the light pass stores only the validation rows
        folder = tmp_path / "runs" / spec.name / results_folder(suite="valstats")
        folder.mkdir(parents=True)
        for epoch in (100, 200):
            rows = [row for row in fake(spec.name, epoch, wide, "kNN", True)[
                "descriptors"] if row["group"] == "valid"]
            (folder / f"epoch_{epoch:03d}.json").write_text(json.dumps(
                {"run": spec.name, "epoch": epoch, "descriptors": rows}))
    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"
    report = tmp_path / "report.md"

    def run(module, step: str, *extra: str) -> str:
        monkeypatch.setattr("sys.argv", [
            "analyze", step, "--results-root", str(tmp_path), "--rules",
            str(first_path), "--out", str(report), *extra])
        module.main()
        return report.read_text()

    def second(step: str, *extra: str) -> str:
        return run(analyze_u, step, "--second-rules", str(second_path), *extra)

    run(analyze_t, "freeze")
    with pytest.raises(SystemExit, match="not the frozen one"):
        second("freeze")                  # the first file must be the frozen one
    with pytest.raises(SystemExit, match="without validation statistics"):
        second("freeze", "--allow-other-rules")       # the T runs are not there
    for spec in seen:
        write(spec, results_folder(suite="step0"), base + wide, True)
    text = second("freeze", "--allow-other-rules")
    frozen = analyze_t.load(second_path)
    assert rules.file_hash(second_path) in text
    assert frozen["amendment"] == "A8" and list(frozen["rules"]) == [analyze_u.TIERS]
    assert len(frozen["training_runs"]) == 6 and frozen["baseline_runs"] == 6
    assert frozen[analyze_u.RECIPE_RULE] == BEST
    with pytest.raises(SystemExit, match="frozen once"):
        second("freeze", "--allow-other-rules")
    with pytest.raises(SystemExit, match="no results of stage u"):
        second("test", "--allow-other-rules")
    for spec in new:
        write(spec, results_folder(suite="step0"), base + wide, True)
    with pytest.raises(SystemExit, match="not the frozen one"):
        second("test")
    text = second("test", "--allow-other-rules")
    assert "2 of 2 runs" in text and "leave the decision rule: none" in text
    assert "**Outcome:" in text and "**The claim is" in text
    assert "The second fit" in text and "the fixed shortlist is useful" in text


def test_the_committed_a8_rules_are_the_frozen_ones():
    assert rules.file_hash(analyze_u.RULES_U_PATH) == analyze_u.RULES_U_SHA256
    first = analyze_t.load(analyze_t.RULES_T_PATH)
    second = analyze_t.load(analyze_u.RULES_U_PATH)
    assert second["amendment"] == "A8" and list(second["rules"]) == [analyze_u.TIERS]
    assert len(second["training_runs"]) == 44 and second["baseline_runs"] == 42
    assert set(second["training_runs"]) == {spec.name for spec in STAGES["a8_train"]}
    # The second fit has the inputs of the rule under test, and other values.
    under_test, fitted_again = (file["rules"][analyze_u.TIERS]
                                for file in (first, second))
    assert sorted(fitted_again.columns) == sorted(under_test.columns)
    assert len(fitted_again.columns) == 137
    assert fitted_again.beta != under_test.beta
