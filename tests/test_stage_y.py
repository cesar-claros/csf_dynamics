import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_stage_t import BEST
from test_stage_v import N_ID, N_UNSEEN, world

from dynamics import analyze_augrc, analyze_v, analyze_y, rules
from dynamics.analyze_selection import FAMILY
from dynamics.config import RECIPES, STAGES, RunSpec, results_folder

NEW = ("Y1", "Y2", "Y3")


def test_recipes_and_stages_of_amendment_a11():
    assert len(STAGES["y"]) == 6 and len(STAGES["a11_train"]) == 12
    assert {spec.recipe for spec in STAGES["a11_train"]} == {
        "H0", "H1", "H2", "H4", "H7", "W3"}
    assert all(RECIPES[spec.recipe].label_smoothing == 0
               and RECIPES[spec.recipe].epochs == 100
               for spec in STAGES["a11_train"] + STAGES["y"])
    settings = [(RECIPES[name].augment, RECIPES[name].weight_decay, RECIPES[name].lr)
                for name in NEW]
    assert settings == [(False, 5e-3, 0.05), (True, 5e-3, 0.05), (True, 5e-3, 0.01)]
    assert [FAMILY[name] for name in NEW] == ["plain", "augmented", "augmented"]
    # No earlier recipe has the settings of a new one.
    for name in NEW:
        assert all(RECIPES[name] != recipe for other, recipe in RECIPES.items()
                   if other != name)
    # Each new recipe is a 10-class recipe of the grid with 100 epochs.
    for name, source in (("Y1", "U3"), ("Y2", "T3"), ("Y3", "X2")):
        assert RECIPES[name] == replace(RECIPES[source], epochs=100)


def table(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"recipe": [name for name in NEW for _ in range(2)],
                         analyze_y.UNDER: values})


def test_decision_rule_of_amendment_a11():
    assert analyze_y.decide(table([0.2, 0.4, 0.9, 1.0, 0.0, 0.3])).startswith("S,")
    # One recipe above 1 point, and the mean inside the limit.
    assert analyze_y.decide(table([0.0, 0.0, 0.1, 0.1, 1.4, 1.6])).startswith("H,")
    # A seed above 1 point does not decide: the recipe mean does.
    assert analyze_y.decide(table([0.0, 0.0, 0.1, 0.1, 0.4, 1.4])).startswith("S,")
    # Two of three recipes above 1 point, or the mean above it.
    assert analyze_y.decide(table([0.0, 0.0, 1.2, 1.2, 1.2, 1.2])).startswith("F,")
    assert analyze_y.decide(table([0.0, 0.0, 0.0, 0.0, 4.0, 4.5])).startswith("F,")
    missing = table([0.2, 0.4, 0.9, 1.0, 0.0, np.nan])
    assert analyze_y.decide(missing).startswith("not decided")


def test_coverage_marks_a_score_that_is_good_in_both_sets():
    old = pd.DataFrame({"NCI": [90.0, 80.0], "kNN": [89.5, 79.5], "Maha": [70.0, 80.0]},
                       index=["H0_seed0", "H1_seed0"])
    new = pd.DataFrame({"NCI": [85.0, 70.0], "kNN": [85.5, 79.4], "Maha": [60.0, 80.0]},
                       index=["Y1_seed0", "Y2_seed0"])
    out = analyze_y.coverage(new, old).set_index("csf")
    assert out.loc["kNN", "covers"] and not out.loc["NCI", "covers"]
    assert not out.loc["Maha", "covers"]
    assert out.loc["NCI", "new_worst"] == pytest.approx(10.0)
    assert out.loc["kNN", "earlier_worst"] == pytest.approx(0.5)


def test_freeze_and_test_on_fabricated_runs(tmp_path, monkeypatch):
    base = rules.load()["targets"]["far"]["rules"]["all"].columns
    first = [RunSpec("H0", 0), RunSpec("H1", 0)]
    second = [RunSpec("H5", 0), RunSpec("H7", 0)]          # H5 has label smoothing
    third = [RunSpec("W3", 0), RunSpec("W4", 0)]           # W4 has label smoothing
    new = [RunSpec("Y1", 0), RunSpec("Y2", 0)]
    inside = [spec for spec in first + second + third
              if RECIPES[spec.recipe].label_smoothing == 0]
    for name, runs in (("h1", first), ("h2", second), ("a9_train_h", first + second),
                       ("w", third), ("a11_train", inside), ("y", new)):
        monkeypatch.setitem(STAGES, name, runs)
    write, run = world(tmp_path, monkeypatch)
    for spec in first + second + third:
        write(spec, results_folder(suite="step0"), base, False)
    earlier, path = tmp_path / "w.json", tmp_path / "choices.json"
    run(analyze_v, "freeze", "--part", "2", "--rules", str(earlier))
    monkeypatch.setattr(analyze_y, "SECONDARY", ("a9_rule", earlier, "", "all"))
    common = ("--choices", str(path))

    text = run(analyze_y, "freeze", *common)
    frozen = json.loads(path.read_text())
    assert rules.file_hash(path) in text
    assert frozen["amendment"] == "A11"
    assert frozen["target"] == analyze_augrc.CORRECT_ONLY
    assert frozen["images"] == {"id": N_ID, "unseen": N_UNSEEN}
    assert frozen["runs"] == sorted(spec.name for spec in inside)
    # Three of the four runs are plain, so the plain score is the one fixed score.
    assert frozen[analyze_y.UNDER] == BEST["plain"]
    assert frozen[analyze_y.RECIPE_RULE] == BEST
    with pytest.raises(SystemExit, match="frozen once"):
        run(analyze_y, "freeze", *common)
    with pytest.raises(SystemExit, match="not the frozen one"):
        run(analyze_y, "test", *common)
    allow = (*common, "--allow-other-files")
    with pytest.raises(SystemExit, match="no results of stage y"):
        run(analyze_y, "test", *allow)
    for spec in new:
        write(spec, results_folder(suite="step0"), base, False)
    with pytest.raises(SystemExit, match="do not store"):
        run(analyze_y, "test", *allow)
    for spec in new:
        write(spec, results_folder(suite="step0"), base, False, stored=True,
              shift=0.01)
    with pytest.raises(SystemExit, match="gate M fails"):
        run(analyze_y, "test", *allow)
    for spec in new:
        write(spec, results_folder(suite="step0"), base, False, stored=True)
    text = run(analyze_y, "test", *allow)
    assert "2 of 2 runs" in text and "**Gate M passes.**" in text
    # The plain score loses 10 points in the augmented recipe Y2.
    assert "**Outcome: F," in text and "covers" in text and "a9_rule" in text


def test_the_committed_a11_choices_are_the_frozen_ones():
    assert rules.file_hash(analyze_y.CHOICES_PATH) == analyze_y.CHOICES_SHA256 != ""
    assert rules.file_hash(analyze_y.SECONDARY[1]) == analyze_y.SECONDARY[2]
    frozen = json.loads(analyze_y.CHOICES_PATH.read_text())
    assert frozen["amendment"] == "A11"
    assert frozen["target"] == analyze_augrc.CORRECT_ONLY
    assert frozen["images"] == {"id": 5000, "unseen": 5000}
    assert set(frozen["runs"]) == {spec.name for spec in STAGES["a11_train"]}
    assert frozen[analyze_y.UNDER] == "NCI"
    assert frozen[analyze_y.RECIPE_RULE] == {"augmented": "NCI", "plain": "NCI"}
    assert frozen["shortlists"]["3"] == ["NCI", "MahaPP", "CTM"]

