import json
from dataclasses import replace

import pytest
from test_stage_t import BEST, fake
from test_stage_v import N_ID, N_UNSEEN, world

from dynamics import analyze_augrc, analyze_t, analyze_v, rules
from dynamics.analyze_selection import FAMILY
from dynamics.config import RECIPES, STAGES, RunSpec, results_folder

NEW = tuple(f"X{i}" for i in range(1, 7))


def test_recipes_and_stages_of_amendment_a10():
    assert len(STAGES["x"]) == 12 and len(STAGES["a10_all"]) == 64
    assert len(STAGES["a10_train"]) == 41
    assert all(RECIPES[spec.recipe].label_smoothing == 0
               for spec in STAGES["a10_train"] + STAGES["x"])
    assert not ({spec.name for spec in STAGES["x"]}
                & {spec.name for spec in STAGES["a10_all"]})
    settings = [(RECIPES[name].augment, RECIPES[name].weight_decay, RECIPES[name].lr)
                for name in NEW]
    assert settings == [(False, 5e-3, 0.01), (True, 5e-3, 0.01), (False, 5e-5, 0.1),
                        (True, 5e-5, 0.1), (False, 5e-3, 0.1), (True, 5e-3, 0.1)]
    assert all(RECIPES[name].epochs == 200 for name in NEW)
    assert [FAMILY[name] for name in NEW] == ["plain", "augmented"] * 3
    # No earlier recipe has the settings of a new one.
    for name in NEW:
        assert all(RECIPES[name] != recipe for other, recipe in RECIPES.items()
                   if other != name)
    # With the new recipes, every cell of the grid without label smoothing has one.
    grid = {(recipe.augment, recipe.weight_decay, recipe.lr)
            for recipe in RECIPES.values()
            if recipe.label_smoothing == 0 and recipe.epochs == 200}
    assert grid == {(kind, decay, rate) for kind in (False, True)
                    for decay in (5e-5, 5e-4, 5e-3) for rate in (0.01, 0.05, 0.1)}


def test_part_3_freeze_and_test_on_fabricated_runs(tmp_path, monkeypatch):
    base = rules.load()["targets"]["far"]["rules"]["all"].columns
    wide = [f"valid/{name}@stage4" for name in ("cdnv", "trace_ratio", "span_share")]
    old = [RunSpec("A", 0), RunSpec("B", 0)]
    late = [RunSpec("C4", 0), RunSpec("C6", 0)]          # C4 has label smoothing
    seen = [RunSpec("T1", 0), RunSpec("T2", 0), RunSpec("U1", 0), RunSpec("U2", 0),
            RunSpec("V1", 0), RunSpec("V3", 0)]
    new = [RunSpec("X1", 0), RunSpec("X2", 0)]
    every = old + late + seen
    inside = [spec for spec in every if RECIPES[spec.recipe].label_smoothing == 0]
    for name, runs in (("e_main", old), ("c", []), ("c2", late),
                       ("a7_train", old + late), ("t", seen[:2]),
                       ("a8_train", old + late + seen[:2]), ("u", seen[2:4]),
                       ("a9_train", old + late + seen[:4]), ("v", seen[4:]),
                       ("a10_all", every), ("a10_train", inside), ("x", new)):
        monkeypatch.setitem(STAGES, name, runs)
    assert [spec.recipe for spec in inside] == ["A", "B", "C6", "T1", "U2", "V3"]
    write, run = world(tmp_path, monkeypatch)
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
    for spec in seen:
        write(spec, results_folder(suite="step0"), base + wide, True)
    earlier, path = tmp_path / "v.json", tmp_path / "x.json"
    run(analyze_v, "freeze", "--part", "1", "--rules", str(earlier))
    part = replace(analyze_v.PARTS[3], earlier=("a9_rule", earlier, "", analyze_t.TIERS))
    monkeypatch.setitem(analyze_v.PARTS, 3, part)
    common = ("--part", "3", "--rules", str(path))

    text = run(analyze_v, "freeze", *common)
    frozen = analyze_t.load(path)
    assert rules.file_hash(path) in text
    assert frozen["amendment"] == "A10"
    assert frozen["target"] == analyze_augrc.CORRECT_ONLY
    assert frozen["images"] == {"id": N_ID, "unseen": N_UNSEEN}
    # The rule under test has the runs without label smoothing only.
    assert frozen["training_runs"] == sorted(spec.name for spec in inside)
    assert frozen["baseline_runs"] == 6
    assert sorted(frozen["rules"]) == sorted([analyze_t.TIERS, analyze_v.WIDE_FIT])
    assert frozen[analyze_v.RECIPE_RULE] == BEST
    with pytest.raises(SystemExit, match="frozen once"):
        run(analyze_v, "freeze", *common)
    for spec in new:
        write(spec, results_folder(suite="step0"), base + wide, True, stored=True)
    with pytest.raises(SystemExit, match="not the frozen one"):
        run(analyze_v, "test", *common)
    text = run(analyze_v, "test", *common, "--allow-other-rules")
    assert text.startswith("# Amendment A10, 10 classes, no label smoothing")
    assert "2 of 2 runs" in text and "**Gate M passes.**" in text
    assert "**Outcome:" in text and analyze_v.WIDE_FIT in text and "a9_rule" in text


def test_the_committed_a10_rules_are_the_frozen_ones():
    part = analyze_v.PARTS[3]
    assert rules.file_hash(part.path) == part.sha256 != ""
    assert rules.file_hash(part.earlier[1]) == part.earlier[2]
    frozen = analyze_t.load(part.path)
    assert frozen["amendment"] == "A10"
    assert frozen["target"] == analyze_augrc.CORRECT_ONLY
    assert frozen["images"] == {"id": 3925, "unseen": 5000}
    assert set(frozen["training_runs"]) == {spec.name for spec in STAGES["a10_train"]}
    assert (len(frozen["training_runs"]), frozen["training_cells"],
            frozen["baseline_runs"]) == (41, 932, 39)
    assert sorted(frozen["rules"]) == sorted([analyze_t.TIERS, analyze_v.WIDE_FIT])
    assert all(len(rule.columns) == 137 for rule in frozen["rules"].values())
    assert frozen[analyze_v.ONE] == "NCI"
    assert frozen["shortlists"]["3"] == ["NCI", "kNN", "PCA_RecError"]

