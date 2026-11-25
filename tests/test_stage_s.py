import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image
from test_build_subset import wnid, write_parquet

from dynamics import build_class_sets as cs
from dynamics import build_sets_parquet as bp
from dynamics import draw_recipes
from dynamics.analyze_selection import FAMILY
from dynamics.config import (
    A15_PER_SET,
    A15_SETS,
    RECIPES,
    STAGES,
    RunSpec,
    a15_class_sets,
    a15_set_of,
    expected_classes,
    parse_run,
    set_folder,
)

HERE = Path(__file__).resolve().parent


def test_frozen_class_sets_of_amendment_a15():
    from dynamics import analyze_s

    frozen = cs.load(bp.PATH)
    counts = json.loads((HERE.parent / "imagenet_class_counts.json").read_text())
    assert frozen == bp.draw_file(counts)            # the draw gives the frozen file
    assert analyze_s.frozen_inputs() == []           # with the recorded hashes
    sets = frozen["sets"]
    every = [name for classes in sets.values() for name in classes]
    assert list(sets) == [str(number) for number in range(1, A15_SETS + 1)]
    assert all(len(classes) == 10 and classes == sorted(classes)
               for classes in sets.values())
    assert len(set(every)) == 320                    # no class in two sets
    assert cs.list_hash(every) == frozen["class_list_sha256"]
    # No Imagenette class, no Imagewoof class, and no class of amendment A14.
    earlier = cs.load()
    used = sorted(name for classes in earlier["sets"].values() for name in classes)
    assert frozen["excluded"] == {"imagenette": earlier["excluded"]["imagenette"],
                                  "imagewoof": earlier["excluded"]["imagewoof"],
                                  "a14": used}
    excluded = [name for group in frozen["excluded"].values() for name in group]
    assert len(excluded) == 60 and not set(every) & set(excluded)
    # Every class has every image, so each set can take the counts of Imagenette.
    assert all(counts["train"][name] == 1300 and counts["val"][name] == 50
               for name in every)
    assert (frozen["train_images"], frozen["val_images"]) == (9469, 3925)
    assert a15_class_sets() == {int(k): tuple(v) for k, v in sets.items()}
    assert expected_classes("S005") == a15_class_sets()[2]
    assert expected_classes("S128") == a15_class_sets()[32]


def s_names() -> list[str]:
    return [f"S{number:03d}" for number in range(1, 129)]


def test_recipes_and_stage_of_amendment_a15():
    stored = json.loads(draw_recipes.A15_PATH.read_text())
    assert stored == draw_recipes.table_a15()        # the draw gives the frozen file
    assert json.loads(draw_recipes.PATH.read_text()) == draw_recipes.table()
    assert list(stored["recipes"]) == s_names() and stored["seed"] == 15
    recipes = [RECIPES[name] for name in s_names()]
    cells = [(recipe.augment, recipe.label_smoothing > 0) for recipe in recipes]
    # Each class set has 4 recipes, one in each cell of kind and label smoothing.
    assert A15_PER_SET * A15_SETS == 128
    for start in range(0, 128, A15_PER_SET):
        assert Counter(cells[start:start + A15_PER_SET]) == dict.fromkeys(
            draw_recipes.CELLS, 1)
    for recipe in recipes:
        assert 1e-5 <= recipe.weight_decay <= 5e-3 and 0.01 <= recipe.lr <= 0.1
        assert recipe.label_smoothing == 0 or 0.05 <= recipe.label_smoothing <= 0.2
        assert (recipe.epochs, recipe.arch) == (200, "resnet18")
    earlier = {recipe for name, recipe in RECIPES.items() if name not in s_names()}
    assert len(set(recipes)) == 128 and not earlier & set(recipes)
    assert all(FAMILY[name] == ("augmented" if RECIPES[name].augment else "plain")
               for name in s_names())
    assert [spec.name for spec in STAGES["s"]] == [f"{n}_seed0" for n in s_names()]
    assert [spec.name for spec in STAGES["s_first"]] == ["S001_seed0"]
    # Run names with three digits, and the earlier names as before.
    assert parse_run("S128_seed0") == RunSpec("S128", 0)
    assert parse_run("Q48_seed0").recipe == "Q48" and parse_run("C1_seed2").seed == 2
    with pytest.raises(ValueError):
        parse_run("S129_seed0")
    assert [a15_set_of(name) for name in ("S001", "S004", "S005", "S128")] == [
        1, 1, 2, 32]
    assert a15_set_of("Q01") is None and a15_set_of("P01") is None
    assert [set_folder(name) for name in ("S001", "S128", "Q13", "P01", "Z1")] == [
        "set01", "set32", "set2", None, None]


@pytest.fixture
def parquet_dir(tmp_path):
    """Fabricated ImageNet files: 6 classes with 14 training and 4 validation
    images, and one class with too few images."""
    folder = tmp_path / "data"
    folder.mkdir()
    labels = [label for label in range(6) for _ in range(14)] + [6] * 5
    half = len(labels) // 2
    for part, chunk in enumerate((labels[:half], labels[half:])):
        write_parquet(folder / f"train-0000{part}-of-00002.parquet", chunk,
                      [f"{wnid(label)}_{part}{i:03d}.JPEG"
                       for i, label in enumerate(chunk)], sizes=(40, 64))
    labels = [label for label in range(7) for _ in range(4)]
    write_parquet(folder / "validation-00000-of-00001.parquet", labels,
                  [f"ILSVRC2012_val_{i:08d}_{wnid(label)}.JPEG"
                   for i, label in enumerate(labels)], sizes=(40, 64))
    return folder


def test_counts_draw_and_build_of_class_sets_from_parquet(parquet_dir, tmp_path,
                                                          monkeypatch):
    counts = bp.count(parquet_dir, workers=2)
    assert counts["train"] == {**{wnid(i): 14 for i in range(6)}, wnid(6): 5}
    assert counts["val"] == {wnid(i): 4 for i in range(7)}
    # The draw takes classes with every image and leaves the earlier classes out.
    monkeypatch.setattr(cs, "FULL_TRAIN", 14)
    monkeypatch.setattr(cs, "FULL_VAL", 4)
    monkeypatch.setattr(cs, "load", lambda path=None: {
        "sets": {"1": [wnid(0)]}, "excluded": {"imagenette": [wnid(1)],
                                               "imagewoof": []}})
    drawn = bp.draw_file(counts, n_sets=2, size=2)
    every = [name for classes in drawn["sets"].values() for name in classes]
    assert drawn["candidates"] == 4 and sorted(every) == [wnid(i) for i in (2, 3, 4, 5)]
    assert drawn["excluded"]["a14"] == [wnid(0)]
    assert drawn["class_list_sha256"] == cs.list_hash(every)
    with pytest.raises(ValueError, match="need 6"):
        bp.draw_file(counts, n_sets=3, size=2)
    # The build writes each set with the image counts and a split with a fixed seed.
    sets = {"1": [wnid(2), wnid(3)], "2": [wnid(4), wnid(5)]}
    out, totals = tmp_path / "sets", (21, 7)
    manifests = bp.build(parquet_dir, out, sets, totals=totals, size=32, workers=2)
    assert sorted(path.name for path in out.iterdir()) == ["set01", "set02"]
    for number, classes in ((1, sets["1"]), (2, sets["2"])):
        folder = out / bp.folder_of(number)
        assert cs.check_set(folder, classes, [wnid(0)], totals) == []
        assert manifests[number]["counts"] == {
            "train": {classes[0]: 11, classes[1]: 10},
            "val": {classes[0]: 4, classes[1]: 3}}
        assert json.loads((folder / "split_manifest.json").read_text()) == manifests[
            number]
        image = next((folder / "train" / classes[0]).iterdir())
        assert min(Image.open(image).size) == 32 and image.name.startswith(
            ("train-0000", "validation-0000"))
    # A second build gives the same split, and another seed gives another one.
    again = bp.build(parquet_dir, tmp_path / "again", sets, totals=totals, size=32,
                     workers=2)
    assert again == manifests
    other = bp.build(parquet_dir, tmp_path / "other", sets, seed=1, totals=totals,
                     size=32, workers=2)
    assert other[1]["split_sha256"] != manifests[1]["split_sha256"]
    with pytest.raises(SystemExit, match="not in the parquet"):
        bp.build(parquet_dir, tmp_path / "bad", {"1": ["n99999999"]}, totals=(1, 1))
    with pytest.raises(ValueError, match="need 19"):
        bp.build(parquet_dir, tmp_path / "few", {"1": [wnid(2)]}, totals=(15, 4))


# ---- the stage runner with one folder for each class set ----

def test_each_run_takes_the_folder_of_its_class_set(tmp_path, monkeypatch):
    from dynamics import config, run_stage

    classes = {1: ("n01", "n02"), 2: ("n03", "n04"), 3: ("n05", "n06")}
    monkeypatch.setattr(config, "a15_class_sets", lambda: classes)
    root = tmp_path / "sets"
    for number in (1, 2):               # the third class set has no folder
        for name in classes[number]:
            (root / f"set{number:02d}" / "train" / name).mkdir(parents=True)
    first, fifth = RunSpec("S001", 0), RunSpec("S005", 0)
    assert run_stage.id_folder(first, None, root) == root / "set01"
    assert run_stage.id_folder(fifth, tmp_path / "other", root) == root / "set02"
    assert run_stage.id_folder(RunSpec("Q13", 0), None, root) == root / "set2"
    assert run_stage.id_folder(RunSpec("Z1", 0), tmp_path / "nette", root) == (
        tmp_path / "nette")
    with pytest.raises(ValueError, match="give --id-dir"):
        run_stage.id_folder(RunSpec("Z1", 0), None, root)
    assert run_stage.wrong_folder([first, fifth], None, root) == []
    # One folder for every run: the second class set does not fit it.
    assert run_stage.wrong_folder([first, fifth], root / "set01") == ["S005_seed0"]
    (root / "set02" / "train" / "n09").mkdir()
    assert run_stage.wrong_folder([first, fifth], None, root) == ["S005_seed0"]
    assert run_stage.wrong_folder([RunSpec("S009", 0)], None, root) == ["S009_seed0"]


# ---- the decision rule, with the class set as the unit ----

def units(rule: list[float], cell: list[float], one: float = 2.0):
    import pandas as pd

    from dynamics import analyze_s

    return pd.DataFrame({analyze_s.RULE: rule, analyze_s.BY_CELL: cell,
                         analyze_s.ONE: [one] * len(rule)},
                        index=pd.Index(range(101, 101 + len(rule)), name="set"))


def test_decision_rule_of_amendment_a15_has_the_class_set_as_the_unit():
    from dynamics import analyze_s

    good = [0.2, 0.4, 0.1, 0.5, 0.3, 0.2, 0.6, 0.1, 0.3, 0.2, 0.4, 0.3]
    worse = [value + 0.8 for value in good]
    assert analyze_s.outcome(units(good, worse)).startswith("A,")
    reading = analyze_s.adds(units(good, worse), analyze_s.RULE, analyze_s.BY_CELL)
    assert reading["supported"] and reading["lower"] == 12 and reading["p"] < 0.01
    assert "supported" in analyze_s.sentence(reading)
    # The same regret as the choice by cell: the rule holds with no shown gain.
    assert analyze_s.outcome(units(good, good)).startswith("H,")
    assert not analyze_s.adds(units(good, good), analyze_s.RULE,
                              analyze_s.BY_CELL)["supported"]
    # Above 1 point in half of the class sets: the rule does not hold.
    half = [1.5] * 6 + [0.0] * 6
    assert analyze_s.outcome(units(half, half)) == "F, the rule is not within 1 point"
    assert analyze_s.outcome(units(half, [value + 2 for value in half])) == (
        "F, the rule is not within 1 point, and it is below the choice by cell")
    assert analyze_s.holds(units(good, worse), analyze_s.RULE)
    assert not analyze_s.holds(units(good, worse), analyze_s.ONE)


def test_class_set_means_gate_t_and_stability():
    import pandas as pd

    from dynamics import analyze_s

    rows = []
    for number in (101, 102, 103, 104):
        for item in range(4):
            recipe = f"S{(number - 101) * 4 + item + 1:03d}"
            rows.append({"run": f"{recipe}_seed0", "recipe": recipe, "set": number,
                         analyze_s.RULE: 0.1 * item + (2.0 if number == 104 else 0.0),
                         analyze_s.BY_CELL: 0.5, analyze_s.ONE: 1.0})
    single = pd.DataFrame(rows).set_index("run")
    means = analyze_s.by_set(single)
    assert means.loc[101, analyze_s.RULE] == pytest.approx(0.15)
    assert means.loc[104, analyze_s.RULE] == pytest.approx(2.15)
    accuracy = pd.Series(0.8, index=single.index)
    kept, failed, small, decided = analyze_s.gate_t(single, accuracy, 4)
    assert len(kept) == 16 and failed == [] and small == [] and decided
    # One recipe below the accuracy leaves; its class set stays with 3 recipes.
    accuracy["S001_seed0"] = 0.5
    kept, failed, small, decided = analyze_s.gate_t(single, accuracy, 4)
    assert failed == ["S001"] and small == [] and len(kept) == 15 and decided
    # Two recipes of one class set leave: the class set leaves, 1 of 4 is allowed.
    accuracy["S002_seed0"] = 0.5
    kept, failed, small, decided = analyze_s.gate_t(single, accuracy, 4)
    assert small == [101] and len(kept) == 12 and decided
    accuracy["S005_seed0"] = accuracy["S006_seed0"] = 0.5
    assert not analyze_s.gate_t(single, accuracy, 4)[3]
    text = analyze_s.stability(single)
    assert "4 class sets" in text and "above the choice by cell in 1 class sets" in text
    assert "below it in 3" in text
    assert [analyze_s.set_of(name) for name in ("Z1", "P01", "Q13", "S001", "S128")
            ] == [0, 0, 2, 101, 132]
    assert analyze_s.cell_of("S001") in analyze_s.CELLS and len(analyze_s.CELLS) == 4
    assert {analyze_s.cell_of(f"S{n:03d}") for n in range(1, 5)} == set(analyze_s.CELLS)
    inputs = analyze_s.recipe_inputs(pd.Index(["S001_seed0", "Z8_seed0"]))
    assert (inputs.to_numpy() > 0).all() and inputs.shape == (2, 5)


# ---- freeze and test on fabricated runs ----

def test_freeze_and_test_of_amendment_a15_on_fabricated_runs(tmp_path, monkeypatch):
    from test_stage_p import meta_of
    from test_stage_z import part_1_world

    from dynamics import analyze_s, analyze_z, config, rules
    from dynamics.config import results_folder

    write, _, columns, earlier = part_1_world(tmp_path, monkeypatch)
    stages = {"z": [RunSpec("Z1", 0), RunSpec("Z2", 0)],
              "p": [RunSpec(f"P{n:02d}", 0) for n in range(1, 5)],
              "q": [RunSpec(f"Q{n:02d}", 0) for n in (1, 2, 13, 14)],
              "s": [RunSpec(f"S{n:03d}", 0) for n in range(1, 9)]}
    for stage, specs in stages.items():
        monkeypatch.setitem(STAGES, stage, specs)
    sets = {1: tuple(f"n1{i}" for i in range(10)), 2: tuple(f"n2{i}" for i in range(10))}
    monkeypatch.setattr(config, "a15_class_sets", lambda: sets)
    path, report = tmp_path / "s.json", tmp_path / "s.md"

    def add(spec: RunSpec, **changes) -> None:
        write(spec, results_folder(suite="step0"), columns, True, kept=True)
        (tmp_path / "runs" / spec.name / "meta.json").write_text(
            json.dumps(meta_of(spec.name, **changes)))

    def run(step: str, *extra: str) -> str:
        monkeypatch.setattr("sys.argv", [
            "analyze_s", step, "--results-root", str(tmp_path), "--rules", str(path),
            "--out", str(report), *extra])
        analyze_s.main()
        return report.read_text()

    for stage in ("z", "p", "q"):
        for spec in stages[stage]:
            add(spec)
    text = run("freeze")
    assert rules.file_hash(path) in text
    frozen = analyze_s.load(path)
    final = len(earlier) + 2 + 4 + 4    # every fabricated run has 200 epochs
    assert frozen["amendment"] == "A15" and len(frozen["training_runs"]) == final
    assert frozen["runs_by_id_set"] == {"0": final - 4, "1": 2, "2": 2}
    assert list(frozen["groups"]) == list(analyze_z.GROUPS)
    far = frozen["groups"]["far"]
    assert set(far[analyze_s.BY_CELL]) == set(analyze_s.CELLS)
    assert set(far["rules"]) == {
        analyze_s.RULE, analyze_s.RULE_CELLS, analyze_s.BOTH,
        *(f"{name}/{flag}" for name in (analyze_s.TWO, analyze_s.TWO_CELLS)
          for flag in (analyze_s.WITH, analyze_s.WITHOUT))}
    assert all(column.startswith("score/") for column in far["rules"][
        analyze_s.RULE].columns)
    assert any(column.startswith("recipe/") for column in far["rules"][
        analyze_s.BOTH].columns)
    with pytest.raises(SystemExit, match="frozen once"):
        run("freeze")
    # The check of the fit: each class set of amendment A14 left out.
    text = run("read")
    assert "one class set of amendment A14 left out" in text and "by_cell" in text
    # The test step.
    with pytest.raises(SystemExit, match="not the frozen one"):
        run("test")
    with pytest.raises(SystemExit, match="no results of stage s"):
        run("test", "--allow-other-rules")
    for spec in stages["s"][:7]:
        add(spec)
    with pytest.raises(SystemExit, match="not complete.*S008_seed0"):
        run("test", "--allow-other-rules")
    add(stages["s"][7])
    # A file of class sets with another hash: gate D stops the test.
    recorded = analyze_s.CLASS_SETS_SHA256
    monkeypatch.setattr(analyze_s, "CLASS_SETS_SHA256", "another hash")
    with pytest.raises(SystemExit, match="gate D fails"):
        run("test", "--allow-other-rules")
    assert "a15_class_sets.json does not have the recorded hash" in report.read_text()
    assert "**Outcome:" not in report.read_text()
    monkeypatch.setattr(analyze_s, "CLASS_SETS_SHA256", recorded)
    text = run("test", "--allow-other-rules")
    assert "**Gate D passes.**" in text and "**Gate S passes.**" in text
    assert "8 of 8 runs, in 2 class sets" in text and "**Gate M passes**" in text
    assert "## Target far: the outcome" in text and "**Outcome:" in text
    assert "Question 1, the statistics add to the recipe" in text
    for title in ("## Stability between class sets", "## Every choice, with the class",
                  "## The recipe as the unit", "## By cell of kind",
                  "## Shortlists and AUGRC", "## Runs", "## Target near (secondary",
                  "## Target textures (secondary"):
        assert title in text
    assert analyze_s.OLD in text and "a12_rule" in text
    # The AUGRC regret has a value for the rule and for the choice by cell.
    line = next(row for row in text.splitlines() if "Regret in AUGRC points" in row)
    assert "nan" not in line and "choice by cell" in line
    # A run with the classes of another set: gate D stops the test.
    add(stages["s"][0], classes=list(sets[2]))
    with pytest.raises(SystemExit, match="gate D fails"):
        run("test", "--allow-other-rules")
    add(stages["s"][0])
    # Every recipe below the accuracy of gate T: the test is not decided.
    monkeypatch.setattr(analyze_z, "MIN_ACCURACY", 0.99)
    with pytest.raises(SystemExit, match="gate T"):
        run("test", "--allow-other-rules")
    assert "**The test is not decided.**" in report.read_text()
    assert "**Outcome:" not in report.read_text()
