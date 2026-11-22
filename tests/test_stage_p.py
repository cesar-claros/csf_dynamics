import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from test_stage_z import part_1_world

from dynamics import analyze_p, analyze_z, draw_recipes, run_stage, train
from dynamics import build_class_sets as cs
from dynamics.analyze_selection import FAMILY
from dynamics.config import (
    A14_EVAL_EPOCHS,
    A14_TRAIN_IMAGES,
    BATCH_SIZE,
    IMAGE_SIZE,
    IMAGENETTE_CLASSES,
    RECIPES,
    STAGES,
    RunSpec,
    a14_class_sets,
    class_set_of,
    expected_classes,
    parse_run,
    results_folder,
)

HERE = Path(__file__).resolve().parent


def a14(letter: str) -> list[str]:
    return [f"{letter}{number:02d}" for number in range(1, 49)]


def test_recipes_of_amendment_a14_are_the_frozen_draw():
    stored = json.loads(draw_recipes.PATH.read_text())
    assert stored == draw_recipes.table()            # the draw gives the frozen file
    assert analyze_p.frozen_inputs() == []           # with the recorded hashes
    assert list(stored["recipes"]) == a14("P") + a14("Q")
    for letter in "PQ":
        recipes = [RECIPES[name] for name in a14(letter)]
        cells = [(recipe.augment, recipe.label_smoothing > 0) for recipe in recipes]
        assert Counter(cells) == dict.fromkeys(draw_recipes.CELLS, 12)
        # Every block of 12 recipes (a class set in part 2) has 3 of each cell.
        for start in range(0, 48, 12):
            assert Counter(cells[start:start + 12]) == dict.fromkeys(
                draw_recipes.CELLS, 3)
        for recipe in recipes:
            assert 1e-5 <= recipe.weight_decay <= 5e-3 and 0.01 <= recipe.lr <= 0.1
            assert recipe.label_smoothing == 0 or 0.05 <= recipe.label_smoothing <= 0.2
            assert (recipe.epochs, recipe.arch, recipe.lr_floor) == (200, "resnet18", 0)
    new = a14("P") + a14("Q")
    assert len({RECIPES[name] for name in new}) == 96
    # No earlier recipe has the settings of a new one, and each has a kind.
    earlier = {recipe for name, recipe in RECIPES.items() if name not in new}
    assert not earlier & {RECIPES[name] for name in new}
    assert all(FAMILY[name] == ("augmented" if RECIPES[name].augment else "plain")
               for name in new)


def test_stages_and_run_names_of_amendment_a14():
    assert [spec.name for spec in STAGES["p"]] == [f"{n}_seed0" for n in a14("P")]
    assert [len(STAGES[f"q{number}"]) for number in range(1, 5)] == [12] * 4
    assert [spec.name for spec in STAGES["q"]] == [f"{n}_seed0" for n in a14("Q")]
    assert STAGES["q"] == [spec for number in range(1, 5)
                           for spec in STAGES[f"q{number}"]]
    assert [spec.name for spec in STAGES["q_first"]] == ["Q01_seed0"]
    assert A14_EVAL_EPOCHS == (RECIPES["P01"].epochs,)
    # Run names with two digits, and the earlier names as before.
    assert parse_run("P07_seed0") == RunSpec("P07", 0)
    assert parse_run("Q48_seed0").recipe == "Q48"
    assert parse_run("A_seed3_hold2") == RunSpec("A", 3, 2)
    assert parse_run("C1_seed2") == RunSpec("C1", 2)
    with pytest.raises(ValueError):
        parse_run("P49_seed0")
    assert [class_set_of(name) for name in ("Q01", "Q12", "Q13", "Q36", "Q37",
                                            "Q48")] == [1, 1, 2, 3, 4, 4]
    assert class_set_of("P01") is None and class_set_of("Z1") is None
    assert expected_classes("P30") == IMAGENETTE_CLASSES
    assert expected_classes("Q13") == a14_class_sets()[2]
    assert expected_classes("Z1") is None and expected_classes("R1") is None
    assert all(class_set_of(spec.recipe) == number for number in range(1, 5)
               for spec in STAGES[f"q{number}"])


def test_frozen_class_sets():
    frozen = cs.load()
    sets = frozen["sets"]
    every = [name for classes in sets.values() for name in classes]
    assert list(sets) == ["1", "2", "3", "4"]
    assert all(len(classes) == 10 and classes == sorted(classes)
               for classes in sets.values())
    assert len(set(every)) == 40                       # no class in two sets
    assert cs.list_hash(every) == frozen["class_list_sha256"]
    forbidden = frozen["excluded"]["imagenette"] + frozen["excluded"]["imagewoof"]
    assert frozen["excluded"]["imagenette"] == list(IMAGENETTE_CLASSES)
    assert len(frozen["excluded"]["imagewoof"]) == 10 and not set(every) & set(forbidden)
    hundred = (HERE.parent / "dynamics" / "frozen" / "imagenet100_classes.txt"
               ).read_text().split()
    assert set(every) <= set(hundred)                  # the folder of Stage H
    assert (frozen["train_images"], frozen["val_images"]) == (A14_TRAIN_IMAGES, 3925)
    assert a14_class_sets() == {int(k): tuple(v) for k, v in sets.items()}


def test_draw_of_class_sets():
    train = {f"n{i:03d}": 1300 for i in range(50)} | {"n900": 1299, "n901": 1300}
    val = {name: 50 for name in train} | {"n901": 49}
    pool = cs.candidates(train, val, ["n000", "n001"])
    assert len(pool) == 48 and not {"n000", "n001", "n900", "n901"} & set(pool)
    sets = cs.draw_sets(pool)
    assert sets == cs.draw_sets(list(reversed(pool)))  # the order does not matter
    assert sets != cs.draw_sets(pool, seed=1)
    every = [name for classes in sets.values() for name in classes]
    assert len(set(every)) == 40 and set(every) <= set(pool)
    with pytest.raises(ValueError, match="need 40"):
        cs.draw_sets(pool[:39])


def test_class_sizes_and_split():
    assert cs.class_sizes(9469) == [947] * 9 + [946]
    assert cs.class_sizes(3925) == [393] * 5 + [392] * 5
    names = [f"img_{i:04d}.JPEG" for i in range(1350)]
    first, second = cs.split_class(names, 947, 393, (14, 1, 0))
    assert len(first) == 947 and len(second) == 393 and not set(first) & set(second)
    assert (first, second) == cs.split_class(list(reversed(names)), 947, 393,
                                             (14, 1, 0))
    assert first != cs.split_class(names, 947, 393, (14, 1, 1))[0]
    with pytest.raises(ValueError, match="need 1340"):
        cs.split_class(names[:1300], 947, 393, (14, 1, 0))


def test_build_and_gate_d_of_a_class_set(tmp_path):
    source, classes = tmp_path / "source", ["n01", "n02", "n03"]
    for name in [*classes, "n09"]:
        for split, count in (("train", 12), ("val", 3)):
            folder = source / split / name
            folder.mkdir(parents=True)
            for item in range(count):
                (folder / f"{split}_{name}_{item:02d}.JPEG").write_bytes(
                    f"{split}{name}{item}".encode())
    totals = (20, 10)
    first = cs.build_set(source, tmp_path / "a", 1, classes, totals=totals)
    assert first["counts"] == {"train": {"n01": 7, "n02": 7, "n03": 6},
                               "val": {"n01": 4, "n02": 3, "n03": 3}}
    assert cs.check_set(tmp_path / "a", classes, ["n09"], totals) == []
    assert json.loads((tmp_path / "a" / "split_manifest.json").read_text()) == first
    # A second build and a build with links give the same split.
    assert cs.build_set(source, tmp_path / "a", 1, classes, totals=totals) == first
    linked = cs.build_set(source, tmp_path / "b", 1, classes, totals=totals, link=True)
    assert linked == first and cs.check_set(tmp_path / "b", classes, [], totals) == []
    place = next((tmp_path / "b" / "train" / "n01").iterdir())
    assert place.is_symlink() and place.read_bytes().startswith(b"train")
    # Another set number gives another split.
    assert cs.build_set(source, tmp_path / "c", 2, classes, totals=totals)[
        "split_sha256"] != first["split_sha256"]
    # Gate D finds a wrong count, an image in both splits, and a forbidden class.
    assert "not 21" in cs.check_set(tmp_path / "a", classes, [], (21, 10))[0]
    assert cs.check_set(tmp_path / "a", classes, ["n02"], totals) == [
        "a class is an Imagenette or an Imagewoof class"]
    assert "not the frozen list" in cs.check_set(tmp_path / "a", ["n01", "n02"], [],
                                                 totals)[0]
    twice = next((tmp_path / "a" / "train" / "n01").iterdir())
    (tmp_path / "a" / "val" / "n01" / twice.name).write_bytes(b"x")
    problems = cs.check_set(tmp_path / "a", classes, [], totals)
    assert any("training and in the validation" in text for text in problems)
    with pytest.raises(ValueError, match="need 16"):
        cs.build_set(source, tmp_path / "d", 1, classes, totals=(30, 16))


def test_training_keeps_the_chosen_checkpoints(tiny_data, tmp_path):
    spec, device = RunSpec("B", 0), torch.device("cpu")
    shared = {"epochs": 1, "workers": 0, "image_size": 32}
    with pytest.raises(ValueError, match="do not hold the last epoch"):
        train.train_run(spec, tiny_data, tmp_path, device, keep_epochs=[0], **shared)
    train.train_run(spec, tiny_data, tmp_path, device, keep_epochs=[1], **shared)
    folder = tmp_path / "runs" / spec.name
    assert [path.name for path in (folder / "checkpoints").iterdir()] == [
        "epoch_001.pt"]
    assert (folder / "train.done").exists()


def test_run_stage_refuses_an_id_folder_with_other_classes(tiny_data, monkeypatch):
    specs = [RunSpec("P01", 0), RunSpec("Z1", 0), RunSpec("Q13", 0)]
    assert run_stage.wrong_folder(specs, tiny_data.imagenette) == ["P01_seed0",
                                                                    "Q13_seed0"]
    assert run_stage.wrong_folder([RunSpec("Z1", 0)], Path("no/such/folder")) == []
    monkeypatch.setattr("sys.argv", [
        "run_stage", "--stage", "q_first", "--out-root", "unused", "--id-dir",
        str(tiny_data.imagenette), "--imagewoof-dir", str(tiny_data.imagewoof),
        "--textures-dir", str(tiny_data.textures), "--device", "cpu"])
    with pytest.raises(SystemExit):
        run_stage.main()


def test_stage_runner_with_one_saved_checkpoint(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "run_stage", "--only", "P01_seed0", "--out-root", str(tmp_path), "--id-dir",
        str(tiny_data.imagenette), "--imagewoof-dir", str(tiny_data.imagewoof),
        "--textures-dir", str(tiny_data.textures), "--epochs", "1", "--image-size",
        "32", "--workers", "0", "--device", "cpu", "--save-epochs", "1",
        "--eval-epochs", "1"])
    run_stage.main()                    # smoke settings: no check of the classes
    folder = tmp_path / "runs" / "P01_seed0"
    assert [path.name for path in (folder / "checkpoints").iterdir()] == [
        "epoch_001.pt"]
    assert [path.name for path in (folder / "results_f64").iterdir()] == [
        "epoch_001.json"]
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["recipe"] == "P01" and meta["lr"] == RECIPES["P01"].lr
    assert meta["weight_decay"] == RECIPES["P01"].weight_decay
    tables = analyze_p.analyze_e.read_runs(tmp_path, "results_f64")
    assert set(tables.metrics["recipe"]) == {"P01"}
    # The same run is not a protocol run: gate D reports it.
    assert analyze_p.gate_d(tmp_path, ["P01_seed0"])["problem"].iloc[0] == (
        "epochs, smoke settings, classes, train_images")


def test_sign_flip_test():
    values = np.array([-1.7, -1.2, -1.6, -1.4, 0.5, 0.0, 0.0, 0.0])
    flips = np.array([np.mean(values * np.array(signs))
                      for signs in itertools.product((1, -1), repeat=len(values))])
    exact = float(np.mean(flips <= values.mean() + 1e-12))
    assert abs(analyze_p.sign_flip(values) - exact) < 0.005
    assert analyze_p.sign_flip(-np.ones(12)) < 0.001
    assert 0.4 < analyze_p.sign_flip(np.array([1.0, -1.0] * 10)) < 0.6
    assert analyze_p.sign_flip(np.ones(12)) > 0.99
    assert np.isnan(analyze_p.sign_flip(np.array([1.0])))
    assert analyze_p.sign_flip(values) == analyze_p.sign_flip(values)   # fixed seed
    supported = {"supported": True}
    assert "at the margin" in analyze_p.margin(supported, 0.07)
    assert "agrees" in analyze_p.margin(supported, 0.01)
    assert "margin" not in analyze_p.margin({"supported": False}, 0.3)


def meta_of(run: str, **changes) -> dict:
    recipe = parse_run(run).recipe
    return {**vars(RECIPES[recipe]), "run": run, "recipe": recipe, "limit": None,
            "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
            "classes": list(expected_classes(recipe) or ["n1", "n2"]),
            "train_images": A14_TRAIN_IMAGES, **changes}


def test_gate_d_reads_the_settings_and_the_classes_of_each_run(tmp_path):
    def write(run: str, **changes) -> None:
        folder = tmp_path / "runs" / run
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "meta.json").write_text(json.dumps(meta_of(run, **changes)))

    good = ["P01_seed0", "Q01_seed0", "Q40_seed0", "Z1_seed0"]
    for run in good:
        write(run)
    old = meta_of("Z2_seed0")
    del old["arch"]                     # a run from before amendment A13
    (tmp_path / "runs" / "Z2_seed0").mkdir()
    (tmp_path / "runs" / "Z2_seed0" / "meta.json").write_text(json.dumps(old))
    assert analyze_p.gate_d(tmp_path, [*good, "Z2_seed0"]).empty
    write("P02_seed0", lr=0.5)
    write("P03_seed0", classes=list(a14_class_sets()[1]))
    write("Q02_seed0", classes=list(a14_class_sets()[2]), train_images=9000)
    write("Q03_seed0", limit=64)
    write("Q04_seed0", arch="resnet50")
    found = analyze_p.gate_d(tmp_path, ["P01_seed0", "P02_seed0", "P03_seed0",
                                        "Q02_seed0", "Q03_seed0", "Q04_seed0"])
    assert dict(zip(found["run"], found["problem"])) == {
        "P02_seed0": "lr", "P03_seed0": "classes",
        "Q02_seed0": "classes, train_images", "Q03_seed0": "smoke settings",
        "Q04_seed0": "arch"}


def test_frozen_rule_on_fabricated_runs_with_random_recipes(tmp_path, monkeypatch):
    write, run_z, columns, _ = part_1_world(tmp_path, monkeypatch)
    monkeypatch.setitem(STAGES, "z", [RunSpec("Z1", 0)])
    path, report = tmp_path / "z.json", tmp_path / "p.md"
    run_z("freeze", 1, path)
    stages = {"p": [RunSpec(f"P{n:02d}", 0) for n in range(1, 5)],
              "q": [RunSpec(f"Q{n:02d}", 0) for n in (1, 2, 13, 14)]}
    for stage, specs in stages.items():
        monkeypatch.setitem(STAGES, stage, specs)

    def run(*extra: str) -> str:
        monkeypatch.setattr("sys.argv", [
            "analyze_p", "--results-root", str(tmp_path), "--rules", str(path),
            "--out", str(report), *extra])
        analyze_p.main()
        return report.read_text()

    def add(spec: RunSpec, **changes) -> None:
        write(spec, results_folder(suite="step0"), columns, True, kept=True)
        (tmp_path / "runs" / spec.name / "meta.json").write_text(
            json.dumps(meta_of(spec.name, **changes)))

    with pytest.raises(SystemExit, match="not the frozen one"):
        run("--part", "1")
    with pytest.raises(SystemExit, match="no results of stage p"):
        run("--part", "1", "--allow-other-rules")
    for spec in stages["p"][:3]:
        add(spec)
    with pytest.raises(SystemExit, match="not complete.*P04_seed0"):
        run("--part", "1", "--allow-other-rules")
    text = run("--part", "1", "--allow-other-rules", "--count-missing-as-failed")
    assert "3 of 4 runs have results" in text and "P04_seed0" in text
    add(stages["p"][3])
    text = run("--part", "1", "--allow-other-rules")
    assert text.startswith("# Amendment A14, part 1, Imagenette")
    assert "**Gate D passes.**" in text and "**Gate S passes.**" in text
    assert "**Gate M passes**" in text and "**Outcome:" in text
    for title in ("## Planned checks on the far sets", "## By cell of kind",
                  "## The rule against the best fixed score in hindsight",
                  "## Where the rule loses", "## Inputs outside the range",
                  "## Recipes"):
        assert title in text
    assert "## By class set" not in text and "Sign-flip test" in text
    assert "The fixed score holds by the limits of question 2" in text
    # The planned readings use the regret of the outcome section.
    found = analyze_p.far(analyze_z.load(path), analyze_p.analyze_h.read_stage(
        tmp_path, ("p",)))
    rule = analyze_z.by_recipe(found.single)[analyze_z.RULE].mean()
    assert f"score_statistics {rule:.2f}" in text and len(found.single) == 4
    # Too many recipes below the accuracy of gate T: the part is not decided.
    monkeypatch.setattr(analyze_z, "MIN_ACCURACY", 0.99)
    with pytest.raises(SystemExit, match="gate T"):
        run("--part", "1", "--allow-other-rules")
    assert "**The part is not decided.**" in report.read_text()
    assert "**Outcome:" not in report.read_text()
    monkeypatch.setattr(analyze_z, "MIN_ACCURACY", 0.60)
    # Part 2, and a run on the classes of another set: gate D stops the part.
    for spec in stages["q"]:
        add(spec)
    text = run("--part", "2", "--allow-other-rules")
    assert "## By class set" in text and "set 1" in text and "set 2" in text
    both = run("--together", "--allow-other-rules")
    assert "8 recipes." in both and "## By part" in both and "Outcome" not in both
    add(stages["q"][2], classes=list(a14_class_sets()[1]))
    with pytest.raises(SystemExit, match="gate D fails"):
        run("--part", "2", "--allow-other-rules")
    assert "**Gate D fails.**" in report.read_text()
    assert "**Outcome:" not in report.read_text()
    with pytest.raises(SystemExit, match="part 2 is not decided"):
        run("--together", "--allow-other-rules")
