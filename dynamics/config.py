"""Fixed values of the Imagenette pilot.

Protocol: ../documentation/imagenette_dynamics_protocol_2026-11-10.md
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

CHECKPOINT_EPOCHS = (0, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80,
                     100, 120, 140, 160, 180, 200)
EPOCHS = 200
IMAGE_SIZE = 160
BATCH_SIZE = 128
MOMENTUM = 0.9
WARMUP_EPOCHS = 5
RRC_SCALE = (0.35, 1.0)

#: Neighbor rank of the kNN score, the region-shape descriptor, and placement.
NEIGHBORS = 10
#: Patch shuffle cuts an image into GRID x GRID patches.
GRID = 4
#: Seeds of the stand-in sets. They are the same for every run.
BLEND_SEED = 0
SHUFFLE_SEED = 0

#: Precision of the fitted CSFs, and the results folder of each precision. The
#: folder of a second extraction for gate G3 adds `_recheck`. Protocol amendment A3
#: adopts float64: float32 fits failed gate G3 on the Stage E runs.
PRECISIONS = {"float32": "results", "float64": "results_f64"}
DEFAULT_PRECISION = "float64"

LAYERS = ("stage1", "stage2", "stage3", "stage4")
#: Networks of torchvision that the pipeline can train, all from a random start.
ARCHS = ("resnet18", "resnet50", "densenet121")
PENULTIMATE = "stage4"
#: Largest number of images that an extra OOD set contributes (evenly thinned).
EXTRA_LIMIT = 5000
#: Set names that the pipeline uses itself. An extra set cannot take one of them.
RESERVED_SETS = ("train", "val", "blend", "shuffle", "imagewoof", "textures",
                 "heldout", "id_errors")


@dataclass(frozen=True)
class Recipe:
    """Training recipe.

    The learning rate warms up to `lr`, follows a cosine from `lr` to `lr_floor`
    until epoch `decay_epochs` (the last epoch if None), and then stays at
    `lr_floor`. `amp` turns mixed precision on a GPU on or off. `arch` is the
    network: one of `ARCHS`.
    """

    augment: bool
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    lr: float = 0.05
    epochs: int = EPOCHS
    lr_floor: float = 0.0
    decay_epochs: int | None = None
    amp: bool = True
    arch: str = "resnet18"


def _long(augment: bool, **changes: Any) -> Recipe:
    """A Stage D recipe: 500 epochs, cosine to 0.01 at epoch 100, constant tail."""
    return Recipe(augment=augment, epochs=500, lr_floor=0.01, decay_epochs=100,
                  **changes)


RECIPES = {
    "A": Recipe(augment=False),
    "B": Recipe(augment=True),
    "C1": Recipe(augment=True, weight_decay=5e-5),
    "C2": Recipe(augment=True, label_smoothing=0.1),
    "C3": Recipe(augment=False, lr=0.01),
    # Amendment A6: the changes of C1 and C2, crossed with the other kind of recipe.
    "C4": Recipe(augment=False, label_smoothing=0.1),
    "C5": Recipe(augment=True, label_smoothing=0.1, weight_decay=5e-5),
    "C6": Recipe(augment=False, weight_decay=5e-5),
    # Amendment A7: test recipes. T1 and T2 combine earlier changes, and T3 and T4
    # use a level that no earlier recipe has.
    "T1": Recipe(augment=True, lr=0.01),
    "T2": Recipe(augment=False, label_smoothing=0.1, weight_decay=5e-5),
    "T3": Recipe(augment=True, weight_decay=5e-3),
    "T4": Recipe(augment=False, label_smoothing=0.2),
    # Amendment A8: test recipes for the all-tiers rule. U1 and U2 combine earlier
    # changes, and U3 and U4 use the levels of T3 and T4 in the other kind.
    "U1": Recipe(augment=False, lr=0.01, label_smoothing=0.1),
    "U2": Recipe(augment=True, lr=0.01, weight_decay=5e-5),
    "U3": Recipe(augment=False, weight_decay=5e-3),
    "U4": Recipe(augment=True, label_smoothing=0.2),
    # Amendment A9, part 1: test recipes for the correct-only target. V1 to V3
    # combine earlier changes, and V4 to V6 use a level that no earlier recipe has.
    "V1": Recipe(augment=False, weight_decay=5e-3, label_smoothing=0.1),
    "V2": Recipe(augment=True, lr=0.01, label_smoothing=0.1),
    "V3": Recipe(augment=False, lr=0.01, weight_decay=5e-5),
    "V4": Recipe(augment=False, lr=0.1),
    "V5": Recipe(augment=True, label_smoothing=0.3),
    "V6": Recipe(augment=True, lr=0.1),
    # Amendment A10: test recipes without label smoothing. They are the six cells
    # of the grid of kind, weight decay, and learning rate that had no recipe.
    "X1": Recipe(augment=False, weight_decay=5e-3, lr=0.01),
    "X2": Recipe(augment=True, weight_decay=5e-3, lr=0.01),
    "X3": Recipe(augment=False, weight_decay=5e-5, lr=0.1),
    "X4": Recipe(augment=True, weight_decay=5e-5, lr=0.1),
    "X5": Recipe(augment=False, weight_decay=5e-3, lr=0.1),
    "X6": Recipe(augment=True, weight_decay=5e-3, lr=0.1),
    # Amendment A9, part 2 (100 classes): V1, V2, V4, and V5 with 100 epochs.
    "W1": Recipe(augment=False, weight_decay=5e-3, label_smoothing=0.1, epochs=100),
    "W2": Recipe(augment=True, lr=0.01, label_smoothing=0.1, epochs=100),
    "W3": Recipe(augment=False, lr=0.1, epochs=100),
    "W4": Recipe(augment=True, label_smoothing=0.3, epochs=100),
    # Amendment A11 (100 classes): a weight decay of 5e-3 without label smoothing,
    # for the check of the fixed NCI score.
    "Y1": Recipe(augment=False, weight_decay=5e-3, epochs=100),
    "Y2": Recipe(augment=True, weight_decay=5e-3, epochs=100),
    "Y3": Recipe(augment=True, weight_decay=5e-3, lr=0.01, epochs=100),
    # Amendment A12, part 1 (10 classes): test recipes for the rule from the score
    # statistics. Z1 to Z4 have label smoothing. Z5 to Z8 have none, with a level
    # that no recipe has.
    "Z1": Recipe(augment=False, label_smoothing=0.1, lr=0.1),
    "Z2": Recipe(augment=True, label_smoothing=0.1, lr=0.1),
    "Z3": Recipe(augment=False, label_smoothing=0.1, weight_decay=5e-5, lr=0.01),
    "Z4": Recipe(augment=True, label_smoothing=0.1, weight_decay=5e-3),
    "Z5": Recipe(augment=False, weight_decay=2e-3),
    "Z6": Recipe(augment=True, weight_decay=2e-3),
    "Z7": Recipe(augment=False, lr=0.02),
    "Z8": Recipe(augment=True, weight_decay=0.0),
    # Amendment A12, part 2 (100 classes, 100 epochs).
    "K1": Recipe(augment=False, label_smoothing=0.1, weight_decay=5e-5, epochs=100),
    "K2": Recipe(augment=True, label_smoothing=0.1, weight_decay=5e-3, epochs=100),
    "K3": Recipe(augment=True, lr=0.01, epochs=100),
    "K4": Recipe(augment=False, weight_decay=5e-3, lr=0.01, epochs=100),
    # Stage H: the eight recipes above with 100 epochs, for the 100-class subset.
    "H0": Recipe(augment=False, epochs=100),
    "H1": Recipe(augment=True, epochs=100),
    "H2": Recipe(augment=True, weight_decay=5e-5, epochs=100),
    "H3": Recipe(augment=True, label_smoothing=0.1, epochs=100),
    "H4": Recipe(augment=False, lr=0.01, epochs=100),
    "H5": Recipe(augment=False, label_smoothing=0.1, epochs=100),
    "H6": Recipe(augment=True, label_smoothing=0.1, weight_decay=5e-5, epochs=100),
    "H7": Recipe(augment=False, weight_decay=5e-5, epochs=100),
    "D1": _long(augment=False),
    "D2": _long(augment=True),
    "D3": _long(augment=False, label_smoothing=0.1),
    "D4": _long(augment=False, weight_decay=5e-5),
    "D5": _long(augment=False, amp=False),
}


#: Amendment A13: the recipes of stage `z` with another architecture.
#: R1 to R8 are ResNet-50, and N1 to N8 are DenseNet-121.
A13_ARCHS = {"R": "resnet50", "N": "densenet121"}
for _letter, _arch in A13_ARCHS.items():
    for _number in range(1, 9):
        RECIPES[f"{_letter}{_number}"] = replace(RECIPES[f"Z{_number}"], arch=_arch)
#: Amendment A13: the checkpoints that are evaluated.
A13_EVAL_EPOCHS = (20, 40, 60, 80, 100, 120, 160, 200)

#: Amendment A14: recipes drawn at random, once, by `dynamics/draw_recipes.py`.
#: P01 to P48 are part 1 (Imagenette). Q01 to Q48 are part 2 (new ID classes).
FROZEN_DIR = Path(__file__).resolve().parent / "frozen"
A14_RECIPES_PATH = FROZEN_DIR / "a14_recipes.json"
A14_CLASS_SETS_PATH = FROZEN_DIR / "a14_class_sets.json"
for _name, _settings in json.loads(A14_RECIPES_PATH.read_text())["recipes"].items():
    RECIPES[_name] = Recipe(**_settings)
#: Amendment A14: recipes of one class set in part 2, and the saved checkpoint.
A14_PER_SET = 12
A14_EVAL_EPOCHS = (EPOCHS,)
#: Image counts of Imagenette. Each class set of amendment A14 takes them.
A14_TRAIN_IMAGES = 9469
A14_VAL_IMAGES = 3925
#: The classes of Imagenette, in the order of their folders.
IMAGENETTE_CLASSES = ("n01440764", "n02102040", "n02979186", "n03000684",
                      "n03028079", "n03394916", "n03417042", "n03425413",
                      "n03445777", "n03888257")


def class_set_of(recipe: str) -> int | None:
    """Number of the class set (1 to 4) of a recipe of amendment A14 part 2."""
    if not re.fullmatch(r"Q\d\d", recipe):
        return None
    return (int(recipe[1:]) - 1) // A14_PER_SET + 1


@cache
def a14_class_sets() -> dict[int, tuple[str, ...]]:
    """The frozen class sets of amendment A14 part 2, by number."""
    stored = json.loads(A14_CLASS_SETS_PATH.read_text())["sets"]
    return {int(number): tuple(classes) for number, classes in stored.items()}


#: Amendment A15: 128 recipes, S001 to S128, drawn once: 4 for each of 32 new class
#: sets. Their class sets are frozen after the image counts come from the HPC.
A15_RECIPES_PATH = FROZEN_DIR / "a15_recipes.json"
A15_CLASS_SETS_PATH = FROZEN_DIR / "a15_class_sets.json"
for _name, _settings in json.loads(A15_RECIPES_PATH.read_text())["recipes"].items():
    RECIPES[_name] = Recipe(**_settings)
A15_PER_SET = 4
A15_SETS = 32


def a15_set_of(recipe: str) -> int | None:
    """Number of the class set (1 to 32) of a recipe of amendment A15."""
    if not re.fullmatch(r"S\d\d\d", recipe):
        return None
    return (int(recipe[1:]) - 1) // A15_PER_SET + 1


@cache
def a15_class_sets() -> dict[int, tuple[str, ...]]:
    """The frozen class sets of amendment A15, by number.

    Raises:
        FileNotFoundError: If the class sets are not drawn yet.
    """
    if not A15_CLASS_SETS_PATH.exists():
        raise FileNotFoundError(
            f"{A15_CLASS_SETS_PATH.name} is not there: draw the class sets of "
            "amendment A15 first (dynamics/build_sets_parquet.py --draw)")
    stored = json.loads(A15_CLASS_SETS_PATH.read_text())["sets"]
    return {int(number): tuple(classes) for number, classes in stored.items()}


def expected_classes(recipe: str) -> tuple[str, ...] | None:
    """The ID classes that a recipe of amendment A14 or A15 must train on.

    Returns None for a recipe of another amendment: those have no such check.
    """
    if re.fullmatch(r"P\d\d", recipe):
        return IMAGENETTE_CLASSES
    number = class_set_of(recipe)
    if number is not None:
        return a14_class_sets()[number]
    number = a15_set_of(recipe)
    return None if number is None else a15_class_sets()[number]


def set_folder(recipe: str) -> str | None:
    """Name of the ID folder of a recipe below the folder of its class sets.

    Amendment A14 has `set1` to `set4`, and amendment A15 has `set01` to `set32`.
    Returns None for a recipe with no class set.
    """
    number = class_set_of(recipe)
    if number is not None:
        return f"set{number}"
    number = a15_set_of(recipe)
    return None if number is None else f"set{number:02d}"


@dataclass(frozen=True)
class RunSpec:
    """One training run: a recipe, a seed, and an optional held-out class."""

    recipe: str
    seed: int
    held_out: int | None = None

    @property
    def name(self) -> str:
        hold = "" if self.held_out is None else f"_hold{self.held_out}"
        return f"{self.recipe}_seed{self.seed}{hold}"


def parse_run(name: str) -> RunSpec:
    """Inverse of `RunSpec.name`.

    Raises:
        ValueError: If the name does not describe a run.
    """
    match = re.fullmatch(r"([A-Z]\d{0,3})_seed(\d+)(?:_hold(\d+))?", name)
    if match is None or match.group(1) not in RECIPES:
        raise ValueError(f"not a run name: {name}")
    hold = match.group(3)
    return RunSpec(match.group(1), int(match.group(2)),
                   None if hold is None else int(hold))


STAGES: dict[str, list[RunSpec]] = {
    "e_main": [RunSpec(recipe, seed) for recipe in ("A", "B") for seed in range(5)],
    "e_heldout": [RunSpec(recipe, 0, held) for recipe in ("A", "B")
                  for held in range(10)],
    "c": ([RunSpec(recipe, seed) for recipe in ("A", "B") for seed in (5, 6, 7)]
          + [RunSpec(recipe, seed) for recipe in ("C1", "C2", "C3")
             for seed in range(3)]),
    "c_first": [RunSpec("C2", 0)],
    "c2": [RunSpec(recipe, seed) for recipe in ("C4", "C5", "C6")
           for seed in range(3)],
    "t": [RunSpec(recipe, seed) for recipe in ("T1", "T2", "T3", "T4")
          for seed in range(2)],
    "u": [RunSpec(recipe, seed) for recipe in ("U1", "U2", "U3", "U4")
          for seed in range(2)],
    "v": [RunSpec(recipe, seed) for recipe in ("V1", "V2", "V3", "V4", "V5", "V6")
          for seed in range(2)],
    "w": [RunSpec(recipe, seed) for recipe in ("W1", "W2", "W3", "W4")
          for seed in range(2)],
    "x": [RunSpec(recipe, seed) for recipe in ("X1", "X2", "X3", "X4", "X5", "X6")
          for seed in range(2)],
    "y": [RunSpec(recipe, seed) for recipe in ("Y1", "Y2", "Y3")
          for seed in range(2)],
    "z": [RunSpec(recipe, seed) for recipe in ("Z1", "Z2", "Z3", "Z4", "Z5", "Z6",
                                                "Z7", "Z8") for seed in range(2)],
    "k": [RunSpec(recipe, seed) for recipe in ("K1", "K2", "K3", "K4")
          for seed in range(2)],
    "r": [RunSpec(f"R{number}", seed) for number in range(1, 9)
          for seed in range(2)],
    "n": [RunSpec(f"N{number}", seed) for number in range(1, 9)
          for seed in range(2)],
    "r_first": [RunSpec("R1", 0)],
    "n_first": [RunSpec("N1", 0)],
    "h_first": [RunSpec("H0", 0)],
    "h1": [RunSpec(recipe, seed) for recipe in ("H0", "H1", "H2", "H3", "H4")
           for seed in range(2)],
    "h2": [RunSpec(recipe, seed) for recipe in ("H5", "H6", "H7")
           for seed in range(2)],
    "d_first": [RunSpec("D1", 0), RunSpec("D2", 0)],
    "d": [RunSpec(recipe, seed) for recipe in ("D1", "D2", "D3", "D4", "D5")
          for seed in range(5)],
}


#: Amendment A14: one run for each recipe. Part 1 is stage `p`. Part 2 has one
#: stage for each class set, because each has its own ID folder. Stage `q` is all
#: of part 2, for the analysis: `run_stage` refuses it, since no folder fits.
STAGES["p"] = [RunSpec(f"P{number:02d}", 0) for number in range(1, 49)]
for _set in range(1, 5):
    STAGES[f"q{_set}"] = [RunSpec(f"Q{number:02d}", 0) for number in range(
        (_set - 1) * A14_PER_SET + 1, _set * A14_PER_SET + 1)]
STAGES["q"] = [spec for _set in range(1, 5) for spec in STAGES[f"q{_set}"]]
STAGES["q_first"] = [RunSpec("Q01", 0)]
#: Amendment A15: one run for each recipe, 4 for each of the 32 new class sets.
#: `run_stage --id-root` gives each run the folder of its class set.
STAGES["s"] = [RunSpec(f"S{number:03d}", 0) for number in range(1, 129)]
STAGES["s_first"] = [RunSpec("S001", 0)]

#: Amendment A7: the earlier 10-class runs that the frozen rules are fitted on.
STAGES["a7_train"] = [*STAGES["e_main"], *STAGES["c"], *STAGES["c2"],
                      *STAGES["d_first"]]
#: Amendment A8: the runs of the second fit of the all-tiers rule.
STAGES["a8_train"] = [*STAGES["a7_train"], *STAGES["t"]]
#: Amendment A9: the runs that the rules of each part are fitted on.
STAGES["a9_train"] = [*STAGES["a8_train"], *STAGES["u"]]
STAGES["a9_train_h"] = [*STAGES["h1"], *STAGES["h2"]]
#: Amendment A10: all earlier 10-class runs, and those without label smoothing,
#: which the rule under test is fitted on.
STAGES["a10_all"] = [*STAGES["a9_train"], *STAGES["v"]]
STAGES["a10_train"] = [spec for spec in STAGES["a10_all"]
                       if RECIPES[spec.recipe].label_smoothing == 0]
#: Amendment A12: all earlier runs of each setting, which the rules are fitted on.
STAGES["a12_train"] = [*STAGES["a10_all"], *STAGES["x"]]
STAGES["a12_train_h"] = [*STAGES["a9_train_h"], *STAGES["w"], *STAGES["y"]]
#: Amendment A11: the earlier 100-class runs without label smoothing, which give
#: the fixed score under test.
STAGES["a11_train"] = [spec for spec in [*STAGES["a9_train_h"], *STAGES["w"]]
                       if RECIPES[spec.recipe].label_smoothing == 0]


@dataclass(frozen=True)
class DataPaths:
    """Dataset folders. Imagenette and Imagewoof hold `train/` and `val/`.

    `extra` holds more OOD sets as (name, folder) pairs. Each contributes at most
    `extra_limit` images.
    """

    imagenette: Path
    imagewoof: Path | None = None
    textures: Path | None = None
    extra: tuple[tuple[str, Path], ...] = ()
    extra_limit: int | None = EXTRA_LIMIT


def results_folder(precision: str = DEFAULT_PRECISION, recheck: bool = False,
                   suite: str = "") -> str:
    """Name of the results folder of a run.

    Args:
        precision: Precision of the fitted CSFs.
        recheck: The folder of the second extraction, for gate G3.
        suite: Name of an evaluation with more sets than the base one, for example
            `step0`. It keeps those results apart from the base results.

    Raises:
        ValueError: If the precision is unknown.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision: {precision}")
    return (PRECISIONS[precision] + (f"_{suite}" if suite else "")
            + ("_recheck" if recheck else ""))


def run_dir(out_root: Path, name: str) -> Path:
    return Path(out_root) / "runs" / name


def checkpoint_path(out_root: Path, name: str, epoch: int) -> Path:
    return run_dir(out_root, name) / "checkpoints" / f"epoch_{epoch:03d}.pt"


def checkpoint_epochs(epochs: int) -> tuple[int, ...]:
    """Checkpoints of a run with `epochs` epochs.

    The protocol grid up to its last epoch, then every 20 epochs, and the final
    epoch. A run of 500 epochs has 38 checkpoints.
    """
    kept = {epoch for epoch in CHECKPOINT_EPOCHS if epoch <= epochs}
    kept |= set(range(CHECKPOINT_EPOCHS[-1] + 20, epochs + 1, 20))
    return tuple(sorted(kept | {epochs}))
