"""Fixed values of the Imagenette pilot.

Protocol: ../documentation/imagenette_dynamics_protocol_2026-11-10.md
"""
from __future__ import annotations

import re
from dataclasses import dataclass
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
    `lr_floor`. `amp` turns mixed precision on a GPU on or off.
    """

    augment: bool
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    lr: float = 0.05
    epochs: int = EPOCHS
    lr_floor: float = 0.0
    decay_epochs: int | None = None
    amp: bool = True


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
    "D1": _long(augment=False),
    "D2": _long(augment=True),
    "D3": _long(augment=False, label_smoothing=0.1),
    "D4": _long(augment=False, weight_decay=5e-5),
    "D5": _long(augment=False, amp=False),
}


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
    match = re.fullmatch(r"([A-Z]\d?)_seed(\d+)(?:_hold(\d+))?", name)
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
    "d_first": [RunSpec("D1", 0), RunSpec("D2", 0)],
    "d": [RunSpec(recipe, seed) for recipe in ("D1", "D2", "D3", "D4", "D5")
          for seed in range(5)],
}


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
