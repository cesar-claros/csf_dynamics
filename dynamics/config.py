"""Fixed values of the Imagenette pilot.

Protocol: ../documentation/imagenette_dynamics_protocol_2026-11-10.md
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class Recipe:
    """Training recipe. The regimes and the Stage C recipes differ only here."""

    augment: bool
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    lr: float = 0.05


RECIPES = {
    "A": Recipe(augment=False),
    "B": Recipe(augment=True),
    "C1": Recipe(augment=True, weight_decay=5e-5),
    "C2": Recipe(augment=True, label_smoothing=0.1),
    "C3": Recipe(augment=False, lr=0.01),
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
}


@dataclass(frozen=True)
class DataPaths:
    """Dataset folders. Imagenette and Imagewoof hold `train/` and `val/`."""

    imagenette: Path
    imagewoof: Path | None = None
    textures: Path | None = None


def results_folder(precision: str = DEFAULT_PRECISION, recheck: bool = False) -> str:
    """Name of the results folder of a run for one precision.

    Raises:
        ValueError: If the precision is unknown.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision: {precision}")
    return PRECISIONS[precision] + ("_recheck" if recheck else "")


def run_dir(out_root: Path, name: str) -> Path:
    return Path(out_root) / "runs" / name


def checkpoint_path(out_root: Path, name: str, epoch: int) -> Path:
    return run_dir(out_root, name) / "checkpoints" / f"epoch_{epoch:03d}.pt"


def checkpoint_epochs(epochs: int) -> tuple[int, ...]:
    """Protocol checkpoints up to `epochs`, plus the final epoch."""
    kept = [epoch for epoch in CHECKPOINT_EPOCHS if epoch <= epochs]
    return tuple(sorted(set(kept) | {epochs}))
