"""Amendment A14: draw the recipes of the two parts, once.

    python -m dynamics.draw_recipes

Each part has 48 recipes. Recipe number i belongs to cell (i - 1) mod 4 of kind
and label smoothing, so each cell has 12 recipes, and every block of 12 recipes
(a class set in part 2) has 3 of each cell. The weight decay and the peak learning
rate are log-uniform in their range, and a recipe with label smoothing takes its
value from a uniform range. Values are rounded to two significant digits.

The draw is written to `dynamics/frozen/a14_recipes.json`, which the configuration
reads. The file is frozen: this script does not write it a second time.

Protocol: ../documentation/random_recipes_new_classes_amendment_2026-11-22.md
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PATH = HERE / "frozen" / "a14_recipes.json"
SEED = 14
#: Letter of the recipes of each part.
PARTS = {1: "P", 2: "Q"}
PER_PART = 48
#: (augmented, with label smoothing), in the order of the recipe numbers.
CELLS = ((False, False), (False, True), (True, False), (True, True))
WEIGHT_DECAY = (1e-5, 5e-3)
LR = (0.01, 0.1)
SMOOTHING = (0.05, 0.20)


def two_digits(value: float) -> float:
    """The value with two significant digits."""
    return float(f"{value:.2g}")


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10 ** rng.uniform(np.log10(low), np.log10(high)))


def draw(part: int, count: int = PER_PART, seed: int = SEED) -> dict[str, dict]:
    """The recipes of one part, by name."""
    rng = np.random.default_rng([seed, part])
    recipes = {}
    for number in range(1, count + 1):
        augment, smooth = CELLS[(number - 1) % len(CELLS)]
        decay = two_digits(log_uniform(rng, *WEIGHT_DECAY))
        rate = two_digits(log_uniform(rng, *LR))
        level = round(float(rng.uniform(*SMOOTHING)), 2)
        recipes[f"{PARTS[part]}{number:02d}"] = {
            "augment": augment, "weight_decay": decay,
            "label_smoothing": level if smooth else 0.0, "lr": rate}
    return recipes


def table() -> dict:
    """The content of the frozen file."""
    return {"amendment": "A14", "seed": SEED,
            "space": {"weight_decay_log_uniform": list(WEIGHT_DECAY),
                      "lr_log_uniform": list(LR),
                      "label_smoothing_uniform": list(SMOOTHING),
                      "cells_augment_smoothing": [list(cell) for cell in CELLS]},
            "recipes": {**draw(1), **draw(2)}}


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    if PATH.exists():
        raise SystemExit(f"{PATH.name} exists: the recipes are drawn once "
                         f"(sha256 {file_hash(PATH)})")
    PATH.write_text(json.dumps(table(), indent=1) + "\n")
    print(f"wrote {PATH.name}, sha256 {file_hash(PATH)}")


if __name__ == "__main__":
    main()
