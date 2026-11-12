"""Tiny image folders with the layout of the real datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dynamics.config import DataPaths


def pattern(rng: np.random.Generator, hue: float, stripes: int,
            size: tuple[int, int] = (72, 88)) -> Image.Image:
    """A noisy striped image whose color and stripe count identify a class."""
    height, width = size
    grid = np.linspace(0, np.pi * stripes, width)[None, :] + rng.uniform(0, np.pi)
    wave = 0.5 + 0.5 * np.sin(grid) * np.ones((height, 1))
    color = np.array([np.cos(2 * np.pi * (hue + shift)) for shift in (0, 1 / 3, 2 / 3)])
    image = 0.5 + 0.35 * wave[..., None] * color + 0.08 * rng.standard_normal(
        (height, width, 3))
    return Image.fromarray((255 * image.clip(0, 1)).astype(np.uint8))


def write_classes(root: Path, split: str, names: list[str], count: int,
                  rng: np.random.Generator, hue_shift: float) -> None:
    for index, name in enumerate(names):
        folder = root / split / name
        folder.mkdir(parents=True)
        for item in range(count):
            pattern(rng, hue_shift + index / len(names), 1 + index % 4).save(
                folder / f"{name}_{item:03d}.JPEG", quality=92)


def make_tiny_data(root: Path, n_classes: int = 10, train: int = 24, val: int = 12
                   ) -> DataPaths:
    """Fake Imagenette, Imagewoof, and Textures folders under `root`."""
    rng = np.random.default_rng(0)
    nette = [f"n{1000 + index}" for index in range(n_classes)]
    woof = [f"n{2000 + index}" for index in range(4)]
    write_classes(root / "imagenette", "train", nette, train, rng, 0.0)
    write_classes(root / "imagenette", "val", nette, val, rng, 0.0)
    write_classes(root / "imagewoof", "train", woof, 2, rng, 0.05)
    write_classes(root / "imagewoof", "val", woof, 15, rng, 0.05)
    textures = root / "textures" / "images" / "dotted"
    textures.mkdir(parents=True)
    for item in range(40):
        noise = rng.uniform(0, 255, (72, 88, 3)).astype(np.uint8)
        Image.fromarray(noise).save(textures / f"dotted_{item:03d}.jpg")
    for name, count in (("places", 30), ("ssb_hard", 18)):
        folder = root / "openood" / name / "images" / "n9999"
        folder.mkdir(parents=True)
        for item in range(count):
            pattern(rng, 0.5 + item / count, 5).save(folder / f"{name}_{item:03d}.jpg")
    return DataPaths(root / "imagenette", root / "imagewoof", root / "textures")


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> DataPaths:
    return make_tiny_data(tmp_path_factory.mktemp("data"))


def synthetic_layers(seed: int = 0, n_classes: int = 6, n_train: int = 600,
                     n_eval: int = 200, spread: float = 1.0, drift: float = 1.0
                     ) -> dict:
    """Rectified Gaussian class clusters at four layer widths, with a linear head.

    Returns a dict like `model.extract`: one tensor per layer, logits, and labels,
    plus the head `w` and `b`.
    """
    import torch

    from dynamics.config import LAYERS

    widths = dict(zip(LAYERS, (16, 24, 32, 48)))
    gen = torch.Generator().manual_seed(seed)
    means = {layer: 3 * torch.randn(n_classes, width, generator=gen)
             for layer, width in widths.items()}

    def sample(count: int, scale: float, pull: float) -> dict:
        labels = torch.arange(count) % n_classes
        out = {layer: torch.relu(means[layer][labels] * pull + scale * torch.randn(
            count, width, generator=gen)) for layer, width in widths.items()}
        out["labels"] = labels
        return out

    train = sample(n_train, 1.0, 1.0)
    last = train[LAYERS[-1]]
    class_means = torch.stack([last[train["labels"] == c].mean(0)
                               for c in range(n_classes)])
    centered = class_means - class_means.mean(0)
    w = 4 * centered / centered.norm(dim=1, keepdim=True).mean() ** 2
    b = -(w @ class_means.mean(0))

    def finish(part: dict) -> dict:
        part["logits"] = part[LAYERS[-1]] @ w.T + b
        return part

    return {"train": finish(train), "val": finish(sample(n_eval, 1.0, 1.0)),
            "far": finish(sample(n_eval, spread * 1.6, drift * 0.3)),
            "w": w, "b": b, "n_classes": n_classes}


@pytest.fixture(scope="session")
def layers() -> dict:
    return synthetic_layers()
