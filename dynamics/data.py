"""Datasets of the pilot: ID splits, external sets, and ID-only stand-in sets.

Every dataset returns images as float tensors in [0, 1]. The model code applies the
normalization, so a stand-in can work in pixel space.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dynamics.config import (
    BLEND_SEED,
    GRID,
    RESERVED_SETS,
    RRC_SCALE,
    SHUFFLE_SEED,
    DataPaths,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    """Every image file below a folder, in sorted order."""
    return sorted(path for path in Path(folder).rglob("*")
                  if path.suffix.lower() in IMAGE_SUFFIXES)


def class_folders(split_dir: Path) -> list[str]:
    """Class identifiers of a split: the sorted names of its subfolders.

    Raises:
        FileNotFoundError: If the split folder has no class folder.
    """
    names = sorted(path.name for path in Path(split_dir).iterdir() if path.is_dir())
    if not names:
        raise FileNotFoundError(f"no class folders in {split_dir}")
    return names


def thin(items: list, limit: int | None) -> list:
    """At most `limit` items, evenly spaced. Smoke runs use this."""
    if limit is None or len(items) <= limit:
        return items
    index = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[i] for i in index]


def eval_transform(size: int) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                               transforms.ToTensor()])


def train_transform(size: int, augment: bool) -> transforms.Compose:
    if not augment:
        return eval_transform(size)
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=RRC_SCALE),
        transforms.RandomHorizontalFlip(), transforms.ToTensor()])


class ImageList(Dataset):
    """Images from a list of files, with integer labels (-1 means no label)."""

    def __init__(self, paths: list[Path], labels: list[int],
                 transform: transforms.Compose) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.labels[index]


class CachedSet(Dataset):
    """A dataset held in memory as uint8 images. Items equal the uncached ones."""

    def __init__(self, images: torch.Tensor, labels: list[int]) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index].float() / 255, self.labels[index]


def cache(dataset: ImageList, workers: int = 4) -> CachedSet:
    """Decode every image of a dataset once.

    The transform ends with ToTensor on 8-bit images, so every value is k / 255 and
    the uint8 copy loses nothing.
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=workers)
    images = [(batch * 255).round().to(torch.uint8) for batch, _ in loader]
    shape = dataset[0][0].shape if len(dataset) else (3, 0, 0)
    return CachedSet(torch.cat(images) if images else torch.empty(0, *shape,
                                                                  dtype=torch.uint8),
                     list(dataset.labels))


def blend_partners(labels: np.ndarray, seed: int = BLEND_SEED) -> np.ndarray:
    """For each image, the index of a random image from a different class."""
    rng = np.random.default_rng(seed)
    partners = np.empty(len(labels), dtype=np.int64)
    for index, label in enumerate(labels):
        partners[index] = rng.choice(np.flatnonzero(labels != label))
    return partners


def patch_permutations(count: int, grid: int = GRID,
                       seed: int = SHUFFLE_SEED) -> np.ndarray:
    """One permutation of the grid * grid patches per image, never the identity."""
    rng = np.random.default_rng(seed)
    identity = np.arange(grid * grid)
    out = np.empty((count, grid * grid), dtype=np.int64)
    for index in range(count):
        permutation = rng.permutation(grid * grid)
        while np.array_equal(permutation, identity):
            permutation = rng.permutation(grid * grid)
        out[index] = permutation
    return out


def shuffle_patches(image: torch.Tensor, permutation: np.ndarray,
                    grid: int = GRID) -> torch.Tensor:
    """Cut a (C, H, W) image into grid * grid patches and reorder them."""
    channels, height, width = image.shape
    rows, cols = height // grid, width // grid
    patches = (image.reshape(channels, grid, rows, grid, cols)
               .permute(1, 3, 0, 2, 4).reshape(grid * grid, channels, rows, cols))
    patches = patches[torch.as_tensor(permutation)]
    return (patches.reshape(grid, grid, channels, rows, cols)
            .permute(2, 0, 3, 1, 4).reshape(channels, height, width))


class BlendSet(Dataset):
    """Pixel average of each image with its partner from a different class."""

    def __init__(self, base: Dataset, partners: np.ndarray) -> None:
        self.base = base
        self.partners = partners

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        first, _ = self.base[index]
        second, _ = self.base[int(self.partners[index])]
        return 0.5 * (first + second), -1


class ShuffleSet(Dataset):
    """Each image with its patches permuted."""

    def __init__(self, base: Dataset, permutations: np.ndarray,
                 grid: int = GRID) -> None:
        self.base = base
        self.permutations = permutations
        self.grid = grid

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, _ = self.base[index]
        return shuffle_patches(image, self.permutations[index], self.grid), -1


def labelled_split(root: Path, split: str, held_out: int | None,
                   limit: int | None = None
                   ) -> tuple[list[Path], list[int], list[str], list[Path]]:
    """Files and labels of the kept classes, and the files of the held-out class.

    Labels follow the sorted order of the kept class identifiers.
    """
    classes = class_folders(Path(root) / split)
    kept = [name for index, name in enumerate(classes) if index != held_out]
    paths: list[Path] = []
    labels: list[int] = []
    for label, name in enumerate(kept):
        files = thin(list_images(Path(root) / split / name),
                     None if limit is None else max(1, limit // len(kept)))
        paths += files
        labels += [label] * len(files)
    held: list[Path] = []
    if held_out is not None:
        held = thin(list_images(Path(root) / split / classes[held_out]), limit)
    return paths, labels, kept, held


def training_set(paths: DataPaths, held_out: int | None, size: int, augment: bool,
                 limit: int | None = None) -> tuple[ImageList, list[str]]:
    """The training split with the training transform, and the kept classes."""
    files, labels, classes, _ = labelled_split(paths.imagenette, "train", held_out,
                                               limit)
    return ImageList(files, labels, train_transform(size, augment)), classes


def evaluation_sets(paths: DataPaths, held_out: int | None, size: int,
                    limit: int | None = None, workers: int = 4
                    ) -> dict[str, Dataset]:
    """Every set that a checkpoint is evaluated on, decoded once and held in memory.

    Keys: train, val, blend, shuffle, and, when available, imagewoof, textures,
    heldout, and the extra sets of `paths.extra`. All sets use the evaluation
    transform. The blend and shuffle sets come from the validation images.

    Raises:
        ValueError: If an extra set has a reserved or repeated name.
        FileNotFoundError: If a set has no images.
    """
    transform = eval_transform(size)
    train_files, train_labels, _, held_train = labelled_split(
        paths.imagenette, "train", held_out, limit)
    val_files, val_labels, _, held_val = labelled_split(
        paths.imagenette, "val", held_out, limit)
    val = cache(ImageList(val_files, val_labels, transform), workers)
    sets: dict[str, Dataset] = {
        "train": cache(ImageList(train_files, train_labels, transform), workers),
        "val": val,
        "blend": BlendSet(val, blend_partners(np.asarray(val_labels))),
        "shuffle": ShuffleSet(val, patch_permutations(len(val))),
    }
    unlabelled: dict[str, list[Path]] = {}
    if paths.imagewoof is not None:
        unlabelled["imagewoof"] = thin(list_images(Path(paths.imagewoof) / "val"),
                                       limit)
    if paths.textures is not None:
        unlabelled["textures"] = thin(list_images(paths.textures), limit)
    if held_out is not None:
        unlabelled["heldout"] = held_train + held_val
    for name, folder in paths.extra:
        if name in RESERVED_SETS or name in unlabelled:
            raise ValueError(f"the set name {name} is reserved or used twice")
        cap = paths.extra_limit if limit is None else limit
        unlabelled[name] = thin(list_images(folder), cap)
    for name, files in unlabelled.items():
        if not files:
            raise FileNotFoundError(f"the set {name} has no images")
        sets[name] = cache(ImageList(files, [-1] * len(files), transform), workers)
    return sets


def manifest(paths: DataPaths) -> dict:
    """Class identifiers and image counts, for gate G5."""
    out: dict = {}
    for name, root in (("imagenette", paths.imagenette),
                       ("imagewoof", paths.imagewoof)):
        if root is None:
            continue
        out[name] = {
            split: {cls: len(list_images(Path(root) / split / cls))
                    for cls in class_folders(Path(root) / split)}
            for split in ("train", "val") if (Path(root) / split).is_dir()}
    if paths.textures is not None:
        out["textures"] = {"images": len(list_images(paths.textures))}
    id_classes = set(class_folders(Path(paths.imagenette) / "train"))
    for name, folder in paths.extra:
        files = list_images(folder)
        parts = {part for path in files for part in path.relative_to(folder).parts}
        used = len(files) if paths.extra_limit is None else min(len(files),
                                                                 paths.extra_limit)
        out[name] = {"images": len(files), "used": used,
                     "id_class_folders": sorted(parts & id_classes)}
    return out
