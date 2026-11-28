"""Amendment A18, part 1: candidate stand-in tasks made from ID validation images.

Each task changes a preprocessed (normalized) image of the fixed validation
subset. A task with pixel values (inverted colours, the geometric mean, noise)
works on the values before the normalization, so it needs the mean and the
standard deviation of the weight. Every random choice has a seed from the task
and the position of the image, so a task gives the same images for every model.

Protocol: ../documentation/standin_tasks_pilot_plan_2026-11-27.md
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf

from dynamics.data import shuffle_patches

TASK_SEED = 18
#: Images of each class in a new task (the existing tasks have 10).
PER_CLASS = 5
#: Tasks that need a pass of images through the network.
IMAGE_TASKS = ("shuffle2", "shuffle8", "shuffle16", "phase", "ampswap", "cutpaste",
               "mosaic", "blend30", "geoblend", "invert", "noise", "blur",
               "highpass", "mask75")
#: Tasks that are made from the features of the validation images.
FEATURE_TASKS = ("featblend",)
NEW_TASKS = IMAGE_TASKS + FEATURE_TASKS
GRIDS = {"shuffle2": 2, "shuffle8": 8, "shuffle16": 16}
#: Tasks that use one image of another class, and the task that uses three.
WITH_PARTNER = ("ampswap", "cutpaste", "blend30", "geoblend")
WITH_THREE = ("mosaic",)
BLEND_SHARE = 0.3
PASTE_AREA = 0.5
BLUR_DIVISOR = 28
MASK_GRID = 14
MASK_SHARE = 0.75


def task_rng(task: str, index: int) -> np.random.Generator:
    """The generator of one image of one task."""
    return np.random.default_rng([TASK_SEED, NEW_TASKS.index(task), index])


def subset_of(kept: np.ndarray, labels: np.ndarray,
              per_class: int = PER_CLASS) -> np.ndarray:
    """The first `per_class` kept positions of each class, in sorted order."""
    kept = np.sort(np.asarray(kept))
    chosen = [kept[labels[kept] == label][:per_class]
              for label in np.unique(labels[kept])]
    return np.sort(np.concatenate(chosen))


def partners_of(labels: np.ndarray, seed: int = TASK_SEED) -> np.ndarray:
    """For each image, the positions of three images of three other classes.

    Raises:
        ValueError: If the labels have fewer than four classes.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 4:
        raise ValueError("the partners need at least four classes")
    rng = np.random.default_rng([seed, 0])
    by_class = {label: np.flatnonzero(labels == label) for label in classes}
    out = np.empty((len(labels), 3), dtype=np.int64)
    for index, label in enumerate(labels):
        others = rng.choice(classes[classes != label], size=3, replace=False)
        out[index] = [rng.choice(by_class[other]) for other in others]
    return out


def shuffle_grid(image: torch.Tensor, grid: int, rng: np.random.Generator
                 ) -> torch.Tensor:
    """Patches of a grid in a random order that is not the identity. The part of
    the image that the grid does not divide stays."""
    identity = np.arange(grid * grid)
    permutation = rng.permutation(grid * grid)
    while np.array_equal(permutation, identity):
        permutation = rng.permutation(grid * grid)
    _, height, width = image.shape
    rows, cols = height - height % grid, width - width % grid
    out = image.clone()
    out[:, :rows, :cols] = shuffle_patches(image[:, :rows, :cols].contiguous(),
                                           permutation, grid)
    return out


def phase_scramble(image: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """The amplitude spectrum of the image with the phase of a noise image. The
    mean of each channel stays."""
    _, height, width = image.shape
    noise = torch.from_numpy(rng.standard_normal((height, width))).to(image.dtype)
    spectrum = torch.fft.fft2(image)
    phase = torch.angle(torch.fft.fft2(noise)).expand_as(spectrum.real).clone()
    phase[:, 0, 0] = torch.angle(spectrum[:, 0, 0])
    return torch.fft.ifft2(torch.polar(spectrum.abs(), phase)).real


def amplitude_swap(image: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
    """The phase of the image with the amplitude spectrum of the partner."""
    amplitude = torch.fft.fft2(partner).abs()
    phase = torch.angle(torch.fft.fft2(image))
    return torch.fft.ifft2(torch.polar(amplitude, phase)).real


def cut_paste(image: torch.Tensor, partner: torch.Tensor, rng: np.random.Generator,
              area: float = PASTE_AREA) -> torch.Tensor:
    """A rectangle with a share `area` of the image, taken from the partner."""
    _, height, width = image.shape
    rows, cols = round(height * area ** 0.5), round(width * area ** 0.5)
    top = int(rng.integers(0, height - rows + 1))
    left = int(rng.integers(0, width - cols + 1))
    out = image.clone()
    out[:, top:top + rows, left:left + cols] = partner[:, top:top + rows,
                                                       left:left + cols]
    return out


def mosaic(image: torch.Tensor, partners: list[torch.Tensor]) -> torch.Tensor:
    """Four quadrants: the image and three partners, each in its own position."""
    _, height, width = image.shape
    row, col = height // 2, width // 2
    out = image.clone()
    out[:, :row, col:] = partners[0][:, :row, col:]
    out[:, row:, :col] = partners[1][:, row:, :col]
    out[:, row:, col:] = partners[2][:, row:, col:]
    return out


def blur(image: torch.Tensor, divisor: int = BLUR_DIVISOR) -> torch.Tensor:
    """A Gaussian blur with a standard deviation of 1/`divisor` of the side."""
    sigma = max(image.shape[1:]) / divisor
    kernel = 2 * round(3 * sigma) + 1
    return tvf.gaussian_blur(image, [kernel, kernel], [sigma, sigma])


def mask_patches(image: torch.Tensor, rng: np.random.Generator,
                 grid: int = MASK_GRID, share: float = MASK_SHARE) -> torch.Tensor:
    """A share of the cells of a grid replaced by the mean of each channel."""
    _, height, width = image.shape
    cells = (torch.arange(height)[:, None] * grid // height) * grid + (
        torch.arange(width)[None, :] * grid // width)
    hidden = torch.zeros(grid * grid, dtype=torch.bool)
    hidden[torch.from_numpy(rng.choice(grid * grid, size=round(share * grid * grid),
                                       replace=False))] = True
    mean = image.mean(dim=(1, 2), keepdim=True).expand_as(image)
    return torch.where(hidden[cells][None], mean, image)


def apply(task: str, image: torch.Tensor, partners: list[torch.Tensor],
          mean: torch.Tensor, std: torch.Tensor, rng: np.random.Generator
          ) -> torch.Tensor:
    """One image of one task.

    Args:
        image: A normalized image, (C, H, W).
        partners: Normalized images of other classes, for the tasks that use them.
        mean, std: The normalization of the weight, each (C, 1, 1).

    Raises:
        ValueError: If the task is not an image task.
    """
    if task in GRIDS:
        return shuffle_grid(image, GRIDS[task], rng)
    if task == "phase":
        return phase_scramble(image, rng)
    if task == "ampswap":
        return amplitude_swap(image, partners[0])
    if task == "cutpaste":
        return cut_paste(image, partners[0], rng)
    if task == "mosaic":
        return mosaic(image, partners)
    if task == "blend30":
        return (1 - BLEND_SHARE) * image + BLEND_SHARE * partners[0]
    if task == "blur":
        return blur(image)
    if task == "highpass":
        return image - blur(image) + image.mean(dim=(1, 2), keepdim=True)
    if task == "mask75":
        return mask_patches(image, rng)
    if task == "geoblend":
        first = (image * std + mean).clamp(0, 1)
        second = (partners[0] * std + mean).clamp(0, 1)
        return (torch.sqrt(first * second) - mean) / std
    if task == "invert":
        return (1 - (image * std + mean).clamp(0, 1) - mean) / std
    if task == "noise":
        pixels = torch.from_numpy(rng.random(tuple(image.shape))).to(image.dtype)
        return (pixels - mean) / std
    raise ValueError(f"{task} is not an image task")


class TaskSet(Dataset):
    """The images of one task, made from a base set of validation images."""

    def __init__(self, base: Dataset, task: str, partners: np.ndarray,
                 mean: torch.Tensor, std: torch.Tensor) -> None:
        if task not in IMAGE_TASKS:
            raise ValueError(f"{task} is not an image task")
        self.base, self.task, self.partners = base, task, partners
        self.mean, self.std = mean, std

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, _ = self.base[index]
        count = 1 if self.task in WITH_PARTNER else 3 if self.task in WITH_THREE else 0
        others = [self.base[int(position)][0]
                  for position in self.partners[index, :count]]
        return apply(self.task, image, others, self.mean, self.std,
                     task_rng(self.task, index)), -1


def feature_blend(features: torch.Tensor, partners: np.ndarray) -> torch.Tensor:
    """The average of the feature of each image and of its first partner."""
    return 0.5 * (features + features[torch.as_tensor(partners[:, 0])])
