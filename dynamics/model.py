"""The networks, their normalization, and the pooled features of their four stages."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from dynamics.config import ARCHS, LAYERS

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BLOCKS = ("stem", *LAYERS)


def build_model(n_classes: int, arch: str = "resnet18") -> torch.nn.Module:
    """A network of torchvision with a random start.

    Raises:
        ValueError: If the pipeline does not know the architecture.
    """
    if arch not in ARCHS:
        raise ValueError(f"unknown architecture: {arch}")
    return getattr(models, arch)(weights=None, num_classes=n_classes)


def head(model: torch.nn.Module) -> torch.nn.Linear:
    """The linear layer that gives the logits from the penultimate feature."""
    return model.classifier if isinstance(model, models.DenseNet) else model.fc


def block_of(name: str) -> str:
    """The block of a convolution, from the name of its module: stem or a stage.

    In a ResNet, the stages are `layer1` to `layer4`. In a DenseNet, a dense block
    is a stage, and a transition belongs to the stage that it leads to.
    """
    parts = name.split(".")
    if parts[0] == "features":
        if parts[1].startswith("denseblock"):
            return f"stage{parts[1][-1]}"
        if parts[1].startswith("transition"):
            return f"stage{int(parts[1][-1]) + 1}"
        return "stem"
    return "stem" if parts[0] == "conv1" else f"stage{parts[0][-1]}"


def weight_norms(model: torch.nn.Module) -> dict[str, float]:
    """L2 norms of the convolution weights, of the batch-norm scales, and of the head.

    In a ResNet every convolution is followed by batch normalization, so its
    output does not depend on the scale of its weights. `conv_norm` is the norm
    over all convolutions, and `conv_norm_<block>` is the norm of one block.
    """
    squared = dict.fromkeys(BLOCKS, 0.0)
    scales = 0.0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            squared[block_of(name)] += float(
                module.weight.detach().double().pow(2).sum())
        elif isinstance(module, torch.nn.BatchNorm2d):
            scales += float(module.weight.detach().double().pow(2).sum())
    out = {f"conv_norm_{block}": value ** 0.5 for block, value in squared.items()}
    out["conv_norm"] = sum(squared.values()) ** 0.5
    out["bn_scale_norm"] = scales ** 0.5
    out["fc_norm"] = float(head(model).weight.detach().double().norm())
    return out


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Standardize [0, 1] images with the ImageNet channel statistics."""
    mean = torch.tensor(MEAN, device=images.device, dtype=images.dtype)
    std = torch.tensor(STD, device=images.device, dtype=images.dtype)
    return (images - mean[:, None, None]) / std[:, None, None]


def dense_stages(model: torch.nn.Module, images: torch.Tensor
                 ) -> tuple[list[torch.Tensor], torch.Tensor]:
    """`forward_stages` for a DenseNet: the pooled output of each dense block.

    The last one is taken after the final normalization and rectification, so it
    is the penultimate feature of the model's own forward pass.
    """
    layers = model.features
    x = layers.pool0(layers.relu0(layers.norm0(layers.conv0(normalize(images)))))
    pooled = []
    for number in (1, 2, 3):
        x = getattr(layers, f"denseblock{number}")(x)
        pooled.append(x.mean(dim=(2, 3)))
        x = getattr(layers, f"transition{number}")(x)
    x = torch.relu(layers.norm5(layers.denseblock4(x)))
    pooled.append(x.mean(dim=(2, 3)))
    return pooled, model.classifier(pooled[-1])


def forward_stages(model: torch.nn.Module, images: torch.Tensor
                   ) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Pooled output of each stage, and the logits.

    The last pooled output is the penultimate feature, and the logits are the head
    applied to it, as in the model's own forward pass.
    """
    if isinstance(model, models.DenseNet):
        return dense_stages(model, images)
    x = model.maxpool(model.relu(model.bn1(model.conv1(normalize(images)))))
    pooled = []
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        x = stage(x)
        pooled.append(x.mean(dim=(2, 3)))
    return pooled, model.fc(pooled[-1])


@torch.no_grad()
def extract(model: torch.nn.Module, dataset: Dataset, device: torch.device,
            batch_size: int = 256, workers: int = 4) -> dict[str, torch.Tensor]:
    """Features of every image of a dataset, in evaluation mode and float32.

    Returns:
        A dict with one (n, d) tensor per layer of `LAYERS`, `logits`, and `labels`.
    """
    model.eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers)
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in (*LAYERS, "logits",
                                                                  "labels")}
    for images, labels in loader:
        pooled, logits = forward_stages(model, images.to(device))
        for name, values in zip(LAYERS, pooled):
            parts[name].append(values.float().cpu())
        parts["logits"].append(logits.float().cpu())
        parts["labels"].append(torch.as_tensor(labels).long())
    return {name: torch.cat(values) for name, values in parts.items()}
