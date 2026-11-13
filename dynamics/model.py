"""ResNet-18, its normalization, and the pooled features of its four stages."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from dynamics.config import LAYERS

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_model(n_classes: int) -> torch.nn.Module:
    """ResNet-18 of torchvision with a random start."""
    return models.resnet18(weights=None, num_classes=n_classes)


def weight_norms(model: torch.nn.Module) -> dict[str, float]:
    """L2 norms of the convolution weights, of the batch-norm scales, and of the head.

    Every convolution of this network is followed by batch normalization, so its
    output does not depend on the scale of its weights. `conv_norm` is the norm
    over all convolutions, and `conv_norm_<block>` is the norm of one block.
    """
    blocks = {"conv1": "stem", "layer1": "stage1", "layer2": "stage2",
              "layer3": "stage3", "layer4": "stage4"}
    squared = dict.fromkeys(blocks.values(), 0.0)
    scales = 0.0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            squared[blocks[name.split(".")[0]]] += float(
                module.weight.detach().double().pow(2).sum())
        elif isinstance(module, torch.nn.BatchNorm2d):
            scales += float(module.weight.detach().double().pow(2).sum())
    out = {f"conv_norm_{block}": value ** 0.5 for block, value in squared.items()}
    out["conv_norm"] = sum(squared.values()) ** 0.5
    out["bn_scale_norm"] = scales ** 0.5
    out["fc_norm"] = float(model.fc.weight.detach().double().norm())
    return out


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Standardize [0, 1] images with the ImageNet channel statistics."""
    mean = torch.tensor(MEAN, device=images.device, dtype=images.dtype)
    std = torch.tensor(STD, device=images.device, dtype=images.dtype)
    return (images - mean[:, None, None]) / std[:, None, None]


def forward_stages(model: torch.nn.Module, images: torch.Tensor
                   ) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Pooled output of each residual stage, and the logits.

    The last pooled output is the penultimate feature, and the logits are the head
    applied to it, as in the model's own forward pass.
    """
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
