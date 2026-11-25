"""Amendment A16: the list of the public ImageNet-1k models, made once.

    python -m dynamics.zoo_list

The list holds every classification weight of torchvision 0.21.0 that was trained
on ImageNet-1k only (`IMAGENET1K_V1` and `IMAGENET1K_V2`) and whose network ends
in a linear layer. Each entry has its family, its lineage (related families that
are left out together), the published accuracy, the preprocessing sizes, and two
tags that need no OOD data:

- `arch_type`: `convolutional` or `transformer`.
- `recipe_tag`, from the documentation of the weight only: `yes` if its recipe has
  label smoothing or a mixing augmentation (mixup or CutMix), `no` if its
  documented command has none, and `ported` if torchvision documents no training
  command for it (weights ported from the paper, from LuaTorch, or from a
  community repository).

The list is written to `dynamics/frozen/a16_models.json`, which is frozen: this
script does not write it a second time.

Protocol: ../documentation/imagenet1k_public_models_protocol_2026-11-24.md
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = HERE / "frozen" / "a16_models.json"
TORCHVISION = "0.21.0"
WEIGHTS = ("IMAGENET1K_V1", "IMAGENET1K_V2")
#: Family of a model name, by the first pattern that matches.
FAMILIES = (
    (r"resnet\d", "resnet"), (r"resnext", "resnext"), (r"wide_resnet", "wide_resnet"),
    (r"densenet", "densenet"), (r"vgg", "vgg"), (r"alexnet", "alexnet"),
    (r"regnet_x", "regnet_x"), (r"regnet_y", "regnet_y"),
    (r"efficientnet_v2", "efficientnet_v2"), (r"efficientnet_b", "efficientnet"),
    (r"mobilenet_v2", "mobilenet_v2"), (r"mobilenet_v3", "mobilenet_v3"),
    (r"shufflenet_v2", "shufflenet_v2"), (r"mnasnet", "mnasnet"),
    (r"convnext", "convnext"), (r"vit_", "vit"), (r"swin_v2", "swin_v2"),
    (r"swin_", "swin"), (r"maxvit", "maxvit"), (r"squeezenet", "squeezenet"),
    (r"googlenet", "googlenet"), (r"inception_v3", "inception_v3"))
#: Related families are one lineage: they are left out of a fit together.
LINEAGES = {
    "resnet": "resnet_kind", "resnext": "resnet_kind", "wide_resnet": "resnet_kind",
    "regnet_x": "regnet", "regnet_y": "regnet",
    "efficientnet": "efficientnet", "efficientnet_v2": "efficientnet",
    "mobilenet_v2": "mobile", "mobilenet_v3": "mobile", "mnasnet": "mobile",
    "shufflenet_v2": "mobile",
    "vgg": "vgg_kind", "alexnet": "vgg_kind",
    "swin": "swin", "swin_v2": "swin",
    "densenet": "densenet", "convnext": "convnext", "vit": "vit",
    "googlenet": "inception_kind", "inception_v3": "inception_kind",
    "maxvit": "maxvit"}
TRANSFORMERS = ("vit", "swin", "swin_v2", "maxvit")
#: Sections of the torchvision classification README (v0.21.0) whose training
#: command has label smoothing with mixup or CutMix.
MIXING_SECTIONS = ("#efficientnet-v2", "#vit_b_16", "#vit_b_32", "#vit_l_16",
                   "#vit_l_32", "#convnext", "#swintransformer",
                   "#swintransformer-v2", "#maxvit")
#: The new recipe of torchvision (label smoothing 0.1, mixup, CutMix).
NEW_RECIPE = ("issues/3995", "new training recipe")
#: Families whose network has no linear head.
NO_LINEAR_HEAD = ("squeezenet",)


def family_of(model: str) -> str:
    """The family of a torchvision model name.

    Raises:
        ValueError: If no pattern matches.
    """
    for pattern, family in FAMILIES:
        if re.match(pattern, model):
            return family
    raise ValueError(f"no family for the model {model}")


def recipe_tag(recipe: str, docs: str) -> str:
    """`yes`, `no`, or `ported`, from the recipe link and the text of the weight."""
    if "ported" in docs or "github.com/pytorch/vision" not in recipe:
        return "ported"
    if (any(mark in recipe or mark in docs for mark in NEW_RECIPE)
            or recipe.endswith(MIXING_SECTIONS)):
        return "yes"
    return "no"


def model_id(model: str, weights: str) -> str:
    return f"{model}-{weights.rsplit('_', 1)[-1].lower()}"


def entries() -> list[dict]:
    """One entry for each weight of the list, from the metadata of torchvision.

    Raises:
        RuntimeError: If the installed torchvision is another version.
    """
    import torchvision
    from torchvision import models

    if not torchvision.__version__.startswith(TORCHVISION):
        raise RuntimeError(f"the list needs torchvision {TORCHVISION}, not "
                           f"{torchvision.__version__}")
    rows = []
    for model in models.list_models(module=models):
        family = family_of(model)
        if family in NO_LINEAR_HEAD:
            continue
        for weights in models.get_model_weights(model):
            if weights.name not in WEIGHTS:
                continue
            meta, preset = weights.meta, weights.transforms()
            metrics = meta["_metrics"]["ImageNet-1K"]
            docs = " ".join((meta.get("_docs") or "").split())
            rows.append({
                "id": model_id(model, weights.name), "model": model,
                "weights": weights.name, "family": family,
                "lineage": LINEAGES[family],
                "arch_type": ("transformer" if family in TRANSFORMERS
                              else "convolutional"),
                "recipe_tag": recipe_tag(meta.get("recipe", ""), docs),
                "acc1": metrics["acc@1"], "acc5": metrics["acc@5"],
                "params": meta["num_params"],
                "crop_size": list(preset.crop_size),
                "resize_size": list(preset.resize_size),
                "recipe": meta.get("recipe", ""), "docs": docs})
    return rows


def table() -> dict:
    """The content of the frozen file."""
    return {"amendment": "A16", "torchvision": TORCHVISION, "models": entries()}


def load(path: Path = PATH) -> list[dict]:
    return json.loads(Path(path).read_text())["models"]


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    if PATH.exists():
        raise SystemExit(f"{PATH.name} exists: the list is made once "
                         f"(sha256 {file_hash(PATH)})")
    PATH.write_text(json.dumps(table(), indent=1) + "\n")
    print(f"wrote {PATH.name}, sha256 {file_hash(PATH)}; "
          f"{len(load())} weights")


if __name__ == "__main__":
    main()
