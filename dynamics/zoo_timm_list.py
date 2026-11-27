"""Amendment A17: the list of the timm models for the test of the frozen rule.

    python -m dynamics.zoo_timm_list draw --results-file results-imagenet.csv
    python -m dynamics.zoo_timm_list freeze --results-file results-imagenet.csv

Needs timm 1.0.31 (the version of the HPC image) and the results file of that
version from the timm repository. `draw` prints the models and writes nothing.
`freeze` writes `dynamics/frozen/a17_models.json` once.

A pretrained weight is eligible if its tag ends in `in1k` with no mark of other
data, of pretraining, of distillation, or of adversarial training, if it is not a
torchvision weight, if it has 1,000 classes and a published accuracy at its
evaluation size, if that size is at most 384 px, and if the head check passes:
the head on its captured input gives the output of the network.

The lineage of a model is its timm module. Modules of an architecture that
torchvision has, and their close relatives, are group S with the lineage of
amendment A16. Every other module is a new lineage (group N). Group R is the
architecture `resnet50` with every eligible weight. Groups N and S take two
weights for each lineage, in the order of a seeded permutation, with different
architecture names.

Protocol: ../documentation/imagenet1k_second_library_amendment_2026-11-26.md
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from dynamics import zoo_timm_tags as tags

HERE = Path(__file__).resolve().parent
PATH = HERE / "frozen" / "a17_models.json"
TIMM = "1.0.31"
#: SHA-256 of `results/results-imagenet.csv` at the tag v1.0.31 of timm.
RESULTS_SHA256 = "30972bf7c7ab451bbd4480ec6116b13d0a84441c57bbc946896fd312264acfcd"
RESULTS_URL = ("https://raw.githubusercontent.com/huggingface/pytorch-image-models/"
               "v1.0.31/results/results-imagenet.csv")
SEED = 17
PER_LINEAGE = 2
MAX_SIZE = 384
N_CLASSES = 1000
#: A part of a tag that starts with one of these marks other data, pretraining,
#: or distillation.
MARK_PREFIXES = ("in21k", "in22k", "in12k", "jft", "ig1b", "yfcc", "laion",
                 "openai", "ssl", "swsl", "lvd", "swag", "dist", "mim", "mae",
                 "dino", "clip", "fcmae", "seer")
#: A part of a tag that is one of these marks fine-tuning, noisy student,
#: AdvProp, or adversarial training.
MARK_PARTS = ("ft", "ns", "ap", "adv")
#: The weights of torchvision: the models of amendment A16.
TORCHVISION_TAGS = ("tv_in1k", "tv2_in1k")
RECIPE_ARCHITECTURE = "resnet50"
#: Group S: modules of an architecture that torchvision has, with the lineage of
#: amendment A16, and their close relatives.
SEEN = {
    "resnet": "resnet_kind", "regnet": "regnet",
    "mobilenetv3": "mobile", "hardcorenas": "mobile",
    "vgg": "vgg_kind", "densenet": "densenet", "convnext": "convnext",
    "vision_transformer": "vit", "deit": "vit", "vision_transformer_relpos": "vit",
    "vision_transformer_hybrid": "vit", "vision_transformer_sam": "vit",
    "swin_transformer": "swin", "swin_transformer_v2": "swin",
    "swin_transformer_v2_cr": "swin", "maxxvit": "maxvit",
    "inception_v3": "inception_kind", "inception_v4": "inception_kind",
    "inception_resnet_v2": "inception_kind",
    "resnetv2": "resnet_kind", "senet": "resnet_kind", "sknet": "resnet_kind",
    "res2net": "resnet_kind", "resnest": "resnet_kind"}
#: Added at the build, before the draw: modules of timm 1.0.31 that hold the ViT
#: architecture and that the table of the plan did not name.
SEEN_ADDED = {"naflexvit": "vit", "eva": "vit", "beit": "vit"}
#: The module `efficientnet` builds EfficientNet and the mobile networks.
EFFICIENTNET_MODULE = "efficientnet"
#: Largest relative difference of the head check.
HEAD_LIMIT = 1e-5
#: Added at the build, before the draw: the small networks that timm trains for
#: its own tests are not models that a person would use.
TEST_PREFIX = "test_"


def tag_eligible(tag: str) -> bool:
    """True for a tag of supervised training on ImageNet-1k from a random start,
    as far as the tag says."""
    parts = tag.split("_")
    if parts[-1] != "in1k" or tag in TORCHVISION_TAGS:
        return False
    return not any(part.startswith(MARK_PREFIXES) or part in MARK_PARTS
                   for part in parts[:-1])


def lineage_of(module: str, architecture: str) -> tuple[str, str]:
    """The group (`S` or `N`) and the lineage of a model."""
    if module == EFFICIENTNET_MODULE:
        return "S", "efficientnet" if "efficientnet" in architecture else "mobile"
    seen = SEEN | SEEN_ADDED
    return ("S", seen[module]) if module in seen else ("N", module)


def read_results(path: Path) -> dict[tuple[str, int], dict]:
    """The rows of the results file of timm, by model name and image size.

    Raises:
        ValueError: If the file does not have the recorded hash.
    """
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != RESULTS_SHA256:
        raise ValueError(f"{Path(path).name} has hash {digest}, not the one of "
                         f"timm {TIMM}")
    with Path(path).open(newline="") as handle:
        return {(row["model"], int(row["img_size"])): row
                for row in csv.DictReader(handle)}


def candidates(results: dict[tuple[str, int], dict]) -> list[dict]:
    """Every weight that the metadata of timm makes eligible (no head check).

    Raises:
        RuntimeError: If the installed timm is another version.
    """
    import timm
    from timm.data import resolve_data_config
    from timm.models import get_pretrained_cfg, model_entrypoint

    if timm.__version__ != TIMM:
        raise RuntimeError(f"the list needs timm {TIMM}, not {timm.__version__}")
    rows = []
    for name in sorted(timm.list_pretrained()):
        architecture, tag = name.split(".", 1)
        if not tag_eligible(tag) or architecture.startswith(TEST_PREFIX):
            continue
        config = get_pretrained_cfg(name).to_dict()
        if config.get("num_classes") != N_CLASSES:
            continue
        data = resolve_data_config(pretrained_cfg=config, use_test_size=True)
        _, height, width = data["input_size"]
        row = results.get((name, height))
        if height != width or height > MAX_SIZE or row is None:
            continue
        if (abs(float(row["crop_pct"]) - data["crop_pct"]) > 1e-6
                or row["interpolation"] != data["interpolation"]):
            continue
        module = model_entrypoint(architecture).__module__.rsplit(".", 1)[-1]
        group, lineage = lineage_of(module, architecture)
        rows.append({
            "id": name, "library": "timm", "model": name,
            "architecture": architecture, "tag": tag, "module": module,
            "family": module, "group": group, "lineage": lineage,
            "acc1": float(row["top1"]), "acc5": float(row["top5"]),
            "params_m": float(row["param_count"].replace(",", "")),
            "crop_size": [height], "crop_pct": data["crop_pct"],
            "crop_mode": data.get("crop_mode", "center"),
            "interpolation": data["interpolation"]})
    return rows


def head_problem(model: torch.nn.Module, size: int, batch: int = 2,
                 seed: int = 0) -> str:
    """Why the head check fails for a network, or an empty text if it passes.

    The head on its captured input must give the output of the network, and the
    captured input must be one vector for each image.
    """
    from dynamics.zoo import head_of  # here: `dynamics.zoo` imports this module

    try:
        head = head_of(model)
    except ValueError as error:
        return str(error)
    captured: list[torch.Tensor] = []
    handle = head.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach()))
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(batch, 3, size, size, generator=generator)
    try:
        with torch.no_grad():
            output = model.eval()(images)
    except Exception as error:  # noqa: BLE001 - any failure of the pass is a reason
        return f"the pass at {size} px fails: {type(error).__name__}"
    finally:
        handle.remove()
    if not isinstance(output, torch.Tensor) or output.shape != (batch, N_CLASSES):
        return "the output is not one row of 1,000 values for each image"
    if len(captured) != 1 or captured[0].ndim != 2 or len(captured[0]) != batch:
        return "the head does not take one vector for each image, once"
    with torch.no_grad():
        again = head(captured[0])
    scale = float(output.abs().max().clamp_min(1e-12))
    gap = float((again - output).abs().max()) / scale
    return "" if gap <= HEAD_LIMIT else (
        f"the head on its input differs from the output (relative {gap:.1e})")


def timm_problem(entry: dict) -> str:
    """The head check of one weight with random weights (no download), or the
    documented reason that removes it."""
    import timm

    if entry["id"] in tags.NOT_ELIGIBLE:
        return tags.NOT_ELIGIBLE[entry["id"]]
    torch.manual_seed(0)
    try:
        model = timm.create_model(entry["model"], pretrained=False)
    except Exception as error:  # noqa: BLE001 - a network that cannot be made
        return f"the network cannot be made: {type(error).__name__}"
    return head_problem(model, entry["crop_size"][0])


def walk_order(names: list[str], lineage: str, seed: int = SEED) -> list[str]:
    """The names of one lineage in the order of its seeded permutation."""
    ordered = sorted(names)
    rng = np.random.default_rng([seed, zlib.crc32(lineage.encode())])
    return [ordered[index] for index in rng.permutation(len(ordered))]


def draw(rows: list[dict], problem: Callable[[dict], str],
         per_lineage: int = PER_LINEAGE, seed: int = SEED
         ) -> tuple[list[dict], list[dict]]:
    """The models of the three groups, and the weights that a check removed.

    Args:
        rows: The eligible weights of `candidates`.
        problem: The reason that removes a weight, or an empty text.

    Returns:
        The chosen entries (group R, then the lineages of N and S by name, each
        in its walk order), and one row for each weight that was looked at and
        removed.
    """
    by_id = {row["id"]: row for row in rows}
    chosen: list[dict] = []
    removed: list[dict] = []

    def passes(row: dict) -> bool:
        reason = problem(row)
        if reason:
            removed.append({"id": row["id"], "group": row["group"],
                            "lineage": row["lineage"], "reason": reason})
        return not reason

    recipes = sorted(row["id"] for row in rows
                     if row["architecture"] == RECIPE_ARCHITECTURE)
    for position, name in enumerate(recipes):
        if passes(by_id[name]):
            chosen.append({**by_id[name], "group": "R", "order": position})
    rest = [row for row in rows if row["architecture"] != RECIPE_ARCHITECTURE]
    for group in ("N", "S"):
        lineages = sorted({row["lineage"] for row in rest if row["group"] == group})
        for lineage in lineages:
            names = [row["id"] for row in rest
                     if row["group"] == group and row["lineage"] == lineage]
            taken: list[str] = []
            for position, name in enumerate(walk_order(names, lineage, seed)):
                row = by_id[name]
                if row["architecture"] in {by_id[t]["architecture"] for t in taken}:
                    continue
                if passes(row):
                    taken.append(name)
                    chosen.append({**row, "order": position})
                if len(taken) == per_lineage:
                    break
    return chosen, removed


def has_attention(model: torch.nn.Module) -> bool:
    """True if a layer of the network has a class name of self-attention over
    positions. It is a first reading: `zoo_timm_tags.TYPE_FIXES` corrects it."""
    for module in model.modules():
        name = type(module).__name__
        if name in tags.CHANNEL_ATTENTION:
            continue
        if any(mark in name for mark in tags.ATTENTION_MARKS):
            return True
    return False


def arch_type_of(model: torch.nn.Module) -> str:
    """`transformer`, `convolutional`, or `other`, from the layers of a network."""
    if has_attention(model):
        return "transformer"
    recurrent = any(isinstance(module, torch.nn.RNNBase) for module in model.modules())
    spatial = sum(1 for module in model.modules()
                  if isinstance(module, torch.nn.Conv2d)
                  and max(module.kernel_size) > 1)
    # One spatial convolution is the patch stem of a network of linear layers.
    return "other" if recurrent or spatial <= 1 else "convolutional"


def with_tags(chosen: list[dict]) -> list[dict]:
    """The entries with `arch_type`, `recipe_tag`, and the source of each.

    Raises:
        ValueError: If a model has no recipe row.
    """
    import timm

    missing = [entry["id"] for entry in chosen if entry["id"] not in tags.RECIPES]
    if missing:
        raise ValueError("no recipe row for: " + ", ".join(missing))
    out = []
    for entry in chosen:
        found = arch_type_of(timm.create_model(entry["model"], pretrained=False))
        fixed = tags.TYPE_FIXES.get(entry["architecture"])
        recipe, source, evidence = tags.RECIPES[entry["id"]]
        out.append({**entry, "arch_type": fixed[0] if fixed else found,
                    "arch_type_code_test": found,
                    "arch_type_note": fixed[1] if fixed else "",
                    "recipe_tag": recipe,
                    "recipe_fields": {field: tags.ROWS[entry["id"]][field]
                                      for field in tags.RECIPE_FIELDS},
                    "recipe_source": source, "recipe_evidence": evidence})
    return out


def pilot_of(models: list[dict]) -> list[str]:
    """The three models of the pilot: the first transformer and the first
    convolutional model of group N in the order of the list, and ResNet-50 A1."""
    ids = []
    for kind in ("transformer", "convolutional"):
        ids.append(next(entry["id"] for entry in models
                        if entry["group"] == "N" and entry["arch_type"] == kind))
    return [*ids, f"{RECIPE_ARCHITECTURE}.a1_in1k"]


def table(results_file: Path) -> dict:
    """The content of the frozen file."""
    chosen, removed = draw(candidates(read_results(results_file)), timm_problem)
    models = with_tags(chosen)
    return {"amendment": "A17", "timm": TIMM, "results_file": RESULTS_URL,
            "results_sha256": RESULTS_SHA256, "seed": SEED,
            "per_lineage": PER_LINEAGE, "pilot": pilot_of(models),
            "models": models, "removed": removed}


def load(path: Path = PATH) -> list[dict]:
    return json.loads(Path(path).read_text())["models"]


def pilot(path: Path = PATH) -> list[str]:
    return json.loads(Path(path).read_text())["pilot"]


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("draw", "freeze"))
    parser.add_argument("--results-file", required=True,
                        help="results/results-imagenet.csv of timm " + TIMM)
    args = parser.parse_args()
    if args.step == "freeze":
        if PATH.exists():
            raise SystemExit(f"{PATH.name} exists: the list is made once "
                             f"(sha256 {file_hash(PATH)})")
        content = table(Path(args.results_file))
        PATH.write_text(json.dumps(content, indent=1) + "\n")
        print(f"wrote {PATH.name}, sha256 {file_hash(PATH)}; "
              f"{len(content['models'])} models")
        return
    rows = candidates(read_results(Path(args.results_file)))
    chosen, removed = draw(rows, timm_problem)
    print(f"{len(rows)} eligible weights by the metadata; {len(chosen)} chosen; "
          f"{len(removed)} removed by a check")
    for entry in chosen:
        print(f"{entry['group']} {entry['lineage']:28s} {entry['id']:52s} "
              f"{entry['crop_size'][0]:4d} px  top-1 {entry['acc1']:.2f}  "
              f"{'' if entry['id'] in tags.RECIPES else 'NO RECIPE ROW'}")
    for row in removed:
        print(f"removed {row['group']} {row['lineage']:20s} {row['id']:52s} "
              f"{row['reason']}")


if __name__ == "__main__":
    main()
