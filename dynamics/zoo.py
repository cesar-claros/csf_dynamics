"""Amendment A16: evaluate public ImageNet-1k models. Nothing is trained.

Run from code/ inside the x9 container on the HPC (see README.md):

    python -m dynamics.zoo --models pilot --imagenet-dir $IMAGENET1K_DIR \
        --textures-dir $TEXTURES_DIR --extra-set inaturalist=$OOD/inaturalist ... \
        --out-root $DYN_ROOT

For each model of the frozen list `dynamics/frozen/a16_models.json`: the weights
are downloaded by torchvision, each image gets the preprocessing of the weight,
and the feature is the input of the last linear layer of the network. The 23
scores are fitted on the fixed training subset and scored on the validation
images, on two stand-in sets made from a fixed subset of the validation images
(class blends and shuffled patches), and on the OOD sets. One result file is
written for each model, <out-root>/zoo/<id>.json. A model with a result file is
skipped, so the command can restart. The log has one line for the features of
each set and one line for each score, with the seconds.

Protocol: ../documentation/imagenet1k_public_models_protocol_2026-11-24.md
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from dynamics import zoo_list
from dynamics.config import (
    DEFAULT_PRECISION,
    EXTRA_LIMIT,
    GRID,
    PRECISIONS,
)
from dynamics.data import (
    BlendSet,
    blend_partners,
    class_folders,
    list_images,
    patch_permutations,
    shuffle_patches,
    thin,
)
from dynamics.metrics import error_metrics, set_metrics
from dynamics.paper_api import PAPER_CSFS, paper_scores
from dynamics.scores import extra_scores

#: The three models of the pilot.
PILOT = ("resnet50-v1", "vit_b_16-v1", "efficientnet_b0-v1")
RESULTS = "zoo"
#: Validation images of each class in the two stand-in sets, and their seed.
STANDIN_PER_CLASS = 10
STANDIN_SEED = 16
#: Counts of a protocol run.
N_CLASSES, TRAIN_IMAGES, VAL_IMAGES = 1000, 100_000, 50_000
#: Set names that the evaluation makes itself.
RESERVED = ("train", "val", "blend", "shuffle", "textures", "id_errors")


def last_linear(model: torch.nn.Module) -> torch.nn.Linear:
    """The last linear layer of a network: its head.

    Raises:
        ValueError: If the network has no linear layer.
    """
    layers = [module for module in model.modules()
              if isinstance(module, torch.nn.Linear)]
    if not layers:
        raise ValueError("the network has no linear layer")
    return layers[-1]


def head_parameters(head: torch.nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
    """The weights and the bias of a head in float32. A head with no bias (the
    last layer of MaxViT) has a bias of 0."""
    weight = head.weight.detach().float().cpu()
    bias = (torch.zeros(head.out_features) if head.bias is None
            else head.bias.detach().float().cpu())
    return weight, bias


def load_model(entry: dict) -> tuple[torch.nn.Module, object]:
    """The network of one entry with its weights, and the preprocessing of the
    weights."""
    from torchvision import models

    weights = models.get_model_weights(entry["model"])[entry["weights"]]
    return models.get_model(entry["model"], weights=weights).eval(), weights.transforms()


class ImageFiles(Dataset):
    """Image files with labels and the preprocessing of one weight."""

    def __init__(self, paths: list[Path], labels: list[int], transform) -> None:
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.labels[index]


def id_files(root: Path, split: str, limit: int | None = None
             ) -> tuple[list[Path], list[int]]:
    """Files and labels of one ID split. Labels follow the sorted class folders,
    which is the class order of the public weights."""
    paths: list[Path] = []
    labels: list[int] = []
    classes = class_folders(Path(root) / split)
    for label, name in enumerate(classes):
        files = thin(list_images(Path(root) / split / name),
                     None if limit is None else max(1, limit // len(classes)))
        paths += files
        labels += [label] * len(files)
    return paths, labels


def standin_indices(labels: np.ndarray, per_class: int = STANDIN_PER_CLASS,
                    seed: int = STANDIN_SEED) -> np.ndarray:
    """Positions of `per_class` images of each class, drawn with a fixed seed."""
    rng = np.random.default_rng(seed)
    kept = []
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        kept.append(rng.choice(positions, size=min(per_class, len(positions)),
                               replace=False))
    return np.sort(np.concatenate(kept))


def shuffle_any(image: torch.Tensor, permutation: np.ndarray,
                grid: int = GRID) -> torch.Tensor:
    """Patch shuffle for any image size: the part that the grid divides is
    shuffled, and the rest of the border stays."""
    _, height, width = image.shape
    rows, cols = height - height % grid, width - width % grid
    out = image.clone()
    out[:, :rows, :cols] = shuffle_patches(image[:, :rows, :cols].contiguous(),
                                           permutation, grid)
    return out


class ShuffleAny(Dataset):
    """Each image with its patches permuted."""

    def __init__(self, base: Dataset, permutations: np.ndarray) -> None:
        self.base, self.permutations = base, permutations

    def __len__(self) -> int:
        return len(self.permutations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, _ = self.base[index]
        return shuffle_any(image, self.permutations[index]), -1


def batch_size_of(crop: int, base: int = 256) -> int:
    """A batch that needs about the memory of `base` images of 224 px."""
    return max(16, int(base * (224 / crop) ** 2))


def extract(model: torch.nn.Module, head: torch.nn.Linear, dataset: Dataset,
            device: torch.device, batch_size: int = 256, workers: int = 8,
            amp: bool = True) -> dict[str, torch.Tensor]:
    """The input of the head, the output of the network, and the labels.

    Raises:
        ValueError: If the input of the head is not one vector for each image.
    """
    captured: list[torch.Tensor] = []
    handle = head.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach()))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers)
    parts: dict[str, list[torch.Tensor]] = {"feature": [], "output": [], "labels": []}
    try:
        with torch.no_grad():
            for images, labels in loader:
                with torch.autocast(device_type=device.type,
                                    enabled=amp and device.type == "cuda"):
                    output = model(images.to(device))
                feature = captured.pop()
                if feature.ndim != 2 or captured:
                    raise ValueError("the head does not take one vector for each "
                                     "image, once for each batch")
                parts["feature"].append(feature.float().cpu())
                parts["output"].append(output.float().cpu())
                parts["labels"].append(torch.as_tensor(labels).long())
    finally:
        handle.remove()
    return {name: torch.cat(values) for name, values in parts.items()}


def timed_scores(train: dict, w: torch.Tensor, b: torch.Tensor,
                 evaluation: torch.Tensor, logits: torch.Tensor, n_classes: int,
                 device: torch.device | None, precision: str, label: str = ""
                 ) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, float]]:
    """The scores of `scores.all_scores`, one score at a time, with the seconds
    of each. A line is printed for each score, so that a log shows the progress.

    Each score has its own seed in the paper pipeline, so one call for each score
    gives the values of one call for all of them.
    """
    dtype = {"float32": torch.float32, "float64": torch.float64}[precision]
    converted = [tensor.to(dtype) for tensor in (
        train["feature"], train["logits"], w, b, evaluation, logits)]
    fit, fit_logits, w_fit, b_fit, rows, row_logits = converted
    scores: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    seconds: dict[str, float] = {}
    for csf in PAPER_CSFS:
        began = time.time()
        values, failed = paper_scores(fit, train["labels"], fit_logits, w_fit, b_fit,
                                      rows, row_logits, n_classes, precision,
                                      names=(csf,))
        scores.update(values)
        errors.update(failed)
        seconds[csf] = round(time.time() - began, 1)
        print(f"[zoo] {label} score {csf}: {seconds[csf]}s"
              + (f" FAILED: {failed[csf]}" if csf in failed else ""), flush=True)
    began = time.time()
    scores.update(extra_scores(train["feature"], evaluation, device))
    seconds["kNN, FeatureNorm"] = round(time.time() - began, 1)
    print(f"[zoo] {label} scores kNN and FeatureNorm: "
          f"{seconds['kNN, FeatureNorm']}s", flush=True)
    return scores, errors, seconds


def evaluate(features: dict[str, dict], w: torch.Tensor, b: torch.Tensor,
             n_classes: int, device: torch.device | None = None,
             precision: str = DEFAULT_PRECISION, label: str = "") -> dict:
    """The metrics of every score on every set, from the extracted features.

    The logits are the head on the feature in float32, so that they are exact for
    the scores. `checks` compares them with the output of the network.

    Args:
        features: Per set name, the dict of `extract`. The sets `train` and `val`
            are required. Every other set is an unseen set.
        label: The name of the model in the progress lines.
    """
    for extracted in features.values():
        extracted["logits"] = extracted["feature"] @ w.T + b
    train, val = features["train"], features["val"]
    unseen = {name: values for name, values in features.items()
              if name not in ("train", "val")}
    order = ["val", *unseen]
    sizes = [len(features[name]["logits"]) for name in order]
    bounds = np.cumsum([0, *sizes])
    scores, errors, seconds = timed_scores(
        train, w, b, torch.cat([features[name]["feature"] for name in order]),
        torch.cat([features[name]["logits"] for name in order]), n_classes, device,
        precision, label)
    correct = (val["logits"].argmax(dim=1) == val["labels"]).numpy()

    def part(values: np.ndarray, name: str) -> np.ndarray:
        position = order.index(name)
        return values[bounds[position]:bounds[position + 1]]

    metrics = []
    for csf, values in scores.items():
        for name in unseen:
            metrics.append({"set": name, "csf": csf, **set_metrics(
                part(values, "val"), correct, part(values, name))})
        metrics.append({"set": "id_errors", "csf": csf,
                        **error_metrics(part(values, "val"), correct)})
    clock = {
        "train_accuracy": float((train["logits"].argmax(dim=1) == train["labels"])
                                .double().mean()),
        "train_loss": float(F.cross_entropy(train["logits"].double(),
                                            train["labels"])),
        "val_accuracy": float(correct.mean()),
        "val_loss": float(F.cross_entropy(val["logits"].double(), val["labels"])),
    }
    return {
        "metrics": metrics,
        "descriptors": [{"group": "clock", "name": name, "layer": "", "value": value}
                        for name, value in clock.items()],
        "checks": {
            "head_gap": float((val["output"] - val["logits"]).abs().max()),
            "argmax_agreement": float((val["output"].argmax(dim=1)
                                       == val["logits"].argmax(dim=1)).double()
                                      .mean())},
        "errors": errors,
        "nonfinite": {csf: int((~np.isfinite(values)).sum())
                      for csf, values in scores.items()},
        "counts": dict(zip(order, sizes)) | {"train": len(train["logits"])},
        "val_accuracy": clock["val_accuracy"],
        "seconds_by_score": seconds,
    }


def datasets(transform, imagenet: Path, textures: Path | None,
             extra: list[tuple[str, Path]], extra_limit: int | None = EXTRA_LIMIT,
             limit: int | None = None) -> dict[str, Dataset]:
    """Every set of the evaluation, with the preprocessing of one weight.

    Raises:
        ValueError: If an extra set has a reserved or repeated name.
        FileNotFoundError: If a set has no images.
    """
    train_files, train_labels = id_files(imagenet, "train", limit)
    val_files, val_labels = id_files(imagenet, "val", limit)
    val = ImageFiles(val_files, val_labels, transform)
    kept = standin_indices(np.asarray(val_labels))
    subset = Subset(val, kept.tolist())
    sets: dict[str, Dataset] = {
        "train": ImageFiles(train_files, train_labels, transform),
        "val": val,
        "blend": BlendSet(subset, blend_partners(np.asarray(val_labels)[kept])),
        "shuffle": ShuffleAny(subset, patch_permutations(len(kept))),
    }
    unlabelled: dict[str, list[Path]] = {}
    if textures is not None:
        unlabelled["textures"] = thin(list_images(textures), limit)
    for name, folder in extra:
        if name in RESERVED or name in unlabelled:
            raise ValueError(f"the set name {name} is reserved or used twice")
        unlabelled[name] = thin(list_images(folder),
                                extra_limit if limit is None else limit)
    for name, files in unlabelled.items():
        if not files:
            raise FileNotFoundError(f"the set {name} has no images")
        sets[name] = ImageFiles(files, [-1] * len(files), transform)
    return sets


def evaluate_model(entry: dict, imagenet: Path, textures: Path | None,
                   extra: list[tuple[str, Path]], out_root: Path,
                   device: torch.device, workers: int = 8,
                   precision: str = DEFAULT_PRECISION, amp: bool = True,
                   extra_limit: int | None = EXTRA_LIMIT, limit: int | None = None,
                   batch_size: int | None = None) -> None:
    """Evaluate one model and write its result file. Does nothing if it exists.

    Raises:
        ValueError: If the head does not have one output for each ID class.
    """
    target = Path(out_root) / RESULTS / f"{entry['id']}.json"
    if target.exists():
        print(f"[zoo] {entry['id']}: already complete", flush=True)
        return
    began = time.time()
    model, transform = load_model(entry)
    model = model.to(device).eval()
    head = last_linear(model)
    sets = datasets(transform, imagenet, textures, extra, extra_limit, limit)
    n_classes = len(class_folders(Path(imagenet) / "train"))
    if head.out_features != n_classes:
        raise ValueError(f"the head has {head.out_features} outputs and the ID data "
                         f"have {n_classes} classes")
    size = batch_size or batch_size_of(entry["crop_size"][0])
    print(f"[zoo] {entry['id']}: {len(sets)} sets, "
          f"{sum(len(dataset) for dataset in sets.values())} images, batch {size}",
          flush=True)
    features = {}
    for name, dataset in sets.items():
        start = time.time()
        features[name] = extract(model, head, dataset, device, size, workers, amp)
        print(f"[zoo] {entry['id']} features of {name}: {len(dataset)} images, "
              f"{time.time() - start:.0f}s", flush=True)
    extracted = time.time()
    model.to("cpu")                     # the scores need the memory of the GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = evaluate(features, *head_parameters(head), n_classes, device, precision,
                   entry["id"])
    import torchvision

    out.update(model=entry["id"], run=entry["id"], epoch=1, n_classes=n_classes,
               feature_dim=int(head.in_features),
               published_accuracy=entry["acc1"] / 100, precision=precision, amp=amp,
               limit=limit, torchvision=torchvision.__version__,
               seconds_extract=round(extracted - began, 1),
               seconds=round(time.time() - began, 1))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(out))
    temporary.replace(target)
    failed = len(out["errors"]) + sum(1 for n in out["nonfinite"].values() if n)
    print(f"[zoo] {entry['id']}: val acc {out['val_accuracy']:.4f} (published "
          f"{out['published_accuracy']:.4f}), feature {out['feature_dim']}, "
          f"{failed} scores with a problem, {out['seconds']}s", flush=True)


def chosen(names: list[str], listed: list[dict]) -> list[dict]:
    """The entries for `pilot`, `all`, or a list of ids.

    Raises:
        ValueError: If an id is not in the frozen list.
    """
    by_id = {entry["id"]: entry for entry in listed}
    if names == ["all"]:
        return listed
    wanted = list(PILOT) if names == ["pilot"] else names
    unknown = [name for name in wanted if name not in by_id]
    if unknown:
        raise ValueError(f"not in the frozen list: {', '.join(unknown)}")
    return [by_id[name] for name in wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", required=True, metavar="ID",
                        help="`pilot`, `all`, or ids of the frozen list")
    parser.add_argument("--imagenet-dir", required=True,
                        help="folder of dynamics.extract_imagenet, with train/ and "
                             "val/")
    parser.add_argument("--textures-dir")
    parser.add_argument("--extra-set", action="append", default=[],
                        metavar="NAME=DIR", help="one more OOD set; repeat it")
    parser.add_argument("--extra-limit", type=int, default=EXTRA_LIMIT)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--precision", choices=sorted(PRECISIONS),
                        default=DEFAULT_PRECISION)
    parser.add_argument("--no-amp", action="store_true",
                        help="extract the features without mixed precision")
    parser.add_argument("--batch-size", type=int,
                        help="default: from the input size of the model")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--limit", type=int, help="smoke runs only: images per set")
    args = parser.parse_args()
    extra = []
    for item in args.extra_set:
        name, separator, folder = item.partition("=")
        if not separator or not name or not Path(folder).is_dir():
            parser.error(f"--extra-set needs NAME=DIR with an existing folder: {item}")
        extra.append((name, Path(folder)))
    if args.limit is None and not (args.textures_dir and extra):
        parser.error("protocol runs need --textures-dir and the extra OOD sets")
    try:
        entries = chosen(args.models, zoo_list.load())
    except ValueError as error:
        parser.error(str(error))
    imagenet = Path(args.imagenet_dir)
    if args.limit is None:
        counts = (len(class_folders(imagenet / "train")),
                  len(id_files(imagenet, "train")[0]), len(id_files(imagenet, "val")[0]))
        if counts != (N_CLASSES, TRAIN_IMAGES, VAL_IMAGES):
            parser.error(f"the ImageNet folder has {counts} classes, training "
                         "images, and validation images, not "
                         f"{(N_CLASSES, TRAIN_IMAGES, VAL_IMAGES)}")
    device = torch.device(args.device)
    print(f"[zoo] {len(entries)} models on {device}"
          + (" (SMOKE SETTINGS, not a protocol run)" if args.limit else ""),
          flush=True)
    failures = []
    for entry in entries:
        try:
            evaluate_model(entry, imagenet,
                           Path(args.textures_dir) if args.textures_dir else None,
                           extra, Path(args.out_root), device, args.workers,
                           args.precision, not args.no_amp, args.extra_limit,
                           args.limit, args.batch_size)
        except Exception:  # noqa: BLE001 - continue with the next model
            failures.append(entry["id"])
            print(f"[zoo] {entry['id']} FAILED\n{traceback.format_exc()}", flush=True)
    print("[zoo] failures: " + (", ".join(failures) if failures else "none"),
          flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
