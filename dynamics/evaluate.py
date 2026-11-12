"""Evaluate the checkpoints of one run.

For each checkpoint: extract the features of every set, score all CSFs, and compute
the performance metrics, the ID-only descriptors, and the placement descriptors.
Writes one JSON file per checkpoint under <out-root>/runs/<run>/<tag>/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dynamics.config import (
    DEFAULT_PRECISION,
    IMAGE_SIZE,
    LAYERS,
    PENULTIMATE,
    DataPaths,
    RunSpec,
    checkpoint_path,
    results_folder,
    run_dir,
)
from dynamics.data import evaluation_sets, manifest
from dynamics.descriptors import (
    convention_descriptors,
    id_descriptors,
    placement_descriptors,
)
from dynamics.metrics import auroc, error_metrics, set_metrics
from dynamics.model import build_model, extract
from dynamics.paper_api import NC_METRICS, paper_nc
from dynamics.scores import all_scores, layer_scores


def independent_logit_scores(logits: torch.Tensor) -> dict[str, np.ndarray]:
    """MSR, MLS, and Energy straight from the logits, for gate G4."""
    logits = logits.double()
    return {
        "MSR": torch.softmax(logits, dim=1).max(dim=1).values.numpy(),
        "MLS": logits.max(dim=1).values.numpy(),
        "Energy": torch.logsumexp(logits, dim=1).numpy(),
    }


def evaluate_features(features: dict[str, dict], w: torch.Tensor, b: torch.Tensor,
                      n_classes: int, device: torch.device | None = None,
                      precision: str = DEFAULT_PRECISION) -> dict:
    """All outputs of one checkpoint from its extracted features.

    Args:
        features: Per set name, the dict that `model.extract` returns. The sets
            `train` and `val` are required. Every other set is an unseen set.
        w: Head weights, (n_classes, d).
        b: Head bias.
        n_classes: Number of ID classes of this model.
        device: Device for the neighbor search.
        precision: Precision of the paper's feature CSFs, float32 or float64.

    Returns:
        A dict with `metrics`, `layer_metrics`, `descriptors`, `placement`,
        `checks`, `errors`, `layer_errors`, `nonfinite`, and `counts`.
        `layer_metrics` holds the head-free CSFs on the layers before the
        penultimate one.
    """
    train, val = features["train"], features["val"]
    unseen = {name: values for name, values in features.items()
              if name not in ("train", "val")}
    order = ["val", *unseen]
    sizes = [len(features[name]["logits"]) for name in order]
    bounds = np.cumsum([0, *sizes])
    evaluation = torch.cat([features[name][PENULTIMATE] for name in order])
    logits = torch.cat([features[name]["logits"] for name in order])
    scores, errors = all_scores(train[PENULTIMATE], train["labels"], train["logits"],
                                w, b, evaluation, logits, n_classes, device,
                                precision)
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

    layer_metrics = []
    layer_errors: dict[str, str] = {}
    for layer in LAYERS[:-1]:
        stacked = torch.cat([features[name][layer] for name in order])
        values_by_csf, failed = layer_scores(train[layer], train["labels"], stacked,
                                             n_classes, device, precision)
        layer_errors.update({f"{layer}:{csf}": text for csf, text in failed.items()})
        for csf, values in values_by_csf.items():
            for name in unseen:
                layer_metrics.append({"layer": layer, "set": name, "csf": csf,
                                      **set_metrics(part(values, "val"), correct,
                                                    part(values, name))})
            layer_metrics.append({"layer": layer, "set": "id_errors", "csf": csf,
                                  **error_metrics(part(values, "val"), correct)})

    independent = independent_logit_scores(logits)
    gaps = [abs(auroc(part(scores[csf], "val"), part(scores[csf], name))
                - auroc(part(values, "val"), part(values, name)))
            for csf, values in independent.items() for name in unseen]
    head_gap = float((train[PENULTIMATE] @ w.T + b - train["logits"]).abs().max())

    train_logits, val_logits = train["logits"].double(), val["logits"].double()
    clock = {
        "train_accuracy": float((train_logits.argmax(dim=1) == train["labels"])
                                .double().mean()),
        "train_loss": float(F.cross_entropy(train_logits, train["labels"])),
        "val_accuracy": float(correct.mean()),
        "val_loss": float(F.cross_entropy(val_logits, val["labels"])),
    }
    try:
        nc = paper_nc(train[PENULTIMATE], train["labels"], w, b, n_classes)
    except Exception as error:  # noqa: BLE001 - keep the other outputs
        nc = dict.fromkeys(NC_METRICS, float("nan"))
        errors["paper_nc"] = f"{type(error).__name__}: {error}"
    descriptors = [{"group": "nc_paper", "name": name, "layer": PENULTIMATE,
                    "value": value} for name, value in nc.items()]
    descriptors += [{"group": "clock", "name": name, "layer": "", "value": value}
                    for name, value in clock.items()]
    descriptors += id_descriptors(train, val, n_classes, device)
    return {
        "metrics": metrics,
        "layer_metrics": layer_metrics,
        "layer_errors": layer_errors,
        "descriptors": descriptors,
        "placement": (placement_descriptors(train, val, unseen, correct, n_classes,
                                            device)
                      + convention_descriptors(train, val, unseen, correct, w,
                                               n_classes)),
        "checks": {"g4_max_gap": float(np.nanmax(gaps)) if gaps else float("nan"),
                   "head_gap": head_gap},
        "errors": errors,
        "nonfinite": {csf: int((~np.isfinite(values)).sum())
                      for csf, values in scores.items()},
        "counts": dict(zip(order, sizes)) | {"train": len(train["logits"])},
        "precision": precision,
    }


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    state = torch.load(path, map_location="cpu")
    model = build_model(len(state["classes"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


def evaluate_run(spec: RunSpec, paths: DataPaths, out_root: Path,
                 device: torch.device, epochs: list[int] | None = None,
                 precision: str = DEFAULT_PRECISION, recheck: bool = False,
                 suite: str = "", workers: int = 4, limit: int | None = None,
                 image_size: int = IMAGE_SIZE, keep_features: bool = False) -> None:
    """Evaluate the checkpoints of one run; skip the checkpoints already evaluated.

    Args:
        epochs: Checkpoint epochs to evaluate. The default is every saved one.
        precision: Precision of the paper's feature CSFs. Each precision has its
            own results folder.
        recheck: Write to the folder of the second extraction, for gate G3.
        suite: Name of an evaluation with extra OOD sets. It has its own folder.

    Raises:
        FileNotFoundError: If the run has no checkpoint.
    """
    folder = run_dir(out_root, spec.name)
    tag = results_folder(precision, recheck, suite)
    saved = sorted(int(path.stem.split("_")[1])
                   for path in (folder / "checkpoints").glob("epoch_*.pt"))
    if not saved:
        raise FileNotFoundError(f"no checkpoints in {folder}")
    todo = [epoch for epoch in (saved if epochs is None else epochs)
            if epoch in saved
            and not (folder / tag / f"epoch_{epoch:03d}.json").exists()]
    if not todo:
        print(f"[eval] {spec.name} ({tag}): nothing to do", flush=True)
        return
    sets = evaluation_sets(paths, spec.held_out, image_size, limit, workers)
    (folder / tag).mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps(manifest(paths), indent=2))
    for epoch in todo:
        began = time.time()
        model, state = load_model(checkpoint_path(out_root, spec.name, epoch), device)
        features = {name: extract(model, dataset, device, workers=0)
                    for name, dataset in sets.items()}
        if keep_features:
            target = folder / "features" / f"epoch_{epoch:03d}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(features, target)
        out = evaluate_features(features, model.fc.weight.detach().float().cpu(),
                                model.fc.bias.detach().float().cpu(),
                                len(state["classes"]), device, precision)
        out.update(run=spec.name, epoch=epoch, n_classes=len(state["classes"]),
                   seconds=round(time.time() - began, 1))
        temporary = folder / tag / f"epoch_{epoch:03d}.tmp"
        temporary.write_text(json.dumps(out))
        temporary.replace(folder / tag / f"epoch_{epoch:03d}.json")
        accuracy = next(row["value"] for row in out["descriptors"]
                        if row["name"] == "val_accuracy")
        print(f"[eval] {spec.name} ({tag}) epoch {epoch}: val acc {accuracy:.4f}, "
              f"{len(out['errors']) + len(out['layer_errors'])} CSF errors, "
              f"{out['seconds']}s", flush=True)
