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
from dynamics.model import build_model, extract, weight_norms
from dynamics.paper_api import NC_METRICS, paper_nc
from dynamics.scores import NO_FIT_CSFS, all_scores, layer_scores

#: Reference sets of the reference check. `train` is the standard one.
REFERENCES = ("train", "train_small", "heldout")
REFERENCE_SEED = 0


def independent_logit_scores(logits: torch.Tensor) -> dict[str, np.ndarray]:
    """MSR, MLS, and Energy straight from the logits, for gate G4."""
    logits = logits.double()
    return {
        "MSR": torch.softmax(logits, dim=1).max(dim=1).values.numpy(),
        "MLS": logits.max(dim=1).values.numpy(),
        "Energy": torch.logsumexp(logits, dim=1).numpy(),
    }


def reference_split(labels: torch.Tensor, n_classes: int,
                    seed: int = REFERENCE_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Fixed class-stratified halves of a set: the rows to fit on, and the rows to test."""
    rng = np.random.default_rng(seed)
    values = labels.numpy()
    fit: list[int] = []
    test: list[int] = []
    for c in range(n_classes):
        rows = rng.permutation(np.flatnonzero(values == c))
        half = len(rows) // 2
        fit += rows[:half].tolist()
        test += rows[half:].tolist()
    return np.sort(np.array(fit, dtype=np.int64)), np.sort(np.array(test,
                                                                    dtype=np.int64))


def matched_subset(labels: torch.Tensor, counts: np.ndarray,
                   seed: int = REFERENCE_SEED) -> np.ndarray:
    """Random rows with the given number of rows of each class, in row order.

    Raises:
        ValueError: If a class has fewer rows than its count.
    """
    rng = np.random.default_rng(seed + 1)
    values = labels.numpy()
    rows = []
    for c, count in enumerate(counts):
        candidates = np.flatnonzero(values == c)
        if len(candidates) < count:
            raise ValueError(f"class {c}: {len(candidates)} rows, {count} needed")
        rows.append(rng.choice(candidates, size=int(count), replace=False))
    return np.sort(np.concatenate(rows))


def reference_metrics(features: dict[str, dict], scores: dict[str, np.ndarray],
                      w: torch.Tensor, b: torch.Tensor, n_classes: int,
                      device: torch.device | None = None,
                      precision: str = DEFAULT_PRECISION
                      ) -> tuple[list[dict], dict[str, str], dict[str, float]]:
    """Metrics of every CSF under three reference sets, on one validation half.

    The validation set is split into a fit half and a test half. `train` fits on all
    training rows, `train_small` on a random training subset with the class counts
    of the fit half, and `heldout` on the fit half. Every reference is tested on the
    test half against each unseen set.

    Args:
        features: As in `evaluate_features`.
        scores: The scores of `evaluate_features` under the training reference, on
            the validation rows and then the rows of each unseen set.

    Returns:
        The metric rows, the error text of each failed CSF as `reference:csf`, and
        the checks `reference_nofit_gap`, `reference_fit`, and `reference_test`.
        The gap is the largest AUROC difference of a no-fit CSF between the
        references. It must be 0, because a no-fit CSF reads no reference.
    """
    train, val = features["train"], features["val"]
    unseen = [name for name in features if name not in ("train", "val")]
    fit, test = reference_split(val["labels"], n_classes)
    counts = np.bincount(val["labels"].numpy()[fit], minlength=n_classes)
    small = matched_subset(train["labels"], counts)
    correct = (val["logits"].argmax(dim=1) == val["labels"]).numpy()[test]
    sizes = [len(features[name]["logits"]) for name in unseen]
    limits = np.cumsum([0, len(test), *sizes])
    n_val = len(val["logits"])
    evaluation = torch.cat([val[PENULTIMATE][test],
                            *[features[name][PENULTIMATE] for name in unseen]])
    logits = torch.cat([val["logits"][test],
                        *[features[name]["logits"] for name in unseen]])
    found = {"train": {csf: np.concatenate([values[:n_val][test], values[n_val:]])
                       for csf, values in scores.items()}}
    errors: dict[str, str] = {}
    for name, rows, source in (("train_small", small, train), ("heldout", fit, val)):
        found[name], failed = all_scores(
            source[PENULTIMATE][rows], source["labels"][rows], source["logits"][rows],
            w, b, evaluation, logits, n_classes, device, precision)
        errors.update({f"{name}:{csf}": text for csf, text in failed.items()})
    out: list[dict] = []
    by_key: dict[tuple[str, str, str], float] = {}
    for reference in REFERENCES:
        for csf, values in found[reference].items():
            for position, name in enumerate(unseen, start=1):
                row = set_metrics(values[:limits[1]], correct,
                                  values[limits[position]:limits[position + 1]])
                out.append({"reference": reference, "set": name, "csf": csf, **row})
                by_key[reference, name, csf] = float(row["auroc"])
    gaps = [abs(by_key[reference, name, csf] - by_key["train", name, csf])
            for reference in REFERENCES[1:] for name in unseen for csf in NO_FIT_CSFS
            if (reference, name, csf) in by_key and ("train", name, csf) in by_key]
    checks = {"reference_nofit_gap": float(np.nanmax(gaps)) if gaps else float("nan"),
              "reference_fit": float(len(fit)), "reference_test": float(len(test))}
    return out, errors, checks


def evaluate_features(features: dict[str, dict], w: torch.Tensor, b: torch.Tensor,
                      n_classes: int, device: torch.device | None = None,
                      precision: str = DEFAULT_PRECISION,
                      reference_check: bool = False) -> dict:
    """All outputs of one checkpoint from its extracted features.

    Args:
        features: Per set name, the dict that `model.extract` returns. The sets
            `train` and `val` are required. Every other set is an unseen set.
        w: Head weights, (n_classes, d).
        b: Head bias.
        n_classes: Number of ID classes of this model.
        device: Device for the neighbor search.
        precision: Precision of the paper's feature CSFs, float32 or float64.
        reference_check: Also score every CSF with other reference sets (see
            `reference_metrics`).

    Returns:
        A dict with `metrics`, `layer_metrics`, `descriptors`, `placement`,
        `checks`, `errors`, `layer_errors`, `nonfinite`, `counts`,
        `reference_metrics`, and `reference_errors`. `layer_metrics` holds the
        head-free CSFs on the layers before the penultimate one. The two reference
        outputs are empty without `reference_check`.
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

    reference_rows: list[dict] = []
    reference_errors: dict[str, str] = {}
    reference_checks: dict[str, float] = {}
    if reference_check:
        reference_rows, reference_errors, reference_checks = reference_metrics(
            features, scores, w, b, n_classes, device, precision)

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
                   "head_gap": head_gap, **reference_checks},
        "reference_metrics": reference_rows,
        "reference_errors": reference_errors,
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
                 image_size: int = IMAGE_SIZE, keep_features: bool = False,
                 reference_check: bool = False) -> None:
    """Evaluate the checkpoints of one run; skip the checkpoints already evaluated.

    Args:
        epochs: Checkpoint epochs to evaluate. The default is every saved one.
        precision: Precision of the paper's feature CSFs. Each precision has its
            own results folder.
        recheck: Write to the folder of the second extraction, for gate G3.
        suite: Name of an evaluation with extra OOD sets. It has its own folder.
        reference_check: Also score every CSF with other reference sets.

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
                                len(state["classes"]), device, precision,
                                reference_check)
        out["descriptors"] += [{"group": "weights", "name": name, "layer": "",
                                "value": value}
                               for name, value in weight_norms(model).items()]
        out.update(run=spec.name, epoch=epoch, n_classes=len(state["classes"]),
                   seconds=round(time.time() - began, 1))
        temporary = folder / tag / f"epoch_{epoch:03d}.tmp"
        temporary.write_text(json.dumps(out))
        temporary.replace(folder / tag / f"epoch_{epoch:03d}.json")
        accuracy = next(row["value"] for row in out["descriptors"]
                        if row["name"] == "val_accuracy")
        failed = len(out["errors"]) + len(out["layer_errors"])
        extra = (f"{len(out['reference_errors'])} reference errors, "
                 if reference_check else "")
        print(f"[eval] {spec.name} ({tag}) epoch {epoch}: val acc {accuracy:.4f}, "
              f"{failed} CSF errors, {extra}{out['seconds']}s", flush=True)
