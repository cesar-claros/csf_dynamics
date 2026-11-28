"""Amendment A18, part 1: evaluate candidate stand-in tasks on 40 public models.

Run from code/ inside the x9 container on the HPC (see README.md):

    python -m dynamics.zoo_pilot --models pilot --imagenet-dir $IMAGENET1K_DIR \
        --out-root $DYN_ROOT

For each model of the frozen list `dynamics/frozen/a18_pilot_models.json`: the
features of the training subset and of the validation images, as in amendments
A16 and A17, and the features of each stand-in task. The three existing tasks
(ID errors, class blends, patches on a 4 by 4 grid) are computed again as a check
of the run. Each new task of `dynamics.standin` has 5 validation images of each
class. The 23 scores are fitted and scored as before. The OOD sets are not run:
the targets come from the stored result files of the same models. One result
file is written for each model, <out-root>/standin_pilot/<id>.json. A model with
a result file is skipped, so the command can restart.

Protocol: ../documentation/standin_tasks_pilot_plan_2026-11-27.md
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from dynamics import standin, standin_list, zoo
from dynamics.config import DEFAULT_PRECISION, PRECISIONS
from dynamics.data import BlendSet, blend_partners, class_folders, patch_permutations

RESULTS = "standin_pilot"
#: The existing image tasks, on the 10 images of each class of amendment A16.
OLD_TASKS = ("blend", "shuffle")


def normalization(entry: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The mean and the standard deviation of the preprocessing of one weight,
    each as (3, 1, 1)."""
    if entry.get("library") == "timm":
        from timm.data import resolve_data_config
        from timm.models import get_pretrained_cfg

        config = resolve_data_config(
            pretrained_cfg=get_pretrained_cfg(entry["model"]).to_dict(),
            use_test_size=True)
        mean, std = config["mean"], config["std"]
    else:
        from torchvision import models

        preset = models.get_model_weights(entry["model"])[entry["weights"]].transforms()
        mean, std = preset.mean, preset.std
    return (torch.tensor(mean, dtype=torch.float32).view(3, 1, 1),
            torch.tensor(std, dtype=torch.float32).view(3, 1, 1))


def task_sets(transform, imagenet: Path, mean: torch.Tensor, std: torch.Tensor,
              limit: int | None = None
              ) -> tuple[dict[str, Dataset], np.ndarray, np.ndarray]:
    """Every image set of the pilot with the preprocessing of one weight.

    Returns:
        The sets (train, val, the two existing image tasks, and the new image
        tasks), the positions in `val` of the images of the new tasks, and the
        partners of these images (positions inside the new subset).
    """
    train_files, train_labels = zoo.id_files(imagenet, "train", limit)
    val_files, val_labels = zoo.id_files(imagenet, "val", limit)
    labels = np.asarray(val_labels)
    val = zoo.ImageFiles(val_files, val_labels, transform)
    kept = zoo.standin_indices(labels)
    old = Subset(val, kept.tolist())
    sets: dict[str, Dataset] = {
        "train": zoo.ImageFiles(train_files, train_labels, transform),
        "val": val,
        "blend": BlendSet(old, blend_partners(labels[kept])),
        "shuffle": zoo.ShuffleAny(old, patch_permutations(len(kept))),
    }
    small = standin.subset_of(kept, labels)
    partners = standin.partners_of(labels[small])
    new = Subset(val, small.tolist())
    for task in standin.IMAGE_TASKS:
        sets[task] = standin.TaskSet(new, task, partners, mean, std)
    return sets, small, partners


def evaluate_model(entry: dict, imagenet: Path, out_root: Path, device: torch.device,
                   workers: int = 8, precision: str = DEFAULT_PRECISION,
                   amp: bool = True, limit: int | None = None,
                   batch_size: int | None = None) -> None:
    """Evaluate one model on every task and write its result file. Does nothing
    if it exists.

    Raises:
        ValueError: If the head does not have one output for each ID class.
    """
    target = Path(out_root) / RESULTS / f"{entry['id']}.json"
    if target.exists():
        print(f"[pilot] {entry['id']}: already complete", flush=True)
        return
    began = time.time()
    model, transform = zoo.load_model(entry)
    model = model.to(device).eval()
    head = zoo.head_of(model)
    n_classes = len(class_folders(Path(imagenet) / "train"))
    if head.out_features != n_classes:
        raise ValueError(f"the head has {head.out_features} outputs and the ID data "
                         f"have {n_classes} classes")
    sets, small, partners = task_sets(transform, imagenet, *normalization(entry),
                                      limit)
    size = batch_size or zoo.batch_size_of(entry["crop_size"][0])
    print(f"[pilot] {entry['id']}: {len(sets)} sets, "
          f"{sum(len(dataset) for dataset in sets.values())} images, batch {size}",
          flush=True)
    features = {}
    for name, dataset in sets.items():
        start = time.time()
        features[name] = zoo.extract(model, head, dataset, device, size, workers, amp)
        print(f"[pilot] {entry['id']} features of {name}: {len(dataset)} images, "
              f"{time.time() - start:.0f}s", flush=True)
    plain = features["val"]["feature"][torch.as_tensor(small)]
    features["featblend"] = {
        "feature": standin.feature_blend(plain, partners),
        "labels": torch.full((len(small),), -1, dtype=torch.long)}
    extracted = time.time()
    model.to("cpu")                     # the scores need the memory of the GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = zoo.evaluate(features, *zoo.head_parameters(head), n_classes, device,
                       precision, entry["id"])
    out.update(model=entry["id"], pool=entry["pool"], n_classes=n_classes,
               feature_dim=int(head.in_features),
               published_accuracy=entry["acc1"] / 100, precision=precision, amp=amp,
               limit=limit, old_tasks=["id_errors", *OLD_TASKS],
               new_tasks=list(standin.NEW_TASKS),
               seconds_extract=round(extracted - began, 1),
               seconds=round(time.time() - began, 1))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(out))
    temporary.replace(target)
    failed = len(out["errors"]) + sum(1 for n in out["nonfinite"].values() if n)
    print(f"[pilot] {entry['id']}: val acc {out['val_accuracy']:.4f} (published "
          f"{out['published_accuracy']:.4f}), feature {out['feature_dim']}, "
          f"{failed} scores with a problem, {out['seconds']}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", required=True, metavar="ID",
                        help="`pilot` (two models), `all`, or ids of the frozen list")
    parser.add_argument("--imagenet-dir", required=True,
                        help="folder of dynamics.extract_imagenet, with train/ and "
                             "val/")
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
    try:
        entries = zoo.chosen(args.models, standin_list.load(), standin_list.pilot()
                             if args.models == ["pilot"] else ())
    except ValueError as error:
        parser.error(str(error))
    imagenet = Path(args.imagenet_dir)
    if args.limit is None:
        counts = (len(class_folders(imagenet / "train")),
                  len(zoo.id_files(imagenet, "train")[0]),
                  len(zoo.id_files(imagenet, "val")[0]))
        wanted = (zoo.N_CLASSES, zoo.TRAIN_IMAGES, zoo.VAL_IMAGES)
        if counts != wanted:
            parser.error(f"the ImageNet folder has {counts} classes, training "
                         f"images, and validation images, not {wanted}")
    device = torch.device(args.device)
    print(f"[pilot] {len(entries)} models on {device}"
          + (" (SMOKE SETTINGS, not a protocol run)" if args.limit else ""),
          flush=True)
    failures = []
    for entry in entries:
        try:
            evaluate_model(entry, imagenet, Path(args.out_root), device, args.workers,
                           args.precision, not args.no_amp, args.limit,
                           args.batch_size)
        except Exception:  # noqa: BLE001 - continue with the next model
            failures.append(entry["id"])
            print(f"[pilot] {entry['id']} FAILED\n{traceback.format_exc()}",
                  flush=True)
    print("[pilot] failures: " + (", ".join(failures) if failures else "none"),
          flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
