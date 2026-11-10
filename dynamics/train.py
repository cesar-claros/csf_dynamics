"""Train one run and keep its checkpoints.

Writes under <out-root>/runs/<run>/: checkpoints/epoch_XXX.pt at the protocol
epochs (epoch 0 is the random start), train_log.csv, meta.json, and train.done.
A stopped run resumes from last.pt. The data order of an epoch depends only on the
seed and the epoch, so it is the same after a resume. The augmentation draws are
not, so a resumed run is not bit-identical to an uninterrupted run.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from dynamics.config import (
    BATCH_SIZE,
    EPOCHS,
    IMAGE_SIZE,
    MOMENTUM,
    RECIPES,
    WARMUP_EPOCHS,
    DataPaths,
    RunSpec,
    checkpoint_epochs,
    checkpoint_path,
    run_dir,
)
from dynamics.data import training_set
from dynamics.model import build_model, normalize


class EpochSampler(Sampler):
    """A shuffle whose order depends only on the seed and the epoch."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def learning_rate(step: int, steps_per_epoch: int, epochs: int, peak: float,
                  warmup_epochs: int = WARMUP_EPOCHS) -> float:
    """Linear warm-up to the peak, then cosine decay to 0 at the last step."""
    warm = warmup_epochs * steps_per_epoch
    total = epochs * steps_per_epoch
    if step < warm:
        return peak * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return 0.5 * peak * (1 + math.cos(math.pi * progress))


def save_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_run(spec: RunSpec, paths: DataPaths, out_root: Path, device: torch.device,
              epochs: int = EPOCHS, workers: int = 8, limit: int | None = None,
              image_size: int = IMAGE_SIZE, batch_size: int = BATCH_SIZE) -> None:
    """Train one run. Does nothing if the run is already complete."""
    folder = run_dir(out_root, spec.name)
    if (folder / "train.done").exists():
        print(f"[train] {spec.name}: already complete", flush=True)
        return
    recipe = RECIPES[spec.recipe]
    # The evaluation switches cuDNN to deterministic mode. Training does not need it.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    dataset, classes = training_set(paths, spec.held_out, image_size, recipe.augment,
                                    limit)
    torch.manual_seed(spec.seed)
    model = build_model(len(classes)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=recipe.lr, momentum=MOMENTUM,
                                weight_decay=recipe.weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sampler = EpochSampler(len(dataset), spec.seed)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=workers, pin_memory=use_amp,
                        persistent_workers=workers > 0)
    steps_per_epoch = len(loader)
    kept = set(checkpoint_epochs(epochs))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").write_text(json.dumps({
        "run": spec.name, "recipe": spec.recipe, "seed": spec.seed,
        "held_out": spec.held_out, "classes": classes, "epochs": epochs,
        "image_size": image_size, "batch_size": batch_size, "limit": limit,
        "train_images": len(dataset), **vars(recipe)}, indent=2))

    def checkpoint(epoch: int) -> None:
        state = {key: value.cpu() for key, value in model.state_dict().items()}
        save_atomic({"model": state, "epoch": epoch, "run": spec.name,
                     "classes": classes},
                    checkpoint_path(out_root, spec.name, epoch))

    start = 0
    last = folder / "last.pt"
    log = folder / "train_log.csv"
    if last.exists():
        state = torch.load(last, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start = state["epoch"]
        with log.open(newline="") as handle:
            rows = list(csv.reader(handle))
        with log.open("w", newline="") as handle:
            csv.writer(handle).writerows(
                [rows[0], *(row for row in rows[1:] if int(row[0]) <= start)])
        print(f"[train] {spec.name}: resume after epoch {start}", flush=True)
    else:
        checkpoint(0)
        with log.open("w", newline="") as handle:
            csv.writer(handle).writerow(["epoch", "lr", "loss", "accuracy", "seconds"])

    for epoch in range(start + 1, epochs + 1):
        model.train()
        sampler.epoch = epoch
        torch.manual_seed(spec.seed * 100003 + epoch)
        began = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        lr = recipe.lr
        for index, (images, labels) in enumerate(loader):
            lr = learning_rate((epoch - 1) * steps_per_epoch + index,
                               steps_per_epoch, epochs, recipe.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(normalize(images))
                loss = F.cross_entropy(logits, labels,
                                       label_smoothing=recipe.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
            seen += len(labels)
        if not math.isfinite(loss_sum):
            raise RuntimeError(f"{spec.name}: the loss is not finite at epoch {epoch}")
        with log.open("a", newline="") as handle:
            csv.writer(handle).writerow([epoch, f"{lr:.6g}", f"{loss_sum / seen:.6f}",
                                         f"{correct / seen:.6f}",
                                         f"{time.time() - began:.1f}"])
        if epoch in kept:
            checkpoint(epoch)
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch}, last)
        print(f"[train] {spec.name} epoch {epoch}/{epochs} loss {loss_sum / seen:.4f} "
              f"acc {correct / seen:.4f} {time.time() - began:.0f}s", flush=True)
    last.unlink(missing_ok=True)
    (folder / "train.done").write_text("done\n")
