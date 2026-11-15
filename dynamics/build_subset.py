"""Build the 100-class ImageNet subset of Stage H from parquet files.

Reads the ImageNet-1k parquet files (columns `image` and `label`), keeps 100
classes, resizes each image so that its shorter side has 160 pixels, and writes
the folder layout of imagenette2-160: <out>/train/<wnid>/ and <out>/val/<wnid>/.

    python -m dynamics.build_subset --parquet-dir /path/to/imagenet1k/data \
        --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR \
        --out /path/to/imagenet100-160

The classes are the 10 Imagenette classes and 90 more, drawn with a fixed seed
from the other classes without the Imagewoof classes. The list is written to
`dynamics/frozen/imagenet100_classes.txt`. If that file exists, it is used.

Protocol: ../documentation/imagenet100_selection_protocol_2026-11-14.md
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

HERE = Path(__file__).resolve().parent
CLASSES_PATH = HERE / "frozen" / "imagenet100_classes.txt"
N_CLASSES = 100
SEED = 0
SIZE = 160
QUALITY = 95
#: Parquet file prefix of each split, and the folder that it becomes.
SPLITS = {"train": "train", "validation": "val"}
WNID = re.compile(r"n\d{8}")


def split_files(folder: Path, prefix: str) -> list[Path]:
    """The parquet files of one split, in name order."""
    return sorted(path for path in Path(folder).rglob("*.parquet")
                  if path.name.startswith(prefix))


def wnid_of(name: str | None) -> str | None:
    """The last WordNet id in a file name, or None."""
    found = WNID.findall(name or "")
    return found[-1] if found else None


def label_names(files: list[Path]) -> dict[int, str]:
    """Label index to WordNet id, from the file names of the training images.

    Raises:
        SystemExit: If the files give no name, or two ids for one label.
    """
    found: dict[int, set[str]] = {}
    for path in files:
        table = pq.ParquetFile(path).read(columns=["label", "image.path"])
        names = table["image"].combine_chunks().field("path").to_pylist()
        for label, name in zip(table["label"].to_pylist(), names):
            wnid = wnid_of(name)
            if wnid is not None:
                found.setdefault(int(label), set()).add(wnid)
    if not found:
        raise SystemExit("the parquet files hold no file name with a WordNet id, so "
                         "the labels cannot be mapped to class folders")
    mixed = {label: ids for label, ids in found.items() if len(ids) > 1}
    if mixed:
        raise SystemExit(f"labels with more than one WordNet id: {mixed}")
    return {label: next(iter(ids)) for label, ids in found.items()}


def choose_classes(available: list[str], keep: list[str], exclude: list[str],
                   count: int = N_CLASSES, seed: int = SEED) -> list[str]:
    """`keep`, and random other classes up to `count`, without `exclude`. Sorted.

    Raises:
        ValueError: If a class of `keep` is not available.
    """
    missing = sorted(set(keep) - set(available))
    if missing:
        raise ValueError(f"classes not in the parquet files: {missing}")
    pool = sorted(set(available) - set(keep) - set(exclude))
    extra = np.random.default_rng(seed).choice(pool, size=count - len(keep),
                                               replace=False)
    return sorted([*keep, *extra.tolist()])


def resize(data: bytes, size: int = SIZE) -> Image.Image:
    """An RGB image whose shorter side has `size` pixels."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    scale = size / min(image.size)
    target = (max(size, round(image.width * scale)),
              max(size, round(image.height * scale)))
    return image.resize(target, Image.Resampling.LANCZOS)


def extract(job: tuple[Path, str, dict[int, str], str, int]) -> Counter:
    """Write the kept images of one parquet file. Returns the count per class."""
    path, out, classes, split, size = job
    counts: Counter = Counter()
    source = pq.ParquetFile(path)
    for group in range(source.num_row_groups):
        labels = source.read_row_group(group, columns=["label"])["label"].to_pylist()
        rows = [row for row, label in enumerate(labels) if label in classes]
        if not rows:
            continue
        table = source.read_row_group(group, columns=["image"]).take(rows)
        for row, image in zip(rows, table["image"].to_pylist()):
            wnid = classes[labels[row]]
            folder = Path(out) / split / wnid
            folder.mkdir(parents=True, exist_ok=True)
            stem = Path(image.get("path") or "").stem or f"{group:04d}_{row:06d}"
            target = folder / f"{path.stem}__{stem}.JPEG"
            if not target.exists():
                resize(image["bytes"], size).save(target, quality=QUALITY)
            counts[wnid] += 1
    return counts


def class_list_hash(classes: list[str]) -> str:
    return hashlib.sha256("\n".join(classes).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--imagenette-dir", required=True)
    parser.add_argument("--imagewoof-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--classes-file", default=str(CLASSES_PATH))
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    files = {folder: split_files(Path(args.parquet_dir), prefix)
             for prefix, folder in SPLITS.items()}
    if not files["train"] or not files["val"]:
        raise SystemExit(f"no train or validation parquet files in {args.parquet_dir}")
    names = label_names(files["train"])
    print(f"[subset] {len(files['train'])} train files, {len(files['val'])} "
          f"validation files, {len(names)} classes", flush=True)
    listing = Path(args.classes_file)
    if listing.exists():
        chosen = listing.read_text().split()
    else:
        def folders(root: str) -> list[str]:
            return sorted(p.name for p in (Path(root) / "train").iterdir()
                          if p.is_dir())
        chosen = choose_classes(sorted(names.values()), folders(args.imagenette_dir),
                                folders(args.imagewoof_dir), count=N_CLASSES)
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text("\n".join(chosen) + "\n")
    unknown = sorted(set(chosen) - set(names.values()))
    if len(chosen) != N_CLASSES or unknown:
        raise SystemExit(f"class list has {len(chosen)} classes; unknown: {unknown}")
    print(f"[subset] class list sha256: {class_list_hash(chosen)}", flush=True)
    kept = {label: wnid for label, wnid in names.items() if wnid in chosen}
    summary = {}
    for split, paths in files.items():
        jobs = [(path, args.out, kept, split, args.size) for path in paths]
        with Pool(args.workers) as pool:
            totals: Counter = Counter()
            for done, counts in enumerate(pool.imap_unordered(extract, jobs), 1):
                totals.update(counts)
                if done % 20 == 0 or done == len(jobs):
                    print(f"[subset] {split}: {done}/{len(jobs)} files, "
                          f"{sum(totals.values())} images", flush=True)
        summary[split] = dict(sorted(totals.items()))
    sizes = {split: (len(counts), sum(counts.values()), min(counts.values()),
                     max(counts.values())) for split, counts in summary.items()}
    (Path(args.out) / "build_manifest.json").write_text(json.dumps({
        "classes": chosen, "class_list_sha256": class_list_hash(chosen),
        "size": args.size, "counts": summary}, indent=1))
    for split, (classes, images, low, high) in sizes.items():
        print(f"[subset] {split}: {classes} classes, {images} images, {low} to "
              f"{high} per class", flush=True)
    if any(size[0] != N_CLASSES for size in sizes.values()):
        raise SystemExit("a split does not have 100 classes")
    print("[subset] done", flush=True)


if __name__ == "__main__":
    main()
