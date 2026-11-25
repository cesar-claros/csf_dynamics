"""Amendment A16: the ImageNet images that the evaluation of public models reads.

    python -m dynamics.extract_imagenet --parquet-dir $PARQUET_DIR --out $IMAGENET1K_DIR

Writes every validation image to <out>/val/<wnid>/ and a fixed subset of the
training images, 100 for each class, to <out>/train/<wnid>/. The bytes of an
image are written as they are in the parquet files, with no resize and no new
encoding, because each public model has its own preprocessing. The subset is
drawn with a fixed seed from the sorted file names of each class, so it is the
same for every model. The script writes `extract_manifest.json` with the counts
and the hash of each list.

Protocol: ../documentation/imagenet1k_public_models_protocol_2026-11-24.md
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dynamics import build_class_sets as cs
from dynamics import build_sets_parquet as bp
from dynamics import build_subset as bs

PER_CLASS = 100
SEED = 16


def choose(names: list[str], count: int, key: tuple[int, ...]) -> list[str]:
    """`count` of the file names of one class, drawn with the seed `key`. Sorted.

    Raises:
        ValueError: If the class has too few images.
    """
    pool = sorted(names)
    if len(pool) < count:
        raise ValueError(f"{len(pool)} images, need {count}")
    picked = np.random.default_rng(list(key)).choice(len(pool), size=count,
                                                     replace=False)
    return sorted(pool[i] for i in picked)


def all_rows(path: Path) -> tuple[str, list[bp.Row]]:
    return str(path), bp.rows_of(path)


def write_raw(job: tuple[str, str, dict[int, dict[int, str]]]) -> int:
    """Write the chosen images of one parquet file as they are. Returns their
    number."""
    path, out, chosen = job
    source = pq.ParquetFile(path)
    written = 0
    for group, rows in sorted(chosen.items()):
        order = sorted(rows)
        table = source.read_row_group(group, columns=["image"]).take(order)
        for row, image in zip(order, table["image"].to_pylist()):
            target = Path(out) / rows[row]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(image["bytes"])
            written += 1
    return written


def extract(parquet_dir: Path, out: Path, per_class: int = PER_CLASS,
            seed: int = SEED, workers: int = 8) -> dict:
    """Write the two folders. Returns the manifest.

    Raises:
        ValueError: If a class has too few training images.
    """
    files = bp.files_of(parquet_dir)
    names = bs.label_names(files["train"])
    classes = sorted(names.values())
    position = {wnid: index for index, wnid in enumerate(classes)}
    listed: dict[str, dict[str, dict[str, tuple[str, int, int]]]] = {
        split: {wnid: {} for wnid in classes} for split in files}
    jobs = [(split, path) for split, paths in files.items() for path in paths]
    with Pool(workers) as pool:
        found = pool.imap(all_rows, [path for _, path in jobs])
        for done, ((split, _), (path, rows)) in enumerate(zip(jobs, found), 1):
            for group, row, label, item in rows:
                listed[split][names[label]][item] = (path, group, row)
            if done % 40 == 0 or done == len(jobs):
                print(f"[extract] listed {done}/{len(jobs)} parquet files",
                      flush=True)
    chosen: dict[str, dict[int, dict[int, str]]] = {}
    lines: dict[str, list[str]] = {split: [] for split in files}
    for split in files:
        for wnid in classes:
            items = listed[split][wnid]
            kept = (choose(list(items), per_class, (seed, position[wnid]))
                    if split == "train" else sorted(items))
            for item in kept:
                path, group, row = items[item]
                chosen.setdefault(path, {}).setdefault(group, {})[row] = (
                    f"{split}/{wnid}/{item}")
                lines[split].append(f"{wnid}/{item}")
    work = [(path, str(out), rows) for path, rows in sorted(chosen.items())]
    written = 0
    with Pool(workers) as pool:
        for done, amount in enumerate(pool.imap_unordered(write_raw, work), 1):
            written += amount
            if done % 40 == 0 or done == len(work):
                print(f"[extract] wrote {written} images from {done}/{len(work)} "
                      "parquet files", flush=True)
    manifest = {"classes": len(classes), "train_per_class": per_class, "seed": seed,
                "counts": {split: len(rows) for split, rows in lines.items()},
                "sha256": {split: cs.list_hash(sorted(rows))
                           for split, rows in lines.items()}}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "extract_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = extract(Path(args.parquet_dir), Path(args.out), workers=args.workers)
    counts = manifest["counts"]
    print(f"[extract] {manifest['classes']} classes, {counts['train']} training "
          f"images (sha256 {manifest['sha256']['train']}), {counts['val']} "
          f"validation images (sha256 {manifest['sha256']['val']})", flush=True)
    if (manifest["classes"] != 1000 or counts["train"] != 1000 * PER_CLASS
            or counts["val"] != 50000):
        raise SystemExit("the counts are not those of ImageNet-1k")
    print("[extract] done", flush=True)


if __name__ == "__main__":
    main()
