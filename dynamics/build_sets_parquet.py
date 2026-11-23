"""Amendment A15: the 32 new ID class sets, from the ImageNet parquet files.

Three steps. The first is on the HPC: it lists the image count of every class.

    python -m dynamics.build_sets_parquet --count --parquet-dir $PARQUET_DIR \
        --out imagenet_class_counts.json

The second is local and is done once: it draws the classes from these counts and
writes the frozen file `dynamics/frozen/a15_class_sets.json`.

    python -m dynamics.build_sets_parquet --draw --counts imagenet_class_counts.json

The third is on the HPC: it builds one folder for each set, `set01` to `set32`,
with the layout of imagenette2-160, and checks gate D.

    python -m dynamics.build_sets_parquet --parquet-dir $PARQUET_DIR --out $A15_SETS_DIR

Each set has 10 classes that are not Imagenette classes, not Imagewoof classes,
and not classes of amendment A14, and no class is in two sets. As in amendment
A14, the 1,350 images of a class are split with a fixed seed, so that a set has
9,469 training and 3,925 validation images. Images are resized as in
`build_subset.py`.

Protocol: ../documentation/several_id_sets_rule_amendment_2026-11-23.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq

from dynamics import build_class_sets as cs
from dynamics import build_subset as bs
from dynamics.config import A15_CLASS_SETS_PATH, A15_SETS

PATH = A15_CLASS_SETS_PATH
SEED = 15
#: A row of a parquet file: its row group, its place in the group, and its label.
Row = tuple[int, int, int, str]


def rows_of(path: Path) -> list[Row]:
    """Every row of one parquet file, with the file name that its image gets."""
    source = pq.ParquetFile(path)
    rows = []
    for group in range(source.num_row_groups):
        table = source.read_row_group(group, columns=["label", "image.path"])
        names = table["image"].combine_chunks().field("path").to_pylist()
        for row, (label, name) in enumerate(zip(table["label"].to_pylist(), names)):
            stem = Path(name or "").stem or f"{group:04d}_{row:06d}"
            rows.append((group, row, int(label), f"{Path(path).stem}__{stem}.JPEG"))
    return rows


def count_file(path: Path) -> Counter:
    """Images of each label in one parquet file."""
    return Counter(label for _, _, label, _ in rows_of(path))


def wanted_rows(job: tuple[Path, frozenset[int]]) -> tuple[str, list[Row]]:
    path, labels = job
    return str(path), [row for row in rows_of(path) if row[2] in labels]


def write_rows(job: tuple[str, str, dict[int, dict[int, str]], int]) -> int:
    """Write the chosen images of one parquet file. Returns their number."""
    path, out, chosen, size = job
    source = pq.ParquetFile(path)
    written = 0
    for group, rows in sorted(chosen.items()):
        order = sorted(rows)
        table = source.read_row_group(group, columns=["image"]).take(order)
        for row, image in zip(order, table["image"].to_pylist()):
            target = Path(out) / rows[row]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                bs.resize(image["bytes"], size).save(target, quality=bs.QUALITY)
            written += 1
    return written


def files_of(parquet_dir: Path) -> dict[str, list[Path]]:
    """The parquet files of each split folder (`train` and `val`).

    Raises:
        SystemExit: If a split has no file.
    """
    files = {folder: bs.split_files(Path(parquet_dir), prefix)
             for prefix, folder in bs.SPLITS.items()}
    if not files["train"] or not files["val"]:
        raise SystemExit(f"no train or validation parquet files in {parquet_dir}")
    return files


def count(parquet_dir: Path, workers: int = 8) -> dict[str, dict[str, int]]:
    """Images of every class in each split, by WordNet id."""
    files = files_of(parquet_dir)
    names = bs.label_names(files["train"])
    out = {}
    with Pool(workers) as pool:
        for split, paths in files.items():
            totals: Counter = Counter()
            for counts in pool.imap_unordered(count_file, paths):
                totals.update(counts)
            out[split] = {names[label]: totals[label] for label in sorted(names)}
    return out


def draw_file(counts: dict[str, dict[str, int]], n_sets: int = A15_SETS,
              size: int = cs.SET_SIZE) -> dict:
    """The content of the frozen file, from the image counts of every class."""
    earlier = cs.load()
    used = [name for classes in earlier["sets"].values() for name in classes]
    excluded = {"imagenette": earlier["excluded"]["imagenette"],
                "imagewoof": earlier["excluded"]["imagewoof"], "a14": sorted(used)}
    pool = cs.candidates(counts["train"], counts["val"],
                         [name for group in excluded.values() for name in group])
    sets = cs.draw_sets(pool, n_sets=n_sets, size=size, seed=SEED)
    every = [name for classes in sets.values() for name in classes]
    return {"amendment": "A15", "seed": SEED, "sets": sets,
            "train_images": cs.TRAIN_IMAGES, "val_images": cs.VAL_IMAGES,
            "candidates": len(pool), "source_classes": len(counts["train"]),
            "excluded": excluded, "class_list_sha256": cs.list_hash(every)}


def folder_of(number: int) -> str:
    return f"set{number:02d}"


def build(parquet_dir: Path, out: Path, sets: dict[str, list[str]], seed: int = SEED,
          totals: tuple[int, int] = (cs.TRAIN_IMAGES, cs.VAL_IMAGES),
          size: int = bs.SIZE, workers: int = 8) -> dict[int, dict]:
    """Write the folder of every class set. Returns the manifest of each set.

    Raises:
        SystemExit: If a class is not in the parquet files.
        ValueError: If a class has too few images or a file name twice.
    """
    files = files_of(parquet_dir)
    names = bs.label_names(files["train"])
    label_of = {wnid: label for label, wnid in names.items()}
    every = [name for classes in sets.values() for name in classes]
    unknown = sorted(set(every) - set(label_of))
    if unknown:
        raise SystemExit(f"classes that are not in the parquet files: {unknown}")
    labels = frozenset(label_of[name] for name in every)
    jobs = [(path, labels) for paths in files.values() for path in paths]
    found: dict[str, dict[str, tuple[str, int, int]]] = {name: {} for name in every}
    with Pool(workers) as pool:
        for done, (path, rows) in enumerate(pool.imap_unordered(wanted_rows, jobs), 1):
            for group, row, label, item in rows:
                if item in found[names[label]]:
                    raise ValueError(f"class {names[label]} has the file name "
                                     f"{item} twice")
                found[names[label]][item] = (path, group, row)
            if done % 40 == 0 or done == len(jobs):
                print(f"[sets] listed {done}/{len(jobs)} parquet files", flush=True)
    chosen: dict[str, dict[int, dict[int, str]]] = {}
    manifests = {}
    for number, classes in ((int(key), value) for key, value in sets.items()):
        sizes = zip(cs.class_sizes(totals[0], len(classes)),
                    cs.class_sizes(totals[1], len(classes)))
        lines: list[str] = []
        counts: dict[str, dict[str, int]] = {split: {} for split in cs.SPLITS}
        for index, (name, (n_train, n_val)) in enumerate(zip(classes, sizes)):
            parts = cs.split_class(list(found[name]), n_train, n_val,
                                   (seed, number, index))
            for split, items in zip(cs.SPLITS, parts):
                counts[split][name] = len(items)
                for item in items:
                    path, group, row = found[name][item]
                    place = f"{folder_of(number)}/{split}/{name}/{item}"
                    chosen.setdefault(path, {}).setdefault(group, {})[row] = place
                    lines.append(f"{split}/{name}/{item}")
        manifests[number] = {"set": number, "classes": classes, "counts": counts,
                             "split_sha256": cs.list_hash(sorted(lines))}
    work = [(path, str(out), rows, size) for path, rows in sorted(chosen.items())]
    written = 0
    with Pool(workers) as pool:
        for done, amount in enumerate(pool.imap_unordered(write_rows, work), 1):
            written += amount
            if done % 40 == 0 or done == len(work):
                print(f"[sets] wrote {written} images from {done}/{len(work)} "
                      "parquet files", flush=True)
    for number, manifest in manifests.items():
        (Path(out) / folder_of(number) / "split_manifest.json").write_text(
            json.dumps(manifest, indent=1))
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--count", action="store_true",
                        help="list the image count of every class (HPC)")
    parser.add_argument("--draw", action="store_true",
                        help="draw the classes and write the frozen file (once)")
    parser.add_argument("--counts", help="with --draw: the file of --count")
    parser.add_argument("--parquet-dir")
    parser.add_argument("--out", help="the counts file, or the folder of the sets")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.draw:
        if PATH.exists():
            raise SystemExit(f"{PATH.name} exists: the classes are drawn once "
                             f"(sha256 {cs.file_hash(PATH)})")
        if not args.counts:
            parser.error("--draw needs --counts")
        content = draw_file(json.loads(Path(args.counts).read_text()))
        PATH.write_text(json.dumps(content, indent=1) + "\n")
        print(f"wrote {PATH.name}, sha256 {cs.file_hash(PATH)}; "
              f"{content['candidates']} candidate classes")
        return
    if not args.parquet_dir or not args.out:
        parser.error("give --parquet-dir and --out, or --draw")
    if args.count:
        counts = count(Path(args.parquet_dir), args.workers)
        Path(args.out).write_text(json.dumps(counts, indent=1) + "\n")
        full = sum(1 for name, amount in counts["train"].items()
                   if amount >= cs.FULL_TRAIN and counts["val"].get(name, 0)
                   >= cs.FULL_VAL)
        print(f"[sets] {len(counts['train'])} classes, "
              f"{sum(counts['train'].values())} training and "
              f"{sum(counts['val'].values())} validation images, {full} classes "
              f"with every image; wrote {args.out}", flush=True)
        return
    frozen = cs.load(PATH)
    every = [name for classes in frozen["sets"].values() for name in classes]
    if (cs.list_hash(every) != frozen["class_list_sha256"]
            or len(set(every)) != len(every)):
        raise SystemExit("the frozen class lists do not have the recorded hash")
    forbidden = [name for group in frozen["excluded"].values() for name in group]
    print(f"[sets] {PATH.name} sha256 {cs.file_hash(PATH)}", flush=True)
    manifests = build(Path(args.parquet_dir), Path(args.out), frozen["sets"],
                      frozen["seed"], workers=args.workers)
    failed = False
    for number, manifest in sorted(manifests.items()):
        problems = cs.check_set(Path(args.out) / folder_of(number),
                                manifest["classes"], forbidden)
        failed |= bool(problems)
        print(f"[sets] {folder_of(number)}: "
              f"{sum(manifest['counts']['train'].values())} training and "
              f"{sum(manifest['counts']['val'].values())} validation images, split "
              f"sha256 {manifest['split_sha256']}, gate D "
              + ("passes" if not problems else f"FAILS: {problems}"), flush=True)
    if failed:
        raise SystemExit("gate D fails")
    print("[sets] done", flush=True)


if __name__ == "__main__":
    main()
