"""Amendment A14: the four new ID class sets of part 2.

Two steps. The first is local and is done once: it draws the classes from the
image counts that an earlier 100-class run recorded, and writes the frozen file.

    python -m dynamics.build_class_sets --draw \
        --counts dyn_results/runs/H0_seed0/manifest.json

The second is on the HPC: it builds one folder for each set from the 100-class
folder of Stage H, with the layout of imagenette2-160, and checks gate D.

    python -m dynamics.build_class_sets --source $IMAGENET100_DIR --out $A14_SETS_DIR

Each set has 10 classes that are not Imagenette or Imagewoof classes, and no
class is in two sets. The 1,350 images of a class (its ImageNet training and
validation images) are split with a fixed seed, so that a set has 9,469 training
and 3,925 validation images, the counts of Imagenette.

Protocol: ../documentation/random_recipes_new_classes_amendment_2026-11-22.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from dynamics.config import A14_TRAIN_IMAGES, A14_VAL_IMAGES, IMAGENETTE_CLASSES
from dynamics.data import IMAGE_SUFFIXES

HERE = Path(__file__).resolve().parent
PATH = HERE / "frozen" / "a14_class_sets.json"
SEED = 14
N_SETS = 4
SET_SIZE = 10
#: Image counts of Imagenette, which each new set takes.
TRAIN_IMAGES = A14_TRAIN_IMAGES
VAL_IMAGES = A14_VAL_IMAGES
#: A class needs every ImageNet image: 1,300 for training and 50 for validation.
FULL_TRAIN = 1300
FULL_VAL = 50
SPLITS = ("train", "val")


def candidates(train: dict[str, int], val: dict[str, int],
               exclude: list[str]) -> list[str]:
    """Classes with every ImageNet image that are not in `exclude`. Sorted."""
    return sorted(name for name, count in train.items()
                  if count >= FULL_TRAIN and val.get(name, 0) >= FULL_VAL
                  and name not in set(exclude))


def draw_sets(pool: list[str], n_sets: int = N_SETS, size: int = SET_SIZE,
              seed: int = SEED) -> dict[str, list[str]]:
    """`n_sets` sets of `size` classes with no class in two sets, each sorted.

    Raises:
        ValueError: If the pool is too small.
    """
    if len(pool) < n_sets * size:
        raise ValueError(f"{len(pool)} classes, need {n_sets * size}")
    drawn = np.random.default_rng(seed).choice(sorted(pool), size=n_sets * size,
                                               replace=False).tolist()
    return {str(index + 1): sorted(drawn[index * size:(index + 1) * size])
            for index in range(n_sets)}


def class_sizes(total: int, classes: int = SET_SIZE) -> list[int]:
    """Images of each class, in class order: equal, and the first take the rest."""
    base, rest = divmod(total, classes)
    return [base + (index < rest) for index in range(classes)]


def split_class(names: list[str], n_train: int, n_val: int,
                key: tuple[int, ...]) -> tuple[list[str], list[str]]:
    """A random split of the file names of one class. `key` seeds the split.

    Raises:
        ValueError: If the class has too few images.
    """
    pool = sorted(names)
    if len(pool) < n_train + n_val:
        raise ValueError(f"{len(pool)} images, need {n_train + n_val}")
    order = np.random.default_rng(list(key)).permutation(len(pool))
    return (sorted(pool[i] for i in order[:n_train]),
            sorted(pool[i] for i in order[n_train:n_train + n_val]))


def list_hash(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path = PATH) -> dict:
    return json.loads(Path(path).read_text())


def images_of(folder: Path) -> dict[str, Path]:
    """Image files of one class folder, by file name."""
    return {path.name: path for path in sorted(Path(folder).iterdir())
            if path.suffix.lower() in IMAGE_SUFFIXES}


def build_set(source: Path, target: Path, number: int, classes: list[str],
              seed: int = SEED, link: bool = False,
              totals: tuple[int, int] = (TRAIN_IMAGES, VAL_IMAGES)) -> dict:
    """Write the folder of one class set. Returns its manifest.

    Raises:
        ValueError: If a class has too few images or a file name twice.
    """
    sizes = list(zip(class_sizes(totals[0], len(classes)),
                     class_sizes(totals[1], len(classes))))
    lines: list[str] = []
    counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for index, (name, (n_train, n_val)) in enumerate(zip(classes, sizes)):
        files: dict[str, Path] = {}
        for split in SPLITS:
            found = images_of(Path(source) / split / name)
            if set(found) & set(files):
                raise ValueError(f"class {name} has a file name in both splits")
            files.update(found)
        chosen = dict(zip(SPLITS, split_class(list(files), n_train, n_val,
                                              (seed, number, index))))
        for split, names in chosen.items():
            folder = Path(target) / split / name
            folder.mkdir(parents=True, exist_ok=True)
            for item in names:
                place = folder / item
                if place.exists() or place.is_symlink():
                    continue
                if link:
                    os.symlink(files[item].resolve(), place)
                else:
                    shutil.copy2(files[item], place)
            counts[split][name] = len(names)
            lines += [f"{split}/{name}/{item}" for item in names]
    manifest = {"set": number, "classes": classes, "counts": counts,
                "split_sha256": list_hash(sorted(lines))}
    (Path(target) / "split_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def check_set(folder: Path, classes: list[str], forbidden: list[str],
              totals: tuple[int, int] = (TRAIN_IMAGES, VAL_IMAGES)) -> list[str]:
    """Gate D for one folder. Returns the problems, and an empty list if none."""
    problems = []
    seen: dict[str, set[str]] = {}
    for split, total in zip(SPLITS, totals):
        root = Path(folder) / split
        found = sorted(path.name for path in root.iterdir() if path.is_dir()
                       ) if root.is_dir() else []
        if found != sorted(classes):
            problems.append(f"{split}: the class folders are not the frozen list")
            continue
        names = {name: set(images_of(root / name)) for name in found}
        count = sum(len(items) for items in names.values())
        if count != total:
            problems.append(f"{split}: {count} images, not {total}")
        seen[split] = {f"{name}/{item}" for name, items in names.items()
                       for item in items}
    if len(seen) == 2 and seen["train"] & seen["val"]:
        problems.append("an image is in the training and in the validation split")
    if set(classes) & set(forbidden):
        problems.append("a class is an Imagenette or an Imagewoof class")
    return problems


def draw_file(counts_path: Path) -> dict:
    """The content of the frozen file, from the manifest of a 100-class run.

    The manifest holds the image counts of the 100-class ID folder under the key
    `imagenette`, and the Imagewoof classes. The Imagenette classes come from the
    configuration.
    """
    stored = json.loads(Path(counts_path).read_text())
    train, val = stored["imagenette"]["train"], stored["imagenette"]["val"]
    woof = sorted(stored["imagewoof"]["train"])
    pool = candidates(train, val, [*IMAGENETTE_CLASSES, *woof])
    sets = draw_sets(pool)
    every = [name for classes in sets.values() for name in classes]
    return {"amendment": "A14", "seed": SEED, "sets": sets,
            "train_images": TRAIN_IMAGES, "val_images": VAL_IMAGES,
            "candidates": len(pool), "source_classes": len(train),
            "excluded": {"imagenette": list(IMAGENETTE_CLASSES), "imagewoof": woof},
            "class_list_sha256": list_hash(every)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--draw", action="store_true",
                        help="draw the classes and write the frozen file (once)")
    parser.add_argument("--counts", help="with --draw: manifest.json of a 100-class "
                                         "run")
    parser.add_argument("--source", help="the 100-class folder of Stage H")
    parser.add_argument("--out", help="folder that gets set1/ to set4/")
    parser.add_argument("--link", action="store_true",
                        help="symbolic links in place of copies")
    args = parser.parse_args()
    if args.draw:
        if PATH.exists():
            raise SystemExit(f"{PATH.name} exists: the classes are drawn once "
                             f"(sha256 {file_hash(PATH)})")
        if not args.counts:
            parser.error("--draw needs --counts")
        PATH.write_text(json.dumps(draw_file(Path(args.counts)), indent=1) + "\n")
        print(f"wrote {PATH.name}, sha256 {file_hash(PATH)}")
        return
    if not args.source or not args.out:
        parser.error("give --source and --out, or --draw")
    frozen = load()
    forbidden = [*frozen["excluded"]["imagenette"], *frozen["excluded"]["imagewoof"]]
    every = [name for classes in frozen["sets"].values() for name in classes]
    if list_hash(every) != frozen["class_list_sha256"] or len(set(every)) != len(every):
        raise SystemExit("the frozen class lists do not have the recorded hash")
    print(f"[sets] {PATH.name} sha256 {file_hash(PATH)}", flush=True)
    failed = False
    for number, classes in frozen["sets"].items():
        target = Path(args.out) / f"set{number}"
        manifest = build_set(Path(args.source), target, int(number), classes,
                             frozen["seed"], args.link)
        problems = check_set(target, classes, forbidden)
        failed |= bool(problems)
        print(f"[sets] set{number}: {sum(manifest['counts']['train'].values())} "
              f"training and {sum(manifest['counts']['val'].values())} validation "
              f"images, split sha256 {manifest['split_sha256']}, gate D "
              + ("passes" if not problems else f"FAILS: {problems}"), flush=True)
    if failed:
        raise SystemExit("gate D fails")
    print("[sets] done", flush=True)


if __name__ == "__main__":
    main()
