"""Train and evaluate the runs of one stage, one run after the other.

Run from code/ inside the x9 container on the HPC (see README.md):

    python -m dynamics.run_stage --stage e_main --out-root "$DYN_ROOT" \
        --imagenette-dir ... --imagewoof-dir ... --textures-dir ...

A run that is complete is skipped, so the command can restart. `--recheck`
evaluates 3 checkpoints of each seed-0 run a second time, for gate G3.
Protocol: ../documentation/imagenette_dynamics_protocol_2026-11-10.md
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import torch

from dynamics.config import (
    DEFAULT_PRECISION,
    EXTRA_LIMIT,
    IMAGE_SIZE,
    PRECISIONS,
    RECIPES,
    STAGES,
    DataPaths,
    RunSpec,
    checkpoint_epochs,
    expected_classes,
    parse_run,
    set_folder,
)
from dynamics.data import class_folders
from dynamics.evaluate import evaluate_run, validation_pass
from dynamics.train import train_run


def recheck_epochs(epochs: int) -> list[int]:
    """An early, a middle, and the final checkpoint."""
    kept = checkpoint_epochs(epochs)
    return sorted({kept[min(4, len(kept) - 1)], kept[len(kept) // 2], kept[-1]})


def id_folder(spec: RunSpec, id_dir: Path | None, id_root: Path | None) -> Path:
    """The ID folder of one run.

    With `id_root`, a run that has a class set takes the folder of its set below
    it. Every other run takes `id_dir`.

    Raises:
        ValueError: If the run has no folder.
    """
    name = set_folder(spec.recipe)
    if id_root is not None and name is not None:
        return Path(id_root) / name
    if id_dir is None:
        raise ValueError(f"{spec.name} has no class set below --id-root; give "
                         "--id-dir")
    return Path(id_dir)


def wrong_folder(specs: list[RunSpec], id_dir: Path | None,
                 id_root: Path | None = None) -> list[str]:
    """Runs of amendments A14 and A15 whose ID classes are not those of their
    folder."""
    wrong = []
    for spec in specs:
        wanted = expected_classes(spec.recipe)
        if wanted is None:
            continue
        folder = id_folder(spec, id_dir, id_root) / "train"
        if not folder.is_dir() or tuple(class_folders(folder)) != wanted:
            wrong.append(spec.name)
    return wrong


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=sorted(STAGES))
    parser.add_argument("--only", nargs="+", metavar="RUN",
                        help="run names, in place of or within a stage")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--imagenette-dir", "--id-dir",
                        help="ID dataset folder with train/ and val/ class folders")
    parser.add_argument("--id-root",
                        help="folder with one ID folder for each class set (set1, "
                             "set01, ...); each run takes the folder of its set")
    parser.add_argument("--imagewoof-dir")
    parser.add_argument("--textures-dir")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--recheck", action="store_true",
                        help="gate G3: evaluate 3 checkpoints of seed-0 runs again")
    parser.add_argument("--precision", choices=sorted(PRECISIONS),
                        default=DEFAULT_PRECISION,
                        help="precision of the fitted CSFs; each has its own "
                             "results folder")
    parser.add_argument("--extra-set", action="append", default=[],
                        metavar="NAME=DIR",
                        help="one more OOD set; repeat the option for more sets")
    parser.add_argument("--extra-limit", type=int, default=EXTRA_LIMIT,
                        help="largest number of images of each extra set")
    parser.add_argument("--suite", default="",
                        help="name of an evaluation with extra sets, for example "
                             "step0; its results get their own folder")
    parser.add_argument("--keep-features", action="store_true")
    parser.add_argument("--valstats-only", action="store_true",
                        help="only store the validation statistics of the saved "
                             "checkpoints (amendment A7); needs the ID folder only")
    parser.add_argument("--eval-epochs", type=int, nargs="+", metavar="EPOCH",
                        help="evaluate only these checkpoints (default: every one)")
    parser.add_argument("--save-epochs", type=int, nargs="+", metavar="EPOCH",
                        help="keep only these checkpoints of a new run (default: "
                             "the protocol epochs); the list needs the last epoch")
    parser.add_argument("--reference-check", action="store_true",
                        help="also fit every CSF on a training subset and on one "
                             "half of the validation set; needs --suite")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--epochs", type=int,
                        help="smoke runs only; the default is the recipe's value")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                        help="smoke runs only")
    parser.add_argument("--limit", type=int, help="smoke runs only: images per set")
    args = parser.parse_args()
    if args.stage is None and not args.only:
        parser.error("give --stage or --only")
    if not args.imagenette_dir and not args.id_root:
        parser.error("give --id-dir or --id-root")
    smoke = (args.epochs, args.image_size, args.limit) != (None, IMAGE_SIZE, None)
    if not smoke and not args.valstats_only and not (args.imagewoof_dir
                                                     and args.textures_dir):
        parser.error("protocol runs need --imagewoof-dir and --textures-dir")
    if bool(args.extra_set) != bool(args.suite):
        parser.error("give --extra-set and --suite together, so that results with "
                     "extra sets stay apart from the base results")
    if args.reference_check and not args.suite:
        parser.error("give --reference-check with --suite, so that its results "
                     "stay apart from the earlier results")
    extra = []
    for item in args.extra_set:
        name, separator, folder = item.partition("=")
        if not separator or not name or not Path(folder).is_dir():
            parser.error(f"--extra-set needs NAME=DIR with an existing folder: {item}")
        extra.append((name, Path(folder)))

    specs: list[RunSpec] = STAGES[args.stage] if args.stage else []
    if args.only:
        wanted = [parse_run(name) for name in args.only]
        specs = [spec for spec in specs if spec in wanted] if args.stage else wanted
    id_dir = Path(args.imagenette_dir) if args.imagenette_dir else None
    id_root = Path(args.id_root) if args.id_root else None
    try:
        folders = {spec.name: id_folder(spec, id_dir, id_root) for spec in specs}
    except ValueError as error:
        parser.error(str(error))
    mismatch = wrong_folder(specs, id_dir, id_root)
    if mismatch and not smoke:
        parser.error("the ID folder does not have the classes of these runs: "
                     + ", ".join(mismatch))
    out_root, device = Path(args.out_root), torch.device(args.device)
    shared = {"workers": args.workers, "limit": args.limit,
              "image_size": args.image_size}
    print(f"[stage] {len(specs)} runs on {device}"
          + (" (SMOKE SETTINGS, not a protocol run)" if smoke else ""), flush=True)
    failures = []
    for spec in specs:
        paths = DataPaths(folders[spec.name],
                          Path(args.imagewoof_dir) if args.imagewoof_dir else None,
                          Path(args.textures_dir) if args.textures_dir else None,
                          tuple(extra), args.extra_limit)
        try:
            if args.valstats_only:
                validation_pass(spec, paths, out_root, device, **shared)
                continue
            if args.recheck:
                if spec.seed == 0 and spec.held_out is None:
                    evaluate_run(spec, paths, out_root, device,
                                 epochs=recheck_epochs(
                                     args.epochs or RECIPES[spec.recipe].epochs),
                                 precision=args.precision, recheck=True,
                                 suite=args.suite, **shared)
                continue
            if not args.skip_train:
                train_run(spec, paths, out_root, device, epochs=args.epochs,
                          keep_epochs=args.save_epochs, **shared)
            if not args.skip_eval:
                evaluate_run(spec, paths, out_root, device,
                             epochs=args.eval_epochs, precision=args.precision,
                             suite=args.suite, keep_features=args.keep_features,
                             reference_check=args.reference_check, **shared)
        except Exception:  # noqa: BLE001 - continue with the next run
            failures.append(spec.name)
            print(f"[stage] {spec.name} FAILED\n{traceback.format_exc()}", flush=True)
    print("[stage] failures: " + (", ".join(failures) if failures else "none"),
          flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
