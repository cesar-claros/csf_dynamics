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
    EPOCHS,
    IMAGE_SIZE,
    STAGES,
    DataPaths,
    RunSpec,
    checkpoint_epochs,
    parse_run,
)
from dynamics.evaluate import RESULTS, evaluate_run
from dynamics.train import train_run

RECHECK = "results_recheck"


def recheck_epochs(epochs: int) -> list[int]:
    """An early, a middle, and the final checkpoint."""
    kept = checkpoint_epochs(epochs)
    return sorted({kept[min(4, len(kept) - 1)], kept[len(kept) // 2], kept[-1]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=sorted(STAGES))
    parser.add_argument("--only", nargs="+", metavar="RUN",
                        help="run names, in place of or within a stage")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--imagenette-dir", required=True)
    parser.add_argument("--imagewoof-dir")
    parser.add_argument("--textures-dir")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--recheck", action="store_true",
                        help="gate G3: evaluate 3 checkpoints of seed-0 runs again")
    parser.add_argument("--keep-features", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="smoke runs only; the protocol value is the default")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                        help="smoke runs only")
    parser.add_argument("--limit", type=int, help="smoke runs only: images per set")
    args = parser.parse_args()
    if args.stage is None and not args.only:
        parser.error("give --stage or --only")
    smoke = (args.epochs, args.image_size, args.limit) != (EPOCHS, IMAGE_SIZE, None)
    if not smoke and not (args.imagewoof_dir and args.textures_dir):
        parser.error("protocol runs need --imagewoof-dir and --textures-dir")

    specs: list[RunSpec] = STAGES[args.stage] if args.stage else []
    if args.only:
        wanted = [parse_run(name) for name in args.only]
        specs = [spec for spec in specs if spec in wanted] if args.stage else wanted
    paths = DataPaths(Path(args.imagenette_dir),
                      Path(args.imagewoof_dir) if args.imagewoof_dir else None,
                      Path(args.textures_dir) if args.textures_dir else None)
    out_root, device = Path(args.out_root), torch.device(args.device)
    shared = {"workers": args.workers, "limit": args.limit,
              "image_size": args.image_size}
    print(f"[stage] {len(specs)} runs on {device}"
          + (" (SMOKE SETTINGS, not a protocol run)" if smoke else ""), flush=True)
    failures = []
    for spec in specs:
        try:
            if args.recheck:
                if spec.seed == 0 and spec.held_out is None:
                    evaluate_run(spec, paths, out_root, device,
                                 epochs=recheck_epochs(args.epochs), tag=RECHECK,
                                 **shared)
                continue
            if not args.skip_train:
                train_run(spec, paths, out_root, device, epochs=args.epochs, **shared)
            if not args.skip_eval:
                evaluate_run(spec, paths, out_root, device, tag=RESULTS,
                             keep_features=args.keep_features, **shared)
        except Exception:  # noqa: BLE001 - continue with the next run
            failures.append(spec.name)
            print(f"[stage] {spec.name} FAILED\n{traceback.format_exc()}", flush=True)
    print("[stage] failures: " + (", ".join(failures) if failures else "none"),
          flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
