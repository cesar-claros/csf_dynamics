"""Print when each run first reaches 100% training accuracy (gate I of Stage D).

The value is the accuracy on the training set in evaluation mode, without
augmentation, at each evaluated checkpoint. Run on the HPC after the first runs:

    python -m dynamics.interpolation --out-root $DYN_ROOT --runs D1_seed0 D2_seed0

Protocol: ../documentation/time_and_collapse_protocol_2026-11-12.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamics.config import DEFAULT_PRECISION, PRECISIONS, results_folder


def train_accuracies(folder: Path) -> dict[int, float]:
    """Epoch to training accuracy, from the result files of one run."""
    found = {}
    for path in sorted(folder.glob("epoch_*.json")):
        out = json.loads(path.read_text())
        for row in out["descriptors"]:
            if row["group"] == "clock" and row["name"] == "train_accuracy":
                found[int(out["epoch"])] = float(row["value"])
    return dict(sorted(found.items()))


def first_interpolation(accuracies: dict[int, float]) -> int | None:
    """The first epoch with 100% training accuracy, or None."""
    reached = [epoch for epoch, value in accuracies.items() if value >= 1.0]
    return min(reached) if reached else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--suite", default="step0")
    parser.add_argument("--precision", choices=sorted(PRECISIONS),
                        default=DEFAULT_PRECISION)
    parser.add_argument("--from-epoch", type=int, default=60,
                        help="first checkpoint to list")
    args = parser.parse_args()
    tag = results_folder(args.precision, suite=args.suite)
    for run in args.runs:
        accuracies = train_accuracies(Path(args.out_root) / "runs" / run / tag)
        first = first_interpolation(accuracies)
        print(f"{run}: " + ("no result files" if not accuracies else
                            f"t0 = {first}" if first is not None else
                            f"never 100% in {len(accuracies)} checkpoints"))
        for epoch, value in accuracies.items():
            if epoch >= args.from_epoch:
                print(f"  epoch {epoch:3d}  train accuracy {value:.5f}")


if __name__ == "__main__":
    main()
