"""Checkpoint selection for OOD detection with ID-only rules.

Can an ID-only index choose a checkpoint better than a trivial rule? Every rule
chooses among the checkpoints whose validation accuracy is near the best of the
run. The regret of a rule is the target at the best such checkpoint minus the
target at the chosen one, in AUROC points.

Run from code/ on results that are already local:

    python -m dynamics.analyze_selection --results-root dyn_results

Protocol: ../documentation/time_and_collapse_protocol_2026-11-12.md, section
"Checkpoint-selection check". Everything here is exploratory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_e
from dynamics.analyze_e import block
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import DEFAULT_PRECISION, PENULTIMATE, PRECISIONS, results_folder

HERE = Path(__file__).resolve().parent
FAMILY = {"A": "plain", "D1": "plain", "C3": "plain",
          "B": "augmented", "D2": "augmented", "C1": "augmented", "C2": "augmented"}
ENVELOPE = "envelope"
TARGETS = (ENVELOPE, "kNN", "NCI", "GradNorm", "MSR", "Maha")
#: Index name: descriptor group, descriptor name, and the direction in which the
#: index moves during training (-1 falls, +1 rises).
INDICES = {"nc1": ("nc_paper", "var_collapse", -1),
           "cdnv_train": ("spread", "cdnv_train", -1),
           "nc1_val_over_train": ("gap", "var_collapse_val_over_train", +1)}
CLOCK = "clock"
FIXED_RULES = ("final", "val_acc", "val_loss", "trivial")
SCHEMES = ("within", "across")
#: A checkpoint is eligible if its validation accuracy is this close to the best.
ACCURACY_SLACK = 0.01
#: Stop criterion: an index must beat `trivial` and `clock` by these margins.
MARGIN_TRIVIAL = 1.0
MARGIN_CLOCK = 0.5


def run_frames(tables: analyze_e.Tables, sets: tuple[str, ...]
               ) -> dict[str, pd.DataFrame]:
    """Per run, one row per checkpoint: clock values, indices, targets, eligibility."""
    clock = tables.descriptors[tables.descriptors["group"] == "clock"].pivot_table(
        index=["run", "epoch"], columns="name", values="value")
    for name, (group, descriptor, _) in INDICES.items():
        rows = tables.descriptors[
            (tables.descriptors["group"] == group)
            & (tables.descriptors["name"] == descriptor)
            & (tables.descriptors["layer"] == PENULTIMATE)]
        if not rows.empty:
            clock[name] = rows.set_index(["run", "epoch"])["value"]
    kept = tables.metrics[tables.metrics["set"].isin(sets)]
    best = kept.groupby(["run", "epoch", "set"])["auroc"].max()
    clock[ENVELOPE] = best.groupby(["run", "epoch"]).mean() * 100
    for csf in TARGETS[1:]:
        clock[csf] = (kept[kept["csf"] == csf].groupby(["run", "epoch"])["auroc"]
                      .mean() * 100)
    frames = {}
    for run, frame in clock.groupby(level="run"):
        frame = frame.droplevel("run").sort_index()
        frame["eligible"] = (frame["val_accuracy"]
                             >= frame["val_accuracy"].max() - ACCURACY_SLACK)
        frames[str(run)] = frame
    return frames


def first_eligible(frame: pd.DataFrame, condition: pd.Series) -> int:
    """The first eligible checkpoint that meets the condition, or the last checkpoint."""
    found = frame.index[frame["eligible"] & condition]
    return int(found[0]) if len(found) else int(frame.index[-1])


def fixed_choices(frame: pd.DataFrame) -> dict[str, int]:
    """The checkpoint of each rule that needs no calibration."""
    eligible = frame[frame["eligible"]]
    return {"final": int(frame.index[-1]),
            "val_acc": int(frame["val_accuracy"].idxmax()),
            "val_loss": int(eligible["val_loss"].idxmin()),
            "trivial": first_eligible(frame, frame["train_accuracy"] >= 1.0)}


def threshold_choice(frame: pd.DataFrame, column: str, threshold: float,
                     direction: int) -> int:
    """The first eligible checkpoint at which a rising or falling column passes a threshold."""
    values = (pd.Series(frame.index, index=frame.index) if column == CLOCK
              else frame[column])
    return first_eligible(frame, direction * (values - threshold) >= 0)


def regret(frame: pd.DataFrame, epoch: int, target: str) -> float:
    """The target at the best eligible checkpoint minus the target at `epoch`."""
    return float(frame.loc[frame["eligible"], target].max() - frame.loc[epoch, target])


def calibrate(frames: list[pd.DataFrame], column: str, direction: int,
              target: str) -> float:
    """The threshold with the lowest mean regret on the calibration runs.

    The candidates are the values of the column at the eligible checkpoints of the
    calibration runs. Of several best candidates, the one that training reaches
    first is taken. NaN if there is no candidate.
    """
    values = [frame.index[frame["eligible"]].to_numpy(dtype=float) if column == CLOCK
              else frame.loc[frame["eligible"], column].to_numpy(dtype=float)
              for frame in frames]
    candidates = np.unique(np.concatenate(values)) if values else np.empty(0)
    candidates = candidates[np.isfinite(candidates)]
    if direction < 0:
        candidates = candidates[::-1]
    best, lowest = float("nan"), np.inf
    for candidate in candidates:
        mean = float(np.mean([regret(frame, threshold_choice(
            frame, column, candidate, direction), target) for frame in frames]))
        if mean < lowest - 1e-12:
            best, lowest = float(candidate), mean
    return best


def evaluate(frames: dict[str, pd.DataFrame], family: dict[str, str]
             ) -> pd.DataFrame:
    """Regret of every rule on every run and target, in long format.

    A calibrated rule has one row per scheme: `within` sets its threshold on the
    other runs of the family of the test run, and `across` on the runs of the
    other families.
    """
    calibrated = {CLOCK: +1, **{name: direction for name, (_, _, direction)
                                in INDICES.items()}}
    rows = []
    for run, frame in frames.items():
        pools = {
            "within": [f for other, f in frames.items()
                       if other != run and family[other] == family[run]],
            "across": [f for other, f in frames.items()
                       if family[other] != family[run]]}
        fixed = fixed_choices(frame)
        for target in (name for name in TARGETS if name in frame):
            for rule, epoch in fixed.items():
                rows.append({"run": run, "family": family[run], "target": target,
                             "rule": rule, "scheme": "none", "epoch": epoch,
                             "regret": regret(frame, epoch, target)})
            for rule, direction in calibrated.items():
                for scheme, pool in pools.items():
                    usable = [f for f in pool if rule == CLOCK or rule in f]
                    if (rule != CLOCK and rule not in frame) or not usable:
                        continue
                    threshold = calibrate(usable, rule, direction, target)
                    if np.isnan(threshold):
                        continue
                    epoch = threshold_choice(frame, rule, threshold, direction)
                    rows.append({"run": run, "family": family[run],
                                 "target": target, "rule": rule, "scheme": scheme,
                                 "epoch": epoch, "threshold": threshold,
                                 "regret": regret(frame, epoch, target)})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, scheme: str, by_family: bool = False
              ) -> pd.DataFrame:
    """Mean regret per rule (rows) and target (columns), for one calibration scheme."""
    kept = results[results["scheme"].isin(["none", scheme])]
    keys = ["family", "rule"] if by_family else ["rule"]
    table = kept.pivot_table(index=keys, columns="target", values="regret")
    order = [rule for rule in (*FIXED_RULES, CLOCK, *INDICES)
             if rule in kept["rule"].unique()]
    table = table.reindex(columns=[t for t in TARGETS if t in table.columns])
    if by_family:
        return table.reindex(order, level="rule").reset_index()
    return table.reindex(order).reset_index()


def verdict(results: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """The stop criterion of the protocol, on the envelope with calibration inside the family."""
    kept = results[(results["target"] == ENVELOPE)
                   & results["scheme"].isin(["none", "within"])]
    mean = kept.groupby("rule")["regret"].mean()
    rows = []
    for name in INDICES:
        if name not in mean.index or CLOCK not in mean.index:
            continue
        rows.append({"index": name, "regret": mean[name],
                     "below_trivial": mean["trivial"] - mean[name],
                     "below_clock": mean[CLOCK] - mean[name]})
    table = pd.DataFrame(rows)
    if table.empty:
        return table, "not decided: no index has a calibration inside its family"
    table["useful"] = ((table["below_trivial"] >= MARGIN_TRIVIAL)
                       & (table["below_clock"] >= MARGIN_CLOCK))
    if table["useful"].any():
        names = ", ".join(table.loc[table["useful"], "index"])
        return table, f"an index is useful for selection: {names}"
    return table, "no index meets both conditions: stop the selection aim"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--suite", default="refcheck",
                        help="evaluation to read; an empty text reads the base one")
    parser.add_argument("--far-sets", nargs="+", default=list(FAR_SETS))
    parser.add_argument("--precision", choices=sorted(PRECISIONS),
                        default=DEFAULT_PRECISION)
    parser.add_argument("--out", default=str(HERE / "outputs" / "selection_report.md"))
    args = parser.parse_args()
    tables = analyze_e.read_runs(Path(args.results_root),
                                 results_folder(args.precision, suite=args.suite))
    if tables.metrics.empty:
        raise SystemExit(f"no results under {args.results_root}/runs")
    frames = run_frames(tables, tuple(args.far_sets))
    recipe = tables.metrics.groupby("run")["recipe"].first()
    family = {run: FAMILY.get(recipe[run], recipe[run]) for run in frames}
    results = evaluate(frames, family)
    table, decision = verdict(results)
    chosen = results[(results["target"] == ENVELOPE)
                     & results["scheme"].isin(["none", "within"])].pivot_table(
        index="run", columns="rule", values="epoch").astype(int)
    oracle = pd.Series({run: int(frame.loc[frame["eligible"], ENVELOPE].idxmax())
                        for run, frame in frames.items()}, name="oracle")
    chosen = chosen.join(oracle).reset_index()
    counts = pd.Series(family).value_counts().rename_axis("family").reset_index(
        name="runs")
    sections = [
        ("Runs", f"Far sets: {', '.join(args.far_sets)}. A checkpoint is eligible if "
                 f"its validation accuracy is at most {100 * ACCURACY_SLACK:.1f} "
                 "point below the best of its run.\n\n" + block(counts)),
        ("Mean regret, calibration inside the family",
         "Rows are rules and columns are targets, in AUROC points over all runs:"
         "\n\n" + block(summarize(results, "within"), 2) + "\n\nBy family:\n\n"
         + block(summarize(results, "within", by_family=True), 2)),
        ("Mean regret, calibration across families",
         block(summarize(results, "across"), 2) + "\n\nBy family:\n\n"
         + block(summarize(results, "across", by_family=True), 2)),
        ("Chosen checkpoints for the envelope",
         "`oracle` is the best eligible checkpoint:\n\n" + block(chosen)),
        ("Stop criterion",
         f"An index is useful if its mean envelope regret is at least "
         f"{MARGIN_TRIVIAL} point below `trivial` and at least {MARGIN_CLOCK} point "
         "below `clock`, with calibration inside the family:\n\n" + block(table, 2)
         + f"\n\n**Result: {decision}.**"),
    ]
    out_dir = Path(args.out).parent / Path(args.out).stem.replace("_report", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "regrets.csv", index=False)
    text = ("# Checkpoint selection with ID-only rules\n\nExploratory. The rules "
            "and the stop criterion were fixed before these results.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
