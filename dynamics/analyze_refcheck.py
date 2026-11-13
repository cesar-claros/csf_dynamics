"""Reference check: features or reference set?

A fitted score takes its statistics from a reference set. This analysis reads an
evaluation with `--reference-check` and compares the decline of each fitted score
under three references: all training features, a training subset, and one half of
the validation set. It also reports the collapse metrics without an inversion,
the weight norms, and which descriptors index the scores through a reset.

Run from code/ after copying the run folders from the HPC:

    python -m dynamics.analyze_refcheck --results-root dyn_results

Protocol: ../documentation/time_and_collapse_protocol_2026-11-12.md, section
"Reference check on the first runs". Everything here is exploratory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_d, analyze_e
from dynamics.analyze_e import block
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import (
    DEFAULT_PRECISION,
    PENULTIMATE,
    PRECISIONS,
    parse_run,
    results_folder,
)
from dynamics.evaluate import REFERENCES
from dynamics.scores import ALL_CSFS, NO_FIT_CSFS

HERE = Path(__file__).resolve().parent
SUITE = "refcheck"
#: The scores with a reading. Both are fitted, and both lost points in the pilot.
PRIMARY = ("kNN", "NCI")
FITTED = tuple(csf for csf in ALL_CSFS if csf not in NO_FIT_CSFS)
#: `start` of a run that never interpolates.
FALLBACK_START = 60
#: A reset: log10 NC1 at least this far above its running minimum.
BACK = 0.1
MIN_DECLINE = 2.0
REFERENCE_SHARE = 0.5
FEATURE_SHARE = 0.8
SIZE_TOLERANCE = 1.0
MONOTONE = 0.9
SHOWN = ("nc_paper/var_collapse", "spread/cdnv_train", "spread/cdnv_val",
         "spread/trace_ratio_train", "spread/within_inside_share_train",
         "spread/val_over_train", "spread/val_over_train_inside",
         "spread/val_over_train_outside", "gap/var_collapse_val_over_train",
         "weights/conv_norm", "weights/fc_norm")


def read_reference(root: Path, tag: str) -> pd.DataFrame:
    """The reference metrics of every run, with AUROC in points."""
    rows = []
    for folder in sorted((Path(root) / "runs").glob("*")):
        try:
            spec = parse_run(folder.name)
        except ValueError:
            continue
        for path in sorted((folder / tag).glob("epoch_*.json")):
            out = json.loads(path.read_text())
            rows += [{"run": spec.name, "recipe": spec.recipe, "epoch": out["epoch"],
                      **row} for row in out.get("reference_metrics", [])]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["auroc"] *= 100
    return frame


def collapse(descriptors: pd.DataFrame) -> pd.DataFrame:
    """log10 of the paper's NC1 per run and epoch, in the column `value`."""
    rows = descriptors[(descriptors["group"] == "nc_paper")
                       & (descriptors["name"] == "var_collapse")]
    out = rows[["run", "recipe", "epoch", "value"]].copy()
    out["value"] = np.log10(out["value"].clip(lower=1e-12))
    return out


def reference_points(clock: pd.DataFrame, nc1: pd.DataFrame,
                     back: float = BACK) -> pd.DataFrame:
    """Per run: `start`, `end`, `final`, and the first reset of collapse.

    `start` is t0, or the first checkpoint from `FALLBACK_START` if the run never
    interpolates. A reset is the first checkpoint after `start` whose log10 NC1 is
    at least `back` above its running minimum. `end` is the last checkpoint before
    the reset, or `final` if there is none.
    """
    t0 = analyze_d.interpolation_epochs(clock)
    rows = []
    for run, group in nc1.groupby("run"):
        series = group.set_index("epoch")["value"].sort_index()
        late = series.index[series.index >= FALLBACK_START]
        if run not in t0.index and len(late) == 0:
            continue
        start = int(t0[run]) if run in t0.index else int(late[0])
        tail = series[series.index >= start]
        low, reset = np.inf, np.nan
        for epoch, value in tail.items():
            if value > low + back:
                reset = epoch
                break
            low = min(low, value)
        before = tail.index[tail.index < reset] if not np.isnan(reset) else tail.index
        rows.append({"run": run, "recipe": group["recipe"].iloc[0], "start": start,
                     "end": int(before[-1]), "final": int(tail.index[-1]),
                     "reset": reset, "interpolates": run in t0.index})
    return pd.DataFrame(rows)


def declines(reference: pd.DataFrame, points: pd.DataFrame,
             sets: tuple[str, ...] = FAR_SETS) -> pd.DataFrame:
    """Per run, CSF, and reference: the far-set AUROC at the three points.

    `decline` is start minus end, and `decline_final` is start minus final.
    """
    columns = ["run", "recipe", "csf", "reference", "start", "end", "final",
               "decline", "decline_final"]
    if reference.empty or points.empty:
        return pd.DataFrame(columns=columns)
    kept = reference[reference["set"].isin(sets)]
    mean = (kept.groupby(["run", "recipe", "csf", "reference", "epoch"])["auroc"]
            .mean().reset_index())
    out = None
    for name in ("start", "end", "final"):
        at = mean.merge(points[["run", name]].rename(columns={name: "epoch"}),
                        on=["run", "epoch"]).rename(columns={"auroc": name})
        at = at.drop(columns="epoch")
        out = at if out is None else out.merge(
            at, on=["run", "recipe", "csf", "reference"])
    assert out is not None
    out["decline"] = out["start"] - out["end"]
    out["decline_final"] = out["start"] - out["final"]
    return out[columns]


def reading(train: float, small: float, heldout: float) -> str:
    """The reading that the protocol fixes for one score of one run."""
    if not np.isfinite([train, small, heldout]).all():
        return "no data"
    note = (" (the subset changes the decline)"
            if abs(small - train) > SIZE_TOLERANCE else "")
    if small < MIN_DECLINE:
        return "no decline to explain" + note
    if heldout <= REFERENCE_SHARE * small:
        return "reference set" + note
    if heldout >= FEATURE_SHARE * small:
        return "features" + note
    return "mixed" + note


def readings(table: pd.DataFrame, column: str = "decline") -> pd.DataFrame:
    """One row per run and primary score, with the three declines and the reading."""
    rows = []
    wide = table.pivot_table(index=["run", "csf"], columns="reference", values=column)
    for (run, csf), row in wide.iterrows():
        if csf not in PRIMARY:
            continue
        values = [row.get(name, np.nan) for name in REFERENCES]
        rows.append({"run": run, "csf": csf, **dict(zip(REFERENCES, values)),
                     "reading": reading(*values)})
    return pd.DataFrame(rows)


def index_table(tables: analyze_e.Tables) -> pd.DataFrame:
    """Every descriptor as one column per `group/name@layer`, by run and epoch.

    Penultimate-layer and layer-free descriptors have no layer suffix.
    """
    rows = tables.descriptors.copy()
    suffix = np.where(rows["layer"].isin(["", PENULTIMATE]), "",
                      "@" + rows["layer"].astype(str))
    rows["index"] = rows["group"] + "/" + rows["name"] + suffix
    wide = rows.pivot_table(index=["run", "epoch"], columns="index", values="value")
    if not tables.placement.empty:
        far = tables.placement[tables.placement["set"].isin(FAR_SETS)
                               & (tables.placement["layer"] == PENULTIMATE)].copy()
        far["index"] = "placement/" + far["name"]
        wide = wide.join(far.pivot_table(index=["run", "epoch"], columns="index",
                                         values="value"))
    return wide


def index_errors(metrics: pd.DataFrame, indices: pd.DataFrame, point: pd.Series,
                 nc1: pd.Series) -> tuple[pd.DataFrame, float, list[int]]:
    """How well each descriptor indexes the scores after a reset, for one run.

    The path is the checkpoints from `start` to `end`. The checkpoints after the
    reset count while their log10 NC1 is inside the range of the path. For each
    descriptor that is monotone on the path, the scores of such a checkpoint are
    predicted from the path at the same descriptor value, by linear interpolation.
    A descriptor that is positive at all of these checkpoints enters as its log10,
    so that NC1 and the other ratios are read on a log scale.

    Returns:
        The mean absolute error per descriptor in AUROC points over the 23 scores
        and the far sets, the same error for the clock (the scores at `end`), and
        the checkpoints after the reset that were used.
    """
    run = point["run"]
    wide = metrics[(metrics["run"] == run) & metrics["set"].isin(FAR_SETS)]
    wide = wide.pivot_table(index="epoch", columns=["csf", "set"], values="auroc") * 100
    epochs = np.array(sorted(wide.index))
    path = [e for e in epochs if point["start"] <= e <= point["end"]]
    after = []
    if not np.isnan(point["reset"]) and len(path) >= 3:
        low, high = nc1[path].min(), nc1[path].max()
        for epoch in epochs[epochs >= point["reset"]]:
            if not low <= nc1[epoch] <= high:
                break
            after.append(int(epoch))
    if not after:
        return pd.DataFrame(columns=["index", "monotone", "error"]), np.nan, after
    table = indices.loc[run]
    rows = []
    for name in table.columns:
        series = table[name]
        if series.loc[path + after].isna().any() or np.ptp(series.loc[path]) == 0:
            continue
        series = series.loc[path + after]
        if (series > 0).all():
            series = np.log10(series)
        x = series.loc[path].to_numpy()
        monotone = abs(pd.Series(x).corr(pd.Series(path, dtype=float),
                                         method="spearman"))
        if monotone < MONOTONE:
            continue
        order = np.argsort(x)
        errors = []
        for epoch in after:
            predicted = np.array([np.interp(series[epoch], x[order],
                                            wide.loc[path, column].to_numpy()[order])
                                  for column in wide.columns])
            errors.append(np.abs(wide.loc[epoch].to_numpy() - predicted).mean())
        rows.append({"index": name, "monotone": monotone,
                     "error": float(np.mean(errors))})
    clock = float(np.mean([np.abs(wide.loc[epoch] - wide.loc[point["end"]]).mean()
                           for epoch in after]))
    return pd.DataFrame(rows).sort_values("error").reset_index(drop=True), clock, after


def determinism(tables: analyze_e.Tables, earlier: analyze_e.Tables) -> str:
    """Largest AUROC gap between this evaluation and an earlier one of the same runs."""
    if earlier.metrics.empty:
        return "No earlier evaluation of these runs is in the results folder."
    keys = ["run", "epoch", "set", "csf"]
    both = tables.metrics.merge(earlier.metrics, on=keys, suffixes=("", "_earlier"))
    if both.empty:
        return "The earlier evaluation shares no checkpoint with this one."
    gap = (both["auroc"] - both["auroc_earlier"]).abs().max()
    return (f"{len(both)} values in common with the `step0` evaluation, largest "
            f"AUROC gap {gap:.2e}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--runs", nargs="+", help="default: every run with results")
    parser.add_argument("--precision", choices=sorted(PRECISIONS),
                        default=DEFAULT_PRECISION)
    parser.add_argument("--out", default=str(HERE / "outputs" / "refcheck_report.md"))
    args = parser.parse_args()
    root = Path(args.results_root)
    tag = results_folder(args.precision, suite=SUITE)

    def only(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not args.runs:
            return frame
        return frame[frame["run"].isin(args.runs)].reset_index(drop=True)

    tables = analyze_e.Tables(**{name: only(frame) for name, frame
                                 in vars(analyze_e.read_runs(root, tag)).items()})
    reference = only(read_reference(root, tag))
    if tables.metrics.empty or reference.empty:
        raise SystemExit(f"no results of the reference check under {root}/runs")
    earlier = analyze_e.read_runs(root, results_folder(args.precision, suite="step0"))
    out_dir = Path(args.out).parent / "refcheck"
    out_dir.mkdir(parents=True, exist_ok=True)

    clock = analyze_e.clock_table(tables.descriptors)
    nc1 = collapse(tables.descriptors)
    points = reference_points(clock, nc1)
    table = declines(reference, points)
    read_end, read_final = readings(table), readings(table, "decline_final")
    fitted = table[table["csf"].isin(FITTED)]
    by_score = pd.DataFrame()
    if not fitted.empty:
        by_score = fitted.pivot_table(index=["run", "csf"], columns="reference",
                                      values=["start", "end", "final"])
        by_score.columns = [f"{value}_{name}" for value, name in by_score.columns]
        by_score = by_score.reset_index()
    checks = tables.checks.groupby("run").agg(
        checkpoints=("epoch", "count"),
        nofit_gap=("reference_nofit_gap", "max"),
        fit_rows=("reference_fit", "first"), test_rows=("reference_test", "first"))

    indices = index_table(tables)
    shown = [name for name in SHOWN if name in indices.columns]
    sections = [
        ("Runs and reference points", (
            "`start` is t0, or epoch 60 for a run that never interpolates. `end` "
            "is the last checkpoint before the first reset of collapse. "
            f"{determinism(tables, earlier)}\n\n" + block(points, 1) + "\n\n"
            "`nofit_gap` is the largest AUROC gap of a no-fit score between the "
            "references. It must be 0:\n\n" + block(checks.reset_index(), 6))),
        ("Reading for the primary scores", (
            "Decline from `start` to `end`, in AUROC points on the four far sets, "
            "under each reference:\n\n" + block(read_end, 2)
            + "\n\nDecline from `start` to the final checkpoint:\n\n"
            + block(read_final, 2))),
        ("Every fitted score", (
            "AUROC at `start`, `end`, and the final checkpoint, under each "
            "reference. These rows have no reading:\n\n" + block(by_score, 1))),
    ]
    for _, point in points.iterrows():
        run = point["run"]
        state = nc1[nc1["run"] == run].set_index("epoch")["value"]
        errors, clock_error, after = index_errors(tables.metrics, indices, point,
                                                  state)
        body = block(indices.loc[run][shown].reset_index(), 3)
        if after:
            named = errors[errors["index"].isin(SHOWN)]
            body += (f"\n\nCheckpoints after the reset: {after}. Error of the clock: "
                     f"{clock_error:.2f} points. The 12 descriptors with the lowest "
                     "error:\n\n" + block(errors.head(12), 2)
                     + "\n\nThe descriptors of the table above:\n\n"
                     + block(named, 2))
            errors.to_csv(out_dir / f"index_errors_{run}.csv", index=False)
        else:
            body += "\n\nThis run has no reset inside the range of its path."
        sections.append((f"Collapse, spread, and weights: {run}", body))

    for name, frame in (("points", points), ("declines", table),
                        ("readings", read_end), ("readings_final", read_final)):
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    indices.reset_index().to_csv(out_dir / "indices.csv", index=False)
    text = ("# Reference check: features or reference set?\n\nExploratory. The "
            "reading rule was fixed before these results.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
