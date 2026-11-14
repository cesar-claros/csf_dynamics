"""Score selection from several ID-only metrics of a checkpoint.

Do several ID-only metrics, used together, choose a score for far OOD better than
one fixed score? One ridge regression maps the metrics of a checkpoint to the
AUROC of every score, and the score with the highest prediction is chosen. The
recipe is never an input.

Run from code/ on results that are already local:

    python -m dynamics.analyze_multimetric --results-root dyn_results

Protocol: ../documentation/time_and_collapse_protocol_2026-11-12.md, section
"Multi-metric test of score selection". Everything here is exploratory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_e, analyze_selection
from dynamics.analyze_e import block
from dynamics.analyze_selection import FAMILY
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import DEFAULT_PRECISION, PENULTIMATE, PRECISIONS, results_folder

HERE = Path(__file__).resolve().parent
#: Inputs without a model: one score for all runs, or one for each family.
FIXED, FAMILY_FIXED = "fixed", "family"
MODELS = ("clock", "nc1", "nc8", "geometry", "all")
INPUTS = (FIXED, FAMILY_FIXED, *MODELS)
NC1 = f"nc_paper/var_collapse@{PENULTIMATE}"
PENALTY = 10.0
#: Secondary model without a linear form: the mean over this many nearest cells.
NEIGHBORS = 10
METHODS = ("ridge", "nearest")
#: Margins of the rules R1, R2, and R3, in AUROC points.
MARGIN_FIXED = 1.0
MARGIN_FAMILY = 0.5
MARGIN_CLOCK = 0.5


def build_cells(tables: analyze_e.Tables, sets: tuple[str, ...]
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The checkpoints from epoch 1, as three tables with one row per cell.

    Returns:
        `meta` with run, epoch, family, and eligibility; `scores` with the mean
        AUROC of each CSF over the sets, in points; and `metrics` with one column
        per ID-only descriptor, named `group/name@layer`.
    """
    rows = tables.descriptors[tables.descriptors["epoch"] >= 1].copy()
    layer = rows["layer"].astype(str)
    rows["column"] = rows["group"] + "/" + rows["name"] + np.where(
        layer == "", "", "@" + layer)
    metrics = rows.pivot_table(index=["run", "epoch"], columns="column",
                               values="value")
    metrics["clock/log_epoch"] = np.log(metrics.index.get_level_values("epoch"))
    kept = tables.metrics[tables.metrics["set"].isin(sets)
                          & (tables.metrics["epoch"] >= 1)]
    scores = (kept.groupby(["run", "epoch", "csf"])["auroc"].mean() * 100).unstack(
        "csf")
    frames = analyze_selection.run_frames(tables, sets)
    eligible = pd.concat({run: frame["eligible"] for run, frame in frames.items()},
                         names=["run", "epoch"])
    recipe = tables.metrics.groupby("run")["recipe"].first()
    index = metrics.index.intersection(scores.index)
    meta = pd.DataFrame({"eligible": eligible.reindex(index).fillna(False)},
                        index=index)
    meta["family"] = [FAMILY.get(recipe[run], recipe[run])
                      for run in index.get_level_values("run")]
    return meta, scores.loc[index], metrics.loc[index]


def transform(metrics: pd.DataFrame) -> pd.DataFrame:
    """Usable inputs: finite, with spread, and on a log scale where positive.

    A column with a non-finite value or with one value only is left out. A column
    that is positive in every cell is replaced by its log10.
    """
    kept = metrics.loc[:, np.isfinite(metrics).all(axis=0)]
    kept = kept.loc[:, kept.nunique() > 1]
    positive = (kept > 0).all(axis=0)
    out = kept.copy()
    out.loc[:, positive] = np.log10(kept.loc[:, positive])
    return out


def input_columns(columns: list[str]) -> dict[str, list[str]]:
    """The metric columns of each model input."""
    clock = [name for name in columns if name.startswith("clock/")]
    weights = [name for name in columns if name.startswith("weights/")]
    return {
        "clock": clock,
        "nc1": [name for name in columns if name == NC1],
        "nc8": [name for name in columns if name.startswith("nc_paper/")],
        "geometry": [name for name in columns if name not in clock + weights],
        "all": list(columns)}


def ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
          penalty: float = PENALTY) -> np.ndarray:
    """Ridge prediction of several targets, with inputs standardized on the training rows."""
    mean, std = train_x.mean(axis=0), train_x.std(axis=0)
    std[std == 0] = 1.0
    fit, offset = (train_x - mean) / std, train_y.mean(axis=0)
    beta = np.linalg.solve(fit.T @ fit + penalty * np.eye(fit.shape[1]),
                           fit.T @ (train_y - offset))
    return (test_x - mean) / std @ beta + offset


def nearest(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
            k: int = NEIGHBORS) -> np.ndarray:
    """Mean target of the k nearest training rows, on standardized inputs.

    It needs no linear form, so it can follow a stage effect that differs between
    recipes.
    """
    mean, std = train_x.mean(axis=0), train_x.std(axis=0)
    std[std == 0] = 1.0
    fit, query = (train_x - mean) / std, (test_x - mean) / std
    distance = ((query[:, None, :] - fit[None, :, :]) ** 2).sum(axis=2)
    rows = np.argsort(distance, axis=1)[:, :min(k, len(fit))]
    return train_y[rows].mean(axis=1)


def logit(points: np.ndarray) -> np.ndarray:
    share = np.clip(points / 100, 1e-3, 1 - 1e-3)
    return np.log(share / (1 - share))


def evaluate(meta: pd.DataFrame, scores: pd.DataFrame, metrics: pd.DataFrame,
             scheme: str, penalty: float = PENALTY, method: str = "ridge"
             ) -> pd.DataFrame:
    """Mean regret of each input on the eligible cells of each test run.

    Args:
        scheme: `run` leaves one run out. `family` trains on the other families.
        method: `ridge` is the model of the protocol. `nearest` is the secondary
            model (see `nearest`).

    Returns:
        One row per test run and input, with the regret in AUROC points and the
        score that the input chose most often.
    """
    inputs = transform(metrics)
    columns = input_columns(list(inputs.columns))
    runs = meta.index.get_level_values("run").to_numpy()
    family = meta["family"].to_numpy()
    eligible = meta["eligible"].to_numpy()
    values, names = scores.to_numpy(), np.array(scores.columns)
    best = values.max(axis=1)
    rows = []
    for run in pd.unique(runs):
        test = (runs == run) & eligible
        own = family[runs == run][0]
        train = (runs != run) if scheme == "run" else (family != own)
        if not test.any() or not train.any():
            continue
        chosen: dict[str, np.ndarray] = {}
        pool = train & eligible
        if pool.any():
            chosen[FIXED] = np.full(test.sum(), values[pool].mean(axis=0).argmax())
        same = pool & (family == own)
        if scheme == "run" and same.any():
            chosen[FAMILY_FIXED] = np.full(test.sum(),
                                           values[same].mean(axis=0).argmax())
        for name in MODELS:
            if columns[name]:
                x = inputs[columns[name]].to_numpy(dtype=np.float64)
                target = logit(values[train])
                predicted = (ridge(x[train], target, x[test], penalty)
                             if method == "ridge" else nearest(x[train], target,
                                                               x[test]))
                chosen[name] = predicted.argmax(axis=1)
        for name, picks in chosen.items():
            actual = values[test][np.arange(test.sum()), picks]
            rows.append({"scheme": scheme, "run": run, "family": own, "input": name,
                         "regret": float((best[test] - actual).mean()),
                         "cells": int(test.sum()),
                         "choice": pd.Series(names[picks]).value_counts().index[0]})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean regret per input: all runs, and each family. Each run has the same weight."""
    if results.empty:
        return pd.DataFrame()
    table = results.pivot_table(index="input", columns="family", values="regret")
    table.insert(0, "all", results.groupby("input")["regret"].mean())
    order = [name for name in INPUTS if name in table.index]
    return table.reindex(order).reset_index()


def verdict(results: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """The reading that the protocol fixes, on the scheme with one run left out."""
    mean = results.groupby("input")["regret"].mean()
    needed = {FIXED, FAMILY_FIXED, "clock", "geometry", "all"}
    if not needed <= set(mean.index):
        return pd.DataFrame(), "not decided: an input is missing"
    model = "geometry" if mean["geometry"] <= mean["all"] else "all"
    rules = pd.DataFrame([
        {"rule": "R1", "compares": f"{model} against fixed",
         "margin": mean[FIXED] - mean[model], "needed": MARGIN_FIXED},
        {"rule": "R2", "compares": f"{model} against family",
         "margin": mean[FAMILY_FIXED] - mean[model], "needed": MARGIN_FAMILY},
        {"rule": "R3", "compares": "geometry against clock",
         "margin": mean["clock"] - mean["geometry"], "needed": MARGIN_CLOCK}])
    rules["holds"] = rules["margin"] >= rules["needed"]
    r1, r2, r3 = rules["holds"]
    if not r1:
        return rules, ("R1 fails: several metrics do not beat one fixed score. The "
                       "selection aim is closed for this setting")
    if not r2:
        return rules, ("R1 holds and R2 fails: the metrics recognise the recipe and "
                       "nothing more. Only new recipes can test this")
    return rules, ("R1 and R2 hold: the metrics also read the training stage. The "
                   + ("geometry adds to the clock group" if r3
                      else "geometry does not add to the clock group"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--suite", default="refcheck",
                        help="evaluation to read; an empty text reads the base one")
    parser.add_argument("--far-sets", nargs="+", default=list(FAR_SETS))
    parser.add_argument("--precision", choices=sorted(PRECISIONS),
                        default=DEFAULT_PRECISION)
    parser.add_argument("--out",
                        default=str(HERE / "outputs" / "multimetric_report.md"))
    args = parser.parse_args()
    tables = analyze_e.read_runs(Path(args.results_root),
                                 results_folder(args.precision, suite=args.suite))
    if tables.metrics.empty:
        raise SystemExit(f"no results under {args.results_root}/runs")
    meta, scores, metrics = build_cells(tables, tuple(args.far_sets))
    by_run = evaluate(meta, scores, metrics, "run")
    by_family = evaluate(meta, scores, metrics, "family")
    rules, decision = verdict(by_run)
    near_run = evaluate(meta, scores, metrics, "run", method="nearest")
    near_family = evaluate(meta, scores, metrics, "family", method="nearest")
    near_rules, near_decision = verdict(near_run)
    columns = input_columns(list(transform(metrics).columns))
    sizes = pd.DataFrame({"input": list(columns),
                          "metrics": [len(v) for v in columns.values()]})
    counts = (meta.groupby("family")
              .agg(runs=("eligible", lambda s: s.index.get_level_values("run")
                         .nunique()), cells=("eligible", "size"),
                   eligible_cells=("eligible", "sum")).reset_index())
    choices = by_run.groupby(["family", "input"])["choice"].agg(
        lambda s: s.value_counts().index[0]).unstack("family").reindex(
        [name for name in INPUTS if name in by_run["input"].unique()]).reset_index()
    penalties = pd.DataFrame([
        {"penalty": value, **evaluate(meta, scores, metrics, "run", value)
         .groupby("input")["regret"].mean().reindex(list(MODELS)).to_dict()}
        for value in (PENALTY / 10, PENALTY, PENALTY * 10)])
    sections = [
        ("Cells and inputs", (
            f"Far sets: {', '.join(args.far_sets)}. A model is trained on every "
            "checkpoint from epoch 1 of the training runs and tested on the "
            "eligible checkpoints of the test run.\n\n" + block(counts) + "\n\n"
            + block(sizes))),
        ("One run left out", (
            "Mean regret in AUROC points. The training runs contain both families, "
            "and the recipe is not an input:\n\n" + block(summarize(by_run), 2)
            + "\n\nThe score that each input chooses most often:\n\n"
            + block(choices))),
        ("One family left out", (
            "The model is trained on one family and tested on the other:\n\n"
            + block(summarize(by_family), 2))),
        ("Sensitivity to the ridge penalty", (
            f"One run left out. The protocol fixes the penalty {PENALTY:g}:\n\n"
            + block(penalties, 2))),
        ("Secondary model: nearest cells", (
            f"The prediction is the mean over the {NEIGHBORS} nearest training "
            "cells. One run left out:\n\n" + block(summarize(near_run), 2)
            + "\n\nOne family left out:\n\n" + block(summarize(near_family), 2)
            + "\n\nThe rules with this model:\n\n" + block(near_rules, 2)
            + f"\n\nWith this model: {near_decision}.")),
        ("Reading", "The rules are judged on the ridge model:\n\n" + block(rules, 2)
         + f"\n\n**Result: {decision}.**"),
    ]
    out_dir = Path(args.out).parent / Path(args.out).stem.replace("_report", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([by_run, by_family]).to_csv(out_dir / "regrets.csv", index=False)
    text = ("# Score selection from several ID-only metrics\n\nExploratory. The "
            "reading was fixed before these results.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
