"""Frozen selection rules of Stage C (amendment A5).

A rule is fitted on the runs that exist before Stage C, written to one JSON file,
and applied to new runs without a new fit. Build the file from local results:

    python -m dynamics.rules --results-root dyn_results

Protocol: ../documentation/stage_c_selection_amendment_2026-11-14.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_e
from dynamics import analyze_multimetric as mm
from dynamics.analyze_selection import FAMILY
from dynamics.analyze_step0 import FAR_SETS
from dynamics.config import DEFAULT_PRECISION, PRECISIONS, results_folder

HERE = Path(__file__).resolve().parent
RULES_PATH = HERE / "frozen" / "stage_c_rules.json"
TARGETS = {"far": FAR_SETS, "near": ("ssb_hard", "ninco", "imagewoof")}
#: Ridge rules in the file. `geometry` is the rule under test.
RIDGE_RULES = ("geometry", "clock", "all")
#: Recipes whose final checkpoints give the baselines B1 and B2.
BASELINE_RECIPES = ("A", "B")
TRAINING_SUITE = "refcheck"


@dataclass
class RidgeRule:
    """One ridge model from ID-only metrics to the logit AUROC of every score.

    `log` marks the columns that enter as log10. `floor` is the smallest positive
    training value of such a column: a new value at or below 0 is set to it.
    """

    columns: list[str]
    log: list[bool]
    floor: list[float | None]
    mean: list[float]
    std: list[float]
    beta: list[list[float]]
    offset: list[float]
    csfs: list[str]

    def inputs(self, metrics: pd.DataFrame) -> np.ndarray:
        """Standardized inputs of the rows of `metrics`. A missing column gives NaN."""
        x = metrics.reindex(columns=self.columns).to_numpy(dtype=np.float64)
        log = np.array(self.log)
        floor = np.array([np.nan if v is None else v for v in self.floor])
        x[:, log] = np.log10(np.where(x[:, log] <= 0, floor[log], x[:, log]))
        return (x - np.array(self.mean)) / np.array(self.std)

    def predict(self, metrics: pd.DataFrame) -> np.ndarray:
        """Predicted logit AUROC, one row per row of `metrics` and one column per score."""
        return self.inputs(metrics) @ np.array(self.beta) + np.array(self.offset)

    def choose(self, metrics: pd.DataFrame) -> list[str | None]:
        """The score with the highest prediction, or None for a row with a non-finite input."""
        predicted = self.predict(metrics)
        return [self.csfs[int(row.argmax())] if np.isfinite(row).all() else None
                for row in predicted]


def fit_ridge_rule(metrics: pd.DataFrame, scores: pd.DataFrame, columns: list[str],
                   penalty: float = mm.PENALTY) -> RidgeRule:
    """Fit a rule on training cells. `metrics` holds the raw values of the columns.

    Raises:
        ValueError: If a column has a non-finite training value.
    """
    x = metrics[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("a training column has a non-finite value")
    log = (x > 0).all(axis=0)
    floor = np.where(log, x.min(axis=0), np.nan)
    x[:, log] = np.log10(x[:, log])
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    fit = (x - mean) / std
    target = mm.logit(scores.to_numpy(dtype=np.float64))
    offset = target.mean(axis=0)
    beta = np.linalg.solve(fit.T @ fit + penalty * np.eye(len(columns)),
                           fit.T @ (target - offset))
    floors: list[float | None] = [None if np.isnan(v) else float(v) for v in floor]
    return RidgeRule(columns, log.tolist(), floors, mean.tolist(), std.tolist(),
                     beta.tolist(), offset.tolist(), list(scores.columns))


def baselines(tables: analyze_e.Tables, sets: tuple[str, ...]) -> dict:
    """B1 and B2 from the final checkpoints of the pilot runs of recipes A and B."""
    kept = tables.metrics[tables.metrics["set"].isin(sets)
                          & tables.metrics["recipe"].isin(BASELINE_RECIPES)]
    last = kept.groupby("run")["epoch"].transform("max") == kept["epoch"]
    scores = (kept[last].groupby(["run", "recipe", "csf"])["auroc"].mean() * 100
              ).unstack("csf")
    best = scores.max(axis=1)
    out: dict = {"runs": sorted(scores.index.get_level_values("run"))}
    one = str(scores.mean().idxmax())
    out["b1"] = {"score": one, "regret": float((best - scores[one]).mean())}
    out["b2"] = {}
    for recipe, part in scores.groupby(level="recipe"):
        own = str(part.mean().idxmax())
        out["b2"][FAMILY[str(recipe)]] = {
            "score": own, "regret": float((part.max(axis=1) - part[own]).mean())}
    return out


def build(tables: analyze_e.Tables, penalty: float = mm.PENALTY) -> dict:
    """Every frozen rule and baseline, as one JSON-ready dict."""
    out: dict = {"amendment": "A5", "penalty": penalty, "targets": {}}
    for target, sets in TARGETS.items():
        meta, scores, metrics = mm.build_cells(tables, sets)
        usable = list(mm.transform(metrics).columns)
        columns = mm.input_columns(usable)
        out["training_runs"] = sorted(set(meta.index.get_level_values("run")))
        out["training_cells"] = len(meta)
        out["targets"][target] = {
            "sets": list(sets),
            "rules": {name: asdict(fit_ridge_rule(metrics, scores, columns[name],
                                                  penalty)) for name in RIDGE_RULES},
            "baselines": baselines(tables, sets)}
    return out


def save(rules: dict, path: Path) -> str:
    """Write the rules with a stable byte form and return the SHA-256 of the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rules, sort_keys=True, separators=(",", ":")) + "\n")
    return file_hash(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path = RULES_PATH) -> dict:
    """The rules file, with each ridge rule as a `RidgeRule`."""
    rules = json.loads(Path(path).read_text())
    for target in rules["targets"].values():
        target["rules"] = {name: RidgeRule(**rule)
                           for name, rule in target["rules"].items()}
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--precision", choices=sorted(PRECISIONS),
                        default=DEFAULT_PRECISION)
    parser.add_argument("--out", default=str(RULES_PATH))
    args = parser.parse_args()
    tables = analyze_e.read_runs(
        Path(args.results_root), results_folder(args.precision, suite=TRAINING_SUITE))
    if tables.metrics.empty:
        raise SystemExit(f"no training results under {args.results_root}/runs")
    rules = build(tables)
    digest = save(rules, Path(args.out))
    print(f"training runs: {len(rules['training_runs'])}, cells: "
          f"{rules['training_cells']}")
    for name, target in rules["targets"].items():
        sizes = {rule: len(spec["columns"]) for rule, spec in target["rules"].items()}
        base = target["baselines"]
        print(f"{name}: inputs {sizes}; B1 {base['b1']}; B2 {base['b2']}")
    print(f"file: {args.out}\nsha256: {digest}")


if __name__ == "__main__":
    main()
