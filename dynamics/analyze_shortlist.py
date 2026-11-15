"""Shortlists of scores from the frozen rules (post hoc check of 2026-11-14).

Do ID-only metrics find a small subset of scores that contains a good one? The
shortlist of a rule is the k scores with the highest predicted AUROC at the final
checkpoint. It is compared with a fixed shortlist, the top scores by mean, and
random subsets of the same size.

    python -m dynamics.analyze_shortlist --results-root dyn_results

Protocol: ../documentation/stage_c2_all_metrics_amendment_2026-11-14.md, section
"Shortlist check". This analysis is post hoc.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_c, analyze_e, rules
from dynamics import analyze_multimetric as mm
from dynamics.analyze_e import block
from dynamics.config import DEFAULT_PRECISION, STAGES, parse_run, results_folder

HERE = Path(__file__).resolve().parent
SIZES = (3, 5, 8)
USEFUL = 1.0
RANDOM_SETS = 2000
#: Rules file, rule names, and the stages whose runs test them.
TESTS = (("first", rules.RULES_PATH, ("all", "geometry"), ("c", "c2"), False),
         ("second", rules.RULES_V2_PATH, ("all",), ("c2",), True))
KNOWN = ("A", "B")


def shortlist_regret(scores: pd.DataFrame, names: list[list[str]] | list[str]
                     ) -> pd.Series:
    """Per run: the best of all scores minus the best of its shortlist.

    `names` is one shortlist for all runs, or one shortlist per run.
    """
    per_run = names if names and isinstance(names[0], list) else [names] * len(scores)
    inside = [scores.iloc[i][list(chosen)].max() for i, chosen in enumerate(per_run)]
    return scores.max(axis=1) - pd.Series(inside, index=scores.index)


def greedy(scores: pd.DataFrame, k: int) -> list[str]:
    """The k scores that a greedy search adds to lower the mean shortlist regret."""
    chosen: list[str] = []
    for _ in range(k):
        rest = [name for name in scores.columns if name not in chosen]
        chosen.append(min(rest, key=lambda name: float(
            shortlist_regret(scores, [*chosen, name]).mean())))
    return chosen


def random_regret(scores: pd.DataFrame, k: int, seed: int = 0) -> pd.Series:
    """Per run: the mean shortlist regret over random subsets of size k."""
    rng = np.random.default_rng(seed)
    values = scores.to_numpy()
    total = np.zeros(len(scores))
    for _ in range(RANDOM_SETS):
        total += values.max(axis=1) - values[:, rng.choice(
            values.shape[1], size=k, replace=False)].max(axis=1)
    return pd.Series(total / RANDOM_SETS, index=scores.index)


def final_scores(tables: analyze_e.Tables, sets: tuple[str, ...]
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores and metrics of the last checkpoint of each run."""
    meta, scores, metrics = mm.build_cells(tables, sets)
    mask = analyze_c.final_mask(meta)
    return scores[mask].droplevel("epoch"), metrics[mask].droplevel("epoch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", default=str(HERE / "outputs" / "shortlist_report.md"))
    args = parser.parse_args()
    root = Path(args.results_root)
    step0 = analyze_e.read_runs(root, results_folder(DEFAULT_PRECISION,
                                                     suite=analyze_c.SUITE))
    sections = []
    for label, path, names, stages, with_c in TESTS:
        frozen = rules.load(path)["targets"]["far"]
        sets = tuple(frozen["sets"])
        wanted = {spec.name for stage in stages for spec in STAGES[stage]}
        new = analyze_e.Tables(**{key: frame[frame["run"].isin(wanted)]
                                  .reset_index(drop=True)
                                  for key, frame in vars(step0).items()})
        scores, metrics = final_scores(new, sets)
        recipe = pd.Series([parse_run(run).recipe for run in scores.index],
                           index=scores.index)
        keep = ~recipe.isin(KNOWN)
        scores, metrics, recipe = scores[keep], metrics[keep], recipe[keep]
        train, _ = final_scores(rules.training_tables(root, with_stage_c=with_c),
                                sets)
        train = train[[parse_run(run).recipe[0] != "D" for run in train.index]]
        rows, by_recipe = [], []
        for k in SIZES:
            lists = {"fixed shortlist": greedy(train, k),
                     "top by mean": list(train.mean().sort_values(
                         ascending=False).index[:k])}
            regrets = {name: shortlist_regret(scores, chosen)
                       for name, chosen in lists.items()}
            regrets["random"] = random_regret(scores, k)
            for name in names:
                predicted = frozen["rules"][name].predict(metrics)
                order = np.argsort(-predicted, axis=1)[:, :k]
                csfs = np.array(frozen["rules"][name].csfs)
                regrets[f"rule {name}"] = shortlist_regret(
                    scores, [list(csfs[row]) for row in order])
            fixed = regrets["fixed shortlist"]
            for name, values in regrets.items():
                mean, low, high = analyze_c.interval((values - fixed).to_numpy())
                rows.append({"k": k, "shortlist": name, "regret": values.mean(),
                             "useful_runs": int((values <= USEFUL).sum()),
                             "runs": len(values), "minus_fixed": mean, "low": low,
                             "high": high})
                by_recipe.append(values.groupby(recipe).mean().rename(
                    (k, name)))
            rows[-1]["scores"] = ""
            rows.append({"k": k, "shortlist": "fixed shortlist is",
                         "scores": ", ".join(lists["fixed shortlist"])})
        table = pd.DataFrame(rows)
        recipes = pd.concat(by_recipe, axis=1).T
        recipes.index.names = ["k", "shortlist"]
        title = f"Rules of the {label} file, tested on stages {' and '.join(stages)}"
        sections.append((title, (
            f"{len(scores)} runs with new recipes, and {len(train)} training runs "
            "for the baselines. Shortlist regret in AUROC points on the far "
            "sets. `minus_fixed` is the difference from the fixed shortlist, with "
            "a 95% interval over runs:\n\n"
            + block(table[table["regret"].notna()].drop(columns="scores"), 2)
            + "\n\n" + "\n".join(
                f"- Fixed shortlist of size {row['k']}: {row['scores']}."
                for _, row in table[table["regret"].isna()].iterrows())
            + "\n\nBy recipe:\n\n" + block(recipes.reset_index(), 2))))
    text = ("# Shortlists of scores from the frozen rules\n\nPost hoc. The sizes, "
            "the baselines, and the reading were fixed before the computation.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
