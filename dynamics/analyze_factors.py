"""Exploratory check: how much each recipe setting changes the order of the scores.

    python -m dynamics.analyze_factors --results-root dyn_results

Two recipes form a matched pair if they differ in one setting only: augmentation,
weight decay, label smoothing, or peak learning rate. For each pair, the check
compares the 23 scores at the final checkpoint, as a mean over the seeds:

- the rank agreement of the two recipes (Spearman);
- the transfer regret: the points that the best score of one recipe loses in the
  other recipe, as a mean of the two directions.

Two seeds of one recipe give the floor of both values. The check uses the 10-class
runs with 200 epochs, with the correct-only target and with all ID images. Every
analysis has seen these runs, so the result is a description.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_augrc as ag
from dynamics import analyze_e, analyze_h
from dynamics import analyze_v as av
from dynamics.analyze_e import block
from dynamics.config import EPOCHS, RECIPES, Recipe

HERE = Path(__file__).resolve().parent
FACTORS = ("augment", "weight_decay", "label_smoothing", "lr")
NC1 = "nc_paper/var_collapse@stage4"
LIMIT = 1.0


def matched_pairs(recipes: dict[str, Recipe]) -> list[tuple[str, str, str]]:
    """Pairs of recipes that differ in exactly one of the factors.

    Returns:
        (factor, first, second) for every pair, in the order of the recipe names.
    """
    others = [f.name for f in fields(Recipe) if f.name not in FACTORS]
    found = []
    for first, second in itertools.combinations(sorted(recipes), 2):
        a, b = recipes[first], recipes[second]
        if any(getattr(a, name) != getattr(b, name) for name in others):
            continue
        changed = [name for name in FACTORS if getattr(a, name) != getattr(b, name)]
        if len(changed) == 1:
            found.append((changed[0], first, second))
    return found


def transfer_regret(first: pd.Series, second: pd.Series) -> float:
    """Loss of the best score of each recipe in the other recipe, as a mean."""
    return float(((second.max() - second[first.idxmax()])
                  + (first.max() - first[second.idxmax()])) / 2)


def compare(first: pd.Series, second: pd.Series) -> dict[str, float]:
    return {"spearman": float(first.corr(second, method="spearman")),
            "transfer_regret": transfer_regret(first, second)}


def final_tables(root: Path) -> tuple[analyze_e.Tables, int, int]:
    """The 10-class runs with 200 epochs: the 50 earlier ones and stage `v`."""
    earlier = av.training_tables(1, root)
    new = analyze_h.read_stage(root, (av.PARTS[1].test_stage,))
    tables = analyze_e.Tables(**{
        key: pd.concat([frame, other], ignore_index=True)
        for (key, frame), other in zip(vars(earlier).items(), vars(new).values())})
    keep = {name for name, recipe in RECIPES.items() if recipe.epochs == EPOCHS}
    tables = analyze_e.Tables(**{
        key: frame if frame.empty else frame[frame["recipe"].isin(keep)]
        .reset_index(drop=True) for key, frame in vars(tables).items()})
    n_id, n_unseen = ag.counts(root, sorted(set(tables.metrics["run"])))
    return tables, n_id, n_unseen


def pair_table(scores: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
    """One row for each matched pair, from the scores of each run."""
    recipe = analyze_h.recipes_of(scores.index)
    mean, collapse = scores.groupby(recipe).mean(), state.groupby(recipe).mean()
    rows = []
    for factor, first, second in matched_pairs(
            {name: RECIPES[name] for name in mean.index}):
        a, b = RECIPES[first], RECIPES[second]
        smoothing = (a.label_smoothing > 0, b.label_smoothing > 0)
        if factor == "label_smoothing":
            case = "two levels" if all(smoothing) else "off against on"
        else:
            case = "with smoothing" if all(smoothing) else "no smoothing"
        rows.append({
            "factor": factor, "case": case, "pair": f"{first}, {second}",
            "levels": f"{getattr(a, factor)} | {getattr(b, factor)}",
            **compare(mean.loc[first], mean.loc[second]),
            "best": f"{mean.loc[first].idxmax()} | {mean.loc[second].idxmax()}",
            "nc1_ratio": float(abs(np.log10(collapse[first] / collapse[second])))})
    return pd.DataFrame(rows)


def seed_floor(scores: pd.DataFrame) -> pd.DataFrame:
    """The same values for two seeds of one recipe: the floor of the comparison."""
    recipe = analyze_h.recipes_of(scores.index)
    rows = []
    for name, part in scores.groupby(recipe):
        for first, second in itertools.combinations(part.index, 2):
            rows.append({"recipe": name, **compare(part.loc[first],
                                                   part.loc[second])})
    return pd.DataFrame(rows)


def summary(pairs: pd.DataFrame, floor: pd.DataFrame) -> pd.DataFrame:
    groups = [((factor, "all"), part) for factor, part in pairs.groupby("factor")]
    groups += list(pairs.groupby(["factor", "case"]))
    rows = [{"factor": factor, "case": case, "pairs": len(part),
             "spearman": part["spearman"].mean(),
             "transfer_regret": part["transfer_regret"].mean(),
             "largest": part["transfer_regret"].max(),
             "above_1_point": int((part["transfer_regret"] > LIMIT).sum()),
             "nc1_ratio": part["nc1_ratio"].mean()}
            for (factor, case), part in groups]
    rows.append({"factor": "seed (floor)", "case": "all", "pairs": len(floor),
                 "spearman": floor["spearman"].mean(),
                 "transfer_regret": floor["transfer_regret"].mean(),
                 "largest": floor["transfer_regret"].max(),
                 "above_1_point": int((floor["transfer_regret"] > LIMIT).sum()),
                 "nc1_ratio": np.nan})
    return pd.DataFrame(rows).sort_values(["factor", "case"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", default=str(HERE / "outputs" / "factors_report.md"))
    args = parser.parse_args()
    tables, n_id, n_unseen = final_tables(Path(args.results_root))
    sections = []
    for target in (ag.CORRECT_ONLY, ag.ALL_ID):
        used = tables if target == ag.ALL_ID else ag.with_target(
            tables, n_id, n_unseen, target)
        scores, metrics = ag.final(used)
        pairs, floor = pair_table(scores, metrics[NC1]), seed_floor(scores)
        sections.append((f"Target {target}", (
            f"{len(scores)} runs of {scores.index.map(lambda r: r[:2]).nunique()} "
            "recipes. `nc1_ratio` is the difference of log10 NC1 of the training "
            "features between the two recipes. By factor:\n\n"
            + block(summary(pairs, floor), 2) + "\n\nEvery pair:\n\n"
            + block(pairs.sort_values(["factor", "transfer_regret"]), 2))))
    text = ("# Exploratory check: the effect of each recipe setting on the order of "
            "the scores\n\nPost hoc, on runs that every analysis has seen.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
