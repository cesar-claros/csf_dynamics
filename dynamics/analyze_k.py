"""Amendment A18, part 1: the readings of the pilot of stand-in tasks.

    python -m dynamics.analyze_k --results-root dyn_results

Reads the result files of `dynamics.zoo_pilot` (<results-root>/standin_pilot) and
the stored result files of the same models from amendments A16 and A17, which
give the targets. No reading needs a fit. The main reading is the rank agreement:
for each model, the rank correlation over the 23 scores between the AUROC on a
stand-in task and the AUROC on a target. A fixed rule accepts at most two new
tasks. No claim comes from the pilot.

Gates: C (the three existing tasks and the accuracy agree with the stored
values) and S (every score has a finite value on every task). A model that fails
a gate leaves. Gate C leaves KPCA_RecError out, by an amendment of 2026-11-27
that was made before any result of the pilot: a control of amendment A17 showed
that this one score is not stable between two runs. Its change is reported apart.

Protocol: ../documentation/standin_tasks_pilot_plan_2026-11-27.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_c, analyze_e, standin_list
from dynamics import analyze_i as ai
from dynamics import analyze_j as aj
from dynamics.analyze_e import block
from dynamics.scores import ALL_CSFS

HERE = Path(__file__).resolve().parent
RESULTS = "standin_pilot"
#: SHA-256 of the frozen list of the 40 models.
MODELS_SHA256 = "213c2fff6ac92657cc873aeb5e6e4b140069f81c926f4ec151fdfc1bdabeb7ca"
OLD_TASKS = ("id_errors", "blend", "shuffle")
POOLS = ("torchvision", "timm")
STORED = {"torchvision": ai.RESULTS, "timm": aj.RESULTS}
PRIMARY = ai.PRIMARY
#: Rule to accept a task.
MARGIN = 0.05
AGREEMENT_LIMIT = 0.90
SPREAD_LIMIT = 0.02
MAX_ACCEPTED = 2
#: A gain on the far sets at which a second evaluation of every model is proposed.
LARGE = 0.15
#: Gate C: largest difference of an AUROC and of the accuracy from the stored value.
CONSISTENCY = 0.01
ACCURACY = 0.0005
#: Scores that gate C leaves out (amendment of 2026-11-27, before any result).
UNSTABLE = ("KPCA_RecError",)
#: The pilot gives no decision if more models of a half leave.
MAX_LEFT = 4


def wide(tables: analyze_e.Tables, tasks: list[str], value: str = "auroc"
         ) -> pd.DataFrame:
    """Per model, the value of each score on each task: columns (task, score)."""
    part = tables.metrics[tables.metrics["set"].isin(tasks)]
    table = part.pivot_table(index="run", columns=["set", "csf"], values=value)
    return table.reindex(columns=pd.MultiIndex.from_product([tasks, list(ALL_CSFS)]))


def stored_tables(root: Path, listed: pd.DataFrame
                  ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.Series]:
    """From the stored result files of the pilot models: the AUROC of the existing
    tasks, the target of each kind of OOD data, and the accuracy."""
    old, accuracy = [], []
    targets: dict[str, list[pd.DataFrame]] = {name: [] for name in ai.GROUPS}
    for pool in POOLS:
        names = list(listed.index[listed["pool"] == pool])
        tables, facts = ai.read(Path(root), STORED[pool])
        tables = ai.keep(tables, names)
        old.append(wide(tables, list(OLD_TASKS)))
        accuracy.append(facts.set_index("model")["val_accuracy"].reindex(names))
        for name, sets in ai.GROUPS.items():
            targets[name].append(ai.targets_of(tables, sets, list(ALL_CSFS)))
    return (pd.concat(old).reindex(listed.index),
            {name: pd.concat(parts).reindex(listed.index)
             for name, parts in targets.items()},
            pd.concat(accuracy).reindex(listed.index))


def rank_agreement(first: pd.DataFrame, second: pd.DataFrame) -> pd.Series:
    """Per model, the rank correlation over the scores between two tables."""
    a = first.rank(axis=1).to_numpy(dtype=np.float64)
    b = second.rank(axis=1).to_numpy(dtype=np.float64)
    a, b = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    scale = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.Series((a * b).sum(axis=1) / scale, index=first.index)


def old_gap(new: pd.DataFrame, old: pd.DataFrame, names: list[str],
            unstable: bool = False) -> pd.Series:
    """Per model, the largest difference of an AUROC of an existing task from the
    stored value: over the scores of gate C, or over the scores that it leaves out."""
    tasks = list(OLD_TASKS)
    gap = (new.loc[names, tasks] - old.loc[names, tasks]).abs()
    inside = gap.columns.get_level_values(1).isin(UNSTABLE)
    return gap.loc[:, inside if unstable else ~inside].max(axis=1)


def gates(listed: pd.DataFrame, facts: pd.DataFrame, new: pd.DataFrame,
          old: pd.DataFrame, accuracy: pd.Series) -> pd.DataFrame:
    """One row for each model that leaves, with the reason."""
    rows = []
    for name in listed.index:
        if name not in facts.index:
            reason = "no result file"
        elif facts.at[name, "failed"]:
            reason = f"gate S: {facts.at[name, 'failed']}"
        elif not np.isfinite(new.loc[name].to_numpy(dtype=np.float64)).all():
            reason = "gate S: a value of a task is not finite"
        elif abs(facts.at[name, "val_accuracy"] - accuracy[name]) > ACCURACY:
            reason = (f"gate C: accuracy {100 * facts.at[name, 'val_accuracy']:.2f}, "
                      f"stored {100 * accuracy[name]:.2f}")
        else:
            gap = float(old_gap(new, old, [name]).iloc[0])
            if gap <= CONSISTENCY:
                continue
            reason = f"gate C: an existing task differs by {gap:.3f} from the stored value"
        rows.append({"model": name, "pool": listed.at[name, "pool"], "reason": reason})
    return pd.DataFrame(rows, columns=["model", "pool", "reason"])


def by_half(values: pd.Series, pool: pd.Series) -> dict[str, float]:
    """The mean of a per-model value in each half and over all models."""
    out = {name: float(values[pool == name].mean()) for name in POOLS}
    out["both"] = float(values.mean())
    return out


def candidates(agreement: dict[tuple[str, str], pd.Series], between: pd.DataFrame,
               spread: pd.Series, tasks: list[str], pool: pd.Series) -> pd.DataFrame:
    """Each new task against the best existing task of each target.

    Args:
        agreement: Per (task, target), the rank agreement of each model.
        between: Mean agreement of each new task (rows) with each existing task.
        spread: Mean spread of each task.
    """
    rows = []
    for target in ai.GROUPS:
        best = max(OLD_TASKS, key=lambda task: agreement[(task, target)].mean())
        for task in tasks:
            difference = agreement[(task, target)] - agreement[(best, target)]
            halves = by_half(difference, pool)
            mean, low, high = analyze_c.interval(difference.to_numpy())
            rows.append({
                "task": task, "target": target, "against": best,
                **{f"gain_{name}": halves[name] for name in POOLS},
                "gain": mean, "low": low, "high": high,
                "agreement_with_existing": float(between.loc[task].max()),
                "spread": float(spread[task]),
                "passes": bool(all(halves[name] >= MARGIN for name in POOLS)
                               and low > 0
                               and between.loc[task].max() < AGREEMENT_LIMIT
                               and spread[task] >= SPREAD_LIMIT)})
    return pd.DataFrame(rows)


def accepted(table: pd.DataFrame, most: int = MAX_ACCEPTED) -> pd.DataFrame:
    """At most `most` tasks that pass, by the largest gain, for different targets
    if that is possible. One row for each accepted task: its best target."""
    passing = table[table["passes"]].sort_values("gain", ascending=False)
    best = passing.drop_duplicates("task")
    if len(best) <= 1 or most == 1:
        return best.head(most)
    first = best.iloc[[0]]
    other = best.iloc[1:]
    different = other[other["target"] != first["target"].iloc[0]]
    rest = pd.concat([different, other]).drop_duplicates("task")
    return pd.concat([first, rest.head(most - 1)])


def direct_regret(task_values: pd.DataFrame, tasks: list[str], target: pd.DataFrame,
                  how: str) -> pd.Series:
    """Per model, the regret on `target` of the direct choice over `tasks`.

    `how` is `mean_rank` (the score with the best mean rank over the tasks) or
    `maximin` (the score with the best worst rank).
    """
    ranks = np.stack([task_values[task].rank(axis=1).to_numpy() for task in tasks])
    merit = ranks.mean(axis=0) if how == "mean_rank" else ranks.min(axis=0)
    pick = merit.argmax(axis=1)
    values = target.to_numpy()
    return pd.Series(values.max(axis=1) - values[np.arange(len(values)), pick],
                     index=target.index)


def readings(listed: pd.DataFrame, new: pd.DataFrame,
             targets: dict[str, pd.DataFrame]) -> dict:
    """Every table of the report, for the models that stay."""
    tasks = [task for task in new.columns.get_level_values(0).unique()
             if task not in OLD_TASKS]
    every = [*OLD_TASKS, *tasks]
    pool = listed["pool"]
    agreement = {(task, target): rank_agreement(new[task], targets[target])
                 for task in every for target in ai.GROUPS}
    table = pd.DataFrame({target: {task: " / ".join(
        f"{value:.2f}" for value in by_half(agreement[(task, target)], pool).values())
        for task in every} for target in ai.GROUPS})
    between = pd.DataFrame({old: {task: float(rank_agreement(new[task], new[old]).mean())
                                  for task in tasks} for old in OLD_TASKS})
    spread = pd.Series({task: float(new[task].std(axis=1).mean()) for task in every})
    level = pd.Series({task: float(new[task].max(axis=1).mean()) for task in every})
    found = candidates(agreement, between, spread, tasks, pool)
    chosen = accepted(found)
    sets = {"the three existing tasks": list(OLD_TASKS)}
    for task in chosen["task"]:
        sets[f"the three and {task}"] = [*OLD_TASKS, task]
    if len(chosen) > 1:
        sets["the three and the accepted tasks"] = [*OLD_TASKS, *chosen["task"]]
    sets["all tasks"] = every
    direct = []
    base = {how: direct_regret(new, list(OLD_TASKS), targets[PRIMARY], how)
            for how in ("mean_rank", "maximin")}
    for label, names in sets.items():
        for how in ("mean_rank", "maximin"):
            regret = direct_regret(new, names, targets[PRIMARY], how)
            mean, low, high = analyze_c.interval((regret - base[how]).to_numpy())
            direct.append({"tasks": label, "choice": how,
                           **by_half(regret, pool), "against_the_three": mean,
                           "low": low, "high": high})
    return {"tasks": tasks, "agreement": table, "between": between, "spread": spread,
            "level": level, "candidates": found, "accepted": chosen,
            "direct": pd.DataFrame(direct)}


def report(listed: pd.DataFrame, root: Path) -> tuple[str, str]:
    """The report of the pilot, and why it gives no decision (an empty text if it
    gives one).

    Raises:
        SystemExit: If no result file is present, or a file is not of the list.
    """
    tables, facts = ai.read(Path(root), RESULTS)
    if facts.empty:
        raise SystemExit(f"no result files under {root}/{RESULTS}")
    facts = facts.set_index("model")
    unknown = sorted(set(facts.index) - set(listed.index))
    if unknown:
        raise SystemExit(f"results of models that are not in the list: {unknown}")
    if facts["limit"].notna().any():
        raise SystemExit("a result file comes from a smoke run")
    tasks = sorted(set(tables.metrics["set"]) - set(OLD_TASKS))
    new = wide(tables, [*OLD_TASKS, *tasks]).reindex(listed.index)
    old, targets, accuracy = stored_tables(root, listed)
    left = gates(listed, facts, new, old, accuracy)
    kept = [name for name in listed.index if name not in set(left["model"])]
    gone = left.groupby("pool").size().reindex(POOLS, fill_value=0)
    stop = "" if (gone <= MAX_LEFT).all() else (
        f"more than {MAX_LEFT} models of a half leave "
        f"({', '.join(f'{name} {gone[name]}' for name in POOLS)})")
    present = [name for name in kept if name in facts.index]
    gap = old_gap(new, old, present)
    apart = old_gap(new, old, [name for name in listed.index if name in facts.index],
                    unstable=True)
    sections = [("Models and gates", (
        f"List: `{standin_list.PATH.name}`, {len(listed)} models. Result files: "
        f"{len(facts)}. Models that stay: {len(kept)} "
        f"({', '.join(f'{name} {int((listed.loc[kept, 'pool'] == name).sum())}' for name in POOLS)})."
        + f"\n\n**Gate C** ({len(ALL_CSFS) - len(UNSTABLE)} scores, without "
        f"{', '.join(UNSTABLE)}). Largest difference of an AUROC of an existing "
        f"task from the stored value, over the models that stay: "
        f"{float(gap.max()) if present else float('nan'):.4f} (limit "
        f"{CONSISTENCY}). **Gate S.** Tasks in the files: {len(tasks)} new and "
        f"{len(OLD_TASKS)} existing."
        + f"\n\n**{', '.join(UNSTABLE)}, reported apart** (amendment of "
        f"2026-11-27, before any result): its largest difference from the stored "
        f"value is {float(apart.max()):.4f} over the models (median "
        f"{float(apart.median()):.4f}), and it is above {CONSISTENCY} for "
        f"{int((apart > CONSISTENCY).sum())} of {len(apart)} models."
        + ("\n\nNo model leaves." if left.empty else
           f"\n\nModels that leave ({len(left)}):\n\n" + block(left))
        + f"\n\nTime of a model: {facts['seconds'].mean() / 60:.1f} minutes on "
        f"average, {facts['seconds'].sum() / 3600:.1f} hours in total."))]
    if stop:
        sections.append(("Decision", f"**The pilot gives no decision:** {stop}."))
    else:
        found = readings(listed.loc[kept], new.loc[kept],
                         {name: frame.loc[kept] for name, frame in targets.items()})
        chosen = found["accepted"]
        sections.append(("Reading 1: rank agreement of each task with each target", (
            "For each model, the rank correlation over the 23 scores between the "
            "AUROC on the task and the AUROC on the target. Mean over the models: "
            "torchvision half / timm half / both.\n\n"
            + block(found["agreement"].reset_index(names="task")))))
        show = found["candidates"].sort_values("gain", ascending=False)
        sections.append(("Each new task against the best existing task", (
            f"`gain` is the mean paired difference of the rank agreement. A task "
            f"passes for a target if the gain is at least {MARGIN} in each half, "
            f"the interval lies above 0, the agreement with each existing task is "
            f"below {AGREEMENT_LIMIT}, and the spread is at least {SPREAD_LIMIT}. "
            "The 15 rows with the largest gain:\n\n" + block(show.head(15), 3))))
        sections.append(("Readings 2 and 3: agreement with existing tasks, spread", (
            "Agreement of each new task with each existing task (the same "
            "statistic, between two tasks), the spread of the 23 AUROC values, "
            "and the AUROC of the best score, each as a mean over the models:\n\n"
            + block(found["between"].assign(
                spread=found["spread"], best_score=found["level"])
                .reset_index(names="task"), 3)
            + "\n\nThe existing tasks: " + ", ".join(
                f"{task} spread {found['spread'][task]:.3f}, best score "
                f"{found['level'][task]:.3f}" for task in OLD_TASKS) + ".")))
        large = chosen[(chosen["target"] == PRIMARY) & (chosen["gain"] >= LARGE)]
        sections.append(("Decision: accepted tasks", (
            ("**No task is accepted.**" if chosen.empty else
             f"**Accepted ({len(chosen)} of at most {MAX_ACCEPTED}):**\n\n"
             + block(chosen[["task", "target", "against", "gain", "low", "high"]], 3))
            + f"\n\nTasks that pass the rule for a target: "
            f"{int(found['candidates']['passes'].sum())} rows, "
            f"{found['candidates'].loc[found['candidates']['passes'], 'task'].nunique()}"
            f" tasks."
            + ("" if large.empty else
               f"\n\n**A gain of {LARGE} or more on the far sets:** "
               f"{', '.join(large['task'])}. The plan proposes a second evaluation "
               "of every model with this task. The user decides."))))
        sections.append(("Reading 4: the direct choice on the far sets", (
            "Mean regret in points of the correct-only AUROC of a choice with no "
            "fit. `mean_rank` takes the score with the best mean rank over the "
            "tasks, and `maximin` the score with the best worst rank. "
            "`against_the_three` is the paired difference from the same choice "
            "over the three existing tasks, with its interval.\n\n"
            + block(found["direct"], 2))))
    text = ("# Amendment A18, part 1: pilot of new stand-in tasks\n\nNo reading "
            "needs a fit, and no claim comes from the pilot. It decides which "
            "choices are frozen for the third-pool test.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    return text, stop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--models-file", default=str(standin_list.PATH))
    parser.add_argument("--allow-other-list", action="store_true",
                        help="accept a list with another hash (tests only)")
    parser.add_argument("--out", default=str(HERE / "outputs"
                                             / "stage_k_pilot_report.md"))
    args = parser.parse_args()
    path = Path(args.models_file)
    if standin_list.file_hash(path) != MODELS_SHA256 and not args.allow_other_list:
        raise SystemExit(f"{path.name} does not have the frozen hash")
    listed = pd.DataFrame(standin_list.load(path)).set_index("id")
    text, stop = report(listed, Path(args.results_root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    if stop:
        raise SystemExit(f"the pilot gives no decision: {stop}")


if __name__ == "__main__":
    main()
