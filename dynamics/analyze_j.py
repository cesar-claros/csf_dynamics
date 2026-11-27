"""Amendment A17: the frozen ImageNet-1k rule on models of another library.

    python -m dynamics.analyze_j freeze --results-root dyn_results
    python -m dynamics.analyze_j test --results-root dyn_results

`freeze` fits every choice on the 100 torchvision models of amendment A16 and
writes `dynamics/frozen/stage_j_rules.json` once: for each target a ridge rule
from the 69 statistics of the scores, one fixed score, two tables from the tag
of a model to a score, a rule from the validation accuracy alone, and a shortlist.

`test` reads the result files of the timm models of the frozen list
`dynamics/frozen/a17_models.json` (<results-root>/zoo_timm) and applies the
frozen choices with no new fit. Group N (new lineages) gives the outcome letter
on the far sets, with the lineage as the unit. Groups R (ResNet-50 with many
recipes) and S (seen lineages) are planned secondary readings.

Gates: W (the measured accuracy of a model is within 0.3 points of the published
one), S (a score or an input of a model has an error or a non-finite value), and
the head check (the head on the feature agrees with the network on the class).
A model that fails one of them leaves. Gate M compares the stored target with the
derived one.

Protocol: ../documentation/imagenet1k_second_library_amendment_2026-11-26.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_e, analyze_h, rules, zoo_list, zoo_timm_list
from dynamics import analyze_i as ai
from dynamics import analyze_multimetric as mm
from dynamics import analyze_p as ap
from dynamics import analyze_s as an
from dynamics import analyze_shortlist as sl
from dynamics import analyze_z as az
from dynamics.analyze_e import block
from dynamics.scores import ALL_CSFS

HERE = Path(__file__).resolve().parent
RULES_PATH = HERE / "frozen" / "stage_j_rules.json"
#: SHA-256 of the frozen choices and of the frozen list of timm models.
RULES_SHA256 = "17761b17b5b75ec1335cbec2d102349b69f67d2e03a2b3a13cc7ecc112c1f89e"
MODELS_SHA256 = "71e835e8ea31d18502bcea2cc93d2e53e89aff9ba1ad9dc9bcc41a9b82176858"
RESULTS = "zoo_timm"
PRIMARY = ai.PRIMARY
RULE, ONE, TAG, TAG_MODELS, ACCURACY = (
    "statistics", "one_score", "by_tag", "by_tag_models", "accuracy_only")
CHOICES = (RULE, ONE, TAG, TAG_MODELS, ACCURACY)
INSIDE, POOLED = "fit_inside", "fit_pooled"
ACCURACY_INPUT = "clock/val_accuracy"
#: A recipe that the library does not document takes the cell `ported`.
CELL_OF_RECIPE = {"unknown": "ported"}
#: Head check: smallest share of images with the same class.
AGREEMENT = 0.99
#: Count gate: fewest lineages of group N, and largest share of removed models.
MIN_LINEAGES = 20
LEFT_SHARE = 0.10
#: Draws of the control with the statistics in a random order.
DRAWS = 500
GROUP_TITLES = {"N": "new lineages", "R": "ResNet-50 with many recipes",
                "S": "seen lineages"}


def fit_tables(root: Path) -> tuple[analyze_e.Tables, pd.DataFrame, dict, pd.Series]:
    """The results of the 100 models of amendment A16, after its checks.

    Returns:
        The tables, the frozen list, the image counts, and the measured accuracy.

    Raises:
        SystemExit: If the list or the results are not those of amendment A16.
    """
    if zoo_list.file_hash(zoo_list.PATH) != ai.MODELS_SHA256:
        raise SystemExit(f"{zoo_list.PATH.name} does not have its frozen hash")
    listed = pd.DataFrame(zoo_list.load()).set_index("id")
    tables, facts = ai.read(Path(root))
    if facts.empty or sorted(facts["model"]) != sorted(listed.index):
        raise SystemExit("the result files are not those of the 100 models of "
                         "amendment A16")
    facts = facts.set_index("model").loc[listed.index]
    gap = (facts["val_accuracy"] - facts["published"]).abs()
    if (facts["limit"].notna().any() or facts["counts"].nunique() != 1
            or facts["failed"].astype(bool).any() or (gap > ai.ACCURACY_LIMIT).any()):
        raise SystemExit("a model of amendment A16 fails a gate of that amendment")
    return tables, listed, json.loads(facts["counts"].iloc[0]), facts["val_accuracy"]


def accuracy_frame(accuracy: pd.Series) -> pd.DataFrame:
    return accuracy.rename(ACCURACY_INPUT).to_frame()


def cells_of(listed: pd.DataFrame) -> pd.Series:
    """The cell of each model in a table by tag."""
    return listed["arch_type"] + "/" + listed["recipe_tag"].replace(CELL_OF_RECIPE)


def build(root: Path) -> dict:
    """The content of the frozen file, from the 100 models of amendment A16."""
    tables, listed, counts, accuracy = fit_tables(root)
    csfs = list(ALL_CSFS)
    metrics = ai.inputs_of(tables, csfs).loc[listed.index]
    weights = ai.balanced(listed["lineage"])
    cell = cells_of(listed)
    rows = {name: (cell == name).to_numpy() for name in sorted(cell.unique())}
    groups = {}
    for name, sets in ai.GROUPS.items():
        scores = ai.targets_of(tables, sets, csfs).loc[listed.index]
        if not np.isfinite(scores.to_numpy()).all():
            raise SystemExit(f"a score has no value on the {name} sets")
        groups[name] = {
            "sets": list(sets),
            "rule": asdict(ai.fit_weighted(metrics, scores, ai.usable(metrics),
                                           weights)),
            "accuracy_rule": asdict(ai.fit_weighted(
                accuracy_frame(accuracy), scores, [ACCURACY_INPUT], weights)),
            ONE: ai.weighted_best(scores, weights),
            TAG: {key: ai.weighted_best(scores[kept], weights[kept])
                  for key, kept in rows.items()},
            TAG_MODELS: {key: str(scores[kept].mean().idxmax())
                         for key, kept in rows.items()},
            "shortlist": sl.greedy(scores, ai.SHORT)}
    return {"amendment": "A17", "target": "correct_only", "penalty": mm.PENALTY,
            "fit_list": zoo_list.PATH.name, "fit_list_sha256": ai.MODELS_SHA256,
            "fit_models": list(listed.index), "images": counts,
            "input_range": {column: [float(metrics[column].min()),
                                     float(metrics[column].max())]
                            for column in metrics.columns},
            "groups": groups}


def load(path: Path = RULES_PATH) -> dict:
    frozen = json.loads(Path(path).read_text())
    for group in frozen["groups"].values():
        for name in ("rule", "accuracy_rule"):
            group[name] = rules.RidgeRule(**group[name])
    return frozen


def choose(group: dict, metrics: pd.DataFrame, accuracy: pd.Series,
           cells: pd.Series) -> pd.DataFrame:
    """The frozen choices for each model. None where the rule has no choice."""
    return pd.DataFrame({
        RULE: group["rule"].choose(metrics),
        ONE: group[ONE],
        TAG: [group[TAG].get(cell, group[ONE]) for cell in cells[metrics.index]],
        TAG_MODELS: [group[TAG_MODELS].get(cell, group[ONE])
                     for cell in cells[metrics.index]],
        ACCURACY: group["accuracy_rule"].choose(
            accuracy_frame(accuracy[metrics.index]))}, index=metrics.index)


def regrets(picks: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: analyze_h.regret_of(scores, list(picks[name]))
                         for name in picks.columns}, index=scores.index)


def gates(listed: pd.DataFrame, facts: pd.DataFrame, usable: pd.Series
          ) -> pd.DataFrame:
    """One row for each model that leaves, with the reason.

    Args:
        listed: The frozen list, by model id.
        facts: The facts of the result files, by model id.
        usable: True for a model whose 69 inputs are all finite.
    """
    rows = []
    for name in listed.index:
        if name not in facts.index:
            reason = "no result file"
        elif abs(facts.at[name, "val_accuracy"]
                 - facts.at[name, "published"]) > ai.ACCURACY_LIMIT:
            reason = (f"gate W: accuracy {100 * facts.at[name, 'val_accuracy']:.2f}, "
                      f"published {100 * facts.at[name, 'published']:.2f}")
        elif facts.at[name, "failed"]:
            reason = f"gate S: {facts.at[name, 'failed']}"
        elif not usable.get(name, False):
            reason = "gate S: an input is not finite"
        elif facts.at[name, "agreement"] < AGREEMENT:
            reason = (f"head check: the same class for "
                      f"{100 * facts.at[name, 'agreement']:.2f}% of the images")
        else:
            continue
        rows.append({"model": name, "group": listed.at[name, "group"],
                     "lineage": listed.at[name, "lineage"], "reason": reason})
    return pd.DataFrame(rows, columns=["model", "group", "lineage", "reason"])


def decided(listed: pd.DataFrame, kept: list[str], group: str) -> str:
    """Why a group has no reading, or an empty text."""
    planned = listed[listed["group"] == group]
    stay = planned.loc[[name for name in planned.index if name in kept]]
    if len(planned) - len(stay) > LEFT_SHARE * len(planned):
        return (f"more than {LEFT_SHARE:.0%} of the models of group {group} leave "
                f"({len(planned) - len(stay)} of {len(planned)})")
    if group == "N" and stay["lineage"].nunique() < MIN_LINEAGES:
        return (f"fewer than {MIN_LINEAGES} lineages of group N stay "
                f"({stay['lineage'].nunique()})")
    return ""


def outcome(units: pd.DataFrame) -> str:
    """A, H, or F for the frozen rule against the frozen fixed score."""
    gain = an.adds(units, RULE, ONE)["supported"]
    if not an.holds(units, RULE):
        return ("F, the frozen rule is not within 1 point"
                + (", and it is below the fixed score" if gain else ""))
    return ("A, the frozen rule adds to the fixed score and holds" if gain
            else "H, the frozen rule holds with no shown gain")


def label(units: pd.DataFrame) -> str:
    """T+, T0, or T- for the frozen rule against the two tables by tag."""
    first, second = an.adds(units, RULE, TAG), an.adds(units, RULE, TAG_MODELS)
    if first["supported"] and second["supported"]:
        return "T+, the rule is below each table by tag"
    if first["low"] > 0 or second["low"] > 0:
        return "T-, a table by tag is below the rule"
    return "T0, the rule is not shown different from the tables by tag"


def decision(units: pd.DataFrame, unit: str) -> str:
    """The planned sentences of one group on one target."""
    main = an.adds(units, RULE, ONE)
    text = (
        f"- **The rule adds** (frozen rule against the frozen fixed score): "
        f"{an.sentence(main)}.\n- **The rule holds:** "
        f"{'yes' if an.holds(units, RULE) else 'no'} (mean {units[RULE].mean():.2f}, "
        f"above {az.HOLD:g} point in {int((units[RULE] > az.HOLD).sum())} of "
        f"{len(units)} {unit}).\n- The fixed score holds: "
        f"{'yes' if an.holds(units, ONE) else 'no'} (mean {units[ONE].mean():.2f}, "
        f"above {az.HOLD:g} point in {int((units[ONE] > az.HOLD).sum())} of "
        f"{len(units)}).\n- The rule against the table by tag with lineage "
        f"weights: {an.sentence(an.adds(units, RULE, TAG))}.\n- The rule against "
        f"the table by tag with model weights: "
        f"{an.sentence(an.adds(units, RULE, TAG_MODELS))}.")
    if main["supported"] and not main["p"] < ap.MARGIN:
        text += (f"\n- The interval lies below 0 and the sign-flip p is not below "
                 f"{ap.MARGIN:g}: **the gain is at the margin.**")
    return text


def refits(metrics: pd.DataFrame, scores: pd.DataFrame, lineage: pd.Series,
           fit_metrics: pd.DataFrame, fit_scores: pd.DataFrame,
           fit_lineage: pd.Series) -> pd.DataFrame:
    """Regret of two rules that saw the new library, with each lineage left out.

    `fit_inside` is fitted on the other lineages of the group. `fit_pooled` is
    fitted on them and on the models of amendment A16.
    """
    out = pd.DataFrame(np.nan, index=scores.index, columns=[INSIDE, POOLED])
    for name in sorted(lineage.unique()):
        test = (lineage == name).to_numpy()
        inside = ai.fit_weighted(metrics[~test], scores[~test],
                                 ai.usable(metrics[~test]),
                                 ai.balanced(lineage[~test]))
        both_metrics = pd.concat([metrics[~test], fit_metrics])
        both_scores = pd.concat([scores[~test], fit_scores])
        both_lineage = pd.concat([lineage[~test], "a16/" + fit_lineage])
        pooled = ai.fit_weighted(both_metrics, both_scores, ai.usable(both_metrics),
                                 ai.balanced(both_lineage))
        for column, rule in ((INSIDE, inside), (POOLED, pooled)):
            out.loc[scores.index[test], column] = analyze_h.regret_of(
                scores[test], rule.choose(metrics[test])).to_numpy()
    return out


def random_order(rule: rules.RidgeRule, metrics: pd.DataFrame, scores: pd.DataFrame,
                 lineage: pd.Series, draws: int = DRAWS, seed: int = 0) -> np.ndarray:
    """Mean regret over the lineages of the frozen rule, with the statistics of
    the models in a random order."""
    rng = np.random.default_rng(seed)
    values = metrics.to_numpy()
    out = []
    for _ in range(draws):
        mixed = pd.DataFrame(values[rng.permutation(len(values))],
                             index=metrics.index, columns=metrics.columns)
        regret = analyze_h.regret_of(scores, rule.choose(mixed))
        out.append(float(regret.groupby(lineage).mean().mean()))
    return np.array(out)


def outside(frozen: dict, metrics: pd.DataFrame) -> pd.Series:
    """For each model, the number of inputs outside the range of the fit."""
    low = np.array([frozen["input_range"][c][0] for c in metrics.columns])
    high = np.array([frozen["input_range"][c][1] for c in metrics.columns])
    values = metrics.to_numpy()
    return pd.Series(((values < low) | (values > high)).sum(axis=1),
                     index=metrics.index)


def augrc_lost(tables: analyze_e.Tables, sets: tuple[str, ...], picks: pd.DataFrame,
               names: tuple[str, ...]) -> dict[str, float]:
    """Mean regret in AUGRC points of each choice over the models of `picks`."""
    part = tables.metrics[tables.metrics["set"].isin(sets)]
    augrc = 100 * part.pivot_table(index="run", columns="csf",
                                   values="augrc_correct").loc[picks.index]
    return {name: float(np.mean([augrc.at[run, picks.at[run, name]]
                                 - augrc.loc[run].min() for run in picks.index]))
            for name in names}


def primary_sections(frozen: dict, fit: tuple, tables: analyze_e.Tables,
                     listed: pd.DataFrame, metrics: pd.DataFrame,
                     accuracy: pd.Series, stops: dict[str, str]
                     ) -> list[tuple[str, str]]:
    """The outcome on the far sets and the planned secondary readings."""
    group = frozen["groups"][PRIMARY]
    sets = tuple(group["sets"])
    csfs = list(ALL_CSFS)
    scores = ai.targets_of(tables, sets, csfs).loc[metrics.index]
    picks = choose(group, metrics, accuracy, cells_of(listed))
    single = regrets(picks, scores)
    regret = scores.rsub(scores.max(axis=1), axis=0)
    sections = []
    part = {name: (listed.loc[metrics.index, "group"] == name).to_numpy()
            for name in ("N", "R", "S")}
    lineage = listed.loc[metrics.index, "lineage"]
    # Group N: the outcome.
    title = "Group N on the far sets: the outcome"
    if stops["N"]:
        sections.append((title, f"**No outcome letter:** {stops['N']}."))
    else:
        new, new_lineage = single[part["N"]], lineage[part["N"]]
        units = new.groupby(new_lineage).mean()
        sizes = new_lineage.value_counts()
        sections.append((title, (
            f"Sets: {', '.join(sets)}. Frozen fixed score: {group[ONE]}. Tables by "
            f"tag: {group[TAG]} (lineage weights), {group[TAG_MODELS]} (model "
            "weights). Regret in points of the correct-only AUROC, with the "
            "lineage as the unit (the mean of its models):\n\n"
            + block(units[[RULE, ONE, TAG, TAG_MODELS]].assign(models=sizes)
                    .reset_index(names="lineage"), 2)
            + "\n\nMeans over the lineages: " + ", ".join(
                f"{name} {units[name].mean():.2f}" for name in CHOICES)
            + "\n\n" + decision(units, "lineages")
            + f"\n\n**Outcome: {outcome(units)}.**\n\n**Second planned comparison: "
            f"{label(units)}.**")))
        sections += new_lineage_sections(frozen, fit, tables, listed, metrics[
            part["N"]], scores[part["N"]], picks[part["N"]], new, new_lineage, regret[
            part["N"]])
    # Group R: one architecture, many recipes.
    title = "Group R, one architecture and many recipes (secondary)"
    if stops["R"]:
        sections.append((title, f"**No reading:** {stops['R']}."))
    elif part["R"].any():
        recipes = single[part["R"]]
        table = recipes[[RULE, ONE, TAG, TAG_MODELS]].assign(
            choice=picks.loc[recipes.index, RULE],
            best=scores.loc[recipes.index].idxmax(axis=1),
            accuracy=accuracy[recipes.index])
        first = an.adds(recipes, RULE, TAG)
        second = an.adds(recipes, RULE, TAG_MODELS)
        inside = first["supported"] and second["supported"]
        sections.append((title, (
            f"{len(recipes)} weights of ResNet-50, with the recipe as the unit. "
            f"Different best scores: {table['best'].nunique()} "
            f"({', '.join(sorted(table['best'].unique()))}).\n\n"
            + block(table.reset_index(names="model"), 2) + "\n\n"
            + decision(recipes, "recipes")
            + "\n\n**The rule adds inside one architecture: "
            + ("yes" if inside else "not shown") + "** (the interval of the rule "
            "minus the choice by tag lies below 0 for each table: "
            + ("yes" if inside else "no") + ").")))
    # Group S: seen lineages.
    title = "Group S, seen lineages with weights of another library (secondary)"
    if stops["S"]:
        sections.append((title, f"**No reading:** {stops['S']}."))
    elif part["S"].any():
        seen = single[part["S"]]
        units = seen.groupby(lineage[part["S"]]).mean()
        sections.append((title, (
            block(units[[RULE, ONE, TAG, TAG_MODELS]].assign(
                models=lineage[part["S"]].value_counts())
                .reset_index(names="lineage"), 2)
            + "\n\nMeans over the lineages: " + ", ".join(
                f"{name} {units[name].mean():.2f}" for name in CHOICES)
            + "\n\n" + decision(units, "lineages")
            + f"\n\nBy the rules of the outcome (no outcome letter here): "
            f"{outcome(units)}; {label(units)}.")))
    models = listed.loc[metrics.index, ["group", "lineage", "arch_type",
                                        "recipe_tag"]].assign(
        accuracy=accuracy[metrics.index], choice=picks[RULE],
        best=scores.idxmax(axis=1), rule=single[RULE], one=single[ONE],
        tag=single[TAG], tag_models=single[TAG_MODELS])
    sections.append(("Models", block(models.reset_index(names="model"), 2)))
    return sections


def new_lineage_sections(frozen: dict, fit: tuple, tables: analyze_e.Tables,
                         listed: pd.DataFrame, metrics: pd.DataFrame,
                         scores: pd.DataFrame, picks: pd.DataFrame,
                         single: pd.DataFrame, lineage: pd.Series,
                         regret: pd.DataFrame) -> list[tuple[str, str]]:
    """The planned secondary readings of group N on the far sets."""
    group = frozen["groups"][PRIMARY]
    units = single.groupby(lineage).mean()
    sections = []
    best = str(regret.mean().idxmin())
    kind = listed.loc[single.index, "arch_type"]
    sections.append((("Group N: the selection problem, the type, and the model "
                      "as the unit (secondary)"), (
        f"- The best fixed score in hindsight inside group N is {best}, with a "
        f"mean regret of {regret[best].mean():.2f} over the models. It leaves "
        f"{int((regret[best] > az.HOLD).sum())} of {len(regret)} models more than "
        f"{az.HOLD:g} point below their best score. The frozen rule leaves "
        f"{int((single[RULE] > az.HOLD).sum())}, and the frozen fixed score "
        f"{int((single[ONE] > az.HOLD).sum())}.\n- With the model as the unit "
        f"({len(single)}): the rule against the fixed score: "
        f"{an.sentence(an.adds(single, RULE, ONE))}; against the table by tag "
        f"with lineage weights: {an.sentence(an.adds(single, RULE, TAG))}; with "
        f"model weights: {an.sentence(an.adds(single, RULE, TAG_MODELS))}.\n\n"
        "By type, over the models:\n\n"
        + block(single[[RULE, ONE, TAG, TAG_MODELS]].groupby(kind).mean()
                .assign(models=kind.value_counts()).reset_index(names="type"), 2))))
    fit_tables_, fit_listed, fit_metrics = fit
    fit_scores = ai.targets_of(fit_tables_, tuple(group["sets"]),
                               list(ALL_CSFS)).loc[fit_listed.index]
    again = refits(metrics, scores, lineage, fit_metrics, fit_scores,
                   fit_listed["lineage"])
    both = pd.concat([units, again.groupby(lineage).mean()], axis=1)
    sections.append(("Group N: fits that saw the new library (secondary)", (
        "Each lineage is left out once. `fit_inside` is a rule fitted on the "
        "other lineages of group N. `fit_pooled` is fitted on them and on the "
        "100 models of amendment A16.\n\n"
        f"- Means over the lineages: frozen rule {both[RULE].mean():.2f}, "
        f"{INSIDE} {both[INSIDE].mean():.2f}, {POOLED} {both[POOLED].mean():.2f}."
        f"\n- {INSIDE} against the frozen rule: "
        f"{an.sentence(an.adds(both, INSIDE, RULE))}. It holds: "
        f"{'yes' if an.holds(both, INSIDE) else 'no'}.\n- {POOLED} against the "
        f"frozen rule: {an.sentence(an.adds(both, POOLED, RULE))}. It holds: "
        f"{'yes' if an.holds(both, POOLED) else 'no'}.\n- {INSIDE} against the "
        f"frozen fixed score: {an.sentence(an.adds(both, INSIDE, ONE))}.")))
    null = random_order(group["rule"], metrics, scores, lineage)
    real = float(units[RULE].mean())
    sections.append(("Group N: negative controls (secondary)", (
        f"- The frozen rule with the statistics of the models in a random order "
        f"({len(null)} draws): mean regret {null.mean():.2f}, 5% to 95% "
        f"{np.quantile(null, 0.05):.2f} to {np.quantile(null, 0.95):.2f}, lowest "
        f"{null.min():.2f}. Draws at or below the frozen rule ({real:.2f}): "
        f"{int((null <= real).sum())} of {len(null)}.\n- A frozen rule from the "
        f"validation accuracy alone against the fixed score: "
        f"{an.sentence(an.adds(units, ACCURACY, ONE))}.\n- The frozen rule "
        f"against it: {an.sentence(an.adds(units, RULE, ACCURACY))}.")))
    short = pd.DataFrame({
        "fixed": sl.shortlist_regret(scores, group["shortlist"]),
        "rule": sl.shortlist_regret(scores, analyze_h.top(group["rule"], metrics,
                                                           ai.SHORT)),
        "random": sl.random_regret(scores, ai.SHORT)}).groupby(lineage).mean()
    lost = augrc_lost(tables, tuple(group["sets"]), picks, (RULE, ONE, TAG))
    sections.append(("Group N: shortlists and AUGRC (secondary)", (
        f"- Shortlists of size {ai.SHORT}, with the lineage as the unit: the rule "
        f"has {short['rule'].mean():.2f}, the frozen shortlist "
        f"({', '.join(group['shortlist'])}) {short['fixed'].mean():.2f}, and a "
        f"random one {short['random'].mean():.2f}. Rule against fixed: "
        f"{an.sentence(an.adds(short, 'rule', 'fixed'))}.\n- Regret in AUGRC "
        f"points over the models: rule {lost[RULE]:.3f}, fixed score "
        f"{lost[ONE]:.3f}, table by tag {lost[TAG]:.3f}.")))
    count = outside(frozen, metrics)
    bins = pd.cut(count, [-1, 0, 5, len(metrics.columns)],
                  labels=["0", "1 to 5", "more than 5"])
    table = pd.DataFrame({"rule": single[RULE], "one": single[ONE]}).groupby(
        bins, observed=True).mean().assign(models=bins.value_counts())
    sections.append(("Group N: inputs outside the range of the fit (secondary)", (
        f"For each model, the number of the {len(metrics.columns)} inputs that "
        f"are outside the range of the 100 models of the fit: median "
        f"{count.median():.0f}, largest {count.max()}. Rank correlation of this "
        f"number with the regret of the rule: "
        f"{count.corr(single[RULE], method='spearman'):.2f}.\n\n"
        + block(table.reset_index(names="inputs outside"), 2))))
    return sections


def other_targets(frozen: dict, tables: analyze_e.Tables, listed: pd.DataFrame,
                  metrics: pd.DataFrame, accuracy: pd.Series, stops: dict[str, str]
                  ) -> list[tuple[str, str]]:
    """Near sets, textures, and ImageNet-O, each with its frozen choices."""
    sections = []
    cells = cells_of(listed)
    for name, group in frozen["groups"].items():
        if name == PRIMARY:
            continue
        scores = ai.targets_of(tables, tuple(group["sets"]),
                               list(ALL_CSFS)).loc[metrics.index]
        single = regrets(choose(group, metrics, accuracy, cells), scores)
        lines = []
        for key in ("N", "S", "R"):
            rows = (listed.loc[metrics.index, "group"] == key).to_numpy()
            if stops[key] or not rows.any():
                continue
            units = single[rows] if key == "R" else single[rows].groupby(
                listed.loc[metrics.index, "lineage"][rows]).mean()
            lines.append(
                f"- **Group {key}** ({len(units)} units): rule against the fixed "
                f"score: {an.sentence(an.adds(units, RULE, ONE))}. The rule holds: "
                f"{'yes' if an.holds(units, RULE) else 'no'}. Table by tag with "
                f"lineage weights {units[TAG].mean():.2f}, with model weights "
                f"{units[TAG_MODELS].mean():.2f}.")
        sections.append((f"Target {name} (secondary, no outcome letter)", (
            f"Sets: {', '.join(group['sets'])}. Frozen fixed score: {group[ONE]}. "
            f"Table by tag: {group[TAG]}.\n\n" + "\n".join(lines))))
    return sections


def test_report(frozen: dict, listed: pd.DataFrame, root: Path,
                allow_missing: bool = False, strict: bool = True) -> tuple[str, str]:
    """The report of the test, and why it stops (an empty text if it does not).

    Raises:
        SystemExit: If the result files are not those of a protocol run.
    """
    tables, facts = ai.read(Path(root), RESULTS)
    if facts.empty:
        raise SystemExit(f"no result files under {root}/{RESULTS}")
    facts = facts.set_index("model")
    unknown = sorted(set(facts.index) - set(listed.index))
    if unknown:
        raise SystemExit(f"results of models that are not in the list: {unknown}")
    missing = sorted(set(listed.index) - set(facts.index))
    if missing and not allow_missing:
        raise SystemExit("the evaluation is not complete; models with no result "
                         f"file: {', '.join(missing)}")
    if strict and facts["limit"].notna().any():
        raise SystemExit("a result file comes from a smoke run")
    counts = [json.loads(text) for text in facts["counts"].unique()]
    if len(counts) != 1 or (strict and counts[0] != frozen["images"]):
        raise SystemExit("the image counts are not those of the frozen file")
    csfs = list(ALL_CSFS)
    present = [name for name in listed.index if name in facts.index]
    every = ai.inputs_of(tables, csfs).reindex(present)
    usable = pd.Series(np.isfinite(every.to_numpy()).all(axis=1), index=present)
    left = gates(listed, facts, usable)
    kept = [name for name in present if name not in set(left["model"])]
    tables = ai.keep(tables, kept)
    sets = tuple(name for group in frozen["groups"].values()
                 for name in group["sets"])
    gap, skipped = ai.metric_gap(tables, counts[0], sets)
    if gap > az.METRIC_LIMIT:
        raise SystemExit(f"gate M fails: the stored and the derived target differ "
                         f"by {gap:.2e}")
    stops = {group: decided(listed, kept, group) for group in ("N", "R", "S")}
    seen = facts.loc[kept]
    difference = (facts["val_accuracy"] - facts["published"]).abs()
    sections = [("Frozen files and gates", (
        f"Rules: `{RULES_PATH.name}`, fitted on {len(frozen['fit_models'])} models "
        f"of amendment A16. List: `{zoo_timm_list.PATH.name}`, {len(listed)} "
        f"models. Result files: {len(facts)}."
        + "\n\nBy group, the planned models and the models that stay:\n\n"
        + block(pd.DataFrame({
            "planned": listed["group"].value_counts(),
            "stay": listed.loc[kept, "group"].value_counts(),
            "lineages": listed.loc[kept].groupby("group")["lineage"].nunique()})
            .fillna(0).astype(int).reset_index(names="group"))
        + f"\n\n**Gate W.** Largest difference between the measured and the "
        f"published accuracy: {100 * difference.max():.2f} points (limit "
        f"{100 * ai.ACCURACY_LIMIT:.1f}). **Gate M passes:** {gap:.2e} (limit "
        f"{az.METRIC_LIMIT:.0e}), rows that could not be compared: {skipped}. "
        f"**Head check:** the lowest agreement of a model that stays is "
        f"{100 * seen['agreement'].min():.2f}%."
        + ("\n\nNo model leaves." if left.empty else
           f"\n\nModels that leave ({len(left)}):\n\n" + block(left))
        + f"\n\nTime of an evaluation: {facts['seconds'].mean() / 60:.1f} minutes "
        f"on average, {facts['seconds'].sum() / 3600:.1f} hours in total."))]
    metrics = every.loc[kept]
    accuracy = seen["val_accuracy"]
    chosen = listed.loc[kept]
    fit_tables_, fit_listed, _, _ = fit_tables(root)
    fit = (fit_tables_, fit_listed,
           ai.inputs_of(fit_tables_, csfs).loc[fit_listed.index])
    sections += primary_sections(frozen, fit, tables, chosen, metrics, accuracy,
                                 stops)
    sections += other_targets(frozen, tables, chosen, metrics, accuracy, stops)
    text = ("# Amendment A17: the frozen ImageNet-1k rule on models of another "
            "library\n\nNothing is trained and nothing is fitted for the outcome: "
            "every choice comes from the frozen file. Group N (new lineages) "
            "gives the outcome letter.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    return text, stops["N"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("freeze", "test"))
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--rules", default=str(RULES_PATH))
    parser.add_argument("--models-file", default=str(zoo_timm_list.PATH))
    parser.add_argument("--allow-other-files", action="store_true",
                        help="accept files with other hashes and other image "
                             "counts (tests only)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="a model with no result file (its evaluation failed) "
                             "is reported and counts as a model that leaves")
    parser.add_argument("--out")
    args = parser.parse_args()
    root, path = Path(args.results_root), Path(args.rules)
    if args.step == "freeze":
        if path.exists():
            raise SystemExit(f"{path.name} exists: the choices are frozen once")
        content = build(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=1) + "\n")
        lines = [f"- Target {name}: fixed score {group[ONE]}; by tag with lineage "
                 f"weights {group[TAG]}; with model weights {group[TAG_MODELS]}; "
                 f"shortlist {group['shortlist']}."
                 for name, group in content["groups"].items()]
        text = (f"# Amendment A17: the frozen choices\n\nFile `{path.name}`, SHA-256 "
                f"`{rules.file_hash(path)}`.\n\n- Models of the fit: "
                f"{len(content['fit_models'])}, in "
                f"{len(set(pd.DataFrame(zoo_list.load())['lineage']))} lineages. "
                f"Inputs of the rule: "
                f"{len(content['groups'][PRIMARY]['rule']['columns'])}.\n"
                + "\n".join(lines) + "\n")
        out = Path(args.out) if args.out else HERE / "outputs" / "stage_j_freeze.md"
        stop = ""
    else:
        models = Path(args.models_file)
        if not args.allow_other_files:
            for file, wanted in ((path, RULES_SHA256), (models, MODELS_SHA256)):
                if not wanted or rules.file_hash(file) != wanted:
                    raise SystemExit(f"{file.name} does not have the frozen hash")
        listed = pd.DataFrame(zoo_timm_list.load(models)).set_index("id")
        text, stop = test_report(load(path), listed, root, args.allow_missing,
                                 strict=not args.allow_other_files)
        out = Path(args.out) if args.out else HERE / "outputs" / "stage_j_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    if stop:
        raise SystemExit(f"no outcome letter: {stop}")


if __name__ == "__main__":
    main()
