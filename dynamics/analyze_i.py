"""Amendment A16: the rule from the score statistics on public ImageNet-1k models.

    python -m dynamics.analyze_i --results-root dyn_results

Reads the result files of `dynamics.zoo`, one for each model of the frozen list.
Each architecture lineage is left out once: a ridge rule from the 69 statistics
of the scores, one fixed score, and a choice by tag are fitted on the models of
the other lineages, with the same total weight for each lineage, and they choose
a score for the models of the left-out lineage. The lineage is the unit.

Gates: W (the measured accuracy of a model is within 0.3 points of the published
one, else the model leaves), S (a score with an error or a non-finite value for a
model is removed for every model), and M (the stored target against the derived
one).

Protocol: ../documentation/imagenet1k_public_models_protocol_2026-11-24.md
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_e, analyze_h, rules, zoo_list
from dynamics import analyze_multimetric as mm
from dynamics import analyze_p as ap
from dynamics import analyze_s as an
from dynamics import analyze_shortlist as sl
from dynamics import analyze_z as az
from dynamics.analyze_e import block
from dynamics.analyze_step0 import FAR_SETS
from dynamics.analyze_t import SCORE_SETS
from dynamics.scores import ALL_CSFS

HERE = Path(__file__).resolve().parent
RESULTS = "zoo"
#: SHA-256 of the frozen list of models, public before any evaluation.
MODELS_SHA256 = "13872261c60b0b6641fa389b981c52f6bd4d72083a2e6a81e2e6d9961b98ea07"
GROUPS = {"far": FAR_SETS, "near": ("ssb_hard", "ninco"), "textures": ("textures",),
          "imagenet_o": ("imagenet_o",)}
PRIMARY = "far"
RULE, ONE, TAG, HINDSIGHT = "statistics", "one_score", "by_tag", "hindsight"
CHOICES = (RULE, ONE, TAG)
#: Gate W: largest difference between the measured and the published accuracy.
ACCURACY_LIMIT = 0.003
#: The test is not decided if more than this share of the models leaves.
LEFT_SHARE = 0.25
#: Image counts of a protocol evaluation.
COUNTS = {"val": 50_000, "train": 100_000, "blend": 10_000, "shuffle": 10_000}
SHORT = 3


def read(root: Path) -> tuple[analyze_e.Tables, pd.DataFrame]:
    """The result files as tables, and one row of facts for each model."""
    parts: dict[str, list[dict]] = {"metrics": [], "descriptors": []}
    facts = []
    for path in sorted((Path(root) / RESULTS).glob("*.json")):
        out = json.loads(path.read_text())
        key = {"run": out["model"], "recipe": out["model"], "seed": 0, "held_out": -1,
               "epoch": 1}
        for name in parts:
            parts[name] += [{**key, **row} for row in out[name]]
        bad = sorted({*out["errors"],
                      *(csf for csf, n in out["nonfinite"].items() if n)})
        facts.append({"model": out["model"], "val_accuracy": out["val_accuracy"],
                      "published": out["published_accuracy"],
                      "feature_dim": out["feature_dim"], "seconds": out["seconds"],
                      "failed": ",".join(bad), "limit": out.get("limit"),
                      "head_gap": out["checks"]["head_gap"],
                      "agreement": out["checks"]["argmax_agreement"],
                      "counts": json.dumps(out["counts"], sort_keys=True)})
    tables = analyze_e.Tables(metrics=pd.DataFrame(parts["metrics"]),
                              descriptors=pd.DataFrame(parts["descriptors"]),
                              placement=pd.DataFrame(), checks=pd.DataFrame())
    return tables, pd.DataFrame(facts)


def keep(tables: analyze_e.Tables, models: list[str]) -> analyze_e.Tables:
    """The same tables with the rows of `models` only."""
    return analyze_e.Tables(**{
        name: frame if frame.empty else frame[frame["run"].isin(models)]
        .reset_index(drop=True) for name, frame in vars(tables).items()})


def metric_gap(tables: analyze_e.Tables, counts: dict[str, int],
               sets: tuple[str, ...]) -> tuple[float, int]:
    """Gate M: the stored correct-only AUROC against the one derived from the
    stored AUGRC and the stored ID-errors AUROC."""
    derived = az.aimed(tables, counts, sets).metrics
    rows = derived["set"].isin(sets).to_numpy()
    kept = tables.metrics.loc[rows, "auroc_correct"].to_numpy(dtype=np.float64)
    other = derived.loc[rows, "auroc"].to_numpy(dtype=np.float64)
    both = np.isfinite(kept) & np.isfinite(other)
    if not both.any():
        raise SystemExit("no row with a stored and a derived target")
    return float(np.abs(kept[both] - other[both]).max()), int((~both).sum())


def inputs_of(tables: analyze_e.Tables, csfs: list[str]) -> pd.DataFrame:
    """The statistics of the scores: per model, the AUROC of each score on each
    stand-in set."""
    frames = []
    for name in SCORE_SETS:
        part = tables.metrics[(tables.metrics["set"] == name)
                              & tables.metrics["csf"].isin(csfs)]
        wide = part.pivot_table(index="run", columns="csf", values="auroc")
        frames.append(wide[csfs].add_prefix(f"score/{name}/"))
    return pd.concat(frames, axis=1)


def targets_of(tables: analyze_e.Tables, sets: tuple[str, ...],
               csfs: list[str]) -> pd.DataFrame:
    """Per model, the mean correct-only AUROC of each score over `sets`, in points."""
    part = tables.metrics[tables.metrics["set"].isin(sets)
                          & tables.metrics["csf"].isin(csfs)]
    return 100 * part.pivot_table(index="run", columns="csf",
                                  values="auroc_correct")[csfs]


def balanced(lineage: pd.Series) -> np.ndarray:
    """Weights with the same total for each lineage, and a mean of 1."""
    size = lineage.map(lineage.value_counts())
    weights = 1.0 / size.to_numpy(dtype=np.float64)
    return weights * len(weights) / weights.sum()


def fit_weighted(metrics: pd.DataFrame, scores: pd.DataFrame, columns: list[str],
                 weights: np.ndarray, penalty: float = mm.PENALTY) -> rules.RidgeRule:
    """The ridge rule of `rules.fit_ridge_rule`, with a weight for each row.

    With equal weights it is the same rule.

    Raises:
        ValueError: If a column has a non-finite training value.
    """
    x = metrics[columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(x).all():
        raise ValueError("a training column has a non-finite value")
    w = np.asarray(weights, dtype=np.float64)
    w = w * len(w) / w.sum()
    log = (x > 0).all(axis=0)
    floor = np.where(log, x.min(axis=0), np.nan)
    x[:, log] = np.log10(x[:, log])
    mean = (w[:, None] * x).sum(axis=0) / w.sum()
    std = np.sqrt((w[:, None] * (x - mean) ** 2).sum(axis=0) / w.sum())
    std[std == 0] = 1.0
    fit = (x - mean) / std
    target = mm.logit(scores.to_numpy(dtype=np.float64))
    offset = (w[:, None] * target).sum(axis=0) / w.sum()
    beta = np.linalg.solve(fit.T @ (w[:, None] * fit) + penalty * np.eye(len(columns)),
                           fit.T @ (w[:, None] * (target - offset)))
    floors: list[float | None] = [None if np.isnan(v) else float(v) for v in floor]
    return rules.RidgeRule(columns, log.tolist(), floors, mean.tolist(), std.tolist(),
                           beta.tolist(), offset.tolist(), list(scores.columns))


def usable(metrics: pd.DataFrame) -> list[str]:
    """The input columns that are finite and have more than one value."""
    return list(mm.transform(metrics).columns)


def weighted_best(scores: pd.DataFrame, weights: np.ndarray) -> str:
    """The score with the best weighted mean AUROC."""
    return str((scores.mul(weights, axis=0).sum() / weights.sum()).idxmax())


def leave_out(metrics: pd.DataFrame, scores: pd.DataFrame, listed: pd.DataFrame,
              weight: Callable[[pd.Series], np.ndarray] = balanced
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The choices for each model with its lineage left out of every fit.

    Args:
        metrics: The statistics of the scores, one row for each model.
        scores: The AUROC of each score on the target, one row for each model.
        listed: `lineage`, `arch_type`, and `recipe_tag` of each model.
        weight: The weight of each model of a fit, from the lineages of the fit.

    Returns:
        The choice of the rule, of the fixed score, and by tag, and the three
        scores of each shortlist (`rule_short`, `fixed_short`), per model.
    """
    picks = pd.DataFrame(index=scores.index, columns=[
        *CHOICES, "rule_short", "fixed_short"], dtype=object)
    cell = listed["arch_type"] + "/" + listed["recipe_tag"]
    for lineage in sorted(listed["lineage"].unique()):
        test = (listed["lineage"] == lineage).to_numpy()
        train = ~test
        weights = weight(listed.loc[train, "lineage"])
        rule = fit_weighted(metrics[train], scores[train], usable(metrics[train]),
                            weights)
        one = weighted_best(scores[train], weights)
        by_cell = {name: weighted_best(scores[train][(cell[train] == name).to_numpy()],
                                       weights[(cell[train] == name).to_numpy()])
                   for name in cell[train].unique()}
        names = scores.index[test]
        picks.loc[names, RULE] = rule.choose(metrics[test])
        picks.loc[names, ONE] = one
        picks.loc[names, TAG] = [by_cell.get(name, one) for name in cell[test]]
        short = sl.greedy(scores[train], SHORT)
        for name, top in zip(names, analyze_h.top(rule, metrics[test], SHORT)):
            picks.at[name, "rule_short"] = top
            picks.at[name, "fixed_short"] = short
    return picks[list(CHOICES)], picks[["rule_short", "fixed_short"]]


def regrets(picks: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Regret of each choice per model, with the best fixed score in hindsight."""
    single = pd.DataFrame({name: analyze_h.regret_of(scores, list(picks[name]))
                           for name in picks.columns}, index=scores.index)
    regret = scores.rsub(scores.max(axis=1), axis=0)
    single[HINDSIGHT] = regret[str(regret.mean().idxmin())]
    return single


def outcome(units: pd.DataFrame) -> str:
    """A, H, or F for the rule against the fixed score, over the lineages."""
    if units.empty or units[[RULE, ONE]].isna().any().any():
        return "not decided: a lineage has no choice"
    gain = an.adds(units, RULE, ONE)["supported"]
    if not an.holds(units, RULE):
        return ("F, the rule is not within 1 point"
                + (", and it is below the fixed score" if gain else ""))
    return ("A, the rule adds to the fixed score" if gain
            else "H, the rule holds with no shown gain")


def target_section(name: str, tables: analyze_e.Tables, metrics: pd.DataFrame,
                   listed: pd.DataFrame, csfs: list[str]
                   ) -> tuple[list[tuple[str, str]], pd.DataFrame, pd.DataFrame]:
    """The reading of one target: its sections, the choices, and the regrets."""
    scores = targets_of(tables, GROUPS[name], csfs).loc[metrics.index]
    picks, short = leave_out(metrics, scores, listed)
    single = regrets(picks, scores)
    units = single.groupby(listed["lineage"]).mean()
    sizes = listed["lineage"].value_counts()
    main, fixed = an.adds(units, RULE, ONE), an.adds(units, RULE, TAG)
    regret = scores.rsub(scores.max(axis=1), axis=0)
    best = str(regret.mean().idxmin())
    title = (f"Target {name}: the outcome" if name == PRIMARY
             else f"Target {name} (secondary, no outcome letter)")
    text = (
        f"Sets: {', '.join(GROUPS[name])}. Regret in points of the correct-only "
        "AUROC, with the lineage as the unit (the mean of its models):\n\n"
        + block(units.assign(models=sizes).reset_index(names="lineage"), 2)
        + "\n\nMeans over the lineages: " + ", ".join(
            f"{choice} {units[choice].mean():.2f}" for choice in units)
        + f"\n\n- **The rule adds** (rule against the fixed score): "
        f"{an.sentence(main)}.\n- **The rule holds:** "
        f"{'yes' if an.holds(units, RULE) else 'no'} (mean {units[RULE].mean():.2f}, "
        f"above {az.HOLD:g} point in {int((units[RULE] > az.HOLD).sum())} of "
        f"{len(units)} lineages).\n- The fixed score holds: "
        f"{'yes' if an.holds(units, ONE) else 'no'} (mean {units[ONE].mean():.2f}, "
        f"above {az.HOLD:g} point in {int((units[ONE] > az.HOLD).sum())} of "
        f"{len(units)}).\n- The rule against the choice by tag: "
        f"{an.sentence(fixed)}.\n- The best fixed score in hindsight is {best}, with "
        f"a mean regret of {regret[best].mean():.2f} over the models. It leaves "
        f"{int((regret[best] > az.HOLD).sum())} of {len(regret)} models more than "
        f"{az.HOLD:g} point below their best score.")
    if name == PRIMARY:
        if main["supported"] and not main["p"] < ap.MARGIN:
            text += (f"\n- The interval lies below 0 and the sign-flip p is not "
                     f"below {ap.MARGIN:g}: **the gain is at the margin.**")
        text += f"\n\n**Outcome: {outcome(units)}.**"
    sections = [(title, text)]
    if name == PRIMARY:
        sections += secondary(tables, metrics, scores, listed, picks, short, single)
    return sections, picks, single


def secondary(tables: analyze_e.Tables, metrics: pd.DataFrame, scores: pd.DataFrame,
              listed: pd.DataFrame, picks: pd.DataFrame, short: pd.DataFrame,
              single: pd.DataFrame) -> list[tuple[str, str]]:
    """The planned secondary readings of the far target."""
    sections = []
    for unit, group in (("family", listed["family"]), ("model", None)):
        units = single if group is None else single.groupby(group).mean()
        sections.append((f"The {unit} as the unit (secondary)", (
            f"{len(units)} units, with the same fits (the lineage is left out):\n\n- "
            f"The rule against the fixed score: "
            f"{an.sentence(an.adds(units, RULE, ONE))}.\n- The rule against the "
            f"choice by tag: {an.sentence(an.adds(units, RULE, TAG))}.")))
    cell = listed["arch_type"] + "/" + listed["recipe_tag"]
    sections.append(("By tag (secondary)", block(
        single.groupby(cell).mean().assign(models=cell.value_counts())
        .reset_index(names="tag"), 2)))
    table = pd.DataFrame({
        "fixed": sl.shortlist_regret(scores, list(short["fixed_short"])),
        "rule": sl.shortlist_regret(scores, list(short["rule_short"])),
        "random": sl.random_regret(scores, SHORT)})
    units = table.groupby(listed["lineage"]).mean()
    reading = an.adds(units, "rule", "fixed")
    part = tables.metrics[tables.metrics["set"].isin(GROUPS[PRIMARY])
                          & tables.metrics["csf"].isin(list(scores.columns))]
    augrc = 100 * part.pivot_table(index="run", columns="csf",
                                   values="augrc_correct").loc[scores.index]
    lost = {name: float(np.mean([augrc.loc[run, picks.loc[run, name]]
                                 - augrc.loc[run].min() for run in scores.index]))
            for name in CHOICES}
    sections.append(("Shortlists and AUGRC (secondary)", (
        f"- Shortlists of size {SHORT}, with the lineage as the unit: the rule has "
        f"{units['rule'].mean():.2f}, the fixed shortlist {units['fixed'].mean():.2f}"
        f", and a random one {units['random'].mean():.2f}. Rule against fixed: "
        f"{an.sentence(reading)}.\n- Regret in AUGRC points over the models: rule "
        f"{lost[RULE]:.3f}, fixed score {lost[ONE]:.3f}, choice by tag "
        f"{lost[TAG]:.3f}.")))
    rule = fit_weighted(metrics, scores, usable(metrics), balanced(listed["lineage"]))
    size = pd.Series(np.sqrt((np.array(rule.beta) ** 2).mean(axis=1)),
                     index=rule.columns).sort_values(ascending=False)
    sections.append(("The inputs that carry the choice (secondary)", (
        "A fit on every model. The size of the standardized weights of each "
        "input, as the root mean square over the scores. The ten largest:\n\n"
        + block(size.head(10).rename("size").rename_axis("input").reset_index(), 3))))
    models = listed[["lineage", "family", "arch_type", "recipe_tag"]].assign(
        choice=picks[RULE], fixed=picks[ONE], best=scores.idxmax(axis=1),
        rule=single[RULE], one=single[ONE], tag=single[TAG])
    sections.append(("Models", block(models.reset_index(names="model").round(2))))
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--models-file", default=str(zoo_list.PATH))
    parser.add_argument("--allow-other-list", action="store_true",
                        help="accept a list with another hash (tests only)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="a model with no result file (its evaluation failed) "
                             "is reported and left out")
    parser.add_argument("--out", default=str(HERE / "outputs" / "stage_i_report.md"))
    args = parser.parse_args()
    path = Path(args.models_file)
    digest = zoo_list.file_hash(path)
    if digest != MODELS_SHA256 and not args.allow_other_list:
        raise SystemExit(f"{path.name} has hash {digest}, not the frozen one")
    listed = pd.DataFrame(zoo_list.load(path)).set_index("id")
    tables, facts = read(Path(args.results_root))
    if facts.empty:
        raise SystemExit(f"no result files under {args.results_root}/{RESULTS}")
    facts = facts.set_index("model")
    unknown = sorted(set(facts.index) - set(listed.index))
    if unknown:
        raise SystemExit(f"results of models that are not in the list: {unknown}")
    missing = sorted(set(listed.index) - set(facts.index))
    if missing and not args.allow_missing:
        raise SystemExit("the evaluation is not complete; models with no result "
                         f"file: {', '.join(missing)}")
    if facts["limit"].notna().any() and not args.allow_other_list:
        raise SystemExit("a result file comes from a smoke run")
    if facts["counts"].nunique() != 1:
        raise SystemExit("the models do not have equal image counts")
    counts = json.loads(facts["counts"].iloc[0])
    wrong = {name: counts.get(name) for name, value in COUNTS.items()
             if counts.get(name) != value}
    if wrong and not args.allow_other_list:
        raise SystemExit(f"the image counts are not those of the protocol: {wrong}")
    gap = (facts["val_accuracy"] - facts["published"]).abs()
    off = facts[gap > ACCURACY_LIMIT]
    kept = [name for name in listed.index if name in facts.index
            and name not in off.index]
    failed = sorted({csf for text in facts.loc[kept, "failed"] if text
                     for csf in text.split(",")})
    csfs = [csf for csf in ALL_CSFS if csf not in failed]
    tables = keep(tables, kept)
    every = tuple(name for sets in GROUPS.values() for name in sets)
    m_gap, skipped = metric_gap(tables, counts, every)
    if m_gap > az.METRIC_LIMIT:
        raise SystemExit(f"gate M fails: the stored and the derived target differ "
                         f"by {m_gap:.2e}")
    left = len(listed) - len(kept)
    sections = [("Frozen list and gates", (
        f"List: `{path.name}`, {len(listed)} models. Result files: {len(facts)}."
        + (f" Models with no result file, left out: {', '.join(missing)}."
           if missing else "")
        + f"\n\n**Gate W.** The measured accuracy is within "
        f"{100 * ACCURACY_LIMIT:.1f} points of the published one for "
        f"{len(facts) - len(off)} of {len(facts)} models (largest difference "
        f"{100 * gap.max():.2f} points)."
        + ("" if off.empty else " Models that leave:\n\n" + block(
            off[["val_accuracy", "published"]].reset_index(), 4))
        + f"\n\n**Gate S.** Scores with an error or a non-finite value for a model, "
        f"removed for every model: {', '.join(failed) or 'none'}. "
        f"{len(csfs)} of {len(ALL_CSFS)} scores stay."
        + f"\n\n**Gate M passes.** Largest difference between the stored and the "
        f"derived target: {m_gap:.2e} (limit {az.METRIC_LIMIT:.0e}). Rows that "
        f"could not be compared: {skipped}."
        + f"\n\nThe head on the feature gives the output of the network: the "
        f"largest difference is {facts.loc[kept, 'head_gap'].max():.3f}, and the "
        f"two agree on the class for at least "
        f"{100 * facts.loc[kept, 'agreement'].min():.2f}% of the images."
        + f"\n\nModels in the analysis: {len(kept)}, in "
        f"{listed.loc[kept, 'lineage'].nunique()} lineages. Time of an evaluation: "
        f"{facts['seconds'].mean() / 60:.1f} minutes on average, "
        f"{facts['seconds'].sum() / 3600:.1f} hours in total."))]
    decided = left <= LEFT_SHARE * len(listed) and len(csfs) >= 2
    if decided:
        chosen = listed.loc[kept]
        metrics = inputs_of(tables, csfs).loc[kept]
        for name in GROUPS:
            sections += target_section(name, tables, metrics, chosen, csfs)[0]
    else:
        sections.append(("Outcome", (
            "More than a quarter of the models leave, or fewer than two scores "
            "stay. **The test is not decided.**")))
    text = ("# Amendment A16: the rule from the score statistics on public "
            "ImageNet-1k models\n\nNothing is trained. Each lineage is left out "
            "once, and every choice comes from the models of the other lineages.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    if not decided:
        raise SystemExit("the test is not decided")


if __name__ == "__main__":
    main()
