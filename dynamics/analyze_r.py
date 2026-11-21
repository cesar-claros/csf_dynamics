"""Amendment A13: the rule from the score statistics on other architectures.

    python -m dynamics.analyze_r --part 1 --results-root dyn_results

Nothing is fitted for the outcome. The rule under test and the fixed score come
from the frozen file of amendment A12 part 1, which was fitted on ResNet-18 runs.
This analysis applies them to runs of another architecture with the same eight
recipes: part 1 is ResNet-50 (stage `r`) and part 2 is DenseNet-121 (stage `n`).

The readings are those of amendment A12, with the recipe as the unit. Gate S is
new: every score must have a finite value at the final checkpoint. A secondary
reading fits the rule inside the new architecture, with one recipe left out.

Protocol: ../documentation/other_architecture_replication_amendment_2026-11-21.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_e, analyze_h, rules
from dynamics import analyze_loro as lo
from dynamics import analyze_t as at
from dynamics import analyze_z as az
from dynamics.analyze_e import block
from dynamics.config import RECIPES, STAGES

HERE = Path(__file__).resolve().parent
#: Label, stage, and architecture of each part.
PARTS = {1: ("part 1, ResNet-50", "r", "resnet50"),
         2: ("part 2, DenseNet-121", "n", "densenet121")}
#: The frozen file of amendment A12 part 1, with the rule and the fixed score.
FROZEN = az.PARTS[1]


def architectures(root: Path, runs: list[str]) -> dict[str, str]:
    """The architecture that the `meta.json` of each run records."""
    found = {}
    for run in runs:
        meta = json.loads((Path(root) / "runs" / run / "meta.json").read_text())
        found[run] = meta.get("arch", "resnet18")
    return found


def score_failures(new: analyze_e.Tables) -> pd.DataFrame:
    """Gate S: final checkpoints where a score has an error or a non-finite value."""
    checks = new.checks
    last = checks.groupby("run")["epoch"].transform("max") == checks["epoch"]
    bad = (checks["failed"] != "") | (checks["nonfinite"] != "")
    return checks.loc[last & bad, ["run", "epoch", "failed", "nonfinite"]].reset_index(
        drop=True)


def inside(new: analyze_e.Tables) -> tuple[str, pd.DataFrame]:
    """Question 3: the rule fitted inside the architecture, one recipe left out.

    Returns:
        The reading, and the regret by recipe. The reading says so if a score has
        no value at some evaluated checkpoint, because the fit then has no target.
    """
    sets = az.GROUPS[az.PRIMARY]
    meta, _, metrics = at.cells(new)
    scores = az.group_scores(az.stored(new, sets), sets, meta.index)
    if not np.isfinite(scores.to_numpy()).all():
        return "not computed: a score has no value at an evaluated checkpoint", (
            pd.DataFrame())
    table = lo.leave_one_recipe_out(meta, scores, metrics)
    names = [c for c in table.columns if c not in ("recipe", "smoothing")]
    recipes = table.groupby("recipe")[names].mean()
    reading = az.adds(recipes, az.RULE, lo.ONE)
    text = (f"the rule has {reading['regret']:.2f} and the fixed score has "
            f"{reading['fixed_regret']:.2f}; difference {reading['difference']:.2f} "
            f"({reading['low']:.2f} to {reading['high']:.2f}); {reading['reading']}")
    return text, recipes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--part", type=int, choices=sorted(PARTS), required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--rules", help="another rules file (tests only)")
    parser.add_argument("--allow-other-rules", action="store_true",
                        help="accept a rules file with another hash (tests only)")
    parser.add_argument("--stage", help="read another stage (dry runs only)")
    parser.add_argument("--derived-target", action="store_true",
                        help="take the target from the old stored values and skip "
                        "gate M (dry runs on earlier runs only)")
    parser.add_argument("--out")
    args = parser.parse_args()
    label, stage, arch = PARTS[args.part]
    dry = bool(args.stage)
    stage = args.stage or stage
    root = Path(args.results_root)
    path = Path(args.rules) if args.rules else FROZEN.path
    out = Path(args.out) if args.out else (
        HERE / "outputs" / f"stage_{stage}_report.md")
    digest = rules.file_hash(path)
    if digest != FROZEN.sha256 and not args.allow_other_rules:
        raise SystemExit(f"{path.name} has hash {digest}, not the frozen one")
    frozen = az.load(path)
    new = analyze_h.read_stage(root, (stage,))
    if new.metrics.empty:
        raise SystemExit(f"no results of stage {stage} under {root}/runs")
    runs = sorted(set(new.metrics["run"]))
    found = architectures(root, runs)
    wrong = {run: value for run, value in found.items() if value != arch}
    if wrong and not dry:
        raise SystemExit(f"runs that are not {arch}: {wrong}")
    counts = az.set_counts(root, runs)
    if frozen["images"] != counts:
        raise SystemExit("the image counts are not those of the frozen file")
    failures = score_failures(new)
    sections = [("Architecture and gate S", (
        f"Architecture in `meta.json`: {sorted(set(found.values()))} for "
        f"{len(runs)} runs. Recipes: "
        f"{sorted({run.split('_')[0] for run in runs})}.\n\n"
        + ("**Gate S passes.** Every score has a finite value at the final "
           "checkpoint of every run." if failures.empty else
           "**Gate S fails.** Final checkpoints with a score that has an error "
           "or a non-finite value:\n\n" + block(failures))))]
    if failures.empty:
        sections += az.test_sections(frozen, new, counts, stage, args.derived_target)
        if args.derived_target:
            sections.append(("Question 3 (secondary)", "Not read in a dry run."))
        else:
            reading, recipes = inside(new)
            sections.append((
                "Question 3 (secondary): the rule fitted inside the architecture", (
                    "Each recipe is left out once. The rule from the score "
                    "statistics and one fixed score come from the other recipes "
                    "of the same architecture, with the checkpoints that were "
                    f"evaluated. Far sets, recipe as the unit: {reading}."
                    + ("" if recipes.empty else "\n\n" + block(
                        recipes.reset_index(), 2)))))
    text = (f"# Amendment A13, {label}: the frozen rule on another architecture\n\n"
            "The rule under test and the fixed score come from the file of "
            f"amendment A12 part 1 (`{path.name}`), with no new fit. The recipes "
            "are those of stage `z`, and the network of this stage is "
            f"{RECIPES[STAGES[stage][0].recipe].arch}.\n"
            + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    if not failures.empty:
        raise SystemExit("gate S fails: the part is not decided")


if __name__ == "__main__":
    main()
