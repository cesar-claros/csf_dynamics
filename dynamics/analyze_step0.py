"""Step 0: more OOD sets and an earlier layer, on the saved Stage E checkpoints.

Two questions, with rules fixed in protocol amendment A4 before the run:

1. Does the far-OOD decline of regime A hold on far sets other than Textures?
2. Does stage 3 keep the far-OOD signature that the penultimate layer loses?

Run from code/ after copying the run folders from the HPC:

    python -m dynamics.analyze_step0 --results-root dyn_results

Protocol: ../documentation/imagenette_dynamics_protocol_2026-11-10.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dynamics import analyze_e
from dynamics.analyze_e import block
from dynamics.config import (
    DEFAULT_PRECISION,
    LAYERS,
    PENULTIMATE,
    PRECISIONS,
    parse_run,
    results_folder,
)
from dynamics.scores import LAYER_CSFS, NO_FIT_CSFS

HERE = Path(__file__).resolve().parent
SUITE = "step0"
#: The four far sets that the rules count. Textures is the set where the decline
#: was found, so it is reported and not counted.
FAR_SETS = ("inaturalist", "places", "sun", "openimage_o")
NEAR_SETS = ("ssb_hard", "ninco", "imagenet_o")
DISCOVERY_SET = "textures"
#: The two checkpoints that the rules compare, and the regime that they read.
EARLY, LATE = 25, 200
REGIME = "A"
#: A decline counts if its mean over the seeds is at least this many AUROC points
#: and it is positive in at least this share of the seeds.
DECLINE = 3.0
SEED_SHARE = 0.8
#: Rule 1 and rule 2 need this many of the four far sets.
SETS_NEEDED = 3
EARLIER_LAYER = "stage3"


def read_layer_metrics(root: Path, tag: str) -> pd.DataFrame:
    """The head-free CSFs on the layers before the penultimate one, per checkpoint."""
    rows = []
    for folder in sorted((Path(root) / "runs").glob("*")):
        try:
            spec = parse_run(folder.name)
        except ValueError:
            continue
        for path in sorted((folder / tag).glob("epoch_*.json")):
            out = json.loads(path.read_text())
            key = {"run": spec.name, "recipe": spec.recipe, "seed": spec.seed,
                   "held_out": -1 if spec.held_out is None else spec.held_out,
                   "epoch": out["epoch"]}
            rows += [{**key, **row} for row in out.get("layer_metrics", [])]
    return pd.DataFrame(rows)


def best_per_cell(long: pd.DataFrame, csfs: list[str]) -> pd.DataFrame:
    """Best AUROC in points over `csfs`, per (run, recipe, epoch, set)."""
    if long.empty:
        return pd.DataFrame(columns=["run", "recipe", "epoch", "set", "best"])
    kept = long[long["csf"].isin(csfs) & (long["held_out"] < 0)]
    best = kept.groupby(["run", "recipe", "epoch", "set"])["auroc"].max() * 100
    return best.rename("best").reset_index()


def change_table(best: pd.DataFrame, early: int = EARLY, late: int = LATE
                 ) -> pd.DataFrame:
    """Per run and set: the best AUROC at the two checkpoints and the decline.

    The decline is NaN for a run that lacks one of the two checkpoints.
    """
    columns = ["run", "recipe", "set", "early", "late", "decline"]
    at = best[best["epoch"].isin([early, late])] if not best.empty else best
    if at.empty:
        return pd.DataFrame(columns=columns)
    table = (at.pivot_table(index=["run", "recipe", "set"], columns="epoch",
                            values="best").reindex(columns=[early, late])
             .rename(columns={early: "early", late: "late"}))
    table["decline"] = table["early"] - table["late"]
    return table.reset_index()[columns]


def summarize_changes(changes: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per set and regime, and the number of seeds with a decline."""
    columns = ["set", "recipe", "early", "late", "decline", "seeds",
               "seeds_declining"]
    if changes.empty:
        return pd.DataFrame(columns=columns)
    group = changes.groupby(["set", "recipe"])
    out = group[["early", "late", "decline"]].mean()
    out["seeds"] = group.size()
    out["seeds_declining"] = group["decline"].apply(lambda d: int((d > 0).sum()))
    return out.reset_index()[columns]


def declines(summary: pd.DataFrame, name: str, regime: str = REGIME) -> bool:
    """Whether a set shows a decline in a regime, by the fixed thresholds."""
    row = summary[(summary["set"] == name) & (summary["recipe"] == regime)]
    if row.empty:
        return False
    row = row.iloc[0]
    return bool(row["decline"] >= DECLINE
                and row["seeds_declining"] >= SEED_SHARE * row["seeds"])


def rule_one(summary: pd.DataFrame) -> tuple[str, list[str]]:
    """Rule 1: the decline of regime A appears on at least 3 of the 4 far sets."""
    present = [name for name in FAR_SETS if name in set(summary["set"])]
    hits = [name for name in present if declines(summary, name)]
    if len(present) < len(FAR_SETS):
        return f"not decided: only {len(present)} of the far sets are present", hits
    verdict = ("the collapse hypothesis survives" if len(hits) >= SETS_NEEDED
               else "the decline does not hold beyond Textures")
    return f"{verdict}: {len(hits)} of {len(present)} far sets", hits


def rule_two(penultimate: pd.DataFrame, earlier: pd.DataFrame
             ) -> tuple[str, pd.DataFrame]:
    """Rule 2: stage 3 keeps what the penultimate layer loses.

    A far set counts if the head-free CSFs decline at the penultimate layer. On
    such a set, the earlier layer passes if it loses less than half of that
    decline and its final AUROC is at least the final AUROC of the penultimate
    layer. The layer route is viable with a pass on at least 3 far sets.
    """
    rows = []
    for name in FAR_SETS:
        last = penultimate[(penultimate["set"] == name)
                           & (penultimate["recipe"] == REGIME)]
        before = earlier[(earlier["set"] == name) & (earlier["recipe"] == REGIME)]
        if last.empty or before.empty:
            continue
        last, before = last.iloc[0], before.iloc[0]
        applies = declines(penultimate, name)
        keeps = bool(before["decline"] < 0.5 * last["decline"])
        higher = bool(before["late"] >= last["late"])
        rows.append({"set": name, "decline_stage4": last["decline"],
                     "decline_stage3": before["decline"],
                     "final_stage4": last["late"], "final_stage3": before["late"],
                     "stage4_declines": applies, "loses_less_than_half": keeps,
                     "final_at_least_stage4": higher,
                     "passes": applies and keeps and higher})
    table = pd.DataFrame(rows)
    if table.empty or int(table["stage4_declines"].sum()) < SETS_NEEDED:
        count = 0 if table.empty else int(table["stage4_declines"].sum())
        return (f"not applicable: the head-free CSFs decline at the penultimate "
                f"layer on only {count} far sets"), table
    passed = int(table["passes"].sum())
    verdict = ("the layer route is viable" if passed >= SETS_NEEDED
               else "the layer route is not viable")
    return f"{verdict}: {passed} of {len(table)} far sets pass", table


def base_gap(step: pd.DataFrame, base: pd.DataFrame) -> float:
    """Largest AUROC gap between this suite and the base results on shared cells."""
    index = ["run", "epoch", "set", "csf"]
    if step.empty or base.empty:
        return float("nan")
    both = step.merge(base, on=index, suffixes=("", "_base"))
    if both.empty:
        return float("nan")
    return float((both["auroc"] - both["auroc_base"]).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--precision", choices=sorted(PRECISIONS),
                        default=DEFAULT_PRECISION)
    parser.add_argument("--out", default=str(HERE / "outputs" / "step0_report.md"))
    args = parser.parse_args()
    root = Path(args.results_root)
    tag = results_folder(args.precision, suite=SUITE)
    tables = analyze_e.stage_e_only(analyze_e.read_runs(root, tag))
    if tables.metrics.empty:
        raise SystemExit(f"no results of the suite {SUITE} under {root}/runs")
    metrics = tables.metrics[tables.metrics["held_out"] < 0]
    layers = read_layer_metrics(root, tag)
    layers = layers[layers["held_out"] < 0] if not layers.empty else layers
    out_dir = Path(args.out).parent / "step0"
    out_dir.mkdir(parents=True, exist_ok=True)

    csfs, dropped = analyze_e.roster(tables, set())
    no_fit = [csf for csf in csfs if csf in NO_FIT_CSFS]
    head_free = [csf for csf in LAYER_CSFS if csf in csfs]
    failed_layers = sorted({key for path in (root / "runs").glob(f"*/{tag}/*.json")
                            for key in json.loads(path.read_text())
                            .get("layer_errors", {})})
    base = analyze_e.read_runs(root, results_folder(args.precision)).metrics
    manifests = sorted((root / "runs").glob("*/manifest.json"))
    manifest = json.loads(manifests[0].read_text()) if manifests else {}
    overlaps = {name: info["id_class_folders"] for name, info in manifest.items()
                if isinstance(info, dict) and info.get("id_class_folders")}
    counts = {name: info["used"] for name, info in manifest.items()
              if isinstance(info, dict) and "used" in info}

    full = summarize_changes(change_table(best_per_cell(metrics, csfs)))
    free = summarize_changes(change_table(best_per_cell(metrics, no_fit)))
    last = summarize_changes(change_table(best_per_cell(metrics, head_free)))
    earlier_rows = (layers[layers["layer"] == EARLIER_LAYER] if not layers.empty
                    else layers)
    earlier = summarize_changes(change_table(best_per_cell(earlier_rows, head_free)))
    verdict_one, hits = rule_one(full)
    verdict_two, layer_table = rule_two(last, earlier)

    order = [DISCOVERY_SET, *FAR_SETS, *NEAR_SETS, "imagewoof", "blend", "shuffle"]
    rank = {name: position for position, name in enumerate(order)}

    def ordered(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        return (frame.assign(_rank=frame["set"].map(rank).fillna(len(rank)))
                .sort_values(["_rank", "recipe"]).drop(columns="_rank"))

    placement = tables.placement[
        (tables.placement["held_out"] < 0) & (tables.placement["name"]
                                              == "outside_share_gap")
        & tables.placement["layer"].isin([EARLIER_LAYER, PENULTIMATE])
        & tables.placement["epoch"].isin([EARLY, LATE])]
    off_span = pd.DataFrame()
    if not placement.empty:
        off_span = (placement.groupby(["set", "recipe", "layer", "epoch"])["value"]
                    .mean().unstack(["layer", "epoch"]).round(3))
        off_span.columns = [f"{layer}_epoch{epoch}" for layer, epoch
                            in off_span.columns]
        off_span = ordered(off_span.reset_index())
    depth = pd.DataFrame()
    if not layers.empty:
        every = pd.concat([
            best_per_cell(layers[layers["layer"] == layer], head_free)
            .assign(layer=layer) for layer in LAYERS[:-1]]
            + [best_per_cell(metrics, head_free).assign(layer=PENULTIMATE)])
        depth = ordered(every[every["epoch"] == LATE]
                        .groupby(["set", "recipe", "layer"])["best"].mean()
                        .unstack("layer").reset_index())
    for name, frame in (("decline_full_roster", full), ("decline_no_fit", free),
                        ("decline_head_free_stage4", last),
                        ("decline_head_free_stage3", earlier),
                        ("rule_two", layer_table), ("off_span_energy", off_span),
                        ("depth_profile", depth)):
        frame.to_csv(out_dir / f"{name}.csv", index=False)

    columns = ["set", "recipe", "early", "late", "decline", "seeds_declining",
               "seeds"]
    gap = base_gap(metrics, base[base["held_out"] < 0] if not base.empty else base)
    sections = [
        ("Runs and checks", (
         f"- Runs: {metrics['run'].nunique()}, checkpoints: "
         f"{metrics[['run', 'epoch']].drop_duplicates().shape[0]}, precision: "
         f"{args.precision}.\n"
         f"- Images used of each extra set: {counts or 'not on record'}.\n"
         "- Extra sets with a folder named like an Imagenette class: "
         f"{overlaps or 'none'}.\n"
         f"- CSFs removed for non-finite scores: "
         f"{dropped['csf'].tolist() or 'none'}. Failed layer CSFs: "
         f"{failed_layers or 'none'}.\n"
         "- Largest AUROC gap to the base results on the shared sets: "
         f"{gap:.2e}.")),
        ("Rule 1: the decline on more far sets",
         f"Best AUROC of the full roster at epoch {EARLY} (`early`) and epoch "
         f"{LATE} (`late`), mean over the seeds. A set declines in regime {REGIME} "
         f"if the mean decline is at least {DECLINE} points and at least "
         f"{SEED_SHARE:.0%} of the seeds decline. Textures is the set where the "
         "decline was found, so the rule does not count it.\n\n"
         + block(ordered(full)[columns] if not full.empty else full, 2)
         + f"\n\n**Rule 1: {verdict_one}.** Far sets with a decline: "
         f"{', '.join(hits) or 'none'}.\n\nThe same for the no-fit roster:\n\n"
         + block(ordered(free)[columns] if not free.empty else free, 2)),
        ("Rule 2: an earlier layer",
         f"Best AUROC of the head-free CSFs ({', '.join(head_free)}) at "
         f"{EARLIER_LAYER} and at the penultimate layer, regime {REGIME}:\n\n"
         + block(layer_table, 2) + f"\n\n**Rule 2: {verdict_two}.**\n\n"
         f"All sets and regimes at {EARLIER_LAYER}:\n\n"
         + block(ordered(earlier)[columns] if not earlier.empty else earlier, 2)),
        ("Energy outside the class-mean span",
         "Mean gap between the set and the ID validation images:\n\n"
         + block(off_span)),
        ("Depth profile at the last epoch",
         "Best AUROC of the head-free CSFs at each layer (descriptive):\n\n"
         + block(depth, 2)),
    ]
    text = ("# Step 0: more OOD sets and an earlier layer\n\nThe hypothesis came "
            "from Stage E, so it is post hoc. The two rules were fixed before this "
            "run.\n" + "".join(f"\n## {title}\n\n{body}\n"
                               for title, body in sections))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
