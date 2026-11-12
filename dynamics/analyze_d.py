"""Stage D analysis: gates and the rules for questions Q1 to Q4.

Q1: does the augmented recipe show the drop once it has the time past
interpolation? Q2: is the re-ordering of the scores tied to collapse or to time?
Q3: do three geometry conventions agree? Q4: are the loss spikes an artefact of
mixed precision?

Run from code/ after copying the run folders from the HPC:

    python -m dynamics.analyze_d --results-root dyn_results

Protocol: ../documentation/time_and_collapse_protocol_2026-11-12.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dynamics import analyze_checks, analyze_e
from dynamics.analyze_e import block
from dynamics.analyze_step0 import DECLINE, FAR_SETS, SEED_SHARE, SETS_NEEDED, SUITE
from dynamics.config import (
    DEFAULT_PRECISION,
    PENULTIMATE,
    PRECISIONS,
    results_folder,
)

HERE = Path(__file__).resolve().parent
CONTROL, AUGMENTED, MORE_COLLAPSE, LESS_COLLAPSE, FLOAT32 = (
    "D1", "D2", "D3", "D4", "D5")
ARMS = (CONTROL, AUGMENTED, MORE_COLLAPSE, LESS_COLLAPSE, FLOAT32)
#: Fixed from check 3 of the pilot, before any Stage D run.
EARLY_STRONG = ("GradNorm", "NCI", "kNN", "PCA_RecError")
LATE_GAINING = ("Maha", "Residual")
#: Scores of the feature outcome Y. They do not read the logits.
FEATURE_OUTCOME = ("kNN", "PCA_RecError")
LOGIT_SCORES = ("MSR", "MLS", "Energy")
#: Reference points on the axis of epochs past interpolation.
TAU = 300
TAU_TOLERANCE = 10
REPLACE_WITHIN = 40
INTERPOLATE_BY = 200
#: An early-strong score drops if its mean drop is at least this many points.
SCORE_DROP = 2.0
SCORES_NEEDED = 3
#: Q2, outcome C: both intervals lie inside plus and minus this many points.
EQUIVALENCE = 1.5
SPIKE_FACTOR = 10.0
SPIKE_WINDOW = 10
RESAMPLES = 2000
CONVENTION_NAMES = ("outside_share_gap", "outside_share_gap_raw", "orthodev",
                    "rowspace_share", "rowspace_share_gap")
#: Descriptive outputs. They support no claim.
GROUPS = ("Y", "early_strong", "late_gaining", "logit")
COURSE_STEP = 20
COURSE_FROM = -40
COURSE_SHOWN = (0, 100, 200, 300)
MIN_TAIL = 6


def interpolation_epochs(clock: pd.DataFrame) -> pd.Series:
    """Per run, the first checkpoint with 100% training accuracy."""
    reached = clock[clock["train_accuracy"] >= 1.0]
    return reached.groupby("run")["epoch"].min()


def nearest(candidates: np.ndarray, target: float, within: float) -> float:
    """The candidate closest to the target, or NaN if none is within the limit."""
    if len(candidates) == 0:
        return float("nan")
    gaps = np.abs(candidates - target)
    best = int(np.argmin(gaps))
    return float(candidates[best]) if gaps[best] <= within else float("nan")


def settle(epoch: float, disturbed: set, converged_epochs: np.ndarray) -> float:
    """The epoch itself, or the nearest converged checkpoint if it is disturbed."""
    if np.isnan(epoch) or epoch not in disturbed:
        return epoch
    return nearest(converged_epochs, epoch, REPLACE_WITHIN)


def reference_epochs(clock: pd.DataFrame, converged: pd.Series,
                     tau: int = TAU) -> pd.DataFrame:
    """Per run: t0, and the checkpoints for tau = 0 and for the given tau.

    A disturbed reference checkpoint is replaced by the nearest converged
    checkpoint within `REPLACE_WITHIN` epochs. A missing value means that the run
    has no usable checkpoint for that reference point.
    """
    t0 = interpolation_epochs(clock)
    disturbed = analyze_e.disturbed_checkpoints(converged)
    rows = []
    for run, group in clock.groupby("run"):
        epochs = np.sort(group["epoch"].to_numpy())
        bad = set(disturbed.loc[disturbed["run"] == run, "epoch"])
        good = np.array([epoch for epoch in epochs
                         if converged.get((run, epoch), False)])
        first = t0.get(run, np.nan)
        start = settle(first, bad, good)
        end = (float("nan") if np.isnan(first)
               else settle(nearest(epochs, first + tau, TAU_TOLERANCE), bad, good))
        rows.append({"run": run, "recipe": group["recipe"].iloc[0], "t0": first,
                     "start": start, "end": end})
    return pd.DataFrame(rows)


def far_score_means(metrics: pd.DataFrame, csfs: list[str]) -> pd.DataFrame:
    """Per run, epoch, and CSF: the mean AUROC over the far sets, in points."""
    kept = metrics[metrics["set"].isin(FAR_SETS) & metrics["csf"].isin(csfs)]
    value = kept.groupby(["run", "recipe", "epoch", "csf"])["auroc"].mean() * 100
    return value.rename("value").reset_index()


def envelopes(metrics: pd.DataFrame, csfs: list[str]) -> pd.DataFrame:
    """Per run, epoch, and far set: the best AUROC over the roster, in points."""
    kept = metrics[metrics["set"].isin(FAR_SETS) & metrics["csf"].isin(csfs)]
    value = kept.groupby(["run", "recipe", "epoch", "set"])["auroc"].max() * 100
    return value.rename("value").reset_index()


def group_mean(scores: pd.DataFrame, names: tuple[str, ...], label: str
               ) -> pd.DataFrame:
    """The mean of some scores per run and epoch, as one pseudo-score."""
    kept = scores[scores["csf"].isin(names)]
    value = kept.groupby(["run", "recipe", "epoch"])["value"].mean()
    return value.reset_index().assign(csf=label)


def drops(table: pd.DataFrame, refs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per run and `by`: the value at tau = 0, at the later point, and the drop."""
    usable = refs.dropna(subset=["start", "end"])
    columns = ["run", "recipe", *by, "start", "end", "drop"]
    if table.empty or usable.empty:
        return pd.DataFrame(columns=columns)
    keys = ["run", "epoch"]
    first = table.merge(usable[["run", "start"]].rename(columns={"start": "epoch"})
                        .astype({"epoch": int}), on=keys)
    last = table.merge(usable[["run", "end"]].rename(columns={"end": "epoch"})
                       .astype({"epoch": int}), on=keys)
    out = first[["run", "recipe", *by, "value"]].merge(
        last[["run", *by, "value"]], on=["run", *by], suffixes=("_first", "_last"))
    out = out.rename(columns={"value_first": "start", "value_last": "end"})
    out["drop"] = out["start"] - out["end"]
    return out[columns]


def summarize_drops(table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean over seeds per arm and `by`, with the number of seeds that drop."""
    columns = ["recipe", *by, "start", "end", "drop", "seeds", "seeds_dropping"]
    if table.empty:
        return pd.DataFrame(columns=columns)
    group = table.groupby(["recipe", *by])
    out = group[["start", "end", "drop"]].mean()
    out["seeds"] = group.size()
    out["seeds_dropping"] = group["drop"].apply(lambda d: int((d > 0).sum()))
    return out.reset_index()[columns]


def count_drops(summary: pd.DataFrame, arm: str, threshold: float) -> int:
    """Rows of one arm whose mean drop and share of dropping seeds pass the rule."""
    rows = summary[summary["recipe"] == arm]
    passed = ((rows["drop"] >= threshold)
              & (rows["seeds_dropping"] >= SEED_SHARE * rows["seeds"]))
    return int(passed.sum())


def shows_effect(envelope: pd.DataFrame, scores: pd.DataFrame, arm: str
                 ) -> tuple[int, int]:
    """Far sets whose envelope drops, and early-strong scores that drop."""
    early = scores[scores["csf"].isin(EARLY_STRONG)]
    return (count_drops(envelope, arm, DECLINE), count_drops(early, arm, SCORE_DROP))


def decide_q1(sets: int, scores: int) -> str:
    """Rule Q1 on the augmented arm."""
    if sets >= SETS_NEEDED and scores >= SCORES_NEEDED:
        return "T, time: the augmented recipe shows the drop and the re-ordering"
    if sets <= 1 and scores <= 1:
        return "A, augmentation: the augmented recipe shows neither"
    return "M, mixed: no claim"


def difference_interval(first: np.ndarray, second: np.ndarray, seed: int = 0
                        ) -> tuple[float, float, float]:
    """Mean of the first arm minus mean of the second, with a 95% interval.

    The seeds of each arm are resampled on their own. NaN if an arm has fewer
    than 2 seeds.
    """
    first, second = (np.asarray(v, dtype=np.float64) for v in (first, second))
    first, second = first[np.isfinite(first)], second[np.isfinite(second)]
    if len(first) < 2 or len(second) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    a = first[rng.integers(0, len(first), (RESAMPLES, len(first)))].mean(axis=1)
    b = second[rng.integers(0, len(second), (RESAMPLES, len(second)))].mean(axis=1)
    gaps = a - b
    return (float(first.mean() - second.mean()), float(np.quantile(gaps, 0.025)),
            float(np.quantile(gaps, 0.975)))


def arm_contrasts(outcome: pd.DataFrame, collapse: pd.DataFrame) -> pd.DataFrame:
    """Gate M and the Q2 contrast of each intervention arm against the control.

    Args:
        outcome: Drops of the feature outcome Y, one row per run.
        collapse: log10 of NC1 at the later reference point, one row per run, in
            the column `value`.
    """
    rows = []
    for arm, sign in ((MORE_COLLAPSE, -1), (LESS_COLLAPSE, +1)):
        shift = difference_interval(
            collapse.loc[collapse["recipe"] == arm, "value"].to_numpy(),
            collapse.loc[collapse["recipe"] == CONTROL, "value"].to_numpy())
        moved = bool(shift[2] < 0) if sign < 0 else bool(shift[1] > 0)
        gap = difference_interval(
            outcome.loc[outcome["recipe"] == arm, "drop"].to_numpy(),
            outcome.loc[outcome["recipe"] == CONTROL, "drop"].to_numpy())
        # More collapse must give a larger drop, and less collapse a smaller one.
        predicted = bool(gap[1] > 0) if sign < 0 else bool(gap[2] < 0)
        inside = bool(gap[1] >= -EQUIVALENCE and gap[2] <= EQUIVALENCE)
        rows.append({"arm": arm,
                     "expected_collapse": "more" if sign < 0 else "less",
                     "log_nc1_shift": shift[0], "shift_low": shift[1],
                     "shift_high": shift[2], "gate_m": moved,
                     "drop_gap": gap[0], "gap_low": gap[1], "gap_high": gap[2],
                     "predicted_sign": predicted, "within_equivalence": inside})
    return pd.DataFrame(rows)


def decide_q2(contrasts: pd.DataFrame) -> str:
    """Rule Q2 on the two intervention arms."""
    passed = contrasts[contrasts["gate_m"]]
    if passed.empty:
        return "inconclusive: no arm passes gate M"
    label = "" if len(passed) == len(contrasts) else " (one arm)"
    if passed["predicted_sign"].all():
        return f"G, geometry: collapse is supported as the driver{label}"
    if len(passed) == len(contrasts) and passed["within_equivalence"].all():
        return "C, clock: collapse changed and the outcome did not"
    return f"inconclusive{label}"


def gated(verdict: str, failed: list[str]) -> str:
    """The verdict of a rule, or `not decided` if a gate of that rule fails."""
    if not failed:
        return verdict
    return f"not decided, because gate {' and gate '.join(failed)} failed"


def direction(changes: np.ndarray) -> str:
    """`falls`, `rises`, or `no clear change`, by the mean and the seed share."""
    changes = np.asarray(changes, dtype=np.float64)
    changes = changes[np.isfinite(changes)]
    if len(changes) == 0:
        return "no data"
    if changes.mean() < 0 and (changes < 0).mean() >= SEED_SHARE:
        return "falls"
    if changes.mean() > 0 and (changes > 0).mean() >= SEED_SHARE:
        return "rises"
    return "no clear change"


def read_conventions(ours: str, orthodev: str) -> str:
    """The reading of the Q3 table that the protocol fixes."""
    if ours == "no data":
        return "no data at the two reference points"
    if ours != "falls":
        return "the pilot's geometric finding does not hold under this schedule"
    if orthodev == "falls":
        return "no conflict of fact: the two statements measure different things"
    if orthodev == "rises":
        return "our result contradicts the fifth collapse property of NECO"
    return "our gap falls, and the NECO metric shows no clear change"


def spike_epochs(losses: dict[int, float], after: float) -> list[int]:
    """Epochs after `after` whose loss exceeds 10 times the median of the 10 before."""
    found = []
    for epoch in sorted(losses):
        window = [losses[e] for e in range(epoch - SPIKE_WINDOW, epoch)
                  if e in losses]
        if epoch <= after or len(window) < SPIKE_WINDOW:
            continue
        if losses[epoch] > SPIKE_FACTOR * float(np.median(window)):
            found.append(epoch)
    return found


def read_losses(root: Path, run: str) -> dict[int, float]:
    path = Path(root) / "runs" / run / "train_log.csv"
    if not path.exists():
        return {}
    with path.open() as handle:
        return {int(row["epoch"]): float(row["loss"]) for row in csv.DictReader(handle)}


def decide_q4(seeds_with_spikes: dict[str, tuple[int, int]]) -> str:
    """Rule Q4 from (seeds with a spike, seeds) of the control and float32 arms."""
    control = seeds_with_spikes.get(CONTROL, (0, 0))
    plain = seeds_with_spikes.get(FLOAT32, (0, 0))
    if plain[1] == 0 or control[1] == 0:
        return "not decided: an arm is missing"
    if plain[0] >= SEED_SHARE * plain[1]:
        return "not a precision artefact: the float32 arm has spikes"
    if plain[0] == 0 and control[0] >= SEED_SHARE * control[1]:
        return "precision artefact: spikes only with mixed precision"
    return "unclear"


def time_course(scores: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """Per arm, score, and tau: the mean over seeds of the far-set AUROC.

    tau is rounded to steps of `COURSE_STEP` epochs, because t0 differs by seed.
    Each seed has the same weight in a step.
    """
    columns = ["recipe", "csf", "tau", "value", "seeds"]
    table = scores.merge(refs.dropna(subset=["t0"])[["run", "t0"]], on="run")
    if table.empty:
        return pd.DataFrame(columns=columns)
    steps = ((table["epoch"] - table["t0"]) / COURSE_STEP).round().astype(int)
    table = table.assign(tau=steps * COURSE_STEP)
    table = table[table["tau"] >= COURSE_FROM]
    per_run = table.groupby(["recipe", "csf", "tau", "run"])["value"].mean()
    group = per_run.groupby(["recipe", "csf", "tau"])
    out = group.mean().rename("value").to_frame()
    out["seeds"] = group.size()
    return out.reset_index()[columns]


def tail_regressions(scores: pd.DataFrame, nc1: pd.DataFrame, refs: pd.DataFrame,
                     names: tuple[str, ...] = GROUPS[:3]) -> pd.DataFrame:
    """Per run and score: the AUROC against tau and log10 NC1 after t0.

    The model is value = a + b * (tau / 100) + c * log10(NC1), by least squares
    on every checkpoint with tau >= 0, the disturbed ones included. tau and NC1
    move together, so the coefficients are unstable. `tau_nc1_corr` shows how
    much. This output supports no claim.
    """
    columns = ["run", "recipe", "csf", "checkpoints", "per_100_epochs",
               "per_log_nc1", "tau_nc1_corr"]
    t0 = refs.dropna(subset=["t0"]).set_index("run")["t0"]
    collapse = nc1.set_index(["run", "epoch"])["value"]
    rows = []
    kept = scores[scores["csf"].isin(names) & scores["run"].isin(t0.index)]
    for (run, recipe, csf), part in kept.groupby(["run", "recipe", "csf"]):
        part = part[part["epoch"] >= t0[run]]
        tau = (part["epoch"].to_numpy(dtype=np.float64) - t0[run]) / 100
        log_nc1 = collapse.reindex(
            pd.MultiIndex.from_product([[run], part["epoch"]])).to_numpy()
        value = part["value"].to_numpy(dtype=np.float64)
        use = np.isfinite(log_nc1) & np.isfinite(value)
        if use.sum() < MIN_TAIL or np.ptp(log_nc1[use]) == 0:
            continue
        design = np.column_stack([np.ones(use.sum()), tau[use], log_nc1[use]])
        fit = np.linalg.lstsq(design, value[use], rcond=None)[0]
        rows.append({"run": run, "recipe": recipe, "csf": csf,
                     "checkpoints": int(use.sum()), "per_100_epochs": fit[1],
                     "per_log_nc1": fit[2],
                     "tau_nc1_corr": float(np.corrcoef(tau[use],
                                                       log_nc1[use])[0, 1])})
    return pd.DataFrame(rows, columns=columns)


def stand_ins_at_end(metrics: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """Check 11 of the pilot at the later reference point of each run."""
    parts = [analyze_checks.stand_in_agreement(metrics[metrics["run"] == row.run],
                                               int(row.end))
             for row in refs.dropna(subset=["end"]).itertuples()]
    parts = [part for part in parts if not part.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--precision", choices=sorted(PRECISIONS),
                        default=DEFAULT_PRECISION)
    parser.add_argument("--out", default=str(HERE / "outputs" / "stage_d_report.md"))
    args = parser.parse_args()
    root = Path(args.results_root)
    tag = results_folder(args.precision, suite=SUITE)

    def stage_d(tables: analyze_e.Tables) -> analyze_e.Tables:
        return analyze_e.Tables(**{
            name: frame if frame.empty
            else frame[frame["recipe"].isin(ARMS)].reset_index(drop=True)
            for name, frame in vars(tables).items()})

    tables = stage_d(analyze_e.read_runs(root, tag))
    if tables.metrics.empty:
        raise SystemExit(f"no Stage D results under {root}/runs")
    recheck = stage_d(analyze_e.read_runs(
        root, results_folder(args.precision, True, SUITE)))
    manifests = sorted((root / "runs").glob("D*/manifest.json"))
    manifest = json.loads(manifests[0].read_text()) if manifests else None
    out_dir = Path(args.out).parent / "stage_d"
    out_dir.mkdir(parents=True, exist_ok=True)

    pilot_gates, unstable = analyze_e.gates(tables, recheck, manifest)
    pilot_gates = pilot_gates[pilot_gates["gate"].str[:2].isin(["G3", "G4", "G5"])]
    csfs, dropped = analyze_e.roster(tables, unstable)
    metrics = tables.metrics
    clock = analyze_e.clock_table(tables.descriptors)
    converged = analyze_e.converged_checkpoints(clock)
    refs = reference_epochs(clock, converged)

    scores = far_score_means(metrics, csfs)
    scores = pd.concat([scores,
                        group_mean(scores, FEATURE_OUTCOME, "Y"),
                        group_mean(scores, EARLY_STRONG, "early_strong"),
                        group_mean(scores, LATE_GAINING, "late_gaining"),
                        group_mean(scores, LOGIT_SCORES, "logit")])
    envelope = envelopes(metrics, csfs)
    score_drops = drops(scores, refs, ["csf"])
    envelope_drops = drops(envelope, refs, ["set"])
    score_summary = summarize_drops(score_drops, ["csf"])
    envelope_summary = summarize_drops(envelope_drops, ["set"])

    nc1 = tables.descriptors[(tables.descriptors["group"] == "nc_paper")
                             & (tables.descriptors["name"] == "var_collapse")]
    nc1 = nc1[["run", "recipe", "epoch", "value"]].copy()
    nc1["value"] = np.log10(nc1["value"].clip(lower=1e-12))
    collapse = drops(nc1, refs, [])
    late_collapse = collapse.rename(columns={"end": "value"})[["run", "recipe",
                                                               "value"]]

    interpolated = refs[refs["recipe"] == AUGMENTED]
    on_time = int((interpolated["t0"] <= INTERPOLATE_BY).sum())
    gate_i = on_time >= SEED_SHARE * max(len(interpolated), 1) and len(interpolated)
    control_sets, control_scores = shows_effect(envelope_summary, score_summary,
                                                CONTROL)
    gate_p = control_sets >= SETS_NEEDED and control_scores >= SCORES_NEEDED
    q1_sets, q1_scores = shows_effect(envelope_summary, score_summary, AUGMENTED)
    contrasts = arm_contrasts(score_drops[score_drops["csf"] == "Y"], late_collapse)

    placement = tables.placement[
        (tables.placement["layer"] == PENULTIMATE)
        & tables.placement["name"].isin(CONVENTION_NAMES)
        & tables.placement["set"].isin(FAR_SETS)]
    by_run = (placement.groupby(["run", "recipe", "epoch", "name"])["value"].mean()
              .reset_index())
    convention_drops = drops(by_run, refs, ["name"])
    convention_rows, readings = [], {}
    for arm in (CONTROL, AUGMENTED):
        part = convention_drops[convention_drops["recipe"] == arm]
        found = {}
        for name in CONVENTION_NAMES:
            change = -part.loc[part["name"] == name, "drop"].to_numpy()
            found[name] = direction(change)
            convention_rows.append({
                "arm": arm, "quantity": name,
                "start": part.loc[part["name"] == name, "start"].mean(),
                "end": part.loc[part["name"] == name, "end"].mean(),
                "direction": found[name]})
        readings[arm] = read_conventions(found["outside_share_gap"],
                                         found["orthodev"])
    conventions = pd.DataFrame(convention_rows)

    spikes, spike_rows = {}, []
    t0 = refs.set_index("run")["t0"]
    for arm in ARMS:
        runs = sorted(refs.loc[(refs["recipe"] == arm) & refs["t0"].notna(), "run"])
        counts = []
        for run in runs:
            flagged = spike_epochs(read_losses(root, run), t0[run])
            counts.append(len(flagged))
            spike_rows.append({"arm": arm, "run": run,
                               "spike_epochs": len(flagged),
                               "first": flagged[0] if flagged else np.nan})
        spikes[arm] = (sum(count > 0 for count in counts), len(runs))
    spike_table = pd.DataFrame(spike_rows)

    course = time_course(scores, refs)
    course_shown = course[course["csf"].isin(GROUPS)
                          & course["tau"].isin(COURSE_SHOWN)]
    if not course_shown.empty:
        course_shown = (course_shown.pivot(index=["recipe", "csf"], columns="tau",
                                           values="value")
                        .rename(columns=lambda tau: f"tau_{tau}").reset_index())
    regressions = tail_regressions(scores, nc1, refs)
    regression_summary = pd.DataFrame()
    if not regressions.empty:
        group = regressions.groupby(["recipe", "csf"])
        regression_summary = group[["per_100_epochs", "per_log_nc1",
                                    "tau_nc1_corr"]].mean()
        regression_summary["seeds"] = group.size()
        regression_summary = regression_summary.reset_index()
    stand_ins = stand_ins_at_end(metrics[metrics["csf"].isin(csfs)], refs)
    stand_in_summary = pd.DataFrame()
    if not stand_ins.empty:
        stand_in_summary = (stand_ins.groupby(["recipe", "stand_in", "target"])
                            ["spearman"].mean().unstack("target").reset_index())

    for name, frame in (("references", refs), ("score_drops", score_summary),
                        ("envelope_drops", envelope_summary),
                        ("q2_contrasts", contrasts), ("q3_conventions", conventions),
                        ("q4_spikes", spike_table), ("time_course", course),
                        ("tail_regressions", regressions),
                        ("stand_ins_at_end", stand_ins)):
        frame.to_csv(out_dir / f"{name}.csv", index=False)

    failed_q2 = [] if gate_p else ["P"]
    failed_q1 = ([] if gate_i else ["I"]) + failed_q2
    shown = score_summary[score_summary["csf"].isin(
        [*EARLY_STRONG, *LATE_GAINING, "Y", "early_strong", "late_gaining",
         "logit"])]
    spike_summary = pd.DataFrame([
        {"arm": arm, "seeds_with_spikes": found, "seeds": total}
        for arm, (found, total) in spikes.items()])
    gate_rows = pd.DataFrame([
        ("Gate I, the augmented arm interpolates by epoch 200",
         "pass" if gate_i else "FAIL", f"{on_time} of {len(interpolated)} seeds"),
        ("Gate P, the control arm shows the pilot's effect",
         "pass" if gate_p else "FAIL",
         (f"envelope drops on {control_sets} of {len(FAR_SETS)} far sets, "
          f"{control_scores} of {len(EARLY_STRONG)} early-strong scores drop")),
    ], columns=["gate", "status", "detail"])
    sections = [
        ("Runs and reference points", (
         f"- Runs: {refs['run'].nunique()}, precision: {args.precision}, roster: "
         f"{len(csfs)} scores"
         + (f" (removed: {dropped['csf'].tolist()})." if not dropped.empty else ".")
         + "\n- `t0` is the first checkpoint with 100% training accuracy. `start` "
         f"and `end` are the checkpoints for tau = 0 and tau = {TAU}. A missing "
         "value means that the run has no usable checkpoint there.\n\n"
         + block(refs, 1))),
        ("Gates", block(pd.concat([gate_rows, pilot_gates], ignore_index=True))),
        ("Q1: time or augmentation",
         f"Drops from tau = 0 to tau = {TAU}, in AUROC points. Envelope by far "
         "set:\n\n" + block(envelope_summary, 2)
         + "\n\nSingle scores and score groups, mean over the far sets:\n\n"
         + block(shown, 2)
         + f"\n\nIn arm {AUGMENTED}, the envelope drops on {q1_sets} of "
         f"{len(FAR_SETS)} far sets, and {q1_scores} of {len(EARLY_STRONG)} "
         "early-strong scores drop.\n\n"
         f"**Q1: {gated(decide_q1(q1_sets, q1_scores), failed_q1)}.**"),
        ("Q2: collapse or time",
         "`log_nc1_shift` is log10 NC1 of the arm minus that of the control at the "
         "later reference point. `drop_gap` is the drop of Y in the arm minus the "
         "drop in the control. Intervals are 95% bootstrap intervals over seeds:"
         "\n\n" + block(contrasts, 3)
         + f"\n\n**Q2: {gated(decide_q2(contrasts), failed_q2)}.**"),
        ("Q3: three conventions",
         "Mean over the far sets, at the two reference points:\n\n"
         + block(conventions, 3) + "\n\n"
         + "\n".join(f"- **Arm {arm}:** {text}." for arm, text in readings.items())),
        ("Q4: loss spikes",
         "Only runs that reach `t0` count as seeds:\n\n"
         + block(spike_summary) + f"\n\n**Q4: {decide_q4(spikes)}.**"),
        ("Descriptive outputs",
         "These tables support no claim. `stage_d/time_course.csv` has every score "
         f"and every step of {COURSE_STEP} epochs.\n\nMean AUROC over the far "
         "sets and the seeds, against the epochs past interpolation:\n\n"
         + block(course_shown, 2)
         + "\n\nRegression inside each run on the checkpoints after `t0`, mean "
         "over seeds. `per_100_epochs` and `per_log_nc1` are the two coefficients "
         "of one model, in AUROC points. A `tau_nc1_corr` near -1 or 1 means that "
         "the data cannot separate the two:\n\n" + block(regression_summary, 2)
         + f"\n\nRank agreement (Spearman) of each stand-in with each true set "
         f"at tau = {TAU}, mean over seeds:\n\n" + block(stand_in_summary, 2)),
    ]
    text = ("# Stage D: time, collapse, and conventions\n\nThe hypotheses come from "
            "the pilot data. The gates and the rules were fixed before these runs."
            "\n" + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
