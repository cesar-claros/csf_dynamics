import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dynamics import analyze_e
from dynamics.config import PENULTIMATE, RunSpec, results_folder
from dynamics.descriptors import PLACEMENT
from dynamics.paper_api import NC_METRICS
from dynamics.scores import ALL_CSFS, NO_FIT_CSFS

EPOCHS = (0, 1, 2, 4)
BEST = {"imagewoof": "Maha", "textures": "MLS"}


def fake_epoch(spec: RunSpec, epoch: int, rng: np.random.Generator,
               broken: str | None = None) -> dict:
    """One result file with the schema of `evaluate_run` and planted structure."""
    sets = ["imagewoof", "textures", "blend", "shuffle", "id_errors"]
    if spec.held_out is not None:
        sets.append("heldout")
    target = {"blend": "imagewoof", "id_errors": "imagewoof", "heldout": "imagewoof",
              "shuffle": "textures"}
    quality = {name: {csf: 0.6 + 0.2 * rng.random() for csf in ALL_CSFS}
               for name in ("imagewoof", "textures")}
    for name, csf in BEST.items():
        quality[name][csf] = 0.95
    metrics = []
    for name in sets:
        for csf in ALL_CSFS:
            base = quality[target.get(name, name)][csf]
            value = float(np.clip(base + 0.01 * rng.standard_normal(), 0.01, 0.99))
            if csf == broken:
                value = float("nan")
            metrics.append({"set": name, "csf": csf, "auroc": value,
                            "fpr95": 1 - value, "aurc": 0.5 - 0.4 * value,
                            "augrc": 0.3 - 0.2 * value})
    progress = epoch / EPOCHS[-1]
    clock = {"train_accuracy": 1.0 if progress >= 0.5 else 0.5 + progress,
             "train_loss": 2.0 * (1 - progress) + 0.01, "val_accuracy": 0.9 * progress,
             "val_loss": 2.0 - progress}
    descriptors = [{"group": "clock", "name": name, "layer": "", "value": value}
                   for name, value in clock.items()]
    for name in NC_METRICS:
        value = 1.0 / (1 + epoch) + 0.01 * rng.random()
        descriptors.append({"group": "nc_paper", "name": name, "layer": PENULTIMATE,
                            "value": value})
        if name.endswith("_uc") or name == "var_collapse":
            descriptors.append({"group": "nc", "name": name, "layer": PENULTIMATE,
                                "value": value})
    descriptors.append({"group": "spectrum", "name": "effective_rank",
                        "layer": PENULTIMATE, "value": 9 + rng.random() - epoch})
    placement = [{"set": name, "layer": PENULTIMATE, "name": key,
                  "value": float(rng.standard_normal() + progress)}
                 for name in sets for key in PLACEMENT]
    placement += [{"set": name, "layer": "logits", "name": "margin_ratio",
                   "value": float(0.5 + 0.1 * rng.random())} for name in sets]
    return {"run": spec.name, "epoch": epoch, "metrics": metrics,
            "descriptors": descriptors, "placement": placement,
            "checks": {"g4_max_gap": 0.0, "head_gap": 0.0}, "errors": {},
            "nonfinite": {csf: int(csf == broken) for csf in ALL_CSFS}}


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("results")
    rng = np.random.default_rng(0)
    specs = ([RunSpec(recipe, seed) for recipe in ("A", "B") for seed in range(3)]
             + [RunSpec(recipe, 0, held) for recipe in ("A", "B")
                for held in range(4)] + [RunSpec("C1", 0)])
    for spec in specs:
        folder = root / "runs" / spec.name
        (folder / results_folder()).mkdir(parents=True)
        for epoch in EPOCHS:
            broken = "pNML" if (spec.name, epoch) == ("B_seed1", 2) else None
            if (spec.name, epoch) == ("A_seed0", 0):
                broken = "GradNorm"
            out = fake_epoch(spec, epoch, rng, broken)
            (folder / results_folder() / f"epoch_{epoch:03d}.json").write_text(
                json.dumps(out))
            if spec.name == "A_seed0" and epoch == 4:
                (folder / results_folder(recheck=True)).mkdir()
                for row in out["metrics"]:
                    if row["csf"] == "KPCA_RecError":
                        row["auroc"] += 0.01
                (folder / results_folder(recheck=True) / "epoch_004.json").write_text(
                    json.dumps(out))
        (folder / "manifest.json").write_text(json.dumps({
            "imagenette": {"train": {"n1": 5, "n2": 5}, "val": {"n1": 2, "n2": 2}},
            "imagewoof": {"val": {"n8": 3, "n9": 3}}, "textures": {"images": 7}}))
    return root


def load(results: Path, recheck: bool = False) -> analyze_e.Tables:
    return analyze_e.stage_e_only(
        analyze_e.read_runs(results, results_folder(recheck=recheck)))


def test_runs_of_other_stages_are_left_out(results):
    everything = analyze_e.read_runs(results, results_folder())
    assert "C1_seed0" in set(everything.metrics["run"])
    tables = load(results)
    for frame in vars(tables).values():
        assert "C1_seed0" not in set(frame["run"])
    assert tables.metrics["run"].nunique() == 14


def test_gates_and_roster(results):
    tables = load(results)
    recheck = load(results, recheck=True)
    manifest = json.loads((results / "runs" / "A_seed0" / "manifest.json")
                          .read_text())
    table, unstable = analyze_e.gates(tables, recheck, manifest)
    status = dict(zip(table["gate"].str.split(",").str[0] + table["gate"].str[3:13],
                      table["status"]))
    assert list(table["gate"].str[:2]) == ["G1", "G1", "G2", "G3", "G4", "G5"]
    assert table["status"].tolist() == ["pass", "pass", "pass", "FAIL", "pass",
                                        "pass"], status
    assert unstable == {"KPCA_RecError"}
    kept, dropped = analyze_e.roster(tables, unstable)
    assert set(dropped["csf"]) == {"KPCA_RecError", "pNML"}
    assert "GradNorm" in kept, "a failure at epoch 0 must not remove a CSF"
    assert len(kept) == len(ALL_CSFS) - 2
    assert analyze_e.nc_cross_check(tables.descriptors) == 0.0


def test_fixed_best_finds_the_planted_csf(results):
    tables = load(results)
    kept, _ = analyze_e.roster(tables, set())
    points = analyze_e.wide(tables.metrics, kept)
    best = analyze_e.fixed_best(points)
    assert set(best["epoch"]) == {1, 2, 4} and set(best["runs"]) == {6}
    for name, csf in BEST.items():
        rows = best[best["set"] == name]
        assert (rows["b1"] == csf).all()
        assert (rows["b1_regret"] < 0.5).all()
        assert (rows["random_regret"] > 10).all()


def test_stand_ins_follow_their_target(results):
    tables = load(results)
    kept, _ = analyze_e.roster(tables, set())
    points = analyze_e.wide(tables.metrics, kept)
    within = analyze_e.within_model_stand_ins(points)
    assert set(within["stand_in"]) == {"blend", "id_errors", "shuffle"}
    assert within["regret"].mean() < 0.5
    held, subsets = analyze_e.held_out_stand_in(points)
    assert set(held["classes"]) == {4} and held["regret"].mean() < 0.5
    assert set(subsets["classes"]) == {1, 2, 3}


def test_a_checkpoint_is_converged_by_its_own_validation_accuracy():
    clock = pd.DataFrame({
        "run": ["r"] * 5 + ["s"] * 3, "epoch": [0, 1, 2, 3, 4, 0, 1, 2],
        "val_accuracy": [0.1, 0.895, 0.80, 0.92, 0.90, 0.2, 0.5, 0.505]})
    converged = analyze_e.converged_checkpoints(clock)
    assert converged["r"].to_dict() == {0: False, 1: True, 2: False, 3: True, 4: True}
    assert converged["s"].to_dict() == {0: False, 1: True, 2: True}
    assert analyze_e.convergence_epochs(converged).to_dict() == {"r": 1, "s": 1}
    disturbed = analyze_e.disturbed_checkpoints(converged)
    assert disturbed.to_dict("records") == [{"run": "r", "epoch": 2}]
    assert analyze_e.is_converged(["r", "r", "s", "x"], [2, 3, 2, 1],
                                  converged).tolist() == [False, True, True, False]


def test_no_fit_roster_and_its_cost(results):
    tables = load(results)
    kept, _ = analyze_e.roster(tables, set())
    points = analyze_e.wide(tables.metrics, kept)
    convergence = analyze_e.converged_checkpoints(
        analyze_e.clock_table(tables.descriptors))
    assert set(analyze_e.convergence_epochs(convergence)) == {4}
    no_fit = [csf for csf in kept if csf in NO_FIT_CSFS]
    assert len(no_fit) == 10
    best = analyze_e.fixed_best(points[no_fit], points, convergence)
    far, near = best[best["set"] == "textures"], best[best["set"] == "imagewoof"]
    assert (far["b1"] == "MLS").all() and (far["cost_vs_full"] < 0.5).all()
    assert (near["cost_vs_full"] > 5).all(), "the planted near winner is fitted"
    assert (best.loc[best["epoch"] < 4, "converged_runs"] == 0).all()
    assert (best.loc[best["epoch"] == 4, "converged_runs"] == 1).all()
    full = analyze_e.fixed_best(points, points, convergence)
    assert (full["cost_vs_full"] == 0).all()


def test_design_regime_measures_the_loss_before_convergence():
    index = pd.MultiIndex.from_tuples(
        [("A_seed0", "A", 0, -1, epoch, "imagewoof") for epoch in (1, 2, 4)],
        names=[*analyze_e.KEYS, "set"])
    points = pd.DataFrame({"MLS": [80.0, 80.0, 80.0], "Maha": [60.0, 70.0, 90.0],
                           "ViM": [50.0, 50.0, 90.0]}, index=index)
    converged = pd.Series({("A_seed0", 1): False, ("A_seed0", 2): False,
                           ("A_seed0", 4): True})
    table = analyze_e.design_regime(points, converged).set_index("csf")
    assert table.loc["MLS", ["unconverged", "converged", "excess"]].tolist() == [
        0, 10, -10]
    assert table.loc["Maha", "excess"] == 15 and table.loc["ViM", "excess"] == 30
    assert table["group"].to_dict() == {"MLS": "no_fit", "Maha": "fit_plain",
                                        "ViM": "fit_absolute"}


def test_fitted_csfs_use_only_converged_cells_in_link_two(results):
    tables = load(results)
    kept, _ = analyze_e.roster(tables, set())
    points = analyze_e.wide(tables.metrics, kept)
    clock = analyze_e.clock_table(tables.descriptors)
    table = analyze_e.link_two(points, tables.placement, clock,
                               analyze_e.converged_checkpoints(clock))
    cells = table.set_index("csf")["cells"]
    assert cells["MLS"] == 6 * 3 * 5 and cells["Maha"] == 6 * 1 * 5
    assert list(table.columns[3:]) == ["base", "clock", "epoch_mean", "placement",
                                       "both"]
    assert set(table["group"]) == {"no_fit", "fit_plain", "fit_relative",
                                   "fit_absolute"}


def test_strict_clock_predicts_from_the_other_runs_at_the_same_epoch():
    rng = np.random.default_rng(0)
    cells = pd.DataFrame([{"run": f"{recipe}{seed}", "recipe": recipe, "epoch": epoch}
                          for recipe in "AB" for seed in range(4)
                          for epoch in (1, 2, 4, 8)])
    trend = np.sin(cells["epoch"].to_numpy()) + (cells["recipe"] == "B").to_numpy()
    noisy = trend + 0.01 * rng.standard_normal(len(cells))
    by = ["recipe", "epoch"]
    assert analyze_e.epoch_mean_r2(cells, noisy, by) > 0.99
    assert analyze_e.epoch_mean_r2(cells, rng.standard_normal(len(cells)), by) < 0.2
    assert np.isnan(analyze_e.epoch_mean_r2(cells, np.ones(len(cells)), by))
    alone = cells[cells["run"] == "A0"]
    assert np.isnan(analyze_e.epoch_mean_r2(alone, noisy[: len(alone)], by))


def test_out_of_run_r2_recovers_a_linear_relation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + 0.01 * rng.standard_normal(120)
    groups = np.repeat(np.arange(6), 20)
    assert analyze_e.out_of_run_r2(x, y, groups, penalty=1e-6) > 0.99
    assert analyze_e.out_of_run_r2(rng.standard_normal((120, 3)), y, groups) < 0.2
    assert np.isnan(analyze_e.out_of_run_r2(x, y, np.zeros(120)))


def test_report_is_written(results, tmp_path, monkeypatch, capsys):
    out = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", ["analyze_e", "--results-root", str(results),
                                     "--out", str(out)])
    analyze_e.main()
    text = out.read_text()
    for title in ("Gates", "Rosters and convergence", "E1", "E2", "Design regime",
                  "E3", "E4", "E5", "E6", "E7"):
        assert f"## {title}" in text
    assert "No-fit roster, 10 CSFs" in text and "Full roster, 21 CSFs" in text
    assert (tmp_path / "stage_e" / "design_regime.csv").exists()
    assert "e_main`: 6 of 10 runs" in text
    assert "Ignored runs of other stages: C1_seed0" in text
    assert (tmp_path / "stage_e" / "auroc_trajectories.csv").exists()
    assert "Traceback" not in capsys.readouterr().out
