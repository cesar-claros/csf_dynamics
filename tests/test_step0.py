import json

import numpy as np
import pandas as pd
import pytest

from dynamics import analyze_step0
from dynamics.config import PENULTIMATE, RunSpec, results_folder
from dynamics.scores import ALL_CSFS, LAYER_CSFS

FAR = analyze_step0.FAR_SETS


def summary(declines: dict[str, float], late: float = 80.0, seeds_declining: int = 5
            ) -> pd.DataFrame:
    return pd.DataFrame([{"set": name, "recipe": "A", "early": late + decline,
                          "late": late, "decline": decline, "seeds": 5,
                          "seeds_declining": seeds_declining}
                         for name, decline in declines.items()])


def test_rule_one_needs_three_of_the_four_far_sets():
    verdict, hits = analyze_step0.rule_one(
        summary({"inaturalist": 8, "places": 5, "sun": 3.0, "openimage_o": 1}))
    assert verdict.startswith("the collapse hypothesis survives: 3 of 4")
    assert hits == ["inaturalist", "places", "sun"]
    verdict, hits = analyze_step0.rule_one(
        summary({"inaturalist": 8, "places": 5, "sun": 2.9, "openimage_o": 1}))
    assert verdict.startswith("the decline does not hold beyond Textures: 2 of 4")
    verdict, _ = analyze_step0.rule_one(summary({"inaturalist": 8, "places": 5}))
    assert verdict.startswith("not decided")


def test_a_decline_needs_most_seeds():
    table = summary(dict.fromkeys(FAR, 8.0), seeds_declining=3)
    assert not analyze_step0.declines(table, "places")
    assert analyze_step0.declines(summary(dict.fromkeys(FAR, 8.0), seeds_declining=4),
                                  "places")
    assert not analyze_step0.declines(summary({"places": 8.0}), "places", regime="B")


def test_rule_two_needs_a_smaller_loss_and_a_higher_final_auroc():
    last = summary(dict.fromkeys(FAR, 10.0), late=80.0)
    keeps = summary(dict.fromkeys(FAR, 2.0), late=85.0)
    verdict, table = analyze_step0.rule_two(last, keeps)
    assert verdict == "the layer route is viable: 4 of 4 far sets pass"
    assert table["passes"].all()

    lower = summary(dict.fromkeys(FAR, 2.0), late=70.0)
    verdict, table = analyze_step0.rule_two(last, lower)
    assert verdict.startswith("the layer route is not viable: 0 of 4")
    assert table["loses_less_than_half"].all()
    assert not table["final_at_least_stage4"].any()

    loses = summary(dict.fromkeys(FAR, 6.0), late=85.0)
    assert analyze_step0.rule_two(last, loses)[0].startswith(
        "the layer route is not viable")

    flat = summary({"inaturalist": 10, "places": 10, "sun": 1, "openimage_o": 1})
    assert analyze_step0.rule_two(flat, keeps)[0].startswith("not applicable")


def fake_epoch(spec: RunSpec, epoch: int, rng: np.random.Generator) -> dict:
    """A result file where regime A loses far-OOD AUROC late, at stage 4 only.

    Stage 3 stays at 0.90, above the late stage-4 value of 0.83, so rule 2 passes.
    """
    sets = ["textures", *FAR, *analyze_step0.NEAR_SETS, "imagewoof"]
    late = epoch == analyze_step0.LATE
    drop = 0.10 if (spec.recipe == "A" and late) else 0.0
    metrics, layer_metrics = [], []
    for name in sets:
        far = name in FAR or name == "textures"
        for csf in ALL_CSFS:
            top = csf == "PCA_RecError"
            value = (0.93 if top else 0.80) - (drop if far else 0.0)
            value += 0.002 * rng.standard_normal()
            metrics.append({"set": name, "csf": csf, "auroc": value,
                            "fpr95": 1 - value, "aurc": 0.1, "augrc": 0.1})
        for layer in ("stage1", "stage2", "stage3"):
            for csf in LAYER_CSFS:
                value = 0.90 + 0.002 * rng.standard_normal()
                layer_metrics.append({"layer": layer, "set": name, "csf": csf,
                                      "auroc": value, "fpr95": 1 - value,
                                      "aurc": 0.1, "augrc": 0.1})
    placement = [{"set": name, "layer": layer, "name": "outside_share_gap",
                  "value": 0.2 - (0.1 if late and layer == PENULTIMATE else 0.0)}
                 for name in sets for layer in ("stage3", PENULTIMATE)]
    descriptors = [{"group": "clock", "name": "val_accuracy", "layer": "",
                    "value": 0.8}]
    return {"run": spec.name, "epoch": epoch, "metrics": metrics,
            "layer_metrics": layer_metrics, "layer_errors": {},
            "descriptors": descriptors, "placement": placement,
            "checks": {"g4_max_gap": 0.0, "head_gap": 0.0}, "errors": {},
            "nonfinite": dict.fromkeys(ALL_CSFS, 0)}


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("step0")
    rng = np.random.default_rng(0)
    tag = results_folder(suite=analyze_step0.SUITE)
    for spec in [RunSpec(recipe, seed) for recipe in ("A", "B") for seed in range(5)]:
        folder = root / "runs" / spec.name
        (folder / tag).mkdir(parents=True)
        for epoch in (analyze_step0.EARLY, 100, analyze_step0.LATE):
            (folder / tag / f"epoch_{epoch:03d}.json").write_text(
                json.dumps(fake_epoch(spec, epoch, rng)))
        (folder / "manifest.json").write_text(json.dumps({
            "imagenette": {"train": {"n1": 5}}, "textures": {"images": 7},
            "places": {"images": 9000, "used": 5000, "id_class_folders": []},
            "sun": {"images": 100, "used": 100, "id_class_folders": ["n1"]}}))
    return root


def test_best_per_cell_and_change_table(results):
    tables = analyze_step0.analyze_e.read_runs(
        results, results_folder(suite=analyze_step0.SUITE))
    best = analyze_step0.best_per_cell(tables.metrics, list(ALL_CSFS))
    changes = analyze_step0.summarize_changes(analyze_step0.change_table(best))
    places = changes[changes["set"] == "places"].set_index("recipe")
    assert places.loc["A", "decline"] == pytest.approx(10, abs=0.5)
    assert places.loc["B", "decline"] == pytest.approx(0, abs=0.5)
    assert places.loc["A", "seeds_declining"] == 5 and places.loc["A", "seeds"] == 5
    near = changes[changes["set"] == "ninco"].set_index("recipe")
    assert abs(near.loc["A", "decline"]) < 0.5
    layers = analyze_step0.read_layer_metrics(
        results, results_folder(suite=analyze_step0.SUITE))
    assert set(layers["layer"]) == {"stage1", "stage2", "stage3"}


def test_report_states_both_verdicts(results, tmp_path, monkeypatch):
    out = tmp_path / "step0_report.md"
    monkeypatch.setattr("sys.argv", ["analyze_step0", "--results-root", str(results),
                                     "--out", str(out)])
    analyze_step0.main()
    text = out.read_text()
    assert "Rule 1: the collapse hypothesis survives: 4 of 4 far sets" in text
    assert "Rule 2: the layer route is viable: 4 of 4 far sets pass" in text
    assert "'sun': ['n1']" in text and "'places': 5000" in text
    assert (tmp_path / "step0" / "depth_profile.csv").exists()
