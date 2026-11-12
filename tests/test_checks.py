import numpy as np
import pandas as pd
import pytest

from dynamics import analyze_checks


def long(values: dict, epoch: int = 200, name: str = "places") -> pd.DataFrame:
    """Rows of the metrics table from {(run, csf): (auroc, fpr95)}."""
    return pd.DataFrame([{"run": run, "recipe": run[0], "epoch": epoch, "set": name,
                          "csf": csf, "auroc": auroc, "fpr95": fpr}
                         for (run, csf), (auroc, fpr) in values.items()])


def test_score_declines_uses_the_step_zero_thresholds():
    rows = []
    for seed in range(5):
        run = f"A_seed{seed}"
        rows.append(long({(run, "kNN"): (0.90, 0.3), (run, "Maha"): (0.60, 0.6),
                          (run, "MSR"): (0.80, 0.4)}, epoch=25))
        late_msr = 0.75 if seed < 3 else 0.82
        rows.append(long({(run, "kNN"): (0.85, 0.3), (run, "Maha"): (0.80, 0.6),
                          (run, "MSR"): (late_msr, 0.4)}, epoch=200))
    table = analyze_checks.score_declines(pd.concat(rows)).set_index("csf")
    assert table.loc["kNN", "drop"] == pytest.approx(5.0)
    assert table.loc["kNN", "declines"]
    assert table.loc["Maha", "drop"] == pytest.approx(-20.0)
    assert not table.loc["Maha", "declines"]
    assert table.loc["MSR", "seeds_dropping"] == 3 and not table.loc["MSR", "declines"]


def test_fixed_score_regret_per_run():
    values = {("A_seed0", "kNN"): (0.90, 0), ("A_seed0", "Maha"): (0.80, 0),
              ("A_seed1", "kNN"): (0.70, 0), ("A_seed1", "Maha"): (0.76, 0)}
    table = analyze_checks.fixed_score_regret(long(values)).set_index("run")
    assert (table["set_fixed"] == "kNN").all()
    assert table.loc["A_seed0", "set_fixed_regret"] == pytest.approx(0.0)
    assert table.loc["A_seed1", "set_fixed_regret"] == pytest.approx(6.0)
    assert table.loc["A_seed1", "winner"] == "Maha"
    assert (table["distance"] == "far").all()


def test_stand_in_agreement_and_pick():
    scores = ("a", "b", "c", "d")
    rows = []
    for name, aurocs in (("blend", (0.9, 0.8, 0.7, 0.6)),
                         ("imagewoof", (0.85, 0.80, 0.75, 0.70)),
                         ("textures", (0.60, 0.70, 0.80, 0.95))):
        rows.append(long({("A_seed0", csf): (value, 0)
                          for csf, value in zip(scores, aurocs)}, name=name))
    table = analyze_checks.stand_in_agreement(pd.concat(rows))
    table = table.set_index("target")
    assert table.loc["imagewoof", "spearman"] == pytest.approx(1.0)
    assert table.loc["imagewoof", "regret"] == pytest.approx(0.0)
    assert table.loc["textures", "spearman"] == pytest.approx(-1.0)
    assert table.loc["textures", "regret"] == pytest.approx(35.0)
    assert table.loc["imagewoof", "distance"] == "near"


def test_run_interval_covers_the_mean():
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    low, high = analyze_checks.run_interval(values)
    assert low < values.mean() < high and 0.1 <= low and high <= 0.5
    assert all(np.isnan(analyze_checks.run_interval(np.array([0.3]))))


def test_metric_choice_measures_the_cost_of_choosing_by_auroc():
    values = {("A_seed0", "kNN"): (0.90, 0.40), ("A_seed0", "Maha"): (0.89, 0.30),
              ("A_seed0", "MSR"): (0.70, 0.80)}
    row = analyze_checks.metric_choice(long(values), {"places": 5000}).iloc[0]
    assert not row["same_best"]
    assert row["fpr_regret"] == pytest.approx(10.0)
    assert row["auroc_regret"] == pytest.approx(1.0)
    assert row["fpr_se"] == pytest.approx(100 * np.sqrt(0.4 * 0.6 / 5000))
    # The two metrics swap the first two scores, so the rank correlation is 0.5.
    assert row["spearman"] == pytest.approx(0.5)
