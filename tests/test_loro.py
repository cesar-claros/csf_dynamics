import numpy as np
import pandas as pd

from dynamics import analyze_loro


def test_input_groups_split_the_columns():
    columns = ["nc_paper/var_collapse@stage4", "spread/cdnv_val@stage4",
               "clock/val_loss", "clock/log_epoch", "derived/val_nc1@stage4",
               "score/blend/kNN", "valid/cdnv@stage4", "weights/fc_norm"]
    groups = analyze_loro.input_groups(columns)
    assert groups["training"] == ["nc_paper/var_collapse@stage4"]
    assert groups["validation"] == ["spread/cdnv_val@stage4", "clock/val_loss",
                                    "derived/val_nc1@stage4"]
    assert groups["score_statistics"] == ["score/blend/kNN"]
    assert groups["clock"] == ["clock/val_loss", "clock/log_epoch"]
    assert "valid/cdnv@stage4" not in groups["all_metrics"]
    assert "weights/fc_norm" in groups["all_metrics"]


def test_leave_one_recipe_out_uses_a_metric_that_tells_the_best_score():
    rng = np.random.default_rng(0)
    rows, scores, metrics = [], [], []
    # In half of the recipes kNN is the best, and one metric tells which half.
    for i, recipe in enumerate(("H0", "H1", "H2", "H4", "H5", "H7")):
        for seed in (0, 1):
            for epoch in (50, 100):
                state = float(i % 2)
                rows.append((f"{recipe}_seed{seed}", epoch))
                scores.append({"kNN": 80.0 + 10 * state, "NCI": 90.0 - 10 * state})
                metrics.append({"nc_paper/var_collapse@stage4": state + 2
                                + 0.01 * rng.normal(),
                                "clock/val_loss": 1 + rng.random(),
                                "spread/cdnv_val@stage4": 1 + rng.random()})
    index = pd.MultiIndex.from_tuples(rows, names=["run", "epoch"])
    meta = pd.DataFrame({"eligible": True}, index=index)
    table = analyze_loro.leave_one_recipe_out(
        meta, pd.DataFrame(scores, index=index), pd.DataFrame(metrics, index=index))
    assert len(table) == 12 and table["recipe"].nunique() == 6
    # The rule from the training statistics follows the state. A fixed score cannot.
    assert table["training"].max() < 1e-6
    assert table[analyze_loro.ONE].mean() >= 5.0
    summary = analyze_loro.summary(table).set_index(["recipes", "choice"])
    assert summary.loc[("all recipes", "training"), "above_1_point"] == 0
    assert summary.loc[("label smoothing", "training"), "n"] == 1      # H5 only
