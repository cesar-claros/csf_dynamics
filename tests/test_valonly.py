import pandas as pd

from dynamics import analyze_valonly as vo


def test_metrics_are_sorted_by_what_they_read():
    expected = {
        "spread/cdnv_val@stage4": "validation", "spread/trace_ratio_val@stage1":
        "validation", "logits/margin_val@logits": "validation",
        "clock/val_accuracy": "validation", "derived/val_nc1@stage3": "validation",
        "spread/cdnv_train@stage4": "training", "nc_paper/var_collapse@stage4":
        "training", "spectrum/effective_rank@stage2": "training",
        "clock/train_loss": "training",
        "spread/val_over_train_inside@stage4": "mixed",
        "gap/var_collapse_val_over_train@stage4": "mixed",
        "norm_angle/class_cosine_val@stage4": "mixed",
        "region/neighbor_radius_ratio@stage4": "mixed",
        "clock/log_epoch": "other", "weights/conv_norm": "other"}
    assert {name: vo.input_set(name) for name in expected} == expected


def test_validation_nc1_is_the_ratio_times_the_training_value():
    metrics = pd.DataFrame({"gap/var_collapse_val_over_train@stage4": [4.0, 10.0],
                            "nc/var_collapse@stage4": [0.5, 0.01]})
    out = vo.with_validation_nc1(metrics)
    assert out["derived/val_nc1@stage4"].tolist() == [2.0, 0.1]
    assert "derived/val_nc1@stage1" not in out
