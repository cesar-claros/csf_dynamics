import pandas as pd
import pytest

from dynamics import analyze_factors
from dynamics.config import RECIPES, Recipe


def test_matched_pairs_differ_in_one_factor():
    recipes = {"A": Recipe(augment=False), "B": Recipe(augment=True),
               "C2": Recipe(augment=True, label_smoothing=0.1),
               "C3": Recipe(augment=False, lr=0.01),
               "H0": Recipe(augment=False, epochs=100)}
    pairs = analyze_factors.matched_pairs(recipes)
    assert pairs == [("augment", "A", "B"), ("lr", "A", "C3"),
                     ("label_smoothing", "B", "C2")]
    # In the real recipes, V2 is T1 with label smoothing and C2 with a lower rate.
    real = analyze_factors.matched_pairs(
        {name: recipe for name, recipe in RECIPES.items() if recipe.epochs == 200})
    assert ("label_smoothing", "T1", "V2") in real and ("lr", "C2", "V2") in real
    assert ("augment", "U1", "V2") in real
    assert all(factor in analyze_factors.FACTORS for factor, _, _ in real)


def test_transfer_regret_and_rank_agreement():
    first = pd.Series({"kNN": 90.0, "Maha": 60.0, "MSR": 80.0})
    same = analyze_factors.compare(first, first + 1.0)
    assert same == {"spearman": pytest.approx(1.0), "transfer_regret": 0.0}
    second = pd.Series({"kNN": 70.0, "Maha": 95.0, "MSR": 80.0})
    out = analyze_factors.compare(first, second)
    # kNN loses 25 points in the second recipe, and Maha loses 30 in the first.
    assert out["transfer_regret"] == pytest.approx(27.5)
    assert out["spearman"] == pytest.approx(-1.0)
