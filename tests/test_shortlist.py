import pandas as pd
import pytest

from dynamics import analyze_shortlist as sl


def scores() -> pd.DataFrame:
    return pd.DataFrame({"a": [90.0, 70.0, 80.0], "b": [70.0, 90.0, 80.0],
                         "c": [85.0, 85.0, 85.0], "d": [60.0, 60.0, 60.0]},
                        index=["r1", "r2", "r3"])


def test_shortlist_regret_for_one_list_and_for_a_list_per_run():
    table = scores()
    assert sl.shortlist_regret(table, ["c"]).tolist() == [5.0, 5.0, 0.0]
    assert sl.shortlist_regret(table, ["a", "b"]).tolist() == [0.0, 0.0, 5.0]
    per_run = sl.shortlist_regret(table, [["a"], ["b"], ["d"]])
    assert per_run.tolist() == [0.0, 0.0, 25.0]


def test_greedy_takes_the_compromise_first_and_then_covers_the_rest():
    assert sl.greedy(scores(), 1) == ["c"]
    assert set(sl.greedy(scores(), 3)) == {"a", "b", "c"}
    assert sl.shortlist_regret(scores(), sl.greedy(scores(), 3)).sum() == 0


def test_random_regret_is_between_the_best_and_the_worst_subset():
    table = scores()
    found = sl.random_regret(table, 2)
    assert (found >= 0).all() and (found < 30).all()
    assert sl.random_regret(table, 4).tolist() == pytest.approx([0.0, 0.0, 0.0])
