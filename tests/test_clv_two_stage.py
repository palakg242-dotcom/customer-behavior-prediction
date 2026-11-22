"""Tests for the two-stage CLV model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.clv_regression import to_log
from src.models.clv_two_stage import combine, evaluate_combined


def test_combine_multiplies_probability_by_expected_spend() -> None:
    spend_log = to_log(pd.Series([100.0, 100.0, 100.0]))
    out = combine(np.array([0.0, 0.25, 1.0]), spend_log)
    assert np.allclose(out, [100.0, 75.0, 0.0])


def test_combine_clips_out_of_range_probabilities() -> None:
    out = combine(np.array([-0.5, 1.5]), to_log(pd.Series([10.0, 10.0])))
    assert np.allclose(out, [10.0, 0.0])


def test_evaluate_combined_is_perfect_for_exact_predictions() -> None:
    y = pd.Series([0.0, 50.0, 200.0, 0.0])
    recency = pd.Series([10, 20, 300, 400])
    res = evaluate_combined(y.to_numpy(), y, recency, 180)
    assert res[("all", "original")]["r2"] == 1.0
    assert res[("all", "log")]["mae"] < 1e-9
    assert ("recency<=180", "original") in res
