"""Tests for the anomaly-detector settings and helpers (no torch needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.anomaly_utils import (
    DEFAULTS,
    INJECTIONS,
    detection_rate,
    flag_rate,
    inject,
    load_anomaly_config,
    pick_threshold,
    top_features,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "monetary_total": [100.0, 200.0],
            "monetary_avg": [50.0, 40.0],
            "avg_basket_qty": [10.0, 12.0],
            "avg_basket_lines": [5.0, 6.0],
            "return_rate": [0.0, 0.1],
        }
    )


def test_threshold_is_a_percentile_of_the_errors() -> None:
    errors = np.arange(1, 101, dtype=float)
    assert pick_threshold(errors, 99.0) == np.percentile(errors, 99.0)
    assert 0.005 <= flag_rate(errors, pick_threshold(errors, 99.0)) <= 0.02


def test_flag_rate_counts_only_values_strictly_above_the_threshold() -> None:
    assert flag_rate(np.array([1.0, 2.0, 3.0, 4.0]), 3.0) == 0.25


def test_top_features_names_the_largest_contributor() -> None:
    errors = np.array([[0.1, 0.9, 0.0], [0.5, 0.2, 0.3]])
    assert top_features(errors, ["a", "b", "c"]) == ["b", "a"]


def test_injections_change_a_copy_not_the_original() -> None:
    df = _frame()
    out = inject(df, "spend_spike", 10)
    assert out["monetary_total"].tolist() == [1000.0, 2000.0]
    assert out["monetary_avg"].tolist() == [500.0, 400.0]
    assert df["monetary_total"].tolist() == [100.0, 200.0]
    assert out["return_rate"].tolist() == df["return_rate"].tolist()


def test_bulk_basket_and_return_spike_touch_the_right_columns() -> None:
    df = _frame()
    bulk = inject(df, "bulk_basket", 5)
    assert bulk["avg_basket_qty"].tolist() == [50.0, 60.0]
    assert bulk["monetary_total"].tolist() == df["monetary_total"].tolist()
    ret = inject(df, "return_spike", 0.95)
    assert ret["return_rate"].tolist() == [0.95, 0.95]


def test_every_injection_kind_has_levels_in_increasing_order() -> None:
    for levels, _fn in INJECTIONS.values():
        assert levels == sorted(levels)
        assert len(levels) >= 2


def test_detection_rate_is_the_share_above_the_threshold() -> None:
    assert detection_rate(np.array([5.0, 6.0, 1.0, 9.0]), 4.0) == 0.75


def test_config_has_sane_defaults() -> None:
    cfg = load_anomaly_config()
    assert set(DEFAULTS) <= set(cfg)
    assert 0 < cfg["threshold_percentile"] < 100
    assert cfg["min_flag_rate"] < cfg["max_flag_rate"]
    assert len(cfg["features"]) >= 3
