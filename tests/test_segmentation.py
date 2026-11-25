"""Tests for the segmentation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.segmentation.cluster import (
    FEATURES,
    ONE_TIME_LAPSED,
    ONE_TIME_RECENT,
    best_k,
    name_all,
    name_segment,
    one_time_mask,
    one_time_names,
    prepare,
    profile_table,
    select_k,
)

REF = {
    "recency_days_median": 100.0,
    "frequency_median": 3.0,
    "monetary_total_median": 500.0,
    "seasonal_concentration_p75": 0.6,
}


def _stats(rec: float, freq: float, money: float, season: float = 0.2) -> dict:
    return {
        "recency_days": rec,
        "frequency": freq,
        "monetary_total": money,
        "seasonal_concentration": season,
    }


def test_prepare_has_no_nan_and_handles_negative_money() -> None:
    df = pd.DataFrame(
        {
            "recency_days": [0, 10, 300],
            "frequency": [1, 5, 40],
            "monetary_total": [-50.0, 200.0, 9000.0],
            "monetary_avg": [-5.0, 40.0, 225.0],
            "return_rate": [0.0, 0.1, np.nan],
            "seasonal_concentration": [0.2, 0.5, 0.9],
        }
    )
    x, _ = prepare(df)
    assert x.shape == (3, len(FEATURES))
    assert not np.isnan(x).any()


def test_select_k_finds_three_separated_groups() -> None:
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [8, 8], [-8, 8]])
    x = np.vstack([c + rng.normal(0, 0.5, size=(100, 2)) for c in centers])
    table = select_k(x, range(2, 7), seed=42)
    assert best_k(table) == 3
    assert {"k", "inertia", "silhouette", "davies_bouldin"} <= set(table.columns)


def test_name_segment_rules() -> None:
    assert name_segment(_stats(20, 10, 2000), REF) == "Loyal high-value"
    assert name_segment(_stats(250, 4, 2000), REF) == "Lapsed high-value"
    assert name_segment(_stats(250, 1, 80), REF) == "Lapsed low-value"
    assert name_segment(_stats(20, 8, 100), REF) == "High-frequency low-value"
    assert name_segment(_stats(20, 1, 900), REF).startswith("Recent high-value")
    assert name_segment(_stats(20, 1, 100), REF) == "Recent low-value"
    seasonal = name_segment(_stats(20, 10, 2000, season=0.8), REF)
    assert seasonal == "Seasonal gift shopper"


def test_name_all_makes_duplicate_names_unique() -> None:
    df = pd.DataFrame(
        {
            "recency_days": [10, 12, 11, 400, 410, 390],
            "frequency": [1, 1, 1, 1, 1, 1],
            "monetary_total": [50.0, 60.0, 55.0, 40.0, 45.0, 42.0],
            "monetary_avg": [50.0] * 6,
            "return_rate": [0.0] * 6,
            "seasonal_concentration": [0.1] * 6,
        }
    )
    names = name_all(df, np.array([0, 0, 0, 1, 1, 1]))
    assert len(set(names.values())) == 2


def test_one_time_buyers_are_split_by_recency() -> None:
    df = pd.DataFrame({"frequency": [1, 1, 5, 1], "recency_days": [30, 300, 10, 180]})
    assert one_time_mask(df).tolist() == [True, True, False, True]
    names = one_time_names(df[one_time_mask(df)], recent_days=180)
    assert names.tolist() == [ONE_TIME_RECENT, ONE_TIME_LAPSED, ONE_TIME_RECENT]


def test_profile_table_covers_every_customer() -> None:
    n = 6
    df = pd.DataFrame({c: np.arange(n, dtype=float) for c in FEATURES})
    df["churn_90d"] = [0, 1, 0, 1, 1, 0]
    df["clv_target"] = np.arange(n) * 10.0
    cfg = {"churn_target": "churn_90d", "clv_target": "clv_target"}
    names = np.array(["A", "A", "A", "B", "B", "B"], dtype=object)
    table = profile_table(df, names, cfg)
    assert table["customers"].sum() == n
    assert abs(table["share"].sum() - 1.0) < 1e-6
    assert set(table.index) == {"A", "B"}


def test_high_return_segment_is_named_for_returns_first() -> None:
    ref = {**REF, "return_rate_p90": 0.1}
    stats = {**_stats(177, 4, 930, season=0.9), "return_rate": 0.45}
    assert name_segment(stats, ref) == "High-return buyer"
    calm = {**stats, "return_rate": 0.0}
    assert name_segment(calm, ref) == "Seasonal gift shopper"
