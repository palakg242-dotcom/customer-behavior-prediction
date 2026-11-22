"""Tests for the modeling split, feature matrix and metric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.common import (
    DEFAULTS,
    compute_metrics,
    make_features,
    split_data,
)


def _toy_frame(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "country": rng.choice(["United Kingdom", "France"], size=n, p=[0.9, 0.1]),
            "recency_days": rng.integers(0, 400, size=n),
            "frequency": rng.integers(1, 30, size=n),
            "clv_historical": rng.random(n) * 1000,
            "engagement_score": rng.random(n),
            "churn_90d": (rng.random(n) < 0.565).astype(int),
            "clv_target": rng.random(n) * 100,
            "next_category": rng.choice(["home_decor", None], size=n),
        }
    )


def test_split_sizes_are_70_15_15() -> None:
    train, val, test = split_data(_toy_frame(), DEFAULTS)
    assert abs(len(train) - 700) <= 2
    assert abs(len(val) - 150) <= 2
    assert abs(len(test) - 150) <= 2


def test_split_has_no_customer_overlap() -> None:
    train, val, test = split_data(_toy_frame(), DEFAULTS)
    ids = [set(part["customer_id"]) for part in (train, val, test)]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert sum(len(s) for s in ids) == 1000


def test_split_is_stratified_and_deterministic() -> None:
    df = _toy_frame()
    a_train, a_val, a_test = split_data(df, DEFAULTS)
    b_train, _, _ = split_data(df, DEFAULTS)
    overall = df["churn_90d"].mean()
    for part in (a_train, a_val, a_test):
        assert abs(part["churn_90d"].mean() - overall) < 0.02
    assert a_train["customer_id"].tolist() == b_train["customer_id"].tolist()


def test_features_exclude_targets_and_ids() -> None:
    x = make_features(_toy_frame())
    for col in ("customer_id", "country", "churn_90d", "clv_target", "next_category"):
        assert col not in x.columns
    assert "is_uk" in x.columns


def test_linear_features_drop_redundant_columns() -> None:
    df = _toy_frame()
    linear = make_features(df, linear=True)
    tree = make_features(df, linear=False)
    assert "clv_historical" not in linear.columns
    assert "engagement_score" not in linear.columns
    assert "clv_historical" in tree.columns


def test_metrics_on_perfect_predictions() -> None:
    y = np.array([0, 1, 1, 0, 1])
    m = compute_metrics(y, y, y.astype(float))
    assert m["recall"] == 1.0
    assert m["roc_auc"] == 1.0
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (3, 2, 0, 0)
