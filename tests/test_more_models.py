"""Tests for the Day 8 models (KNN, LightGBM, CLV regression)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.clv_regression import (
    from_log,
    regression_metrics,
    to_log,
    tune_rf_regressor,
)
from src.models.more_classifiers import grid_size, tune_knn


def _toy_xy(n: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.random((n, 4)), columns=list("abcd"))
    y = pd.Series((x["a"] + 0.2 * rng.random(n) > 0.6).astype(int))
    return x, y


def test_grid_size_multiplies_options() -> None:
    assert grid_size({"a": [1, 2], "b": [1, 2, 3]}) == 6


def test_knn_search_returns_a_working_pipeline() -> None:
    x, y = _toy_xy()
    search = tune_knn(x, y, seed=42, n_iter=2)
    proba = search.best_estimator_.predict_proba(x)[:, 1]
    assert proba.shape == (len(x),)
    assert search.best_score_ > 0.5


def test_lightgbm_search_if_installed() -> None:
    pytest.importorskip("lightgbm")
    from src.models.more_classifiers import tune_lightgbm

    x, y = _toy_xy()
    search = tune_lightgbm(x, y, seed=42, n_iter=2)
    assert search.best_estimator_.predict_proba(x)[:, 1].shape == (len(x),)


def test_log_transform_round_trip_and_never_negative() -> None:
    y = pd.Series([0.0, 10.0, 250.0])
    assert np.allclose(from_log(to_log(y)), y.to_numpy())
    assert (from_log(np.array([-5.0, 0.0])) >= 0).all()


def test_regression_metrics_on_perfect_predictions() -> None:
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0


def test_rf_regressor_search_learns_a_signal() -> None:
    x, _ = _toy_xy()
    y_log = to_log(pd.Series(x["a"] * 100))
    search = tune_rf_regressor(x, y_log, seed=42, n_iter=2)
    assert search.best_score_ > 0.5
