"""Tests for the tuned tree models (small, fast searches on toy data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.tree_models import top_features, tune_random_forest


def _toy_xy(n: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.random((n, 4)), columns=list("abcd"))
    y = pd.Series((x["a"] + 0.2 * rng.random(n) > 0.6).astype(int))
    return x, y


def test_random_forest_search_returns_a_working_model() -> None:
    x, y = _toy_xy()
    search = tune_random_forest(x, y, seed=42, n_iter=2)
    proba = search.best_estimator_.predict_proba(x)[:, 1]
    assert proba.shape == (len(x),)
    assert 0.5 < search.best_score_ <= 1.0


def test_top_features_finds_the_informative_column() -> None:
    x, y = _toy_xy()
    search = tune_random_forest(x, y, seed=42, n_iter=2)
    top = top_features(search.best_estimator_, list(x.columns), n=2)
    assert top.index[0] == "a"


def test_xgboost_search_if_installed() -> None:
    pytest.importorskip("xgboost")
    from src.models.tree_models import tune_xgboost

    x, y = _toy_xy()
    model, search = tune_xgboost(x[:300], y[:300], x[300:], y[300:], 42, 2)
    assert model.predict_proba(x)[:, 1].shape == (len(x),)
    assert search.best_score_ > 0.5
