"""Tests for the SHAP helpers, plus a check that shap works with XGBoost here."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_utils import (
    additivity_error,
    as_2d,
    mean_abs_importance,
    pick_customers,
    top_drivers,
)


def test_as_2d_handles_2d_3d_and_scalar_base_values() -> None:
    v2 = np.ones((4, 3))
    values, base = as_2d(SimpleNamespace(values=v2, base_values=0.5))
    assert values.shape == (4, 3) and base.tolist() == [0.5] * 4
    v3 = np.stack([-np.ones((4, 3)), np.ones((4, 3))], axis=2)
    b2 = np.column_stack([np.zeros(4), np.full(4, 0.7)])
    values, base = as_2d(SimpleNamespace(values=v3, base_values=b2))
    assert (values == 1).all() and base.tolist() == [0.7] * 4


def test_additivity_error_is_zero_for_exact_attributions() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 5))
    base = np.full(50, -0.3)
    proba = 1 / (1 + np.exp(-(base + values.sum(axis=1))))
    assert additivity_error(base, values, proba) < 1e-12
    assert additivity_error(base, values, np.clip(proba + 0.1, 0, 1)) > 0.01


def test_pick_customers_returns_low_high_and_borderline() -> None:
    proba = np.linspace(0.01, 0.99, 200)
    picks = pick_customers(proba)
    low, mid, high = (proba[picks[k]] for k in ("low_risk", "borderline", "high_risk"))
    assert low < mid < high
    assert abs(proba[picks["borderline"]] - 0.5) < 0.01
    assert proba[picks["low_risk"]] < 0.1 and proba[picks["high_risk"]] > 0.9


def test_importance_ranks_features_and_reads_the_direction() -> None:
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"a": rng.random(300), "b": rng.random(300), "c": np.ones(300)})
    values = np.column_stack([3 * x["a"] - 1.5, -0.5 * x["b"], np.zeros(300)])
    table = mean_abs_importance(values, x)
    assert table["feature"].tolist() == ["a", "b", "c"]
    direction = dict(zip(table["feature"], table["direction"]))
    assert direction["a"] == "higher value -> higher churn risk"
    assert direction["b"] == "higher value -> lower churn risk"
    assert direction["c"] == "mixed or non-linear"


def test_top_drivers_orders_by_absolute_effect() -> None:
    row = pd.Series([1.0, 2.0, 3.0])
    out = top_drivers(np.array([0.1, -0.9, 0.4]), ["a", "b", "c"], row, k=2)
    assert [d["feature"] for d in out] == ["b", "c"]
    assert out[0]["value"] == 2.0 and out[0]["shap"] == -0.9


def test_shap_values_add_up_for_a_small_xgboost_model() -> None:
    shap = pytest.importorskip("shap")
    xgboost = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.random((300, 4)), columns=list("abcd"))
    y = (x["a"] + 0.3 * rng.random(300) > 0.7).astype(int)
    model = xgboost.XGBClassifier(n_estimators=30, max_depth=3, random_state=0)
    model.fit(x, y)
    values, base = as_2d(shap.TreeExplainer(model)(x))
    assert values.shape == (300, 4)
    proba = model.predict_proba(x)[:, 1]
    assert additivity_error(base, values, proba) < 1e-3
