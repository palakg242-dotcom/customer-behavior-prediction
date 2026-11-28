"""Tests for the partial dependence and LIME-vs-SHAP helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.explainability.xai_utils import compare_rankings, make_grid, pdp_values


def _data(n: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.random((n, 3)), columns=["a", "b", "c"])
    y = (2 * x["a"] - x["b"] + 0.2 * rng.normal(size=n) > 0.5).astype(int)
    return x, y


def test_grid_is_sorted_unique_and_includes_the_maximum() -> None:
    values = np.array([0.0] * 50 + [1.0] * 50)
    grid = make_grid(values)
    assert grid.tolist() == sorted(set(grid.tolist()))
    assert grid.max() == 1.0 and grid.min() >= 0.0


def test_pdp_rises_with_a_feature_the_model_uses_positively() -> None:
    x, y = _data()
    model = LogisticRegression().fit(x, y)
    frame = pdp_values(model, x, "a", make_grid(x["a"].to_numpy(), n=10))
    probs = frame["avg_churn_probability"].to_numpy()
    assert (np.diff(probs) >= -1e-12).all()
    assert probs[-1] > probs[0] + 0.2
    flat = pdp_values(model, x, "c", make_grid(x["c"].to_numpy(), n=10))
    assert flat["avg_churn_probability"].max() - flat["avg_churn_probability"].min() < 0.1


def test_pdp_does_not_modify_the_data() -> None:
    x, y = _data()
    before = x.copy()
    pdp_values(LogisticRegression().fit(x, y), x, "a", np.array([0.0, 1.0]))
    pd.testing.assert_frame_equal(x, before)


def test_compare_rankings_counts_overlap_and_sign_agreement() -> None:
    names = ["a", "b", "c", "d"]
    lime_pairs = [(0, 0.5), (1, -0.4), (2, 0.1), (3, -0.05)]
    shap_row = np.array([0.9, 0.3, -0.2, 0.0])
    out = compare_rankings(lime_pairs, shap_row, names, k=2)
    assert out["lime_top"] == ["a", "b"]
    assert out["shap_top"] == ["a", "b"]
    assert out["overlap"] == 2
    assert out["sign_agreement"] == 0.5  # a agrees, b has opposite signs


def test_compare_rankings_is_nan_without_shared_features() -> None:
    names = ["a", "b", "c", "d"]
    out = compare_rankings([(0, 1.0)], np.array([0.0, 0.0, 0.0, 2.0]), names, k=1)
    assert out["overlap"] == 0
    assert np.isnan(out["sign_agreement"])


def test_lime_returns_feature_index_weight_pairs() -> None:
    lime_tabular = pytest.importorskip("lime.lime_tabular")
    x, y = _data()
    model = LogisticRegression().fit(x, y)
    explainer = lime_tabular.LimeTabularExplainer(
        x.to_numpy(),
        feature_names=list(x.columns),
        class_names=["stay", "churn"],
        mode="classification",
        random_state=0,
    )

    def predict_fn(rows: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(rows, columns=x.columns))

    exp = explainer.explain_instance(
        x.iloc[0].to_numpy(), predict_fn, num_features=3, num_samples=500
    )
    pairs = exp.as_map()[1]
    assert len(pairs) == 3
    assert all(0 <= int(i) < 3 for i, _w in pairs)
