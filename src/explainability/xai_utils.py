"""Helpers for LIME and partial dependence plots. No lime or shap import needed."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def make_grid(values: np.ndarray, n: int = 30) -> np.ndarray:
    """Grid for a partial dependence plot: quantiles of the data plus its maximum.

    Quantiles follow the data, so crowded regions get more points. Adding the
    maximum keeps an extreme value (such as 1.0 for a share) on the plot.
    """
    v = np.asarray(values, dtype=float)
    grid = np.quantile(v, np.linspace(0.01, 0.99, n))
    return np.unique(np.append(grid, v.max()))


def pdp_values(model: Any, x: pd.DataFrame, feature: str, grid: np.ndarray) -> pd.DataFrame:
    """Average churn probability when ``feature`` is set to each grid value.

    Every customer is given the grid value in turn, all other features stay as
    they are, and the predicted probabilities are averaged.
    """
    rows = []
    for value in grid:
        shifted = x.copy()
        shifted[feature] = value
        average = float(model.predict_proba(shifted)[:, 1].mean())
        rows.append({"feature": feature, "value": float(value), "avg_churn_probability": average})
    return pd.DataFrame(rows)


def compare_rankings(
    lime_pairs: list[tuple[int, float]],
    shap_row: np.ndarray,
    feature_names: list[str],
    k: int = 5,
) -> dict[str, Any]:
    """How well do LIME and SHAP agree for one customer?

    ``lime_pairs`` are (feature index, weight) pairs, with positive weights
    pushing towards churn. Returns the overlap of the two top-k lists and how
    often the shared features have the same sign (the same direction of effect).
    """
    lime_top = sorted(lime_pairs, key=lambda p: -abs(p[1]))[:k]
    lime_names = [feature_names[int(i)] for i, _ in lime_top]
    shap_order = np.argsort(-np.abs(np.asarray(shap_row)))[:k]
    shap_names = [feature_names[int(i)] for i in shap_order]
    shared = [name for name in lime_names if name in shap_names]
    lime_sign = {feature_names[int(i)]: np.sign(w) for i, w in lime_pairs}
    same_sign = [lime_sign[name] == np.sign(shap_row[feature_names.index(name)]) for name in shared]
    return {
        "lime_top": lime_names,
        "shap_top": shap_names,
        "overlap": len(shared),
        "k": k,
        "sign_agreement": float(np.mean(same_sign)) if same_sign else float("nan"),
    }
