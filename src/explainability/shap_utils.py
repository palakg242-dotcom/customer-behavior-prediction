"""Helpers for the SHAP analysis. Nothing here needs the shap package."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def as_2d(explanation: Any) -> tuple[np.ndarray, np.ndarray]:
    """SHAP values (rows x features) and one base value per row, in log-odds.

    Some model/library combinations return a third axis, one slice per class;
    the slice for the positive (churn) class is used then.
    """
    values = np.asarray(explanation.values, dtype=float)
    base = np.asarray(explanation.base_values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 1]
    if base.ndim == 2:
        base = base[:, 1]
    if base.ndim == 0:
        base = np.full(values.shape[0], float(base))
    return values, base


def additivity_error(base: np.ndarray, values: np.ndarray, proba: np.ndarray) -> float:
    """Largest gap between the model's probability and sigmoid(base + sum SHAP).

    SHAP values are exact attributions, so this should be tiny. A large value
    means the explanation does not match the model being explained.
    """
    margin = np.asarray(base) + np.asarray(values).sum(axis=1)
    rebuilt = 1.0 / (1.0 + np.exp(-margin))
    return float(np.abs(rebuilt - np.asarray(proba)).max())


def pick_customers(proba: np.ndarray, low_q: float = 0.05, high_q: float = 0.95) -> dict[str, int]:
    """Row positions of a low-risk, a high-risk and a borderline customer.

    Quantiles (not the extremes) are used so that the examples are typical
    of their group; borderline is the customer closest to 50% churn risk.
    """
    p = np.asarray(proba, dtype=float)

    def nearest(target: float) -> int:
        return int(np.abs(p - target).argmin())

    return {
        "low_risk": nearest(float(np.quantile(p, low_q))),
        "high_risk": nearest(float(np.quantile(p, high_q))),
        "borderline": nearest(0.5),
    }


def mean_abs_importance(values: np.ndarray, x: pd.DataFrame) -> pd.DataFrame:
    """Global importance (mean |SHAP|) and the direction of each feature's effect."""
    rows = []
    for j, name in enumerate(x.columns):
        col = pd.Series(x.iloc[:, j].to_numpy(dtype=float))
        shap_col = pd.Series(values[:, j])
        if col.nunique() < 2 or shap_col.nunique() < 2:
            corr = 0.0
        else:
            corr = float(col.corr(shap_col, method="spearman"))
        if corr > 0.2:
            direction = "higher value -> higher churn risk"
        elif corr < -0.2:
            direction = "higher value -> lower churn risk"
        else:
            direction = "mixed or non-linear"
        rows.append(
            {
                "feature": name,
                "mean_abs_shap": float(np.abs(values[:, j]).mean()),
                "spearman_value_vs_shap": round(corr, 3),
                "direction": direction,
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    return out.reset_index(drop=True)


def top_drivers(
    values_row: np.ndarray, feature_names: list[str], x_row: pd.Series, k: int = 3
) -> list[dict[str, Any]]:
    """The k features with the largest absolute SHAP value for one customer."""
    order = np.argsort(-np.abs(np.asarray(values_row)))[:k]
    return [
        {
            "feature": feature_names[i],
            "value": float(x_row.iloc[i]),
            "shap": float(values_row[i]),
        }
        for i in order
    ]
