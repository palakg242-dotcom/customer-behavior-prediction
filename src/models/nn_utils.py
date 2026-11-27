"""Helpers shared by the neural-network models. No torch import is needed here."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Right-skewed columns get a log1p transform before scaling. Tree models do not
# need this, but neural networks train much better on roughly symmetric inputs.
SKEWED = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "clv_historical",
    "avg_basket_qty",
    "avg_basket_lines",
    "gap_variance",
]


def log_skewed(df: pd.DataFrame) -> pd.DataFrame:
    """log1p of the skewed columns (negative values are treated as 0)."""
    out = df.copy()
    for col in SKEWED:
        if col in out.columns:
            out[col] = np.log1p(out[col].clip(lower=0))
    return out


def build_preprocessor() -> Pipeline:
    """Log transform, median imputation and standardization.

    Fit it on the training rows only, then use ``transform`` on validation and
    test rows, so that no information flows from them into the scaling.
    """
    return Pipeline(
        [
            ("log", FunctionTransformer(log_skewed, validate=False)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def error_overlap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """How much two classifiers' mistakes overlap.

    A low error correlation means the second model makes different mistakes
    and could serve as a useful second opinion.
    """
    y = np.asarray(y_true)
    wrong_a = np.asarray(pred_a) != y
    wrong_b = np.asarray(pred_b) != y
    both = int((wrong_a & wrong_b).sum())
    if wrong_a.std() == 0 or wrong_b.std() == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(wrong_a, wrong_b)[0, 1])
    return {
        "errors_a": int(wrong_a.sum()),
        "errors_b": int(wrong_b.sum()),
        "both_wrong": both,
        "error_correlation": corr,
    }
