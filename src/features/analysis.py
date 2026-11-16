"""Small analysis helpers used by the EDA and feature notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def vif_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Variance inflation factor per column (VIF above about 10 signals multicollinearity)."""
    data = df[columns].astype(float)
    rows = []
    for col in columns:
        others = data.drop(columns=[col])
        r2 = LinearRegression().fit(others, data[col]).score(others, data[col])
        rows.append({"feature": col, "vif": np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def target_correlations(df: pd.DataFrame, columns: list[str], targets: list[str]) -> pd.DataFrame:
    """Pearson correlation of every feature with each target."""
    return df[columns + targets].corr().loc[columns, targets]
