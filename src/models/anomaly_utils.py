"""Settings and helpers for the anomaly detector. No torch import is needed here."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.models.common import ROOT

DEFAULTS: dict[str, Any] = {
    "features": [
        "frequency",
        "monetary_total",
        "monetary_avg",
        "avg_basket_qty",
        "avg_basket_lines",
        "return_rate",
        "discount_sensitivity",
        "gap_variance",
    ],
    "hidden": 16,
    "bottleneck": 3,
    "dropout": 0.0,
    "max_epochs": 200,
    "patience": 15,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "threshold_percentile": 99.0,
    "min_flag_rate": 0.005,
    "max_flag_rate": 0.02,
    "review_rows": 20,
}


def load_anomaly_config() -> dict[str, Any]:
    """The ``anomaly`` section of config/config.yaml, with defaults filled in."""
    cfg = dict(DEFAULTS)
    path = ROOT / "config" / "config.yaml"
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg.update(raw.get("anomaly", {}) or {})
    return cfg


def pick_threshold(errors: np.ndarray, percentile: float) -> float:
    """Reconstruction-error threshold: a percentile of the training errors."""
    return float(np.percentile(np.asarray(errors, dtype=float), percentile))


def flag_rate(errors: np.ndarray, threshold: float) -> float:
    """Share of rows whose error is above the threshold."""
    return float((np.asarray(errors) > threshold).mean())


def top_features(per_feature_errors: np.ndarray, names: list[str]) -> list[str]:
    """For each row, the feature that contributes most to the error."""
    return [names[i] for i in np.asarray(per_feature_errors).argmax(axis=1)]


def _scale_columns(columns: list[str]) -> Callable[[pd.DataFrame, float], pd.DataFrame]:
    def apply(df: pd.DataFrame, level: float) -> pd.DataFrame:
        out = df.copy()
        for col in columns:
            if col in out.columns:
                out[col] = out[col] * level
        return out

    return apply


def _set_return_rate(df: pd.DataFrame, level: float) -> pd.DataFrame:
    out = df.copy()
    if "return_rate" in out.columns:
        out["return_rate"] = level
    return out


Injection = tuple[list[float], Callable[[pd.DataFrame, float], pd.DataFrame]]

# Synthetic anomalies used only to check the threshold: kind -> (levels, function).
INJECTIONS: dict[str, Injection] = {
    "spend_spike": ([2, 5, 10], _scale_columns(["monetary_total", "monetary_avg"])),
    "bulk_basket": ([2, 5, 10], _scale_columns(["avg_basket_qty", "avg_basket_lines"])),
    "return_spike": ([0.3, 0.6, 0.95], _set_return_rate),
}


def inject(df: pd.DataFrame, kind: str, level: float) -> pd.DataFrame:
    """A copy of ``df`` with one kind of synthetic anomaly applied to every row."""
    return INJECTIONS[kind][1](df, level)


def detection_rate(errors: np.ndarray, threshold: float) -> float:
    """Share of injected rows that the detector flags."""
    return flag_rate(errors, threshold)
