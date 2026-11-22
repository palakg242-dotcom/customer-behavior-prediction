"""Shared helpers for the modeling stage: config, split, features, metrics, logging."""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "val_size": 0.15,
    "test_size": 0.15,
    "churn_target": "churn_90d",
    "clv_target": "clv_target",
    "features_path": "data/processed/customer_features.csv",
    "experiment_log": "models_artifacts/experiment_log.csv",
    "recent_recency_days": 180,
}

# Columns that are never model inputs.
NON_FEATURES = ["customer_id", "country", "churn_90d", "clv_target", "next_category"]

# Dropped for linear models only (VIF / redundancy found in Week 1 EDA).
LINEAR_DROP = ["clv_historical", "engagement_score"]


def load_modeling_config() -> dict[str, Any]:
    """Read the ``modeling`` section of config/config.yaml, filling in defaults."""
    cfg = dict(DEFAULTS)
    path = ROOT / "config" / "config.yaml"
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg.update(raw.get("modeling", {}) or {})
    return cfg


def load_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load the customer-level feature table."""
    return pd.read_csv(ROOT / cfg["features_path"])


def split_data(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation/test split (default 70/15/15) on the churn label."""
    target = cfg["churn_target"]
    seed = int(cfg["seed"])
    val_size, test_size = float(cfg["val_size"]), float(cfg["test_size"])
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val[target], random_state=seed
    )
    parts = (train, val, test)
    train, val, test = (part.reset_index(drop=True) for part in parts)
    return train, val, test


def make_features(df: pd.DataFrame, linear: bool = False) -> pd.DataFrame:
    """Build the model input matrix.

    Numeric columns are used as they are. ``country`` becomes a simple UK / non-UK
    flag because most customers are in the UK. For linear models, the redundant
    ``clv_historical`` and ``engagement_score`` are dropped.
    """
    cols = [c for c in df.columns if c not in NON_FEATURES]
    if linear:
        cols = [c for c in cols if c not in LINEAR_DROP]
    x = df[cols].copy()
    x["is_uk"] = (df["country"] == "United Kingdom").astype(int)
    return x


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Accuracy, precision, recall, F1, ROC-AUC and the confusion matrix counts."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if both_classes else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_two_views(
    model: Any, x: pd.DataFrame, y: pd.Series, recency: pd.Series, max_recency: int
) -> dict[str, dict[str, float]]:
    """Metrics on all customers and on recently active customers only."""
    pred = model.predict(x)
    proba = model.predict_proba(x)[:, 1]
    out = {"all": compute_metrics(y, pred, proba)}
    mask = (recency <= max_recency).to_numpy()
    out[f"recency<={max_recency}"] = compute_metrics(y.to_numpy()[mask], pred[mask], proba[mask])
    return out


def log_run(
    cfg: dict[str, Any],
    model_name: str,
    split_name: str,
    subset: str,
    metrics: dict[str, float],
    train_seconds: float,
    params: str,
) -> None:
    """Append one row to the experiment log CSV (created on first use)."""
    path = ROOT / cfg["experiment_log"]
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_name,
        "split": split_name,
        "subset": subset,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        "train_seconds": round(train_seconds, 2),
        "params": params,
    }
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    logger.info("logged %s | %s | %s", model_name, split_name, subset)
