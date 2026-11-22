"""Day 8 (part 2): customer lifetime value (CLV) regression.

Run from the project root:  python -m src.models.clv_regression

About 3,000 customers have a CLV target of 0, so the models are trained on
log1p(target) and their predictions are converted back with expm1. Metrics are
reported on both scales: the log scale (how well the model ranks and sizes
customers) and the original currency scale (what the business sees, where a few
very large customers dominate RMSE and R-squared).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.common import (
    ROOT,
    load_dataset,
    load_modeling_config,
    make_features,
    split_data,
)

logger = logging.getLogger(__name__)

CLV_LOG = "models_artifacts/clv_experiment_log.csv"

RF_REG_GRID: dict[str, list[Any]] = {
    "n_estimators": [200, 300],
    "max_depth": [4, 6, 8, 12, None],
    "min_samples_leaf": [1, 5, 10, 20],
    "max_features": ["sqrt", 0.5, 1.0],
}


def to_log(y: pd.Series) -> np.ndarray:
    """log1p transform of the (non-negative) CLV target."""
    return np.log1p(np.clip(np.asarray(y, dtype=float), 0, None))


def from_log(pred: np.ndarray) -> np.ndarray:
    """Convert log-scale predictions back to currency, never below zero."""
    return np.clip(np.expm1(np.asarray(pred, dtype=float)), 0, None)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and R-squared."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate(
    model: Any, x: pd.DataFrame, y: pd.Series, recency: pd.Series, max_rec: int
) -> dict[tuple[str, str], dict[str, float]]:
    """Metrics on both scales, for all customers and for recent ones only."""
    pred_log = model.predict(x)
    pred = from_log(pred_log)
    y_arr = np.asarray(y, dtype=float)
    y_log = to_log(y)
    masks = {
        "all": np.ones(len(y_arr), dtype=bool),
        f"recency<={max_rec}": (recency <= max_rec).to_numpy(),
    }
    out = {}
    for subset, m in masks.items():
        out[(subset, "log")] = regression_metrics(y_log[m], pred_log[m])
        out[(subset, "original")] = regression_metrics(y_arr[m], pred[m])
    return out


def tune_rf_regressor(
    x: pd.DataFrame, y_log: np.ndarray, seed: int, n_iter: int
) -> RandomizedSearchCV:
    """Random Forest regressor tuned by 5-fold CV on the log-scale R-squared."""
    base = RandomForestRegressor(random_state=seed, n_jobs=1)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base,
        RF_REG_GRID,
        n_iter=n_iter,
        scoring="r2",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(x, y_log)
    return search


def log_row(row: dict[str, Any]) -> None:
    """Append one row to the CLV experiment log (created on first use)."""
    path = ROOT / CLV_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run(n_iter: int = 20) -> pd.DataFrame:
    """Train Ridge and Random Forest CLV models and report validation metrics."""
    cfg = load_modeling_config()
    seed = int(cfg["seed"])
    max_rec = int(cfg["recent_recency_days"])
    target = cfg["clv_target"]

    df = load_dataset(cfg)
    train, val, _test = split_data(df, cfg)
    x_train, x_val = make_features(train), make_features(val)
    y_train_log = to_log(train[target])
    logger.info("share of customers with CLV target 0: %.3f", (df[target] == 0).mean())

    models: dict[str, tuple[Any, str, float]] = {}

    start = time.perf_counter()
    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
        ]
    )
    ridge.fit(x_train, y_train_log)
    models["ridge"] = (ridge, "alphas=0.1..100", time.perf_counter() - start)

    start = time.perf_counter()
    search = tune_rf_regressor(x_train, y_train_log, seed, n_iter)
    params = json.dumps(
        {"cv_r2_log": round(float(search.best_score_), 4), **search.best_params_},
        default=str,
    )
    models["random_forest_reg"] = (
        search.best_estimator_,
        params,
        time.perf_counter() - start,
    )

    rows = []
    out_dir = ROOT / "models_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (model, params, seconds) in models.items():
        results = evaluate(model, x_val, val[target], val["recency_days"], max_rec)
        for (subset, scale), metrics in results.items():
            rows.append({"model": name, "subset": subset, "scale": scale, **metrics})
            log_row(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "model": name,
                    "split": "validation",
                    "subset": subset,
                    "scale": scale,
                    **{k: round(v, 4) for k, v in metrics.items()},
                    "train_seconds": round(seconds, 2),
                    "params": params,
                }
            )
        joblib.dump(
            {"model": model, "columns": list(x_train.columns), "target": "log1p"},
            out_dir / f"clv_{name}.joblib",
        )
        print(f"\n{name}: {params}")

    result = pd.DataFrame(rows)
    print("\nValidation results (CLV)")
    print(result.round(3).to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=20, help="search iterations")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(n_iter=args.n_iter)
