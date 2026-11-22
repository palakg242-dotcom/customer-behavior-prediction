"""Day 8b: two-stage customer lifetime value (CLV) model.

Run from the project root:  python -m src.models.clv_two_stage

About 56% of customers have a CLV target of 0 (they do not buy again), which
makes a single regression hard. Here the problem is split in two:

    expected CLV = P(customer stays active) x expected spend if active

Stage 1 is the churn model trained on Day 7/8 (P(active) = 1 - P(churn)).
Stage 2 is a Random Forest regressor trained only on customers who stayed
active, on log1p(spend). Scores are reported on the validation split.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src.models.clv_regression import (
    from_log,
    log_row,
    regression_metrics,
    to_log,
    tune_rf_regressor,
)
from src.models.common import (
    ROOT,
    load_dataset,
    load_modeling_config,
    make_features,
    split_data,
)

logger = logging.getLogger(__name__)

STAGE1_CHOICES = ["xgboost", "random_forest", "lightgbm"]


def combine(p_churn: np.ndarray, spend_log: np.ndarray) -> np.ndarray:
    """Expected CLV = P(active) x expected spend if active (never below zero)."""
    p_active = 1.0 - np.clip(np.asarray(p_churn, dtype=float), 0.0, 1.0)
    return p_active * from_log(spend_log)


def evaluate_combined(
    clv_pred: np.ndarray, y: pd.Series, recency: pd.Series, max_rec: int
) -> dict[tuple[str, str], dict[str, float]]:
    """Metrics on both scales for all customers and for recent customers."""
    y_arr = np.asarray(y, dtype=float)
    masks = {
        "all": np.ones(len(y_arr), dtype=bool),
        f"recency<={max_rec}": (recency <= max_rec).to_numpy(),
    }
    out = {}
    for subset, m in masks.items():
        out[(subset, "log")] = regression_metrics(to_log(y_arr[m]), np.log1p(clv_pred[m]))
        out[(subset, "original")] = regression_metrics(y_arr[m], clv_pred[m])
    return out


def _log(name: str, subset: str, scale: str, m: dict, secs: float, params: str) -> None:
    log_row(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": name,
            "split": "validation",
            "subset": subset,
            "scale": scale,
            **{k: round(v, 4) for k, v in m.items()},
            "train_seconds": round(secs, 2),
            "params": params,
        }
    )


def run(n_iter: int = 20, stage1: str = "xgboost") -> pd.DataFrame:
    """Train stage 2, combine it with the saved churn model and report metrics."""
    cfg = load_modeling_config()
    seed = int(cfg["seed"])
    max_rec = int(cfg["recent_recency_days"])
    churn_col, clv_col = cfg["churn_target"], cfg["clv_target"]

    df = load_dataset(cfg)
    train, val, _test = split_data(df, cfg)
    x_train, x_val = make_features(train), make_features(val)

    saved = joblib.load(ROOT / "models_artifacts" / f"{stage1}.joblib")
    churn_model, columns = saved["model"], saved["columns"]
    p_churn_val = churn_model.predict_proba(x_val[columns])[:, 1]

    active = (train[churn_col] == 0).to_numpy()
    logger.info("stage 2 trains on %d active customers", int(active.sum()))
    start = time.perf_counter()
    search = tune_rf_regressor(x_train[active], to_log(train.loc[active, clv_col]), seed, n_iter)
    seconds = time.perf_counter() - start
    reg = search.best_estimator_
    params = json.dumps(
        {
            "stage1": stage1,
            "cv_r2_log_active": round(float(search.best_score_), 4),
            **search.best_params_,
        },
        default=str,
    )

    spend_log = reg.predict(x_val)
    clv_pred = combine(p_churn_val, spend_log)
    name = f"two_stage_{stage1}_rf"
    rows = []
    for (subset, scale), m in evaluate_combined(
        clv_pred, val[clv_col], val["recency_days"], max_rec
    ).items():
        rows.append({"model": name, "subset": subset, "scale": scale, **m})
        _log(name, subset, scale, m, seconds, params)

    # Stage 2 on its own, for the customers who really stayed active.
    act_val = (val[churn_col] == 0).to_numpy()
    y_act = np.asarray(val.loc[act_val, clv_col], dtype=float)
    for scale, yt, yp in [
        ("log", to_log(y_act), spend_log[act_val]),
        ("original", y_act, from_log(spend_log[act_val])),
    ]:
        m = regression_metrics(yt, yp)
        rows.append({"model": "stage2_only", "subset": "active", "scale": scale, **m})
        _log("stage2_only", "active", scale, m, seconds, params)

    out_dir = ROOT / "models_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"stage1": stage1, "regressor": reg, "columns": list(x_train.columns)},
        out_dir / "clv_two_stage.joblib",
    )
    print(f"\nstage 2 (active customers only): {params}")

    result = pd.DataFrame(rows)
    print("\nValidation results (two-stage CLV)")
    print(result.round(3).to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=20, help="search iterations")
    parser.add_argument("--stage1", choices=STAGE1_CHOICES, default="xgboost")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(n_iter=args.n_iter, stage1=args.stage1)
