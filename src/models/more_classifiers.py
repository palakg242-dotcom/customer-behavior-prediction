"""Day 8 (part 1): LightGBM and KNN churn models.

Run from the project root:  python -m src.models.more_classifiers

LightGBM is compared with XGBoost on accuracy and training time. KNN is the
distance-based comparison model, trained on scaled features, to check that the
tree-model gains are real and not an artifact of scaling. Hyperparameters are
tuned with 5-fold stratified CV on the training split; scores are reported on
the validation split. The test split is not used here.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.common import (
    ROOT,
    evaluate_two_views,
    load_dataset,
    load_modeling_config,
    log_run,
    make_features,
    split_data,
)
from src.models.tree_models import cv_search, top_features

logger = logging.getLogger(__name__)

LGBM_GRID: dict[str, list[Any]] = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [200, 400],
    "min_child_samples": [10, 20, 40],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
}

KNN_GRID: dict[str, list[Any]] = {
    "clf__n_neighbors": [5, 11, 21, 31, 51],
    "clf__weights": ["uniform", "distance"],
    "clf__p": [1, 2],
}


def grid_size(grid: dict[str, list[Any]]) -> int:
    """Number of combinations in a parameter grid."""
    return math.prod(len(values) for values in grid.values())


def tune_knn(x: pd.DataFrame, y: pd.Series, seed: int, n_iter: int) -> RandomizedSearchCV:
    """Scaler + KNN, tuned by CV. The scaler is fitted inside each CV fold."""
    base = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
    n_iter = min(n_iter, grid_size(KNN_GRID))
    return cv_search(base, KNN_GRID, x, y, seed, n_iter)


def tune_lightgbm(x: pd.DataFrame, y: pd.Series, seed: int, n_iter: int) -> RandomizedSearchCV:
    """LightGBM classifier tuned by CV."""
    from lightgbm import LGBMClassifier

    base = LGBMClassifier(subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1)
    n_iter = min(n_iter, grid_size(LGBM_GRID))
    return cv_search(base, LGBM_GRID, x, y, seed, n_iter)


def run(n_iter: int = 20) -> pd.DataFrame:
    """Tune, evaluate, log and save LightGBM and KNN."""
    cfg = load_modeling_config()
    seed = int(cfg["seed"])
    max_rec = int(cfg["recent_recency_days"])
    target = cfg["churn_target"]

    df = load_dataset(cfg)
    train, val, _test = split_data(df, cfg)
    y_train, y_val = train[target], val[target]

    rows = []
    for name in ["lightgbm", "knn"]:
        # KNN is distance-based, so it drops the two redundant columns.
        linear = name == "knn"
        x_train = make_features(train, linear=linear)
        x_val = make_features(val, linear=linear)

        logger.info("tuning %s (n_iter=%d) ...", name, n_iter)
        start = time.perf_counter()
        if name == "lightgbm":
            search = tune_lightgbm(x_train, y_train, seed, n_iter)
        else:
            search = tune_knn(x_train, y_train, seed, n_iter)
        seconds = time.perf_counter() - start
        model = search.best_estimator_

        params = json.dumps(
            {"cv_auc": round(float(search.best_score_), 4), **search.best_params_},
            default=str,
        )
        views = evaluate_two_views(model, x_val, y_val, val["recency_days"], max_rec)
        for subset, metrics in views.items():
            log_run(cfg, name, "validation", subset, metrics, seconds, params)
            rows.append({"model": name, "subset": subset, **metrics})

        out_dir = ROOT / "models_artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": model, "columns": list(x_train.columns)},
            out_dir / f"{name}.joblib",
        )
        print(f"\n{name}: CV AUC={search.best_score_:.3f} | {params}")
        print(f"search time: {seconds:.1f} s")
        if name == "lightgbm":
            print(top_features(model, list(x_train.columns)).round(3).to_string())

    result = pd.DataFrame(rows)
    cols = ["model", "subset", "accuracy", "precision", "recall", "f1", "roc_auc"]
    print("\nValidation results")
    print(result[cols].round(3).to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=20, help="search iterations")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(n_iter=args.n_iter)
