"""Day 7: tuned Random Forest and XGBoost churn models.

Run from the project root:  python -m src.models.tree_models

Hyperparameters are searched with 5-fold stratified cross-validation on the
training split only. Models are then scored on the validation split. The test
split is not used here (it is touched once, on Day 10).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.models.common import (
    ROOT,
    evaluate_two_views,
    load_dataset,
    load_modeling_config,
    log_run,
    make_features,
    split_data,
)

logger = logging.getLogger(__name__)

RF_GRID: dict[str, list[Any]] = {
    "n_estimators": [200, 300, 500],
    "max_depth": [4, 6, 8, 12, None],
    "min_samples_leaf": [1, 5, 10, 20],
    "max_features": ["sqrt", 0.5],
}

XGB_GRID: dict[str, list[Any]] = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
}


def cv_search(
    estimator: Any,
    grid: dict[str, list[Any]],
    x: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_iter: int,
) -> RandomizedSearchCV:
    """Randomized search scored by ROC-AUC with 5-fold stratified CV."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator,
        grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x, y)
    return search


def tune_random_forest(x: pd.DataFrame, y: pd.Series, seed: int, n_iter: int) -> RandomizedSearchCV:
    """Tune a class-weighted Random Forest."""
    base = RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1)
    return cv_search(base, RF_GRID, x, y, seed, n_iter)


def tune_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
    n_iter: int,
) -> tuple[Any, RandomizedSearchCV]:
    """Tune XGBoost by CV, then refit the best setting with early stopping.

    The CV search uses a fixed number of trees. The final model is refitted on
    the training split with up to 1000 trees and stops early when the
    validation AUC has not improved for 30 rounds.
    """
    from xgboost import XGBClassifier

    base = XGBClassifier(
        n_estimators=300,
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
    )
    search = cv_search(base, XGB_GRID, x_train, y_train, seed, n_iter)
    final = XGBClassifier(
        **search.best_params_,
        n_estimators=1000,
        eval_metric="auc",
        early_stopping_rounds=30,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    final.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return final, search


def top_features(model: Any, columns: list[str], n: int = 8) -> pd.Series:
    """Most important features of a fitted tree model."""
    imp = pd.Series(model.feature_importances_, index=columns)
    return imp.sort_values(ascending=False).head(n)


def run(n_iter: int = 20) -> pd.DataFrame:
    """Tune, evaluate, log and save Random Forest and XGBoost."""
    cfg = load_modeling_config()
    seed = int(cfg["seed"])
    max_rec = int(cfg["recent_recency_days"])
    target = cfg["churn_target"]

    df = load_dataset(cfg)
    train, val, _test = split_data(df, cfg)
    x_train, x_val = make_features(train), make_features(val)
    y_train, y_val = train[target], val[target]

    rows = []
    jobs = ["random_forest", "xgboost"]
    for name in jobs:
        logger.info("tuning %s (n_iter=%d) ...", name, n_iter)
        start = time.perf_counter()
        if name == "random_forest":
            search = tune_random_forest(x_train, y_train, seed, n_iter)
            model = search.best_estimator_
        else:
            model, search = tune_xgboost(x_train, y_train, x_val, y_val, seed, n_iter)
        seconds = time.perf_counter() - start

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
        print(f"\n{name}: CV AUC={search.best_score_:.3f} | best params: {params}")
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
