"""Day 6: baseline churn models (Logistic Regression and Decision Tree).

Run from the project root:  python -m src.models.baselines

Models are evaluated on the validation split only. The test split is left
untouched until the final comparison on Day 10.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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


def build_logistic(seed: int) -> Pipeline:
    """Scaler + Logistic Regression. The scaler is fitted on the training data only."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
            ),
        ]
    )


def build_tree(seed: int) -> DecisionTreeClassifier:
    """Shallow decision tree, kept simple on purpose to limit overfitting."""
    return DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=seed
    )


def run() -> pd.DataFrame:
    """Train both baselines, log validation metrics and save the fitted models."""
    cfg = load_modeling_config()
    seed = int(cfg["seed"])
    max_rec = int(cfg["recent_recency_days"])
    target = cfg["churn_target"]

    df = load_dataset(cfg)
    train, val, _test = split_data(df, cfg)
    logger.info(
        "rows: train=%d val=%d test=%d | churn rate train=%.3f",
        len(train),
        len(val),
        len(_test),
        train[target].mean(),
    )

    specs: list[tuple[str, Any, bool, str]] = [
        ("logistic_regression", build_logistic(seed), True, "class_weight=balanced"),
        ("decision_tree", build_tree(seed), False, "max_depth=5,min_samples_leaf=20"),
    ]

    rows = []
    for name, model, linear, params in specs:
        x_train = make_features(train, linear=linear)
        x_val = make_features(val, linear=linear)
        start = time.perf_counter()
        model.fit(x_train, train[target])
        seconds = time.perf_counter() - start

        views = evaluate_two_views(model, x_val, val[target], val["recency_days"], max_rec)
        for subset, metrics in views.items():
            log_run(cfg, name, "validation", subset, metrics, seconds, params)
            rows.append({"model": name, "subset": subset, **metrics})

        out_dir = ROOT / "models_artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": model, "columns": list(x_train.columns)},
            out_dir / f"{name}.joblib",
        )

    result = pd.DataFrame(rows)
    cols = ["model", "subset", "accuracy", "precision", "recall", "f1", "roc_auc"]
    print(result[cols].round(3).to_string(index=False))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
