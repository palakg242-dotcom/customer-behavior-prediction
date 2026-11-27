"""Tests for the ANN helpers and (when torch is installed) the training loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.nn_utils import (
    SKEWED,
    build_preprocessor,
    error_overlap,
    log_skewed,
)


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "recency_days": rng.integers(0, 400, n),
            "frequency": rng.integers(1, 40, n),
            "monetary_total": rng.gamma(2.0, 500.0, n),
            "return_rate": rng.random(n),
        }
    )


def test_log_skewed_clips_negative_values_and_ignores_missing_columns() -> None:
    df = pd.DataFrame({"monetary_total": [-50.0, 0.0, 99.0], "other": [1, 2, 3]})
    out = log_skewed(df)
    assert out["monetary_total"].iloc[0] == 0.0
    assert out["monetary_total"].iloc[2] == pytest.approx(np.log(100.0))
    assert out["other"].tolist() == [1, 2, 3]
    assert "monetary_total" in SKEWED


def test_preprocessor_uses_training_statistics_only() -> None:
    train, val = _frame(seed=0), _frame(seed=1).assign(frequency=1000)
    pre = build_preprocessor()
    a_train = pre.fit_transform(train)
    scaler_mean_before = pre.named_steps["scale"].mean_.copy()
    pre.transform(val)
    assert np.allclose(pre.named_steps["scale"].mean_, scaler_mean_before)
    assert np.allclose(a_train.mean(axis=0), 0.0, atol=1e-6)


def test_preprocessor_fills_missing_values() -> None:
    train = _frame()
    train.loc[0, "return_rate"] = np.nan
    out = build_preprocessor().fit_transform(train)
    assert not np.isnan(out).any()


def test_error_overlap_counts_shared_mistakes() -> None:
    y = np.array([1, 1, 0, 0, 1, 0])
    a = np.array([1, 0, 0, 1, 1, 0])  # wrong on rows 1 and 3
    b = np.array([1, 0, 0, 0, 0, 0])  # wrong on rows 1 and 4
    res = error_overlap(y, a, b)
    assert (res["errors_a"], res["errors_b"], res["both_wrong"]) == (2, 2, 1)


def test_error_overlap_is_nan_when_a_model_never_errs() -> None:
    y = np.array([1, 0, 1, 0])
    res = error_overlap(y, y, np.array([0, 0, 1, 0]))
    assert np.isnan(res["error_correlation"])


def test_ann_trains_and_predicts_probabilities() -> None:
    pytest.importorskip("torch")
    from src.models.ann import ANNClassifier, train_ann

    rng = np.random.default_rng(0)
    x = _frame(600)
    y = (np.log1p(x["recency_days"]) + rng.normal(0, 0.3, 600) > 4.5).astype(int)
    pre = build_preprocessor()
    a_all = pre.fit_transform(x).astype(np.float32)
    net, hist = train_ann(
        a_all[:400],
        y.to_numpy()[:400],
        a_all[400:],
        y.to_numpy()[400:],
        seed=0,
        max_epochs=40,
        patience=5,
    )
    assert {"epoch", "train_loss", "val_loss"} <= set(hist.columns)
    assert hist["val_loss"].min() < 0.6931  # better than a coin flip
    model = ANNClassifier(pre, net)
    proba = model.predict_proba(x)
    assert proba.shape == (600, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert set(np.unique(model.predict(x))) <= {0, 1}


def test_early_stopping_ends_before_max_epochs_on_pure_noise() -> None:
    pytest.importorskip("torch")
    from src.models.ann import train_ann

    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 4)).astype(np.float32)
    y = rng.integers(0, 2, 300)
    _net, hist = train_ann(x[:200], y[:200], x[200:], y[200:], 0, 100, 4)
    assert len(hist) < 100
