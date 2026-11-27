"""Tests for the autoencoder (skipped when PyTorch is not installed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.models.autoencoder import AnomalyDetector, train_autoencoder
from src.models.nn_utils import build_preprocessor


def _toy(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    noise = rng.normal(scale=0.05, size=(n, 4))
    return pd.DataFrame(
        {
            "f1": a + noise[:, 0],
            "f2": a + noise[:, 1],  # f2 follows f1
            "f3": b + noise[:, 2],
            "f4": -b + noise[:, 3],  # f4 mirrors f3
        }
    )


def _fit():
    df = _toy()
    pre = build_preprocessor()
    x_train = pre.fit_transform(df[:450]).astype(np.float32)
    x_val = pre.transform(df[450:]).astype(np.float32)
    net, hist = train_autoencoder(
        x_train, x_val, seed=0, hidden=8, bottleneck=2, max_epochs=150, patience=15
    )
    return df, pre, net, hist


def test_training_reduces_the_reconstruction_loss() -> None:
    _df, _pre, _net, hist = _fit()
    assert hist["train_loss"].iloc[-1] < hist["train_loss"].iloc[0]
    assert hist["val_loss"].min() < 0.5


def test_customers_that_break_the_usual_pattern_score_higher() -> None:
    df, pre, net, _hist = _fit()
    detector = AnomalyDetector(pre, net, list(df.columns))
    normal = detector.score(df[450:])
    odd = df[450:].copy()
    odd["f1"], odd["f2"] = 8.0, -8.0  # f2 no longer follows f1
    odd_scores = detector.score(odd)
    assert np.median(odd_scores) > 10 * np.median(normal)
    assert detector.feature_errors(df[450:]).shape == (150, 4)


def test_detector_flags_rows_above_its_threshold() -> None:
    df, pre, net, _hist = _fit()
    detector = AnomalyDetector(pre, net, list(df.columns))
    detector.threshold = float(np.percentile(detector.score(df[:450]), 99))
    odd = df[450:].copy()
    odd["f3"], odd["f4"] = 8.0, 8.0
    assert detector.flag(odd).mean() > 0.9
    assert detector.flag(df[450:]).mean() < 0.1
