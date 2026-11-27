"""Tests for the LSTM model (skipped when PyTorch is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.sequences import Sequences

torch = pytest.importorskip("torch")

from src.models.lstm import (
    LSTMClassifier,
    SeqNet,
    fit_scaler,
    scale,
    train_lstm,
)


def _toy(n: int = 300, steps: int = 6, features: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, steps + 1, n).astype(np.int64)
    x = rng.normal(size=(n, steps, features)).astype(np.float32)
    real = np.arange(steps)[None, :] < lengths[:, None]
    x[~real] = 0.0
    # label depends on the mean of feature 0 over the real steps
    y = ((x[:, :, 0] * real).sum(axis=1) / lengths > 0).astype(int)
    return x, lengths, y


def test_scaler_uses_real_steps_only_and_keeps_padding_zero() -> None:
    x, lengths, _ = _toy()
    mean, std = fit_scaler(x, lengths)
    scaled = scale(x, lengths, mean, std)
    real = np.arange(x.shape[1])[None, :] < lengths[:, None]
    assert np.allclose(scaled[real].mean(axis=0), 0.0, atol=1e-4)
    assert np.all(scaled[~real] == 0.0)


def test_lstm_ignores_padding_values() -> None:
    x, lengths, _ = _toy(n=20)
    net = SeqNet(x.shape[2])
    net.eval()
    noisy = x.copy()
    real = np.arange(x.shape[1])[None, :] < lengths[:, None]
    noisy[~real] = 99.0
    with torch.no_grad():
        a = net(torch.from_numpy(x), torch.from_numpy(lengths))
        b = net(torch.from_numpy(noisy), torch.from_numpy(lengths))
    assert torch.allclose(a, b, atol=1e-5)


def test_lstm_trains_and_predicts_probabilities() -> None:
    x, lengths, y = _toy()
    mean, std = fit_scaler(x[:200], lengths[:200])
    xs = scale(x, lengths, mean, std)
    net, hist = train_lstm(
        xs[:200],
        lengths[:200],
        y[:200],
        xs[200:],
        lengths[200:],
        y[200:],
        seed=0,
        max_epochs=40,
        patience=6,
    )
    assert {"epoch", "train_loss", "val_loss"} <= set(hist.columns)
    assert hist["val_loss"].min() < 0.6931  # better than a coin flip
    seq = Sequences(x, lengths, np.arange(len(x)), ["a", "b", "c"])
    model = LSTMClassifier(net, mean, std)
    proba = model.predict_proba(seq)
    assert proba.shape == (len(x), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert set(np.unique(model.predict(seq))) <= {0, 1}
