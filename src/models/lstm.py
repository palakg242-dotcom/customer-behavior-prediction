"""Day 12: LSTM churn model on each customer's purchase sequence (PyTorch).

Run from the project root:  python -m src.models.lstm

Input: the customer's last 20 orders before the cutoff date (amount, gap in days,
number of items, days to the cutoff and the category mix of each order). The
model sees only this sequence, no tabular features, so it shows how much the
purchase order and timing alone can say about churn. It is evaluated on the
validation split. The test split is not used here.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from src.features.sequences import (
    MAX_LEN,
    TRANSACTIONS_PATH,
    Sequences,
    build_orders,
    build_sequences,
    load_feature_settings,
)
from src.models.common import (
    ROOT,
    evaluate_two_views,
    load_dataset,
    load_modeling_config,
    log_run,
    make_features,
    split_data,
)
from src.models.nn_utils import error_overlap

logger = logging.getLogger(__name__)

TX_COLUMNS = [
    "invoice",
    "description",
    "quantity",
    "invoice_date",
    "customer_id",
    "has_customer",
    "is_cancellation",
    "is_return",
    "is_non_product",
    "line_total",
]


class SeqNet(nn.Module):
    """One LSTM layer; the last hidden state feeds a small classifier head."""

    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return the churn logit for each sequence (padding is ignored)."""
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.head(self.dropout(hidden[-1])).squeeze(1)


def fit_scaler(x: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of every feature over the real (unpadded) steps of x."""
    real = np.arange(x.shape[1])[None, :] < lengths[:, None]
    steps = x[real]
    mean = steps.mean(axis=0)
    std = steps.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def scale(x: np.ndarray, lengths: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize real steps and keep the padding at exactly zero."""
    out = (x - mean) / std
    out[~(np.arange(x.shape[1])[None, :] < lengths[:, None])] = 0.0
    return out.astype(np.float32)


class LSTMClassifier:
    """The network plus the training-set scaler, with a scikit-learn-like API."""

    def __init__(self, net: SeqNet, mean: np.ndarray, std: np.ndarray) -> None:
        self.net, self.mean, self.std = net, mean, std

    def predict_proba(self, seq: Sequences) -> np.ndarray:
        """Return [P(stay), P(churn)] for every customer in ``seq``."""
        x = scale(seq.x, seq.lengths, self.mean, self.std)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(x), torch.from_numpy(seq.lengths))
            p = torch.sigmoid(logits).numpy()
        return np.column_stack([1.0 - p, p])

    def predict(self, seq: Sequences) -> np.ndarray:
        """Predicted class at a 0.5 threshold."""
        return (self.predict_proba(seq)[:, 1] >= 0.5).astype(int)


def train_lstm(
    x_train: np.ndarray,
    len_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    len_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    hidden: int = 32,
    dropout: float = 0.3,
    max_epochs: int = 100,
    patience: int = 12,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[SeqNet, pd.DataFrame]:
    """Train with early stopping on validation loss; restore the best weights.

    ``x_train`` and ``x_val`` must already be scaled.
    """
    torch.manual_seed(seed)
    net = SeqNet(x_train.shape[2], hidden=hidden, dropout=dropout)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    lt = torch.from_numpy(np.asarray(len_train, dtype=np.int64))
    yt = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    xv = torch.from_numpy(np.asarray(x_val, dtype=np.float32))
    lv = torch.from_numpy(np.asarray(len_val, dtype=np.int64))
    yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(xt, lt, yt),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    history: list[dict[str, float]] = []
    best_loss, best_state, waited = float("inf"), None, 0
    for epoch in range(1, max_epochs + 1):
        net.train()
        total, seen = 0.0, 0
        for xb, lb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(net(xb, lb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(xv, lv), yv).item()
        history.append({"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss})
        if val_loss < best_loss - 1e-4:
            best_loss, waited = val_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, pd.DataFrame(history)


def save_loss_curve(history: pd.DataFrame, best_epoch: int) -> None:
    """Training vs validation loss, with the early-stopping epoch marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train (dropout on)")
    ax.plot(history["epoch"], history["val_loss"], label="validation")
    ax.axvline(best_epoch, color="gray", linestyle="--", label="best epoch")
    ax.set(title="LSTM loss curve", xlabel="epoch", ylabel="binary cross-entropy")
    ax.legend()
    fig.tight_layout()
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "lstm_loss_curve.png", dpi=150)
    plt.close(fig)


def load_orders(settings: dict[str, Any]) -> pd.DataFrame:
    """Read the cleaned transactions and build the pre-cutoff order table."""
    tx = pd.read_csv(ROOT / TRANSACTIONS_PATH, usecols=TX_COLUMNS, dtype={"invoice": str})
    return build_orders(tx, settings["cutoff"], settings["categories"])


def run() -> pd.DataFrame:
    """Build sequences, train the LSTM, report validation metrics, save it."""
    cfg = load_modeling_config()
    seed = int(cfg["seed"])
    max_rec = int(cfg["recent_recency_days"])
    target = cfg["churn_target"]
    settings = load_feature_settings()

    df = load_dataset(cfg)
    train, val, _test = split_data(df, cfg)
    orders = load_orders(settings)

    per_customer = orders.groupby("customer_id").size()
    freq = df.set_index("customer_id")["frequency"].astype(int)
    same = (per_customer.reindex(freq.index).fillna(0).astype(int) == freq).mean()
    print(
        f"orders before cutoff: {len(orders)} | customers with orders: "
        f"{len(per_customer)} of {len(df)}"
    )
    print(f"orders match the feature table's frequency for {same:.1%} of customers")

    seq_train = build_sequences(orders, train["customer_id"].astype("int64").to_numpy(), MAX_LEN)
    seq_val = build_sequences(orders, val["customer_id"].astype("int64").to_numpy(), MAX_LEN)
    print(
        f"sequence length: median {int(np.median(seq_train.lengths))}, "
        f"{(seq_train.lengths >= MAX_LEN).mean():.1%} of customers are cut to {MAX_LEN}"
    )

    mean, std = fit_scaler(seq_train.x, seq_train.lengths)
    x_train = scale(seq_train.x, seq_train.lengths, mean, std)
    x_val = scale(seq_val.x, seq_val.lengths, mean, std)

    start = time.perf_counter()
    net, history = train_lstm(
        x_train,
        seq_train.lengths,
        train[target].to_numpy(),
        x_val,
        seq_val.lengths,
        val[target].to_numpy(),
        seed,
    )
    seconds = time.perf_counter() - start

    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    n_params = sum(p.numel() for p in net.parameters())
    params = json.dumps(
        {
            "hidden": 32,
            "max_len": MAX_LEN,
            "step_features": len(seq_train.feature_names),
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "parameters": n_params,
        }
    )
    print(
        f"LSTM: {n_params} parameters | {len(history)} epochs run | "
        f"best epoch {best_epoch} | best val loss {history['val_loss'].min():.4f}"
    )

    model = LSTMClassifier(net, mean, std)
    views = evaluate_two_views(model, seq_val, val[target], val["recency_days"], max_rec)
    rows = []
    for subset, metrics in views.items():
        log_run(cfg, "lstm", "validation", subset, metrics, seconds, params)
        rows.append({"model": "lstm", "subset": subset, **metrics})

    out_dir = ROOT / "models_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "max_len": MAX_LEN,
            "feature_names": seq_train.feature_names,
        },
        out_dir / "lstm.joblib",
    )
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    history.to_csv(docs / "lstm_history.csv", index=False)
    save_loss_curve(history, best_epoch)

    result = pd.DataFrame(rows)
    cols = ["model", "subset", "accuracy", "precision", "recall", "f1", "roc_auc"]
    print("\nLSTM validation results")
    print(result[cols].round(3).to_string(index=False))

    try:
        xgb = joblib.load(out_dir / "xgboost.joblib")
        xgb_pred = xgb["model"].predict(make_features(val)[xgb["columns"]])
        overlap = error_overlap(val[target].to_numpy(), model.predict(seq_val), xgb_pred)
        print("\nLSTM vs XGBoost mistakes on the validation split")
        for key, value in overlap.items():
            print(f"  {key}: {round(value, 3)}")
    except FileNotFoundError:
        print("\n(xgboost.joblib not found; skipped the error-overlap check)")
    return result


if __name__ == "__main__":
    # Import under the real module name so the saved objects can be unpickled.
    from src.models.lstm import run as run_lstm

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_lstm()
