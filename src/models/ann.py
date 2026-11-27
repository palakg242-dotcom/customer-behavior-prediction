"""Day 11: feed-forward neural network (ANN) for churn, in PyTorch.

Run from the project root:  python -m src.models.ann

The network uses the same engineered features as the classical models.
It has batch normalization, dropout, L2 weight decay and early stopping on the
validation loss, as the PRD requires. It is compared with the tree models on the
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
from torch.utils.data import DataLoader, TensorDataset

from src.models.common import (
    ROOT,
    evaluate_two_views,
    load_dataset,
    load_modeling_config,
    log_run,
    make_features,
    split_data,
)
from src.models.nn_utils import build_preprocessor, error_overlap

logger = logging.getLogger(__name__)


class ChurnNet(nn.Module):
    """Two hidden layers, each with batch norm, ReLU and dropout."""

    def __init__(self, n_inputs: int, hidden: tuple[int, int] = (64, 32), dropout: float = 0.3):
        super().__init__()
        h1, h2 = hidden
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the churn logit for each row."""
        return self.layers(x).squeeze(1)


class ANNClassifier:
    """Wraps the preprocessor and the network behind a scikit-learn-like API."""

    def __init__(self, preprocessor: Any, net: ChurnNet) -> None:
        self.preprocessor = preprocessor
        self.net = net

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return [P(stay), P(churn)] for each row of a feature DataFrame."""
        data = self.preprocessor.transform(x).astype(np.float32)
        self.net.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net(torch.from_numpy(data))).numpy()
        return np.column_stack([1.0 - p, p])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predicted class at a 0.5 threshold."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def train_ann(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    max_epochs: int = 200,
    patience: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
) -> tuple[ChurnNet, pd.DataFrame]:
    """Train with early stopping on validation loss; restore the best weights."""
    torch.manual_seed(seed)
    net = ChurnNet(x_train.shape[1], dropout=dropout)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    xv = torch.from_numpy(np.asarray(x_val, dtype=np.float32))
    yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(xt, yt),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # batch norm needs more than one row per batch
        generator=torch.Generator().manual_seed(seed),
    )

    history: list[dict[str, float]] = []
    best_loss, best_state, waited = float("inf"), None, 0
    for epoch in range(1, max_epochs + 1):
        net.train()
        total, seen = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(xv), yv).item()
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
    ax.set(title="ANN loss curve", xlabel="epoch", ylabel="binary cross-entropy")
    ax.legend()
    fig.tight_layout()
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "ann_loss_curve.png", dpi=150)
    plt.close(fig)


def run() -> pd.DataFrame:
    """Train the ANN, report validation metrics and save the model."""
    cfg = load_modeling_config()
    seed = int(cfg["seed"])
    max_rec = int(cfg["recent_recency_days"])
    target = cfg["churn_target"]

    df = load_dataset(cfg)
    train, val, _test = split_data(df, cfg)
    x_train, x_val = make_features(train), make_features(val)
    y_train, y_val = train[target].to_numpy(), val[target].to_numpy()

    pre = build_preprocessor()
    a_train = pre.fit_transform(x_train).astype(np.float32)
    a_val = pre.transform(x_val).astype(np.float32)

    start = time.perf_counter()
    net, history = train_ann(a_train, y_train, a_val, y_val, seed)
    seconds = time.perf_counter() - start

    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    n_params = sum(p.numel() for p in net.parameters())
    params = json.dumps(
        {
            "layers": "64-32",
            "dropout": 0.3,
            "weight_decay": 1e-4,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "parameters": n_params,
        }
    )
    print(
        f"ANN: {n_params} parameters | {len(history)} epochs run | "
        f"best epoch {best_epoch} | best val loss {history['val_loss'].min():.4f}"
    )

    model = ANNClassifier(pre, net)
    views = evaluate_two_views(model, x_val, val[target], val["recency_days"], max_rec)
    rows = []
    for subset, metrics in views.items():
        log_run(cfg, "ann", "validation", subset, metrics, seconds, params)
        rows.append({"model": "ann", "subset": subset, **metrics})

    out_dir = ROOT / "models_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": list(x_train.columns)}, out_dir / "ann.joblib")
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    history.to_csv(docs / "ann_history.csv", index=False)
    save_loss_curve(history, best_epoch)

    result = pd.DataFrame(rows)
    cols = ["model", "subset", "accuracy", "precision", "recall", "f1", "roc_auc"]
    print("\nANN validation results")
    print(result[cols].round(3).to_string(index=False))

    # Do the ANN and XGBoost make the same mistakes?
    try:
        xgb = joblib.load(out_dir / "xgboost.joblib")
        xgb_pred = xgb["model"].predict(make_features(val)[xgb["columns"]])
        overlap = error_overlap(y_val, model.predict(x_val), xgb_pred)
        print("\nANN vs XGBoost mistakes on the validation split")
        for key, value in overlap.items():
            print(f"  {key}: {round(value, 3)}")
    except FileNotFoundError:
        print("\n(xgboost.joblib not found; skipped the error-overlap check)")
    return result


if __name__ == "__main__":
    # Import the module under its real name so that the saved objects can be
    # loaded later (classes defined in __main__ cannot be unpickled elsewhere).
    from src.models.ann import run as run_ann

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_ann()
