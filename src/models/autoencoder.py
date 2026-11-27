"""Day 13: autoencoder for anomaly detection on spending patterns (PyTorch).

Run from the project root:  python -m src.models.autoencoder

The autoencoder is trained without labels on scaled spending-pattern features of
the training customers. A customer whose reconstruction error is above a
threshold is flagged for manual review (possible fraud, data-entry errors or
extreme outliers). The threshold is a percentile of the training errors, and it
is validated in three ways, because there are no fraud labels:

1. the flag rate on held-out (validation) customers should stay close to the
   rate expected from the percentile;
2. the flagged customers are inspected and compared with the others;
3. synthetic anomalies of growing size are injected into the validation
   customers, to see how large a deviation the threshold catches.

Hyperparameters and thresholds come from the ``anomaly`` section of config.yaml.
The test split is not used here.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.anomaly_utils import (
    INJECTIONS,
    detection_rate,
    flag_rate,
    inject,
    load_anomaly_config,
    pick_threshold,
    top_features,
)
from src.models.common import (
    ROOT,
    load_dataset,
    load_modeling_config,
    split_data,
)
from src.models.nn_utils import build_preprocessor

logger = logging.getLogger(__name__)


class AutoEncoder(nn.Module):
    """Features -> small bottleneck -> features."""

    def __init__(
        self, n_features: int, hidden: int = 16, bottleneck: int = 3, dropout: float = 0.0
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input."""
        return self.decoder(self.encoder(x))


class AnomalyDetector:
    """Preprocessor + autoencoder + threshold, scoring customers from a DataFrame."""

    def __init__(
        self,
        preprocessor: Any,
        net: AutoEncoder,
        features: list[str],
        threshold: float = float("inf"),
    ) -> None:
        self.preprocessor = preprocessor
        self.net = net
        self.features = features
        self.threshold = threshold

    def feature_errors(self, df: pd.DataFrame) -> np.ndarray:
        """Squared reconstruction error of every feature (scaled space)."""
        x = torch.from_numpy(self.preprocessor.transform(df[self.features]).astype(np.float32))
        self.net.eval()
        with torch.no_grad():
            return ((self.net(x) - x) ** 2).numpy()

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Anomaly score: mean squared reconstruction error per customer."""
        return self.feature_errors(df).mean(axis=1)

    def flag(self, df: pd.DataFrame) -> np.ndarray:
        """True where the score is above the threshold."""
        return self.score(df) > self.threshold


def train_autoencoder(
    x_train: np.ndarray,
    x_val: np.ndarray,
    seed: int,
    hidden: int = 16,
    bottleneck: int = 3,
    dropout: float = 0.0,
    max_epochs: int = 200,
    patience: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[AutoEncoder, pd.DataFrame]:
    """Train with early stopping on validation reconstruction loss."""
    torch.manual_seed(seed)
    net = AutoEncoder(x_train.shape[1], hidden, bottleneck, dropout)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    xt = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    xv = torch.from_numpy(np.asarray(x_val, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(xt),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history: list[dict[str, float]] = []
    best_loss, best_state, waited = float("inf"), None, 0
    for epoch in range(1, max_epochs + 1):
        net.train()
        total, seen = 0.0, 0
        for (xb,) in loader:
            optimizer.zero_grad()
            loss = loss_fn(net(xb), xb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(xv), xv).item()
        history.append({"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, waited = val_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, pd.DataFrame(history)


def save_plots(history: pd.DataFrame, val_scores: np.ndarray, threshold: float) -> None:
    """Loss curve, and the distribution of validation scores with the threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss"], label="validation")
    ax.set(title="Autoencoder loss curve", xlabel="epoch", ylabel="reconstruction MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(docs / "autoencoder_loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(val_scores + 1e-9), bins=50, color="tab:blue", alpha=0.8)
    ax.axvline(np.log10(threshold), color="red", linestyle="--", label="threshold")
    ax.set(
        title="Validation reconstruction errors",
        xlabel="log10(reconstruction error)",
        ylabel="customers",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(docs / "anomaly_error_hist.png", dpi=150)
    plt.close(fig)


def run() -> dict[str, Any]:
    """Train, set and validate the threshold, score every customer, save outputs."""
    cfg = load_modeling_config()
    acfg = load_anomaly_config()
    seed = int(cfg["seed"])
    features = list(acfg["features"])

    df = load_dataset(cfg)
    train, val, _test = split_data(df, cfg)

    pre = build_preprocessor()
    a_train = pre.fit_transform(train[features]).astype(np.float32)
    a_val = pre.transform(val[features]).astype(np.float32)

    net, history = train_autoencoder(
        a_train,
        a_val,
        seed,
        hidden=int(acfg["hidden"]),
        bottleneck=int(acfg["bottleneck"]),
        dropout=float(acfg["dropout"]),
        max_epochs=int(acfg["max_epochs"]),
        patience=int(acfg["patience"]),
        batch_size=int(acfg["batch_size"]),
        lr=float(acfg["learning_rate"]),
        weight_decay=float(acfg["weight_decay"]),
    )
    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"Autoencoder: {len(features)} features | {n_params} parameters | "
        f"{len(history)} epochs run | best epoch {best_epoch} | "
        f"best val loss {history['val_loss'].min():.4f}"
    )

    detector = AnomalyDetector(pre, net, features)
    train_scores = detector.score(train)
    threshold = pick_threshold(train_scores, float(acfg["threshold_percentile"]))
    detector.threshold = threshold
    val_scores = detector.score(val)

    # Check 1: flag rate on held-out customers.
    train_rate = flag_rate(train_scores, threshold)
    val_rate = flag_rate(val_scores, threshold)
    lo, hi = float(acfg["min_flag_rate"]), float(acfg["max_flag_rate"])
    ok = lo <= val_rate <= hi
    print(f"\nThreshold (percentile {acfg['threshold_percentile']}): {threshold:.3f}")
    print(
        f"Check 1 - flag rate: train {train_rate:.2%} | validation {val_rate:.2%} "
        f"(accepted range {lo:.1%} to {hi:.1%}): {'OK' if ok else 'OUT OF RANGE'}"
    )

    # Check 2: who gets flagged?
    scores = detector.score(df)
    errors = detector.feature_errors(df)
    out = df[["customer_id"]].copy()
    out["anomaly_score"] = scores
    out["is_anomaly"] = scores > threshold
    out["top_feature"] = top_features(errors, features)
    flagged = out["is_anomaly"].to_numpy()
    print(f"\nCheck 2 - flagged customers overall: {int(flagged.sum())} of {len(df)}")
    profile = pd.DataFrame(
        {
            "flagged_median": df.loc[flagged, features].median(),
            "others_median": df.loc[~flagged, features].median(),
        }
    ).round(2)
    print(profile.to_string())
    print("\nMain feature behind the flags:")
    print(out.loc[flagged, "top_feature"].value_counts().to_string())

    review = out[flagged].merge(df[["customer_id", *features]], on="customer_id")
    review = review.sort_values("anomaly_score", ascending=False)
    review.head(int(acfg["review_rows"])).to_csv(ROOT / "docs" / "anomaly_review.csv", index=False)
    print("\nTop 10 accounts for manual review")
    show = [
        "customer_id",
        "anomaly_score",
        "top_feature",
        "monetary_total",
        "return_rate",
        "avg_basket_qty",
    ]
    print(review[show].head(10).round(2).to_string(index=False))

    # Check 3: synthetic anomalies of growing size.
    rows = []
    for kind, (levels, _fn) in INJECTIONS.items():
        for level in levels:
            rate = detection_rate(detector.score(inject(val, kind, level)), threshold)
            rows.append({"anomaly": kind, "level": level, "detected": round(rate, 3)})
    injection = pd.DataFrame(rows)
    print(
        "\nCheck 3 - share of injected anomalies above the threshold "
        f"(normal validation customers flagged: {val_rate:.1%})"
    )
    print(injection.to_string(index=False))

    out.to_csv(ROOT / "data" / "processed" / "customer_anomalies.csv", index=False)
    joblib.dump(
        {"model": detector, "features": features, "threshold": threshold},
        ROOT / "models_artifacts" / "autoencoder.joblib",
    )
    history.to_csv(ROOT / "docs" / "autoencoder_history.csv", index=False)
    summary = {
        "threshold": threshold,
        "threshold_percentile": float(acfg["threshold_percentile"]),
        "flag_rate_train": round(train_rate, 4),
        "flag_rate_validation": round(val_rate, 4),
        "flag_rate_in_accepted_range": bool(ok),
        "flagged_customers": int(flagged.sum()),
        "injection_detection": injection.to_dict(orient="records"),
    }
    (ROOT / "docs" / "anomaly_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_plots(history, val_scores, threshold)
    return summary


if __name__ == "__main__":
    # Import under the real module name so the saved objects can be unpickled.
    from src.models.autoencoder import run as run_autoencoder

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_autoencoder()
