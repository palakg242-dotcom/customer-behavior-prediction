"""Day 9: customer segmentation with K-Means and a Gaussian Mixture Model.

Run from the project root:  python -m src.segmentation.cluster

Customers who bought only once have no purchase gaps, trend or seasonality (for
example their seasonal concentration is always 1.0), so they cannot be clustered
on those features. They are split by a simple recency rule into "recent" and
"lapsed" one-time buyers. All repeat customers (2+ orders) are clustered:

* K-Means is the main method. The number of clusters is chosen from the elbow
  (inertia) and the silhouette score, not fixed in advance.
* A Gaussian Mixture Model (components chosen by BIC) is the second algorithm.
  It tests whether K-Means' round-cluster assumption hides overlapping or
  stretched segments.

Each segment gets a business-readable name and a profile table. Churn and CLV
are used only to describe segments, never to build them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.models.common import ROOT, load_dataset, load_modeling_config

logger = logging.getLogger(__name__)

FEATURES = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "return_rate",
    "seasonal_concentration",
]
# Skewed columns are log-transformed before scaling.
LOG_COLUMNS = ["recency_days", "frequency", "monetary_total", "monetary_avg"]

MIN_K = 3  # fewer than 3 segments is not useful for marketing
K_RANGE = range(2, 9)

ONE_TIME_RECENT = "One-time buyer (recent)"
ONE_TIME_LAPSED = "One-time buyer (lapsed)"


def one_time_mask(df: pd.DataFrame) -> np.ndarray:
    """True for customers with a single order."""
    return (df["frequency"] <= 1).to_numpy()


def one_time_names(df: pd.DataFrame, recent_days: int) -> np.ndarray:
    """Name one-time buyers by whether their only order was recent."""
    return np.where(df["recency_days"] <= recent_days, ONE_TIME_RECENT, ONE_TIME_LAPSED)


def prepare(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Log-transform skewed columns (negatives become 0) and standardize."""
    x = df[FEATURES].astype(float).copy()
    for col in LOG_COLUMNS:
        x[col] = np.log1p(x[col].clip(lower=0))
    x = x.fillna(0.0)
    scaler = StandardScaler()
    return scaler.fit_transform(x), scaler


def select_k(x: np.ndarray, k_range: range, seed: int) -> pd.DataFrame:
    """Inertia (elbow), silhouette and Davies-Bouldin for each k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(x, km.labels_)),
                "davies_bouldin": float(davies_bouldin_score(x, km.labels_)),
            }
        )
    return pd.DataFrame(rows)


def best_k(table: pd.DataFrame) -> int:
    """Highest silhouette among k >= MIN_K."""
    ok = table[table["k"] >= MIN_K]
    return int(ok.loc[ok["silhouette"].idxmax(), "k"])


def fit_gmm(x: np.ndarray, k_range: range, seed: int) -> tuple[GaussianMixture, pd.DataFrame]:
    """Fit Gaussian Mixture Models and keep the one with the lowest BIC."""
    rows, best, best_bic = [], None, np.inf
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
        gmm.fit(x)
        bic = float(gmm.bic(x))
        rows.append({"components": k, "bic": bic})
        if bic < best_bic:
            best, best_bic = gmm, bic
    return best, pd.DataFrame(rows)


def name_segment(stats: Mapping[str, float], ref: Mapping[str, float]) -> str:
    """Business name from a segment's median behavior vs. all customers.

    ``stats`` holds the segment medians; ``ref`` holds the overall medians, the
    75th percentile of seasonal concentration and the 90th percentile of the
    return rate. A segment that returns far more than others is named for that.
    """
    returns = stats.get("return_rate", 0.0)
    if returns >= 0.2 and returns > ref.get("return_rate_p90", 1.0):
        return "High-return buyer"
    lapsed = stats["recency_days"] > ref["recency_days_median"]
    frequent = stats["frequency"] > ref["frequency_median"]
    big = stats["monetary_total"] > ref["monetary_total_median"]
    if stats["seasonal_concentration"] > ref["seasonal_concentration_p75"]:
        return "Seasonal gift shopper"
    if not lapsed and frequent and big:
        return "Loyal high-value"
    if lapsed and big:
        return "Lapsed high-value"
    if lapsed:
        return "Lapsed low-value"
    if frequent:
        return "High-frequency low-value"
    if big:
        return "Recent high-value, low-frequency"
    return "Recent low-value"


def name_all(df: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Name every segment; duplicate names get the segment number appended."""
    ref = {
        "recency_days_median": df["recency_days"].median(),
        "frequency_median": df["frequency"].median(),
        "monetary_total_median": df["monetary_total"].median(),
        "seasonal_concentration_p75": df["seasonal_concentration"].quantile(0.75),
        "return_rate_p90": df["return_rate"].quantile(0.90),
    }
    medians = df.assign(_seg=labels).groupby("_seg")[FEATURES].median()
    names = {int(seg): name_segment(row, ref) for seg, row in medians.iterrows()}
    counts = pd.Series(names).value_counts()
    return {seg: f"{n} ({seg})" if counts[n] > 1 else n for seg, n in names.items()}


def profile_table(
    df: pd.DataFrame, segment_names: np.ndarray, cfg: Mapping[str, Any]
) -> pd.DataFrame:
    """Size and median behavior per segment, plus churn rate and mean CLV."""
    frame = df.assign(segment_name=segment_names)
    grouped = frame.groupby("segment_name")
    out = grouped[FEATURES].median().round(2)
    out.insert(0, "customers", grouped.size())
    out.insert(1, "share", (grouped.size() / len(df)).round(3))
    out["churn_rate"] = grouped[cfg["churn_target"]].mean().round(3)
    out["mean_clv_target"] = grouped[cfg["clv_target"]].mean().round(1)
    return out.sort_values("customers", ascending=False)


def save_plots(
    x: np.ndarray, labels: np.ndarray, names: dict[int, str], table: pd.DataFrame
) -> None:
    """Elbow + silhouette chart and a 2-D PCA view of the segments."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(table["k"], table["inertia"], marker="o")
    axes[0].set(title="Elbow (inertia)", xlabel="k", ylabel="inertia")
    axes[1].plot(table["k"], table["silhouette"], marker="o", color="tab:green")
    axes[1].set(title="Silhouette score", xlabel="k", ylabel="silhouette")
    fig.tight_layout()
    fig.savefig(docs / "kmeans_k_selection.png", dpi=150)
    plt.close(fig)

    coords = PCA(n_components=2, random_state=0).fit_transform(x)
    fig, ax = plt.subplots(figsize=(7, 5))
    for seg in sorted(names):
        m = labels == seg
        ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.6, label=names[seg])
    ax.set(title="Repeat-customer segments (PCA view)", xlabel="PC1", ylabel="PC2")
    ax.legend(markerscale=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(docs / "segments_pca.png", dpi=150)
    plt.close(fig)


def run() -> pd.DataFrame:
    """Segment all customers, save outputs and print the profile table."""
    cfg = load_modeling_config()
    seed = int(cfg["seed"])
    recent_days = int(cfg["recent_recency_days"])
    df = load_dataset(cfg)

    once = one_time_mask(df)
    repeat = df[~once].reset_index(drop=True)
    print(
        f"one-time buyers: {int(once.sum())} ({once.mean():.1%}) | "
        f"repeat customers clustered: {len(repeat)}"
    )
    x, scaler = prepare(repeat)

    k_table = select_k(x, K_RANGE, seed)
    k = best_k(k_table)
    print("\nK-Means model selection (repeat customers)")
    print(k_table.round(3).to_string(index=False))
    print(f"\nchosen k = {k} (highest silhouette with k >= {MIN_K})")

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
    labels = km.labels_
    names = name_all(repeat, labels)

    gmm, gmm_table = fit_gmm(x, K_RANGE, seed)
    gmm_labels = gmm.predict(x)
    summary = {
        "one_time_buyers": int(once.sum()),
        "repeat_customers": len(repeat),
        "kmeans": {
            "k": k,
            "silhouette": round(float(silhouette_score(x, labels)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(x, labels)), 4),
        },
        "gmm": {
            "components": int(gmm.n_components),
            "silhouette": round(float(silhouette_score(x, gmm_labels)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(x, gmm_labels)), 4),
        },
        "agreement_adjusted_rand_index": round(float(adjusted_rand_score(labels, gmm_labels)), 4),
    }
    print("\nGMM (BIC by number of components)")
    print(gmm_table.round(1).to_string(index=False))
    if gmm.n_components == max(K_RANGE):
        print("note: BIC is still falling at the largest size tried, so the data")
        print("is not described well by a few Gaussians; K-Means is preferred.")
    print("\nComparison (repeat customers)")
    print(json.dumps(summary, indent=2))

    segment_names = np.empty(len(df), dtype=object)
    segment_names[once] = one_time_names(df[once], recent_days)
    segment_names[~once] = [names[int(s)] for s in labels]
    profiles = profile_table(df, segment_names, cfg)
    print("\nSegment profiles")
    print(profiles.to_string())

    out = df[["customer_id"]].copy()
    out["segment_name"] = segment_names
    gmm_full = pd.array([pd.NA] * len(df), dtype="Int64")
    gmm_full[~once] = gmm_labels
    out["gmm_cluster"] = gmm_full
    out.to_csv(ROOT / "data" / "processed" / "customer_segments.csv", index=False)

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    profiles.to_csv(docs / "segment_profiles.csv")
    k_table.to_csv(docs / "kmeans_k_selection.csv", index=False)
    (docs / "segmentation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    models_dir = ROOT / "models_artifacts"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": km,
            "features": FEATURES,
            "names": names,
            "one_time_recent_days": recent_days,
        },
        models_dir / "kmeans_segments.joblib",
    )
    save_plots(x, labels, names, k_table)
    return profiles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
