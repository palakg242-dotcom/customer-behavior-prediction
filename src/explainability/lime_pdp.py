"""Day 15 (part 1): LIME, partial dependence plots and a check on one feature.

Run from the project root:  python -m src.explainability.lime_pdp

* LIME explains the same three validation customers as the SHAP analysis (low
  risk, high risk, borderline) and is compared with SHAP: do both name the same
  main features, and push in the same direction?
* Partial dependence plots show how the average predicted churn probability of
  the XGBoost model changes with one feature, the others left as they are.
* A short check shows whether seasonal_concentration = 1 simply marks customers
  with a single order (then it is not a real seasonality signal).

The test split is not used. Settings come from the ``explanation`` section of
config.yaml.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.explainability.explain_text import load_explanation_config
from src.explainability.shap_utils import as_2d, pick_customers
from src.explainability.xai_utils import (
    compare_rankings,
    make_grid,
    pdp_values,
)
from src.models.common import (
    ROOT,
    load_dataset,
    load_modeling_config,
    make_features,
    split_data,
)

logger = logging.getLogger(__name__)
DOCS = ROOT / "docs"


def _label_value(item: Any, label: int = 1) -> float:
    """LIME stores some results per label (dict) and some directly."""
    if isinstance(item, dict):
        item = item[label]
    return float(np.ravel(item)[0])


def seasonal_check(df: pd.DataFrame, target: str) -> dict[str, float]:
    """Is seasonal_concentration = 1 just a marker for single-order customers?"""
    top = df["seasonal_concentration"] >= 0.999
    single = df["frequency"] <= 1
    out = {
        "share_with_seasonal_1": float(top.mean()),
        "median_frequency_when_seasonal_1": float(df.loc[top, "frequency"].median()),
        "share_single_order_among_seasonal_1": float(single[top].mean()),
        "share_seasonal_1_among_single_order": float(top[single].mean()),
        "churn_rate_seasonal_1": float(df.loc[top, target].mean()),
        "churn_rate_other": float(df.loc[~top, target].mean()),
    }
    print("\nCheck - is seasonal_concentration = 1 a marker for one-order customers?")
    for key, value in out.items():
        print(f"  {key}: {value:.3f}")
    return out


def run() -> dict[str, Any]:
    """LIME vs SHAP for three customers, partial dependence plots, seasonal check."""
    cfg = load_modeling_config()
    ecfg = load_explanation_config()
    seed = int(cfg["seed"])
    bundle = joblib.load(ROOT / "models_artifacts" / "xgboost.joblib")
    model, columns = bundle["model"], list(bundle["columns"])

    df = load_dataset(cfg)
    train, val, _test = split_data(df, cfg)
    val = val.reset_index(drop=True)
    x_train = make_features(train)[columns].reset_index(drop=True)
    x_val = make_features(val)[columns].reset_index(drop=True)
    proba = model.predict_proba(x_val)[:, 1]
    picks = pick_customers(proba)  # the same three customers as in the SHAP step
    DOCS.mkdir(exist_ok=True)

    # LIME versus SHAP.
    shap_values, _base = as_2d(shap.TreeExplainer(model)(x_val))
    lime_explainer = LimeTabularExplainer(
        x_train.to_numpy(dtype=float),
        feature_names=columns,
        class_names=["stay", "churn"],
        mode="classification",
        random_state=seed,
    )

    def predict_fn(rows: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(rows, columns=columns))

    k = int(ecfg["compare_top_k"])
    rows = []
    for label, i in picks.items():
        exp = lime_explainer.explain_instance(
            x_val.iloc[i].to_numpy(dtype=float),
            predict_fn,
            num_features=int(ecfg["lime_features"]),
            num_samples=int(ecfg["lime_samples"]),
        )
        fig = exp.as_pyplot_figure(label=1)
        fig.savefig(DOCS / f"lime_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        comparison = compare_rankings(exp.as_map()[1], shap_values[i], columns, k=k)
        rows.append(
            {
                "example": label,
                "customer_id": int(val["customer_id"].iloc[i]),
                "model_probability": round(float(proba[i]), 3),
                "lime_local_prediction": round(_label_value(exp.local_pred), 3),
                "lime_local_r2": round(_label_value(exp.score), 3),
                f"top{k}_overlap": comparison["overlap"],
                "sign_agreement": round(comparison["sign_agreement"], 2),
                "lime_top": ", ".join(comparison["lime_top"][:3]),
                "shap_top": ", ".join(comparison["shap_top"][:3]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(DOCS / "lime_vs_shap.csv", index=False)
    print("LIME vs SHAP for the three example customers (validation split)")
    print(table.to_string(index=False))

    # Partial dependence plots.
    pdp_frames = []
    features = [f for f in ecfg["pdp_features"] if f in columns]
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    axes = np.atleast_1d(axes)
    average = float(model.predict_proba(x_train)[:, 1].mean())
    for ax, feature in zip(axes, features):
        grid = make_grid(x_train[feature].to_numpy())
        frame = pdp_values(model, x_train, feature, grid)
        pdp_frames.append(frame)
        ax.plot(frame["value"], frame["avg_churn_probability"], marker="o", markersize=3)
        ax.axhline(average, color="gray", linestyle="--", linewidth=0.8)
        ax.set(xlabel=feature, ylabel="average churn probability", ylim=(0, 1))
        ax.set_title(f"Partial dependence: {feature}", fontsize=10)
    fig.tight_layout()
    fig.savefig(DOCS / "pdp.png", dpi=150)
    plt.close(fig)
    pdp = pd.concat(pdp_frames, ignore_index=True)
    pdp.round(4).to_csv(DOCS / "pdp_values.csv", index=False)
    print("\nPartial dependence: churn probability at the low, middle and high end")
    for feature, part in pdp.groupby("feature", sort=False):
        probs = part["avg_churn_probability"].to_numpy()
        print(
            f"  {feature}: {probs[0]:.2f} (lowest value) -> "
            f"{probs[len(probs) // 2]:.2f} (middle) -> {probs[-1]:.2f} (highest value)"
        )

    seasonal = seasonal_check(df, cfg["churn_target"])
    summary = {
        "lime_vs_shap": rows,
        "pdp_features": features,
        "seasonal_check": seasonal,
    }
    (DOCS / "lime_pdp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
