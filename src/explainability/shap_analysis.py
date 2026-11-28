"""Day 14: SHAP explanations for the XGBoost churn model.

Run from the project root:  python -m src.explainability.shap_analysis

Explains the selected churn model on the validation split (the test split is
not used). SHAP values are in log-odds of churn: positive values push a customer
towards churn, negative values towards staying. Outputs go to docs/:

* shap_summary.png and shap_importance_bar.png - global importance;
* shap_waterfall_*.png and shap_force_*.png - one low-risk, one high-risk and
  one borderline customer;
* shap_global_importance.csv, shap_customer_explanations.csv, shap_summary.json.
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

from src.explainability.shap_utils import (
    additivity_error,
    as_2d,
    mean_abs_importance,
    pick_customers,
    top_drivers,
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


def _save_current_figure(name: str) -> None:
    fig = plt.gcf()
    fig.savefig(DOCS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run() -> dict[str, Any]:
    """Compute SHAP values, check them, draw the plots and save the tables."""
    cfg = load_modeling_config()
    bundle = joblib.load(ROOT / "models_artifacts" / "xgboost.joblib")
    model, columns = bundle["model"], list(bundle["columns"])

    df = load_dataset(cfg)
    _train, val, _test = split_data(df, cfg)
    val = val.reset_index(drop=True)
    x = make_features(val)[columns].reset_index(drop=True)
    proba = model.predict_proba(x)[:, 1]

    explainer = shap.TreeExplainer(model)
    values, base = as_2d(explainer(x))
    explanation = shap.Explanation(
        values=values, base_values=base, data=x.to_numpy(), feature_names=columns
    )

    error = additivity_error(base, values, proba)
    status = "OK" if error < 1e-3 else "CHECK THIS"
    print("SHAP check - base value + sum of SHAP matches the model's probability")
    print(f"  largest difference on {len(x)} customers: {error:.2e} ({status})")

    DOCS.mkdir(exist_ok=True)
    importance = mean_abs_importance(values, x)
    importance.round(4).to_csv(DOCS / "shap_global_importance.csv", index=False)
    print("\nGlobal importance (mean |SHAP|, log-odds)")
    print(importance.head(10).round(3).to_string(index=False))

    shap.plots.beeswarm(explanation, max_display=14, show=False)
    _save_current_figure("shap_summary.png")
    shap.plots.bar(explanation, max_display=14, show=False)
    _save_current_figure("shap_importance_bar.png")

    picks = pick_customers(proba)
    rows = []
    for label, i in picks.items():
        shap.plots.waterfall(explanation[i], max_display=12, show=False)
        _save_current_figure(f"shap_waterfall_{label}.png")
        try:
            shap.plots.force(base[i], values[i], x.iloc[i], matplotlib=True, show=False)
            _save_current_figure(f"shap_force_{label}.png")
        except Exception as exc:  # the force plot is the fussiest SHAP plot
            plt.close("all")
            print(
                f"(force plot for {label} skipped: {type(exc).__name__}; "
                "the waterfall plot has the same information)"
            )
        drivers = top_drivers(values[i], columns, x.iloc[i], k=3)
        rows.append(
            {
                "example": label,
                "customer_id": int(val["customer_id"].iloc[i]),
                "churn_probability": round(float(proba[i]), 3),
                "actually_churned": int(val[cfg["churn_target"]].iloc[i]),
                "base_probability": round(float(1 / (1 + np.exp(-base[i]))), 3),
                **{
                    f"driver_{k + 1}": (f"{d['feature']}={d['value']:.2f} ({d['shap']:+.2f})")
                    for k, d in enumerate(drivers)
                },
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(DOCS / "shap_customer_explanations.csv", index=False)
    print("\nThree example customers (validation split)")
    print(table.to_string(index=False))

    summary = {
        "customers_explained": len(x),
        "additivity_max_error": error,
        "top_features": importance["feature"].head(5).tolist(),
        "examples": rows,
    }
    (DOCS / "shap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
