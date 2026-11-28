"""Day 15 (part 2): a plain-language explanation for every customer.

Run from the project root:  python -m src.explainability.explain_all

For each customer the XGBoost churn probability and the top SHAP drivers are
turned into one sentence by a template. The result is saved to
data/processed/customer_explanations.csv, to be used by the API and dashboard.
Customers in the training split were part of the model's training data, so
their probabilities are in-sample; the ``split`` column says which is which.
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from src.explainability.explain_text import (
    explain_customer,
    load_explanation_config,
    risk_level,
)
from src.explainability.shap_utils import as_2d, top_drivers
from src.models.common import (
    ROOT,
    load_dataset,
    load_modeling_config,
    make_features,
    split_data,
)

logger = logging.getLogger(__name__)


def run() -> pd.DataFrame:
    """Explain every customer and save the table."""
    cfg = load_modeling_config()
    ecfg = load_explanation_config()
    bundle = joblib.load(ROOT / "models_artifacts" / "xgboost.joblib")
    model, columns = bundle["model"], list(bundle["columns"])

    df = load_dataset(cfg)
    train, val, test = split_data(df, cfg)
    split_of: dict[int, str] = {}
    for name, part in (("train", train), ("validation", val), ("test", test)):
        split_of.update({int(c): name for c in part["customer_id"]})

    x = make_features(df)[columns].reset_index(drop=True)
    proba = model.predict_proba(x)[:, 1]
    values, base = as_2d(shap.TreeExplainer(model)(x))
    base_proba = 1.0 / (1.0 + np.exp(-base))

    k = int(ecfg["drivers"])
    rows: list[dict[str, Any]] = []
    for i, customer_id in enumerate(df["customer_id"].astype("int64")):
        drivers = top_drivers(values[i], columns, x.iloc[i], k=k)
        rows.append(
            {
                "customer_id": int(customer_id),
                "split": split_of[int(customer_id)],
                "churn_probability": round(float(proba[i]), 4),
                "risk_level": risk_level(float(proba[i]), ecfg),
                "explanation": explain_customer(
                    int(customer_id), float(proba[i]), float(base_proba[i]), drivers, ecfg
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "processed" / "customer_explanations.csv", index=False)

    print(f"explained {len(out)} customers")
    print(out["risk_level"].value_counts().to_string())
    print("\nExample explanations")
    for level in ("high", "moderate", "low"):
        sample = out[out["risk_level"] == level].iloc[0]
        print(f"\n[{level}] {sample['explanation']}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
