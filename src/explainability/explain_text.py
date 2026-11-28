"""Plain-language explanations built from SHAP drivers. No shap import needed."""

from __future__ import annotations

from typing import Any

import yaml

from src.models.common import ROOT

DEFAULTS: dict[str, Any] = {
    "high_risk_at": 0.7,
    "low_risk_at": 0.4,
    "drivers": 3,
    "pdp_features": ["engagement_score", "recency_days", "seasonal_concentration"],
    "lime_samples": 5000,
    "lime_features": 10,
    "compare_top_k": 5,
}

PHRASES: dict[str, str] = {
    "recency_days": "the last purchase was {v:.0f} days before the cutoff date",
    "engagement_score": "the engagement score is {v:.0f}",
    "seasonal_concentration": "{v:.0%} of the purchases fall in a single month",
    "frequency": "the customer placed {v:.0f} orders",
    "clv_historical": "past spending is {v:,.0f}",
    "monetary_total": "total spending is {v:,.0f}",
    "monetary_avg": "the average order value is {v:,.0f}",
    "gap_variance": "the gaps between orders vary (variance {v:.0f})",
    "frequency_trend": "the ordering trend is {v:+.2f}",
    "return_rate": "{v:.0%} of the items were returned",
    "discount_sensitivity": "the discount sensitivity is {v:.2f}",
    "avg_basket_qty": "an average basket holds {v:.0f} items",
    "avg_basket_lines": "an average basket has {v:.0f} lines",
    "product_popularity": "the product popularity score is {v:.2f}",
}


def load_explanation_config() -> dict[str, Any]:
    """The ``explanation`` section of config/config.yaml, with defaults filled in."""
    cfg = dict(DEFAULTS)
    path = ROOT / "config" / "config.yaml"
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg.update(raw.get("explanation", {}) or {})
    return cfg


def risk_level(proba: float, cfg: dict[str, Any]) -> str:
    """'high', 'moderate' or 'low' from the churn probability and the config."""
    if proba >= float(cfg["high_risk_at"]):
        return "high"
    if proba < float(cfg["low_risk_at"]):
        return "low"
    return "moderate"


def describe_feature(name: str, value: float) -> str:
    """A short phrase for one feature value."""
    if name == "seasonal_concentration" and value >= 0.999:
        # Always true for customers with a single order, so it is not real seasonality.
        return (
            "all of the purchases fall in a single month "
            "(always the case for customers with one order)"
        )
    if name in PHRASES:
        return PHRASES[name].format(v=value)
    if name.startswith("cat_share_"):
        category = name.removeprefix("cat_share_").replace("_", " ")
        return f"{value:.0%} of the purchased items are {category}"
    if name == "is_uk":
        where = "in" if value >= 0.5 else "outside"
        return f"the customer is {where} the UK"
    return f"{name} is {value:.2f}"


def explain_customer(
    customer_id: int,
    proba: float,
    base_proba: float,
    drivers: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> str:
    """One plain-language paragraph on why a customer has this churn risk."""
    reasons = [
        f"{describe_feature(d['feature'], d['value'])} "
        f"({'raises' if d['shap'] > 0 else 'lowers'} the risk)"
        for d in drivers
    ]
    return (
        f"Customer {customer_id} has a {risk_level(proba, cfg)} churn risk "
        f"({proba:.0%}; the average customer is at {base_proba:.0%}). "
        f"Main reasons: {'; '.join(reasons)}. "
        "These are patterns the model uses, not proven causes."
    )
