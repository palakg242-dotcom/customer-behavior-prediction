"""Tests for the plain-language explanation template."""

from __future__ import annotations

from src.explainability.explain_text import (
    DEFAULTS,
    describe_feature,
    explain_customer,
    load_explanation_config,
    risk_level,
)

CFG = {"high_risk_at": 0.7, "low_risk_at": 0.4}


def test_risk_levels_follow_the_config_thresholds() -> None:
    assert risk_level(0.76, CFG) == "high"
    assert risk_level(0.70, CFG) == "high"
    assert risk_level(0.50, CFG) == "moderate"
    assert risk_level(0.31, CFG) == "low"


def test_known_unknown_and_category_features_get_readable_phrases() -> None:
    assert describe_feature("recency_days", 634) == (
        "the last purchase was 634 days before the cutoff date"
    )
    assert describe_feature("cat_share_home_decor", 0.25) == (
        "25% of the purchased items are home decor"
    )
    assert describe_feature("is_uk", 0.0) == "the customer is outside the UK"
    assert describe_feature("something_new", 1.5) == "something_new is 1.50"


def test_explanation_names_probability_reasons_and_direction() -> None:
    drivers = [
        {"feature": "engagement_score", "value": 8.8, "shap": 0.35},
        {"feature": "recency_days", "value": 8.0, "shap": -0.23},
    ]
    text = explain_customer(17318, 0.759, 0.565, drivers, CFG)
    assert "Customer 17318 has a high churn risk (76%" in text
    assert "average customer is at 56%" in text or "average customer is at 57%" in text
    assert "the engagement score is 9 (raises the risk)" in text
    lowers = "the last purchase was 8 days before the cutoff date (lowers the risk)"
    assert lowers in text
    assert text.endswith("not proven causes.")


def test_default_config_is_consistent() -> None:
    cfg = load_explanation_config()
    assert set(DEFAULTS) <= set(cfg)
    assert 0 < cfg["low_risk_at"] < cfg["high_risk_at"] < 1
    assert cfg["drivers"] >= 1


def test_full_seasonal_concentration_is_not_described_as_seasonality() -> None:
    text = describe_feature("seasonal_concentration", 1.0)
    assert "one order" in text
    assert describe_feature("seasonal_concentration", 0.73) == (
        "73% of the purchases fall in a single month"
    )
