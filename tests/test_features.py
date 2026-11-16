"""Feature engineering tests, including the point-in-time (no target leakage) checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    BASE_FEATURES,
    TARGET_COLUMNS,
    build_customer_features,
    category_of,
    feature_columns,
)
from src.features.preprocess import FeaturePreprocessor
from tests.helpers import CATEGORIES, CUTOFF, HORIZON, hand_transactions


def build(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build the feature table with the test configuration."""
    return build_customer_features(transactions, CUTOFF, HORIZON, CATEGORIES)


def features_only(table: pd.DataFrame) -> pd.DataFrame:
    """Keep only the id and the model input columns (drop targets)."""
    return table[["customer_id", "country", *feature_columns(CATEGORIES)]].set_index("customer_id")


def test_hand_calculated_features_customer_1() -> None:
    row = build(hand_transactions()).set_index("customer_id").loc[1]
    assert row["frequency"] == 2
    assert row["recency_days"] == 39  # last purchase 2011-08-01 09:00 -> cutoff 2011-09-10
    assert row["monetary_total"] == pytest.approx(50.0)
    assert row["monetary_avg"] == pytest.approx(25.0)
    assert row["clv_historical"] == pytest.approx(45.0)  # 50 sales - 5 returned
    assert row["return_rate"] == pytest.approx(1 / 7)  # 1 returned of 7 sold
    assert row["avg_basket_qty"] == pytest.approx(3.5)
    assert row["avg_basket_lines"] == pytest.approx(1.5)
    assert row["gap_variance"] == pytest.approx(0.0)  # a single gap has zero variance
    assert row["discount_sensitivity"] == pytest.approx(0.0)  # constant prices
    assert row["cat_share_kitchen"] == pytest.approx(30 / 50)
    assert row["cat_share_bags"] == pytest.approx(20 / 50)


def test_hand_calculated_features_customer_2() -> None:
    row = build(hand_transactions()).set_index("customer_id").loc[2]
    assert row["frequency"] == 1
    assert row["recency_days"] == 242
    assert row["country"] == "Germany"


def test_targets_come_from_the_label_window() -> None:
    table = build(hand_transactions()).set_index("customer_id")
    assert table.loc[1, "churn_90d"] == 0 and table.loc[1, "clv_target"] == pytest.approx(5.0)
    assert table.loc[2, "churn_90d"] == 1 and table.loc[2, "clv_target"] == pytest.approx(0.0)


def test_customer_first_seen_after_cutoff_is_excluded() -> None:
    assert 3 not in set(build(hand_transactions())["customer_id"])


def test_features_ignore_all_future_rows() -> None:
    """LEAKAGE TEST: removing every post-cutoff row must not change any feature."""
    tx = hand_transactions()
    with_future = features_only(build(tx))
    history_only = features_only(build(tx[tx["invoice_date"] < pd.Timestamp(CUTOFF)]))
    pd.testing.assert_frame_equal(with_future, history_only)


def test_features_do_not_change_when_future_data_changes() -> None:
    """LEAKAGE TEST: adding huge future purchases must not change any feature."""
    tx = hand_transactions()
    extra = tx.iloc[[0, 5]].copy()
    extra["invoice_date"] = pd.Timestamp("2011-09-20")
    extra["quantity"] = 5000
    extra["line_total"] = extra["quantity"] * extra["price"]
    pd.testing.assert_frame_equal(
        features_only(build(tx)), features_only(build(pd.concat([tx, extra], ignore_index=True)))
    )


def test_purchase_exactly_at_cutoff_is_future() -> None:
    """The cutoff is exclusive for features: a purchase at the cutoff instant is a label event."""
    tx = hand_transactions()
    at_cutoff = tx.iloc[[5]].copy()
    at_cutoff["invoice"] = "99999"
    at_cutoff["invoice_date"] = pd.Timestamp(CUTOFF)
    table = build(pd.concat([tx, at_cutoff], ignore_index=True)).set_index("customer_id")
    assert table.loc[2, "frequency"] == 1  # not counted as history
    assert table.loc[2, "churn_90d"] == 0  # but it counts as a purchase in the label window


def test_purchases_after_the_window_do_not_change_the_label() -> None:
    tx = hand_transactions()
    late = tx.iloc[[5]].copy()
    late["invoice"] = "88888"
    late["invoice_date"] = pd.Timestamp(CUTOFF) + pd.Timedelta(days=HORIZON + 5)
    table = build(pd.concat([tx, late], ignore_index=True)).set_index("customer_id")
    assert table.loc[2, "churn_90d"] == 1


def test_target_columns_are_not_model_inputs() -> None:
    inputs = feature_columns(CATEGORIES)
    assert not set(TARGET_COLUMNS) & set(inputs)
    assert set(BASE_FEATURES) <= set(inputs)


def test_recency_is_never_negative_and_scores_are_bounded() -> None:
    table = build(hand_transactions())
    assert (table["recency_days"] >= 0).all()
    assert table["engagement_score"].between(0, 100).all()
    assert table["return_rate"].between(0, 1).all()


def test_category_shares_leave_out_the_reference_category() -> None:
    """Shares sum to 1, so 'other' must not be a model input (perfect collinearity)."""
    table = build(hand_transactions())
    shares = [c for c in table.columns if c.startswith("cat_share_")]
    assert "cat_share_other" not in shares
    assert (table[shares].sum(axis=1) <= 1 + 1e-9).all()


def test_category_keywords_match_whole_words_and_plurals() -> None:
    assert category_of("RED MUGS", CATEGORIES) == "kitchen"
    assert category_of("LARGE BAG", CATEGORIES) == "bags"
    assert category_of("BAGUETTE TOY", CATEGORIES) == "other"


def test_preprocessor_is_fit_on_train_only_and_handles_unseen_country() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "country": ["UK", "UK", "Germany"]})
    test = pd.DataFrame({"x": [100.0], "country": ["Spain"]})
    prep = FeaturePreprocessor(["x"], top_countries=1).fit(train)
    assert prep.scaler_.mean_[0] == pytest.approx(2.0)  # statistics come from train only
    out = prep.transform(test)
    assert out.loc[0, "country_Other"] == 1.0
    assert list(out.columns) == ["x", "country_UK", "country_Other"]
    assert np.isfinite(out.to_numpy()).all()
