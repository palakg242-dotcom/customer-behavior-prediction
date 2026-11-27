"""Tests for the purchase-sequence builder, including the leakage check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.features.sequences import (
    BASE_FEATURES,
    assign_categories,
    build_orders,
    build_sequences,
    category_columns,
)

CUTOFF = pd.Timestamp("2011-09-10")
CATS = {"christmas": ["CHRISTMAS"], "toys_games": ["TOY"]}


def _tx(rows: list[dict]) -> pd.DataFrame:
    base = {
        "description": "MUG",
        "quantity": 2,
        "line_total": 10.0,
        "has_customer": True,
        "is_cancellation": False,
        "is_return": False,
        "is_non_product": False,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_post_cutoff_orders_do_not_change_the_sequences() -> None:
    before = [
        {"customer_id": 1.0, "invoice": "A", "invoice_date": "2011-07-01 10:00:00"},
        {"customer_id": 1.0, "invoice": "B", "invoice_date": "2011-08-01 10:00:00"},
    ]
    after = [
        {"customer_id": 1.0, "invoice": "C", "invoice_date": "2011-09-10 00:00:00"},
        {"customer_id": 1.0, "invoice": "D", "invoice_date": "2011-10-20 10:00:00"},
        {"customer_id": 2.0, "invoice": "E", "invoice_date": "2011-11-01 10:00:00"},
    ]
    clean = build_orders(_tx(before), CUTOFF, CATS)
    leaky = build_orders(_tx(before + after), CUTOFF, CATS)
    pdt.assert_frame_equal(clean, leaky)
    assert 2 not in set(leaky["customer_id"])  # only has post-cutoff orders


def test_returns_cancellations_non_products_and_anonymous_rows_are_ignored() -> None:
    day = "2011-06-01 10:00:00"
    rows = [
        {"customer_id": 1.0, "invoice": "A", "invoice_date": day},
        {"customer_id": 1.0, "invoice": "R", "invoice_date": day, "is_return": True},
        {"customer_id": 1.0, "invoice": "C", "invoice_date": day, "is_cancellation": True},
        {"customer_id": 1.0, "invoice": "P", "invoice_date": day, "is_non_product": True},
        {"customer_id": 1.0, "invoice": "N", "invoice_date": day, "quantity": -3},
        {"customer_id": np.nan, "invoice": "X", "invoice_date": day, "has_customer": False},
    ]
    orders = build_orders(_tx(rows), CUTOFF, CATS)
    assert orders["invoice"].tolist() == ["A"]


def test_gap_days_and_days_to_cutoff() -> None:
    rows = [
        {"customer_id": 1.0, "invoice": "A", "invoice_date": "2011-07-01 00:00:00"},
        {"customer_id": 1.0, "invoice": "B", "invoice_date": "2011-07-11 00:00:00"},
        {"customer_id": 1.0, "invoice": "C", "invoice_date": "2011-09-09 00:00:00"},
    ]
    orders = build_orders(_tx(rows), CUTOFF, CATS)
    assert orders["gap_days"].tolist() == [0.0, 10.0, 60.0]
    assert orders["days_to_cutoff"].iloc[-1] == 1.0


def test_category_shares_sum_to_one_per_order() -> None:
    rows = [
        {
            "customer_id": 1.0,
            "invoice": "A",
            "invoice_date": "2011-07-01 10:00:00",
            "description": "CHRISTMAS TREE",
        },
        {
            "customer_id": 1.0,
            "invoice": "A",
            "invoice_date": "2011-07-01 10:00:00",
            "description": "BLUE MUG",
        },
    ]
    orders = build_orders(_tx(rows), CUTOFF, CATS)
    cols = category_columns(CATS)
    assert orders[cols].sum(axis=1).iloc[0] == 1.0
    assert orders["cat_christmas"].iloc[0] == 0.5
    assert orders["cat_other"].iloc[0] == 0.5


def test_first_matching_category_wins() -> None:
    desc = pd.Series(["CHRISTMAS TOY", "TOY CAR", "PLAIN"])
    out = assign_categories(desc, CATS)
    assert out.tolist() == ["christmas", "toys_games", "other"]


def test_sequences_keep_the_most_recent_orders_and_pad_with_zeros() -> None:
    rows = [
        {"customer_id": 1.0, "invoice": f"I{i}", "invoice_date": f"2011-0{i}-01 00:00:00"}
        for i in range(1, 6)
    ]
    rows.append({"customer_id": 2.0, "invoice": "Z", "invoice_date": "2011-08-01 00:00:00"})
    orders = build_orders(_tx(rows), CUTOFF, CATS)
    seq = build_sequences(orders, np.array([1, 2, 99]), max_len=3)
    assert seq.x.shape == (3, 3, len(BASE_FEATURES) + len(category_columns(CATS)))
    assert seq.lengths.tolist() == [3, 1, 1]
    last = BASE_FEATURES.index("days_to_cutoff")
    assert seq.x[0, 2, last] < seq.x[0, 0, last]  # last step is the most recent order
    assert np.all(seq.x[1, 1:] == 0)  # padding
    assert np.all(seq.x[2] == 0)  # unknown customer
