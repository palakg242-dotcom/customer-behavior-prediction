"""Small hand-built and synthetic datasets shared by the tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import coerce_types, standardize_columns

CUTOFF = "2011-09-10"
HORIZON = 90
CATEGORIES = {"kitchen": ["MUG"], "bags": ["BAG"]}


def hand_transactions() -> pd.DataFrame:
    """Customer-level lines whose features can be calculated by hand.

    Customer 1 (UK): orders 2011-06-01 (2 lines) and 2011-08-01, one return on 2011-08-05,
    and a purchase on 2011-10-01 (after the cutoff, so churn_90d = 0).
    Customer 2 (Germany): one order on 2011-01-10, nothing afterwards (churn_90d = 1).
    Customer 3: first purchase 2011-10-05 (after the cutoff, so excluded from the table).
    """
    rows = [
        ("10001", "A1", "RED MUG", 2, "2011-06-01 10:00", 5.0, 1, "United Kingdom"),
        ("10001", "A2", "BLUE BAG", 1, "2011-06-01 10:00", 20.0, 1, "United Kingdom"),
        ("10002", "A1", "RED MUG", 4, "2011-08-01 09:00", 5.0, 1, "United Kingdom"),
        ("C10003", "A1", "RED MUG", -1, "2011-08-05 09:00", 5.0, 1, "United Kingdom"),
        ("10004", "A1", "RED MUG", 1, "2011-10-01 09:00", 5.0, 1, "United Kingdom"),
        ("10010", "A2", "BLUE BAG", 3, "2011-01-10 12:00", 20.0, 2, "Germany"),
        ("10020", "A1", "RED MUG", 2, "2011-10-05 09:00", 5.0, 3, "United Kingdom"),
    ]
    cols = [
        "invoice", "stock_code", "description", "quantity",
        "invoice_date", "price", "customer_id", "country",
    ]  # fmt: skip
    df = pd.DataFrame(rows, columns=cols)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["customer_id"] = df["customer_id"].astype("int64")
    df["line_total"] = df["quantity"] * df["price"]
    return df


def messy_raw() -> pd.DataFrame:
    """A small raw-looking table (Excel-style column names) containing every known problem."""
    rows = [
        ("I1", "A1", "red mug ", 2, "2011-01-05", 5.0, 1.0, "UK"),
        ("I1", "A1", "red mug ", 2, "2011-01-05", 5.0, 1.0, "UK"),  # exact duplicate
        ("I2", "A1", "RED MUG", 1, "2011-01-06", 5.0, np.nan, "UK"),  # no customer id
        ("I3", "B1", "BLUE BAG", 1, "2011-01-07", 0.0, 1.0, "UK"),  # zero price
        ("C4", "A1", "RED MUG", -1, "2011-01-08", 5.0, 1.0, "UK"),  # return
        ("I5", "POST", "POSTAGE", 1, "2011-01-09", 15.0, 1.0, "UK"),  # non-product code
        ("I6", "A2", "HUGE", 80995, "2011-01-10", 1.0, 1.0, "UK"),  # data-entry error
        ("I7", "gift_0001_20", "GIFT VOUCHER", 1, "2011-01-11", 10.0, 1.0, "UK"),  # prefix
    ]
    cols = [
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country",
    ]  # fmt: skip
    return coerce_types(standardize_columns(pd.DataFrame(rows, columns=cols)))


def synthetic_raw(n_customers: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic Online-Retail-like raw data (Excel-style column names) with realistic mess."""
    rng = np.random.default_rng(seed)
    products = [
        ("P1", "RED MUG"), ("P2", "BLUE BAG"), ("P3", "CHRISTMAS CANDLE"),
        ("P4", "GARDEN BIRD FEEDER"), ("P5", "GREETING CARD"), ("P6", "TOY ROBOT"),
    ]  # fmt: skip
    start, end = pd.Timestamp("2010-01-01"), pd.Timestamp("2011-12-09 12:00")
    rows: list[tuple] = []
    invoice_no = 500000
    for cust in range(12000, 12000 + n_customers):
        country = rng.choice(["United Kingdom", "Germany", "France"], p=[0.8, 0.1, 0.1])
        when = start + pd.Timedelta(days=float(rng.uniform(0, 400)))
        gap_scale = float(rng.uniform(15, 90))
        while when < end:
            invoice_no += 1
            for code, desc in products[: int(rng.integers(1, 5))]:
                qty = int(rng.integers(1, 12))
                price = float(rng.choice([2.5, 3.0, 4.5, 9.0]))
                rows.append((str(invoice_no), code, desc, qty, when, price, float(cust), country))
            if rng.random() < 0.08:  # a cancellation of the first line
                rows.append(
                    (f"C{invoice_no}", "P1", "RED MUG", -1, when, 2.5, float(cust), country)
                )
            when += pd.Timedelta(days=float(rng.exponential(gap_scale)))
    rows.append(
        ("500001", "P1", "RED MUG", 1, pd.Timestamp("2010-02-02"), 2.5, np.nan, "United Kingdom")
    )
    rows.append(
        (
            "500002",
            "POST",
            "POSTAGE",
            1,
            pd.Timestamp("2010-02-03"),
            18.0,
            12000.0,
            "United Kingdom",
        )
    )
    rows.append(
        ("500003", "P2", "BLUE BAG", 1, pd.Timestamp("2010-02-04"), 0.0, 12000.0, "United Kingdom")
    )
    rows.append(("599999", "P1", "RED MUG", 1, end, 2.5, 12000.0, "United Kingdom"))
    rows.append(rows[5])  # exact duplicate line
    cols = [
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country",
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=cols)
