"""Order-level purchase sequences for the LSTM. No torch needed here.

Every customer's orders placed strictly before the cutoff date are turned into
a sequence of steps. One step is one order (one invoice), described by:

* amount, gap in days since the previous order, number of items and days
  between the order and the cutoff (all log1p-transformed), and
* the share of the order's lines that fall in each product category.

Only orders before the cutoff are used, so the sequences cannot see the churn
window (the same rule as the tabular features).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.models.common import ROOT

TRANSACTIONS_PATH = "data/interim/transactions_clean.csv.gz"
MAX_LEN = 20  # keep the most recent 20 orders of each customer
BASE_FEATURES = ["amount", "gap_days", "n_items", "days_to_cutoff"]


@dataclass
class Sequences:
    """Zero-padded sequences for a list of customers."""

    x: np.ndarray  # (customers, MAX_LEN, features), log-transformed, not scaled
    lengths: np.ndarray  # number of real steps per customer (at least 1)
    customer_ids: np.ndarray
    feature_names: list[str]


def load_feature_settings() -> dict[str, Any]:
    """Cutoff date and product-category keywords from config/config.yaml."""
    with (ROOT / "config" / "config.yaml").open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    feats = raw["features"]
    return {
        "cutoff": pd.Timestamp(str(feats["cutoff_date"])),
        "categories": dict(feats["categories"]),
    }


def category_columns(categories: dict[str, list[str]]) -> list[str]:
    """Names of the per-order category share columns (plus 'other')."""
    return [f"cat_{name}" for name in [*categories, "other"]]


def assign_categories(description: pd.Series, categories: dict[str, list[str]]) -> pd.Series:
    """Category of each line from keywords in its description (first match wins)."""
    upper = description.fillna("").astype(str).str.upper()
    out = pd.Series("other", index=description.index, dtype=object)
    for name, words in categories.items():
        pattern = "|".join(re.escape(str(w).upper()) for w in words)
        hit = upper.str.contains(pattern, regex=True) & (out == "other")
        out[hit] = name
    return out


def build_orders(
    tx: pd.DataFrame, cutoff: pd.Timestamp, categories: dict[str, list[str]]
) -> pd.DataFrame:
    """One row per (customer, invoice) for real purchases before the cutoff."""
    cat_cols = category_columns(categories)
    keep = (
        tx["has_customer"]
        & ~tx["is_cancellation"]
        & ~tx["is_return"]
        & ~tx["is_non_product"]
        & (tx["quantity"] > 0)
    )
    lines = tx[keep].copy()
    lines["invoice_date"] = pd.to_datetime(lines["invoice_date"])
    lines = lines[lines["invoice_date"] < cutoff].copy()
    if lines.empty:
        return pd.DataFrame(
            columns=["customer_id", "invoice", "order_date", *BASE_FEATURES, *cat_cols]
        )
    lines["customer_id"] = lines["customer_id"].astype("int64")
    category = assign_categories(lines["description"], categories)
    shares = pd.get_dummies(category).reindex(columns=[*categories, "other"], fill_value=0)
    shares = shares.astype(float)
    shares.columns = cat_cols
    lines = pd.concat(
        [lines[["customer_id", "invoice", "invoice_date", "line_total", "quantity"]], shares],
        axis=1,
    )
    named = {c: (c, "sum") for c in cat_cols}
    orders = (
        lines.groupby(["customer_id", "invoice"], sort=False)
        .agg(
            order_date=("invoice_date", "min"),
            amount=("line_total", "sum"),
            n_items=("quantity", "sum"),
            n_lines=("quantity", "size"),
            **named,
        )
        .reset_index()
    )
    for col in cat_cols:
        orders[col] = orders[col] / orders["n_lines"]
    orders = orders.sort_values(
        ["customer_id", "order_date", "invoice"], kind="mergesort"
    ).reset_index(drop=True)
    gaps = orders.groupby("customer_id")["order_date"].diff()
    orders["gap_days"] = gaps.dt.total_seconds().div(86400).fillna(0.0)
    orders["days_to_cutoff"] = (cutoff - orders["order_date"]).dt.total_seconds() / 86400
    return orders.drop(columns=["n_lines"])


def build_sequences(
    orders: pd.DataFrame, customer_ids: np.ndarray, max_len: int = MAX_LEN
) -> Sequences:
    """Pad or truncate each customer's orders to ``max_len`` steps.

    Customers with no orders get a single all-zero step (length 1).
    """
    cat_cols = [c for c in orders.columns if c.startswith("cat_")]
    names = [*BASE_FEATURES, *cat_cols]
    values = orders[names].to_numpy(dtype=float).copy()
    values[:, : len(BASE_FEATURES)] = np.log1p(np.clip(values[:, : len(BASE_FEATURES)], 0, None))
    positions = orders.groupby("customer_id").indices  # ordered by date already
    x = np.zeros((len(customer_ids), max_len, len(names)), dtype=np.float32)
    lengths = np.zeros(len(customer_ids), dtype=np.int64)
    for i, cid in enumerate(customer_ids):
        pos = positions.get(int(cid))
        if pos is None:
            continue
        take = pos[-max_len:]
        x[i, : len(take)] = values[take]
        lengths[i] = len(take)
    return Sequences(
        x=x,
        lengths=np.maximum(lengths, 1),
        customer_ids=np.asarray(customer_ids),
        feature_names=names,
    )
