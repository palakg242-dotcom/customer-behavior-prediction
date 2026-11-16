"""Customer-level feature engineering with strict point-in-time discipline.

Every feature is computed ONLY from transactions strictly before the cutoff date. The targets
(churn_90d, clv_target) are computed ONLY from the window [cutoff, cutoff + horizon).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

BASE_FEATURES: list[str] = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "clv_historical",
    "avg_basket_qty",
    "avg_basket_lines",
    "frequency_trend",
    "gap_variance",
    "seasonal_concentration",
    "return_rate",
    "discount_sensitivity",
    "engagement_score",
]
TARGET_COLUMNS: list[str] = ["churn_90d", "clv_target"]


def feature_columns(categories: dict[str, list[str]]) -> list[str]:
    """Return the model input columns (base features plus category affinity shares).

    The shares of all categories (including "other") sum to 1, so "other" is left out as the
    reference category. Keeping it would make the columns perfectly collinear (infinite VIF).
    """
    return BASE_FEATURES + [f"cat_share_{name}" for name in categories]


def category_of(description: str, categories: dict[str, list[str]]) -> str:
    """Map a product description to the first category whose keyword appears in it."""
    tokens = set(re.findall(r"[A-Z]+", str(description).upper()))
    tokens |= {t[:-1] for t in tokens if len(t) > 3 and t.endswith("S")}
    for name, keywords in categories.items():
        if tokens & {k.upper() for k in keywords}:
            return name
    return "other"


def _assign_categories(descriptions: pd.Series, categories: dict[str, list[str]]) -> pd.Series:
    """Categorize each distinct description once, then map it back to all rows."""
    lookup = {d: category_of(d, categories) for d in descriptions.dropna().unique()}
    return descriptions.map(lookup).fillna("other")


def _frequency_trend(orders: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Slope of monthly order counts (orders per month, per month) up to the cutoff."""
    last_month = (cutoff - pd.Timedelta(days=1)).to_period("M")
    months = orders["invoice_date"].dt.to_period("M")
    slopes: dict[int, float] = {}
    for customer, group in orders.groupby("customer_id"):
        counts = months.loc[group.index].value_counts()
        full = pd.period_range(counts.index.min(), last_month, freq="M")
        y = counts.reindex(full, fill_value=0).to_numpy(dtype=float)
        slopes[int(customer)] = float(np.polyfit(np.arange(len(y)), y, 1)[0]) if len(y) > 1 else 0.0
    return pd.Series(slopes, name="frequency_trend")


def build_customer_features(
    transactions: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    horizon_days: int,
    categories: dict[str, list[str]],
    discount_threshold: float = 0.9,
) -> pd.DataFrame:
    """Build one row per customer with features (before cutoff) and targets (after cutoff).

    Only customers with at least one purchase before the cutoff are included.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    horizon_end = cutoff_ts + pd.Timedelta(days=horizon_days)

    history = transactions[transactions["invoice_date"] < cutoff_ts]
    future = transactions[
        (transactions["invoice_date"] >= cutoff_ts) & (transactions["invoice_date"] < horizon_end)
    ]
    sales = history[history["quantity"] > 0].copy()
    returns = history[history["quantity"] < 0]
    if sales.empty:
        raise ValueError("No purchases before the cutoff date; check cutoff_date in the config.")

    by_customer = sales.groupby("customer_id")
    index = by_customer.size().index
    feats = pd.DataFrame(index=index)
    feats.index.name = "customer_id"

    feats["country"] = by_customer["country"].agg(lambda s: s.mode().iloc[0])
    feats["frequency"] = by_customer["invoice"].nunique()
    feats["recency_days"] = (cutoff_ts - by_customer["invoice_date"].max()).dt.days
    feats["monetary_total"] = by_customer["line_total"].sum()
    feats["monetary_avg"] = feats["monetary_total"] / feats["frequency"]

    returned_value = -returns.groupby("customer_id")["line_total"].sum()
    returned_qty = -returns.groupby("customer_id")["quantity"].sum()
    sold_qty = by_customer["quantity"].sum()
    feats["clv_historical"] = feats["monetary_total"] - returned_value.reindex(index).fillna(0.0)
    feats["return_rate"] = (returned_qty.reindex(index).fillna(0.0) / sold_qty).clip(0.0, 1.0)

    invoices = (
        sales.groupby(["customer_id", "invoice"])
        .agg(
            items=("quantity", "sum"),
            lines=("stock_code", "nunique"),
            invoice_date=("invoice_date", "min"),
        )
        .reset_index()
        .sort_values(["customer_id", "invoice_date"])
    )
    feats["avg_basket_qty"] = invoices.groupby("customer_id")["items"].mean()
    feats["avg_basket_lines"] = invoices.groupby("customer_id")["lines"].mean()

    invoices["gap_days"] = (
        invoices.groupby("customer_id")["invoice_date"].diff().dt.total_seconds() / 86400
    )
    feats["gap_variance"] = invoices.groupby("customer_id")["gap_days"].var(ddof=0).fillna(0.0)
    feats["frequency_trend"] = _frequency_trend(invoices, cutoff_ts).reindex(index).fillna(0.0)

    calendar_spend = sales.groupby(["customer_id", sales["invoice_date"].dt.month])[
        "line_total"
    ].sum()
    share = calendar_spend / calendar_spend.groupby(level=0).transform("sum")
    feats["seasonal_concentration"] = (share**2).groupby(level=0).sum()

    sales["category"] = _assign_categories(sales["description"], categories)
    spend = sales.pivot_table(
        index="customer_id", columns="category", values="line_total", aggfunc="sum", fill_value=0.0
    )
    shares = spend.div(spend.sum(axis=1), axis=0)
    for name in categories:  # "other" is the reference category and is not exported
        feats[f"cat_share_{name}"] = shares[name] if name in shares.columns else 0.0

    median_price = sales.groupby("stock_code")["price"].transform("median")
    discounted = sales["quantity"].where(sales["price"] < discount_threshold * median_price, 0)
    feats["discount_sensitivity"] = discounted.groupby(sales["customer_id"]).sum() / sold_qty

    feats["engagement_score"] = 100 * pd.concat(
        [
            feats["recency_days"].rank(pct=True, ascending=False),
            feats["frequency"].rank(pct=True),
            feats["monetary_total"].rank(pct=True),
        ],
        axis=1,
    ).mean(axis=1)

    future_sales = future[future["quantity"] > 0].groupby("customer_id")
    feats["clv_target"] = future_sales["line_total"].sum().reindex(index).fillna(0.0)
    future_orders = future_sales["invoice"].nunique().reindex(index).fillna(0)
    feats["churn_90d"] = (future_orders == 0).astype(int)

    ordered = ["country", *feature_columns(categories), *TARGET_COLUMNS]
    return feats[ordered].reset_index()
