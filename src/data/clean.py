"""Data cleaning rules. Every rule is counted so the pipeline report can document it."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def standardize_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case, trim and unify descriptions: each StockCode gets its most common description."""
    out = df.copy()
    desc = out["description"].astype("string").str.upper()
    desc = desc.str.replace(r"\s+", " ", regex=True).str.strip()
    known = pd.DataFrame({"stock_code": out["stock_code"], "desc": desc})
    known = known[known["desc"].notna() & known["desc"].ne("")]
    most_common = known.groupby("stock_code")["desc"].agg(lambda s: s.value_counts().idxmax())
    out["description"] = out["stock_code"].map(most_common).fillna(desc).fillna("")
    return out


def iqr_bounds(values: pd.Series, multiplier: float) -> tuple[float, float]:
    """Return the (lower, upper) IQR fences of a numeric series."""
    q1, q3 = values.quantile([0.25, 0.75])
    spread = q3 - q1
    return float(q1 - multiplier * spread), float(q3 + multiplier * spread)


def clean_transactions(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the raw transaction table.

    Rules: exact duplicates dropped; price <= 0 dropped (adjustments); extreme quantities or
    prices dropped as data-entry errors; returns (negative quantity) KEPT and flagged; rows
    without Customer ID FLAGGED (not dropped); non-product codes FLAGGED; bulk quantities and
    high prices FLAGGED with IQR fences (kept, because bulk orders can be legitimate).
    """
    rules = cfg["cleaning"]
    report: dict[str, Any] = {"rows_in": len(df)}
    out = df.copy()

    duplicated = out.duplicated(keep="first")
    report["exact_duplicates_removed"] = int(duplicated.sum())
    out = out.loc[~duplicated]

    bad_price = out["price"].isna() | (out["price"] <= 0)
    report["non_positive_price_removed"] = int(bad_price.sum())
    out = out.loc[~bad_price]

    entry_error = (
        out["quantity"].isna()
        | (out["quantity"].abs() > rules["max_quantity_abs"])
        | (out["price"] > rules["max_price"])
    )
    report["data_entry_errors_removed"] = int(entry_error.sum())
    out = out.loc[~entry_error].copy()

    out = standardize_descriptions(out)

    out["is_cancellation"] = out["invoice"].str.upper().str.startswith("C").fillna(False)
    out["is_return"] = out["quantity"] < 0
    out["has_customer"] = out["customer_id"].notna()

    non_product = {c.upper() for c in rules["non_product_codes"]}
    prefixes = tuple(p.upper() for p in rules["non_product_prefixes"])
    out["is_non_product"] = out["stock_code"].isin(non_product) | out["stock_code"].str.startswith(
        prefixes
    ).fillna(False)

    sold = out.loc[out["quantity"] > 0, "quantity"]
    _, q_high = iqr_bounds(sold, rules["iqr_multiplier"])
    _, p_high = iqr_bounds(out["price"], rules["iqr_multiplier"])
    out["is_bulk_quantity"] = out["quantity"] > q_high
    out["is_high_price"] = out["price"] > p_high
    out["line_total"] = out["quantity"] * out["price"]

    report.update(
        rows_out=len(out),
        rows_without_customer_id=int((~out["has_customer"]).sum()),
        return_lines=int(out["is_return"].sum()),
        non_product_lines=int(out["is_non_product"].sum()),
        bulk_quantity_flagged=int(out["is_bulk_quantity"].sum()),
        high_price_flagged=int(out["is_high_price"].sum()),
        quantity_upper_fence=q_high,
        price_upper_fence=p_high,
    )
    logger.info("stage=clean rows_in=%d rows_out=%d", report["rows_in"], report["rows_out"])
    return out.reset_index(drop=True), report


def select_customer_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only product lines that belong to an identified customer (customer-level modeling)."""
    mask = df["has_customer"] & ~df["is_non_product"]
    out = df.loc[mask].copy()
    out["customer_id"] = out["customer_id"].astype("int64")
    return out.reset_index(drop=True)
