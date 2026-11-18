"""Automated data validation. Checks fail loudly with DataValidationError."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when a validation rule is violated."""


def validate_raw(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate schema, null rates and the date range of the loaded raw data."""
    rules = cfg["validation"]
    errors: list[str] = []

    missing = [c for c in rules["required_columns"] if c not in df.columns]
    if missing:
        logger.error("stage=validate_raw status=failed missing_columns=%s", missing)
        raise DataValidationError(f"Missing required columns: {missing}")
    if df.empty:
        logger.error("stage=validate_raw status=failed reason=empty_dataset")
        raise DataValidationError("The dataset is empty.")

    null_rates = {c: float(df[c].isna().mean()) for c in rules["required_columns"]}
    for col, limit in rules["max_null_rate"].items():
        if null_rates[col] > limit:
            errors.append(f"null rate of '{col}' is {null_rates[col]:.4f} (limit {limit})")

    for col in ("quantity", "price"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"column '{col}' is not numeric")

    low, high = df["invoice_date"].min(), df["invoice_date"].max()
    if low < pd.Timestamp(rules["date_min"]) or high > pd.Timestamp(rules["date_max"]):
        errors.append(f"dates {low} to {high} outside [{rules['date_min']}, {rules['date_max']}]")

    if errors:
        logger.error("stage=validate_raw status=failed errors=%s", errors)
        raise DataValidationError("Raw data validation failed: " + "; ".join(errors))
    logger.info("stage=validate_raw status=ok rows=%d", len(df))
    return {"null_rates": null_rates, "date_min": str(low), "date_max": str(high)}


def validate_clean(df: pd.DataFrame) -> None:
    """Validate the cleaned transaction table before feature engineering."""
    errors: list[str] = []
    needed = ["has_customer", "is_non_product", "is_cancellation", "line_total"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logger.error("stage=validate_clean status=failed missing_columns=%s", missing)
        raise DataValidationError(f"Cleaned table lacks columns: {missing}")
    if (df["price"] <= 0).any():
        errors.append("rows with price <= 0 remain")
    if not np.isfinite(df["line_total"]).all():
        errors.append("line_total contains non-finite values")
    if df["invoice_date"].isna().any():
        errors.append("invoice_date contains missing values")
    if errors:
        logger.error("stage=validate_clean status=failed errors=%s", errors)
        raise DataValidationError("Clean data validation failed: " + "; ".join(errors))
    logger.info("stage=validate_clean status=ok rows=%d", len(df))
