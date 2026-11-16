"""Load the Online Retail II data and standardize its column names and types."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import resolve_path
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_ALIASES = {
    "customerid": "customer_id",
    "invoicedate": "invoice_date",
    "stockcode": "stock_code",
    "unitprice": "price",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case (e.g. 'Customer ID' and 'CustomerID' -> 'customer_id')."""
    renamed = {}
    for col in df.columns:
        name = re.sub(r"[^0-9a-z]+", "_", str(col).strip().lower()).strip("_")
        renamed[col] = _ALIASES.get(name, name)
    return df.rename(columns=renamed)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Give every column a consistent type and sort the table chronologically."""
    out = df.copy()
    out["invoice"] = out["invoice"].astype("string").str.strip()
    out["stock_code"] = out["stock_code"].astype("string").str.strip().str.upper()
    out["description"] = out["description"].astype("string")
    out["country"] = out["country"].astype("string").str.strip()
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["invoice_date"] = pd.to_datetime(out["invoice_date"], errors="coerce")
    out["customer_id"] = pd.to_numeric(out["customer_id"], errors="coerce").astype("Int64")
    return out.sort_values("invoice_date", kind="stable").reset_index(drop=True)


def _read_excel(path: Path) -> pd.DataFrame:
    """Read every sheet of the workbook (2009-2010 and 2010-2011) and stack them."""
    logger.info("stage=load source=excel path=%s (this can take a few minutes)", path)
    sheets = pd.read_excel(path, sheet_name=None)
    return pd.concat(sheets.values(), ignore_index=True)


def _read_ucimlrepo(repo_id: int) -> pd.DataFrame:
    """Download the dataset through the ucimlrepo package (Option B in the PRD)."""
    from ucimlrepo import fetch_ucirepo  # imported lazily: only needed as a fallback

    logger.info("stage=load source=ucimlrepo id=%s", repo_id)
    return fetch_ucirepo(id=repo_id).data.features


def load_raw_transactions(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load both sheets, standardize names and types, and return one chronological table.

    Uses data/raw/online_retail_II.xlsx when it exists, otherwise falls back to ucimlrepo.
    """
    excel_path = resolve_path(cfg["paths"]["raw_excel"])
    if excel_path.exists():
        raw = _read_excel(excel_path)
    else:
        try:
            raw = _read_ucimlrepo(int(cfg["data"]["ucimlrepo_id"]))
        except Exception as exc:  # network or package problems
            raise FileNotFoundError(
                f"Dataset not found at {excel_path} and ucimlrepo failed ({exc}). "
                "Download online_retail_II.xlsx from "
                "https://archive.ics.uci.edu/dataset/502/online+retail+ii "
                "and place it in data/raw/."
            ) from exc
    df = coerce_types(standardize_columns(raw))
    logger.info("stage=load rows=%d columns=%d", len(df), df.shape[1])
    return df
