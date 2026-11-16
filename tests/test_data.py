"""Validation, cleaning and end-to-end pipeline tests (all use small synthetic data)."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from src.data.clean import clean_transactions, select_customer_lines
from src.data.load import coerce_types, standardize_columns
from src.data.validate import DataValidationError, validate_clean, validate_raw
from src.pipeline import run_pipeline
from src.utils.config import load_config
from tests.helpers import messy_raw, synthetic_raw


def test_standardize_columns_accepts_both_spellings() -> None:
    a = standardize_columns(pd.DataFrame(columns=["Customer ID", "InvoiceDate", "Price"]))
    b = standardize_columns(pd.DataFrame(columns=["CustomerID", "Invoice Date", "Price"]))
    assert list(a.columns) == list(b.columns) == ["customer_id", "invoice_date", "price"]


def test_validate_raw_passes_on_good_data() -> None:
    cfg = load_config()
    report = validate_raw(coerce_types(standardize_columns(synthetic_raw())), cfg)
    assert "null_rates" in report


def test_validate_raw_fails_loudly() -> None:
    cfg = load_config()
    good = coerce_types(standardize_columns(synthetic_raw()))
    with pytest.raises(DataValidationError):
        validate_raw(good.drop(columns=["price"]), cfg)
    broken = good.copy()
    broken["customer_id"] = pd.NA  # null rate 100% > 30%
    with pytest.raises(DataValidationError):
        validate_raw(broken, cfg)
    out_of_range = good.copy()
    out_of_range.loc[0, "invoice_date"] = pd.Timestamp("2020-01-01")
    with pytest.raises(DataValidationError):
        validate_raw(out_of_range, cfg)


def test_cleaning_rules_are_applied_and_counted() -> None:
    cleaned, report = clean_transactions(messy_raw(), load_config())
    assert report["rows_in"] == 8
    assert report["exact_duplicates_removed"] == 1
    assert report["non_positive_price_removed"] == 1
    assert report["data_entry_errors_removed"] == 1
    assert report["rows_out"] == 5
    assert report["return_lines"] == 1  # returns are kept, not dropped
    assert report["rows_without_customer_id"] == 1  # flagged, not dropped
    assert report["non_product_lines"] == 2  # POST and the GIFT_ voucher
    assert "RED MUG" in set(cleaned["description"])  # standardized
    validate_clean(cleaned)


def test_customer_selection_removes_unidentified_and_non_product_lines() -> None:
    cleaned, _ = clean_transactions(messy_raw(), load_config())
    selected = select_customer_lines(cleaned)
    assert len(selected) == 2
    assert selected["customer_id"].dtype == "int64"
    assert selected["quantity"].min() < 0  # the return survived


def test_pipeline_end_to_end_on_synthetic_data(tmp_path: Path) -> None:
    cfg = copy.deepcopy(load_config())
    raw_path = tmp_path / "raw.xlsx"
    data = synthetic_raw()
    half = len(data) // 2
    with pd.ExcelWriter(raw_path) as writer:  # two sheets, like the real workbook
        data.iloc[:half].to_excel(writer, sheet_name="Year 2009-2010", index=False)
        data.iloc[half:].to_excel(writer, sheet_name="Year 2010-2011", index=False)
    cfg["paths"].update(
        raw_excel=str(raw_path),
        interim_transactions=str(tmp_path / "interim.csv.gz"),
        processed_features=str(tmp_path / "features.csv"),
        pipeline_report=str(tmp_path / "report.json"),
    )
    report = run_pipeline(cfg)
    features = pd.read_csv(tmp_path / "features.csv")
    assert report["customers"] == len(features) > 50
    assert features["customer_id"].is_unique
    assert features["churn_90d"].isin([0, 1]).all()
    assert features.drop(columns=["country"]).notna().all().all()
