"""Validation, cleaning and end-to-end pipeline tests (all use small synthetic data)."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_transactions, select_customer_lines
from src.data.load import coerce_types, load_raw_transactions, standardize_columns
from src.data.validate import DataValidationError, validate_clean, validate_raw
from src.features.analysis import target_correlations, vif_table
from src.features.preprocess import FeaturePreprocessor
from src.pipeline import check_label_window, run_pipeline
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
    assert features.drop(columns=["country", "next_category"]).notna().all().all()


def test_validate_clean_fails_on_bad_rows() -> None:
    cleaned, _ = clean_transactions(messy_raw(), load_config())
    broken = cleaned.copy()
    broken.loc[0, "price"] = -1.0
    with pytest.raises(DataValidationError):
        validate_clean(broken)
    with pytest.raises(DataValidationError):
        validate_clean(cleaned.drop(columns=["line_total"]))


def test_label_window_must_fit_inside_the_data() -> None:
    cleaned, _ = clean_transactions(messy_raw(), load_config())  # data ends in January 2011
    check_label_window(cleaned, "2010-12-01", 30)  # fits: no error
    with pytest.raises(DataValidationError):
        check_label_window(cleaned, "2011-06-01", 90)


def test_missing_dataset_gives_a_helpful_error() -> None:
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["raw_excel"] = "data/raw/does_not_exist.xlsx"
    with (
        patch("src.data.load._read_ucimlrepo", side_effect=RuntimeError("not available")),
        pytest.raises(FileNotFoundError),
    ):
        load_raw_transactions(cfg)


def test_vif_detects_collinearity_and_correlations_have_the_right_shape() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    df = pd.DataFrame(
        {"a": x, "b": x * 2 + rng.normal(scale=0.01, size=300), "c": rng.normal(size=300)}
    )
    df["target"] = x + rng.normal(size=300)
    vif = vif_table(df, ["a", "b", "c"]).set_index("feature")["vif"]
    assert vif["a"] > 100 and vif["b"] > 100 and vif["c"] < 2
    corr = target_correlations(df, ["a", "c"], ["target"])
    assert corr.shape == (2, 1)


def test_preprocessor_can_be_saved_and_loaded(tmp_path: Path) -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "country": ["UK", "UK", "France"]})
    prep = FeaturePreprocessor(["x"], top_countries=2).fit(train)
    prep.save(tmp_path / "prep.joblib")
    loaded = FeaturePreprocessor.load(tmp_path / "prep.joblib")
    pd.testing.assert_frame_equal(prep.transform(train), loaded.transform(train))
    unscaled = loaded.transform(train, scale=False)
    assert unscaled["x"].tolist() == [1.0, 2.0, 3.0]  # tree models get raw values
