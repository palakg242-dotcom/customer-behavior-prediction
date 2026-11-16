"""End-to-end data pipeline: raw data -> clean transactions -> customer feature table.

Run from the project root with a single command:

    python -m src.pipeline
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from src.data.clean import clean_transactions, select_customer_lines
from src.data.load import load_raw_transactions
from src.data.validate import DataValidationError, validate_clean, validate_raw
from src.features.build_features import build_customer_features
from src.utils.config import load_config, resolve_path
from src.utils.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def check_label_window(transactions: pd.DataFrame, cutoff: str, horizon_days: int) -> None:
    """Fail loudly if the churn window would extend past the last observed transaction day."""
    last_day = transactions["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    window_end = pd.Timestamp(cutoff) + pd.Timedelta(days=horizon_days)
    if window_end > last_day:
        raise DataValidationError(
            f"Label window ends {window_end.date()} but data ends {last_day.date()}; "
            "choose an earlier cutoff_date or a shorter horizon_days."
        )


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the full pipeline for a configuration dictionary and return the pipeline report."""
    configure_logging(cfg["logging"]["level"])
    feat_cfg = cfg["features"]

    raw = load_raw_transactions(cfg)
    raw_report = validate_raw(raw, cfg)

    clean, clean_report = clean_transactions(raw, cfg)
    validate_clean(clean)
    check_label_window(clean, feat_cfg["cutoff_date"], feat_cfg["horizon_days"])

    interim_path = resolve_path(cfg["paths"]["interim_transactions"])
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(interim_path, index=False)

    customer_lines = select_customer_lines(clean)
    features = build_customer_features(
        customer_lines,
        cutoff=feat_cfg["cutoff_date"],
        horizon_days=feat_cfg["horizon_days"],
        categories=feat_cfg["categories"],
        discount_threshold=feat_cfg["discount_threshold"],
    )
    out_path = resolve_path(cfg["paths"]["processed_features"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)

    report: dict[str, Any] = {
        "raw": raw_report,
        "cleaning": clean_report,
        "customers": len(features),
        "churn_rate": float(features["churn_90d"].mean()),
        "cutoff_date": feat_cfg["cutoff_date"],
        "horizon_days": feat_cfg["horizon_days"],
    }
    report_path = resolve_path(cfg["paths"]["pipeline_report"])
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "stage=done customers=%d churn_rate=%.3f", report["customers"], report["churn_rate"]
    )
    return report


def run(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load the YAML config and run the pipeline."""
    return run_pipeline(load_config(config_path))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the customer data pipeline.")
    parser.add_argument("--config", default="config/config.yaml")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
