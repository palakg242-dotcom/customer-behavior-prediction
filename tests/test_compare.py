"""Tests for the model-selection rules and helpers in src.models.compare."""

from __future__ import annotations

import pandas as pd

from src.models.compare import select_churn_model, select_clv_model, to_markdown


def _churn_rows(entries: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    """entries: (model, roc_auc, precision, recent_recall)."""
    rows = []
    for model, auc, precision, recent_recall in entries:
        rows.append(
            {
                "model": model,
                "subset": "all",
                "roc_auc": auc,
                "precision": precision,
                "recall": 0.7,
            }
        )
        rows.append(
            {
                "model": model,
                "subset": "recency<=180",
                "roc_auc": auc,
                "precision": precision,
                "recall": recent_recall,
            }
        )
    return pd.DataFrame(rows)


def test_clear_auc_winner_is_selected() -> None:
    table = _churn_rows([("a", 0.90, 0.7, 0.2), ("b", 0.80, 0.7, 0.9)])
    assert select_churn_model(table) == "a"


def test_near_tie_goes_to_better_recent_recall() -> None:
    table = _churn_rows([("rf", 0.813, 0.79, 0.445), ("xgb", 0.814, 0.71, 0.716)])
    assert select_churn_model(table) == "xgb"


def test_model_outside_tie_band_is_not_considered() -> None:
    table = _churn_rows([("best", 0.814, 0.7, 0.1), ("lgbm", 0.807, 0.7, 0.9)])
    assert select_churn_model(table) == "best"


def test_clv_selection_uses_original_scale_r2() -> None:
    rows = [
        ("ridge", "all", "original", 0.33),
        ("ridge", "all", "log", 0.90),
        ("two_stage", "all", "original", 0.42),
        ("two_stage", "all", "log", -0.30),
        ("two_stage", "recency<=180", "original", 0.99),
    ]
    table = pd.DataFrame(rows, columns=["model", "subset", "scale", "r2"])
    assert select_clv_model(table) == "two_stage"


def test_to_markdown_has_header_divider_and_rows() -> None:
    text = to_markdown(pd.DataFrame({"model": ["a", "b"], "auc": [0.8, 0.9]}))
    lines = text.splitlines()
    assert lines[0] == "| model | auc |"
    assert lines[1] == "| --- | --- |"
    assert len(lines) == 4
