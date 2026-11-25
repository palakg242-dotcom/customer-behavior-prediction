"""Day 10: final model comparison and the one-time test-set evaluation.

Step A (safe, uses validation data only):
    python -m src.models.compare

    Rebuilds the comparison of every saved churn model and CLV model on the
    validation split, applies the selection rules, and writes tables and a ROC
    chart to docs/. The test split is not touched.

Step B (uses the test split, once):
    python -m src.models.compare --evaluate-test

    Scores only the two selected models (best churn model and best CLV model)
    on the test split and saves docs/test_results.json. The script refuses to
    run this step a second time unless --force is given, so the test result
    cannot be tuned by repeated peeking.

Selection rules, fixed before the test split is looked at:
* Churn: highest validation ROC-AUC. Models within 0.005 AUC of the best are
  treated as tied, and the tie is broken by recall on recently active customers
  (recency <= 180 days), then by precision.
* CLV: highest validation R-squared on the original currency scale.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.models.clv_regression import evaluate as evaluate_clv
from src.models.clv_two_stage import combine, evaluate_combined
from src.models.common import (
    ROOT,
    evaluate_two_views,
    load_dataset,
    load_modeling_config,
    make_features,
    split_data,
)

logger = logging.getLogger(__name__)

CHURN_MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "knn",
]
CLV_MODELS = ["ridge", "random_forest_reg", "two_stage"]
AUC_TIE = 0.005
TARGETS = {"roc_auc": 0.80, "recall": 0.70, "clv_r2": 0.60}


def load_bundle(name: str) -> dict[str, Any]:
    """Load a saved model bundle (model, columns, ...) from models_artifacts."""
    return joblib.load(ROOT / "models_artifacts" / f"{name}.joblib")


def churn_table(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Metrics of every saved churn model on ``df`` (all and recent customers)."""
    max_rec = int(cfg["recent_recency_days"])
    full = make_features(df)
    rows = []
    for name in CHURN_MODELS:
        bundle = load_bundle(name)
        x = full[bundle["columns"]]
        views = evaluate_two_views(
            bundle["model"], x, df[cfg["churn_target"]], df["recency_days"], max_rec
        )
        for subset, metrics in views.items():
            rows.append({"model": name, "subset": subset, **metrics})
    return pd.DataFrame(rows)


def select_churn_model(table: pd.DataFrame) -> str:
    """Highest ROC-AUC; near-ties broken by recent-customer recall, then precision."""
    overall = table[table["subset"] == "all"]
    recent = table[table["subset"] != "all"].set_index("model")["recall"]
    top = overall["roc_auc"].max()
    tied = overall[overall["roc_auc"] >= top - AUC_TIE].copy()
    tied["recent_recall"] = tied["model"].map(recent).fillna(0.0)
    tied = tied.sort_values(["recent_recall", "precision"], ascending=False)
    return str(tied.iloc[0]["model"])


def predict_two_stage(df: pd.DataFrame, stage1: dict, two: dict) -> np.ndarray:
    """Expected CLV = P(active) x expected spend, for every row of ``df``."""
    full = make_features(df)
    p_churn = stage1["model"].predict_proba(full[stage1["columns"]])[:, 1]
    spend_log = two["regressor"].predict(full[two["columns"]])
    return combine(p_churn, spend_log)


def clv_scores(df: pd.DataFrame, cfg: dict[str, Any], name: str) -> dict:
    """CLV metrics of one saved model, keyed by (subset, scale)."""
    max_rec = int(cfg["recent_recency_days"])
    y, recency = df[cfg["clv_target"]], df["recency_days"]
    if name == "two_stage":
        two = load_bundle("clv_two_stage")
        pred = predict_two_stage(df, load_bundle(two["stage1"]), two)
        return evaluate_combined(pred, y, recency, max_rec)
    bundle = load_bundle(f"clv_{name}")
    x = make_features(df)[bundle["columns"]]
    return evaluate_clv(bundle["model"], x, y, recency, max_rec)


def clv_table(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Metrics of every saved CLV model on ``df``."""
    rows = []
    for name in CLV_MODELS:
        for (subset, scale), m in clv_scores(df, cfg, name).items():
            rows.append({"model": name, "subset": subset, "scale": scale, **m})
    return pd.DataFrame(rows)


def select_clv_model(table: pd.DataFrame) -> str:
    """Highest R-squared on the original scale, all customers."""
    sel = table[(table["subset"] == "all") & (table["scale"] == "original")]
    return str(sel.loc[sel["r2"].idxmax(), "model"])


def to_markdown(df: pd.DataFrame) -> str:
    """Small markdown table writer (avoids needing the tabulate package)."""
    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy()]
    return "\n".join([header, divider, *body])


def search_seconds() -> pd.Series:
    """Latest logged tuning/fit time per churn model, from the experiment log."""
    cfg = load_modeling_config()
    log = pd.read_csv(ROOT / cfg["experiment_log"])
    return log.groupby("model")["train_seconds"].last()


def save_roc_chart(val: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """ROC curves of all churn models on the validation split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = make_features(val)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in CHURN_MODELS:
        bundle = load_bundle(name)
        proba = bundle["model"].predict_proba(full[bundle["columns"]])[:, 1]
        fpr, tpr, _ = roc_curve(val[cfg["churn_target"]], proba)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set(title="Churn ROC curves (validation)", xlabel="FPR", ylabel="TPR")
    ax.legend(fontsize=8)
    fig.tight_layout()
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "roc_validation.png", dpi=150)
    plt.close(fig)


def validation_report(cfg: dict[str, Any]) -> tuple[str, str]:
    """Step A: compare models on validation data and return (churn, clv) picks."""
    df = load_dataset(cfg)
    _train, val, _test = split_data(df, cfg)

    churn = churn_table(val, cfg)
    churn["search_seconds"] = churn["model"].map(search_seconds()).round(1)
    clv = clv_table(val, cfg)
    best_churn, best_clv = select_churn_model(churn), select_clv_model(clv)

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    churn_out = churn.round(3)
    clv_out = clv.round(3)
    churn_out.to_csv(docs / "model_comparison.csv", index=False)
    clv_out.to_csv(docs / "clv_comparison.csv", index=False)
    ranked = churn_out[churn_out["subset"] == "all"].sort_values("roc_auc", ascending=False)
    md = (
        "## Churn models (validation split)\n\n"
        + to_markdown(churn_out.sort_values(["subset", "roc_auc"], ascending=False))
        + "\n\n## CLV models (validation split)\n\n"
        + to_markdown(clv_out)
        + "\n"
    )
    (docs / "model_comparison.md").write_text(md, encoding="utf-8")
    save_roc_chart(val, cfg)

    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    print("Churn models, ranked by ROC-AUC (validation, all customers)")
    print(ranked[[*cols, "search_seconds"]].to_string(index=False))
    recent = churn_out[churn_out["subset"] != "all"]
    print("\nChurn models, recently active customers only (recency <= 180)")
    print(recent[cols].to_string(index=False))
    print("\nCLV models (validation)")
    print(clv_out.to_string(index=False))
    print(f"\nSelected churn model: {best_churn}")
    print(f"Selected CLV model:   {best_clv}")
    print("Test split not used in this step.")
    return best_churn, best_clv


def test_evaluation(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Step B: score the two selected models on the test split, once."""
    path = ROOT / "docs" / "test_results.json"
    if path.exists() and not force:
        raise SystemExit(
            "docs/test_results.json already exists: the test split was already used.\n"
            "Re-testing would make the result unreliable. Use --force only if you "
            "have a real reason, and say so in the report."
        )
    best_churn, best_clv = validation_report(cfg)
    df = load_dataset(cfg)
    _train, _val, test = split_data(df, cfg)

    churn_bundle = load_bundle(best_churn)
    churn_views = evaluate_two_views(
        churn_bundle["model"],
        make_features(test)[churn_bundle["columns"]],
        test[cfg["churn_target"]],
        test["recency_days"],
        int(cfg["recent_recency_days"]),
    )
    clv_views = clv_scores(test, cfg, best_clv)

    churn_all = churn_views["all"]
    clv_all = clv_views[("all", "original")]
    checks = {
        "churn_roc_auc >= 0.80": bool(churn_all["roc_auc"] >= TARGETS["roc_auc"]),
        "churn_recall >= 0.70": bool(churn_all["recall"] >= TARGETS["recall"]),
        "clv_r2_original >= 0.60": bool(clv_all["r2"] >= TARGETS["clv_r2"]),
    }
    result = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_rows": len(test),
        "churn_model": best_churn,
        "churn_test": {k: {m: round(v, 4) for m, v in d.items()} for k, d in churn_views.items()},
        "clv_model": best_clv,
        "clv_test": {
            f"{s}|{sc}": {m: round(v, 4) for m, v in d.items()} for (s, sc), d in clv_views.items()
        },
        "targets_met": checks,
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== TEST SET (used once) ===")
    print(f"churn model: {best_churn}")
    print(
        pd.DataFrame(churn_views)
        .T.round(3)[["accuracy", "precision", "recall", "f1", "roc_auc"]]
        .to_string()
    )
    print(f"\nCLV model: {best_clv}")
    print(pd.DataFrame(clv_views).T.round(3).to_string())
    print("\nTargets:")
    for name, ok in checks.items():
        print(f"  {'MET    ' if ok else 'NOT MET'}  {name}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_modeling_config()
    if args.evaluate_test:
        test_evaluation(config, force=args.force)
    else:
        validation_report(config)
