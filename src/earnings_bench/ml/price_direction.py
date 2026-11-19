"""Model 1 — Post-earnings price-direction classifier.

Target: next-day stock return sign (up / down / flat) after the earnings call.
Features: engineered from fct_extraction_ground_truth via earnings_bench.ml.features.
Model: XGBoost with scikit-learn pipeline; logistic baseline for comparison.
Evaluation: time-based walk-forward CV. AUC, F1, calibration curve.
Interpretability: SHAP summary + dependence plots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import shap
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
except ImportError:
    LogisticRegression = XGBClassifier = Pipeline = StandardScaler = None  # type: ignore[assignment]
    TimeSeriesSplit = f1_score = roc_auc_score = shap = None  # type: ignore[assignment]

from earnings_bench.ml.mlflow_utils import configure_mlflow, start_run

FEATURE_COLS = [
    "revenue_beat_pct",
    "eps_beat_pct",
    "n_guidance_items",
    "hedge_density",
    "n_segment_ups",
    "n_segment_downs",
    "n_macro_mentions",
    "sentiment_balance",
]


@dataclass
class PriceDirectionReport:
    model: str
    folds: int
    mean_auc: float
    mean_f1: float
    per_fold_auc: list[float]
    per_fold_f1: list[float]


def _prepare_xy(features: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    merged = features.merge(labels, on=["ticker", "fiscal_year", "fiscal_quarter"])
    merged = merged.dropna(subset=FEATURE_COLS + ["direction_label"])
    merged = merged.sort_values(["fiscal_year", "fiscal_quarter", "ticker"]).reset_index(drop=True)
    X = merged[FEATURE_COLS]
    y = (merged["direction_label"] == "up").astype(int)  # binary up vs not-up
    return X, y


def train_walk_forward(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    model_name: str = "xgboost",
    n_splits: int = 5,
    seed: int = 42,
) -> PriceDirectionReport:
    """Walk-forward time-based CV. Logs metrics + model to MLflow."""
    if XGBClassifier is None:
        raise ImportError("Install the `ml` extras: scikit-learn, xgboost, shap, mlflow")

    X, y = _prepare_xy(features, labels)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs: list[float] = []
    f1s: list[float] = []

    configure_mlflow()
    with start_run(run_name=f"price_direction_{model_name}", model_tag=model_name) as run:
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            pipe = _build_pipeline(model_name, seed)
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            auc = roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else float("nan")
            f1 = f1_score(y_te, pred, zero_division=0)
            aucs.append(auc)
            f1s.append(f1)
            _log_fold_metrics(i, auc, f1)

        _log_final_metrics(aucs, f1s)

        # Refit on everything and log a SHAP summary
        final = _build_pipeline(model_name, seed)
        final.fit(X, y)
        _log_shap_summary(final, X, model_name)

        import mlflow

        mlflow.sklearn.log_model(final, name="model")

    return PriceDirectionReport(
        model=model_name,
        folds=n_splits,
        mean_auc=float(np.nanmean(aucs)),
        mean_f1=float(np.nanmean(f1s)),
        per_fold_auc=aucs,
        per_fold_f1=f1s,
    )


def _build_pipeline(model_name: str, seed: int) -> Pipeline:
    if model_name == "xgboost":
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=seed,
            eval_metric="logloss",
        )
    elif model_name == "logistic":
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _log_fold_metrics(fold: int, auc: float, f1: float) -> None:
    import mlflow

    mlflow.log_metric(f"fold_{fold}_auc", auc)
    mlflow.log_metric(f"fold_{fold}_f1", f1)


def _log_final_metrics(aucs: list[float], f1s: list[float]) -> None:
    import mlflow

    mlflow.log_metric("mean_auc", float(np.nanmean(aucs)))
    mlflow.log_metric("mean_f1", float(np.nanmean(f1s)))


def _log_shap_summary(pipe: Pipeline, X: pd.DataFrame, model_name: str) -> None:
    if model_name != "xgboost":
        return
    import mlflow

    clf = pipe.named_steps["clf"]
    scaled = pipe.named_steps["scaler"].transform(X)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(scaled)
    mean_abs = np.abs(shap_values).mean(axis=0)
    for name, val in zip(FEATURE_COLS, mean_abs, strict=True):
        mlflow.log_metric(f"shap_mean_abs_{name}", float(val))
