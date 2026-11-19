"""Model 2 — Guidance-hedge intensity scorer.

Task: regress the hedge density of guidance language against subsequent actual
results (how much the company under- or over-shot the guidance midpoint).

Features: linguistic features from guidance passages (hedge-word counts,
modal-verb counts, subjectivity scores from VADER + a custom financial lexicon).

Model: ridge regression — interpretable linear baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    Ridge = TimeSeriesSplit = Pipeline = StandardScaler = None  # type: ignore[assignment]
    mean_absolute_error = r2_score = None  # type: ignore[assignment]

from earnings_bench.ml.mlflow_utils import start_run

HEDGE_WORDS = {
    "approximately", "about", "roughly", "around", "near", "nearly",
    "we expect", "we anticipate", "we believe", "we think", "we project",
    "likely", "possibly", "potentially", "could", "may", "might", "should",
    "assuming", "subject to", "depending on", "range", "between",
    "modest", "measured", "cautious",
}
MODAL_VERBS = {"could", "may", "might", "should", "would", "can"}


@dataclass
class HedgeReport:
    folds: int
    mean_mae: float
    mean_r2: float
    coefficients: dict[str, float]


def extract_features(guidance_rows: pd.DataFrame) -> pd.DataFrame:
    """One feature row per (ticker, fy, fq)."""
    rows = []
    for (ticker, fy, fq), g in guidance_rows.groupby(["ticker", "fiscal_year", "fiscal_quarter"]):
        text = " ".join(str(q) for q in g["source_quote"].dropna()).lower()
        words = re.findall(r"[a-z']+", text)
        total = max(len(words), 1)
        hedge_count = sum(1 for phrase in HEDGE_WORDS if phrase in text)
        modal_count = sum(1 for w in words if w in MODAL_VERBS)
        range_width = _mean_range_width(g)
        rows.append(
            {
                "ticker": ticker,
                "fiscal_year": int(fy),
                "fiscal_quarter": int(fq),
                "hedge_word_density": hedge_count / total,
                "modal_density": modal_count / total,
                "mean_guidance_range_width": range_width,
                "n_guidance_items": len(g),
            }
        )
    return pd.DataFrame(rows)


def _mean_range_width(g: pd.DataFrame) -> float:
    widths = []
    for _, row in g.iterrows():
        low, high = row.get("value_low"), row.get("value_high")
        if pd.notna(low) and pd.notna(high) and float(low) != 0:
            widths.append(abs(float(high) - float(low)) / abs(float(low)))
    return float(np.mean(widths)) if widths else 0.0


def train_hedge_regression(
    features: pd.DataFrame,
    actuals_vs_guidance: pd.DataFrame,
    n_splits: int = 5,
    alpha: float = 1.0,
    seed: int = 42,
) -> HedgeReport:
    """Regress guidance-miss magnitude on hedge features. Logs to MLflow."""
    if Ridge is None:
        raise ImportError("Install the `ml` extras: scikit-learn, mlflow")

    merged = features.merge(actuals_vs_guidance, on=["ticker", "fiscal_year", "fiscal_quarter"])
    merged = merged.dropna(subset=["miss_magnitude"])
    merged = merged.sort_values(["fiscal_year", "fiscal_quarter", "ticker"]).reset_index(drop=True)

    feature_cols = ["hedge_word_density", "modal_density", "mean_guidance_range_width", "n_guidance_items"]
    X = merged[feature_cols]
    y = merged["miss_magnitude"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes: list[float] = []
    r2s: list[float] = []

    with start_run(run_name="hedge_intensity_ridge", model_tag="ridge"):
        for i, (tr, te) in enumerate(tscv.split(X)):
            pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=seed))])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[te])
            mae = float(mean_absolute_error(y.iloc[te], pred))
            r2 = float(r2_score(y.iloc[te], pred))
            maes.append(mae)
            r2s.append(r2)
            import mlflow
            mlflow.log_metric(f"fold_{i}_mae", mae)
            mlflow.log_metric(f"fold_{i}_r2", r2)

        final = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=seed))])
        final.fit(X, y)
        coefs = dict(zip(feature_cols, final.named_steps["ridge"].coef_, strict=True))

        import mlflow
        mlflow.log_metric("mean_mae", float(np.mean(maes)))
        mlflow.log_metric("mean_r2", float(np.mean(r2s)))
        for k, v in coefs.items():
            mlflow.log_metric(f"coef_{k}", float(v))

    return HedgeReport(
        folds=n_splits,
        mean_mae=float(np.mean(maes)),
        mean_r2=float(np.mean(r2s)),
        coefficients=coefs,
    )
