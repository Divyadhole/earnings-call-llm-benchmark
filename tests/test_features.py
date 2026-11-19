"""Tests for ML feature engineering."""

import pandas as pd

from earnings_bench.ml.features import compute_features


def _facts_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL", "fiscal_year": 2025, "fiscal_quarter": 3,
                "record_type": "headline", "metric_or_topic": "revenue", "value": 94.9,
                "direction": None, "hedge_phrases": None, "source_quote": "Revenue was $94.9B",
            },
            {
                "ticker": "AAPL", "fiscal_year": 2025, "fiscal_quarter": 3,
                "record_type": "guidance", "metric_or_topic": "revenue", "value": 95.0,
                "direction": None, "hedge_phrases": "we expect|approximately",
                "source_quote": "We expect approximately $95B",
            },
            {
                "ticker": "AAPL", "fiscal_year": 2025, "fiscal_quarter": 3,
                "record_type": "segment", "metric_or_topic": "revenue", "value": 25.0,
                "direction": "up", "hedge_phrases": None, "source_quote": "Services grew 14%",
            },
            {
                "ticker": "AAPL", "fiscal_year": 2025, "fiscal_quarter": 3,
                "record_type": "commentary", "metric_or_topic": "ai", "value": None,
                "direction": "up", "hedge_phrases": None, "source_quote": "Bullish on AI",
            },
            {
                "ticker": "AAPL", "fiscal_year": 2025, "fiscal_quarter": 3,
                "record_type": "commentary", "metric_or_topic": "china", "value": None,
                "direction": "down", "hedge_phrases": None, "source_quote": "China softness",
            },
        ]
    )


def test_compute_features_shape() -> None:
    features = compute_features(_facts_df())
    assert len(features) == 1
    row = features.iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["n_guidance_items"] == 1
    assert row["n_segment_ups"] == 1
    assert row["n_segment_downs"] == 0
    assert row["n_macro_mentions"] == 1   # 'china' is macro
    assert row["sentiment_balance"] == 0  # 1 up - 1 down


def test_hedge_density_positive() -> None:
    features = compute_features(_facts_df())
    assert features.iloc[0]["hedge_density"] > 0
