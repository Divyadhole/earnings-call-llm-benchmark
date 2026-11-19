"""Feature engineering for the three ML models.

Pulls from fct_extraction_ground_truth and derives per-call feature rows:
    (ticker, fiscal_year, fiscal_quarter) → {
        revenue_beat,        # reported revenue - consensus estimate
        eps_beat,
        guidance_change_pct, # new guidance vs prior
        n_guidance_items,
        hedge_density,       # hedge_phrases / total guidance words
        n_segment_ups,
        n_segment_downs,
        n_macro_mentions,
        sentiment_pos_minus_neg,
    }
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MACRO_TOPICS = {"china", "macro", "fx", "supply_chain", "labor", "regulatory"}


@dataclass
class FeatureRow:
    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    revenue_beat_pct: float | None
    eps_beat_pct: float | None
    n_guidance_items: int
    hedge_density: float
    n_segment_ups: int
    n_segment_downs: int
    n_macro_mentions: int
    sentiment_balance: int


def compute_features(facts: pd.DataFrame, consensus: pd.DataFrame | None = None) -> pd.DataFrame:
    """Derive the feature matrix from the ground-truth fact table.

    facts: rows from fct_extraction_ground_truth (long format)
    consensus: optional (ticker, fy, fq, metric, estimate) table for beat/miss
    """
    rows: list[FeatureRow] = []
    group_cols = ["ticker", "fiscal_year", "fiscal_quarter"]
    for (ticker, fy, fq), group in facts.groupby(group_cols):
        headline = group[group["record_type"] == "headline"]
        guidance = group[group["record_type"] == "guidance"]
        segments = group[group["record_type"] == "segment"]
        commentary = group[group["record_type"] == "commentary"]

        rev_beat = _beat_pct(headline, consensus, ticker, fy, fq, "revenue")
        eps_beat = _beat_pct(headline, consensus, ticker, fy, fq, "eps_diluted")

        hedge_words = sum(len(str(p).split("|")) for p in guidance["hedge_phrases"].dropna())
        total_guidance_tokens = sum(len(str(q).split()) for q in guidance["source_quote"].dropna()) or 1

        rows.append(
            FeatureRow(
                ticker=ticker,
                fiscal_year=int(fy),
                fiscal_quarter=int(fq),
                revenue_beat_pct=rev_beat,
                eps_beat_pct=eps_beat,
                n_guidance_items=len(guidance),
                hedge_density=hedge_words / total_guidance_tokens,
                n_segment_ups=int((segments["direction"] == "up").sum()),
                n_segment_downs=int((segments["direction"] == "down").sum()),
                n_macro_mentions=int(commentary["metric_or_topic"].isin(MACRO_TOPICS).sum()),
                sentiment_balance=int(
                    (commentary["direction"] == "up").sum()
                    - (commentary["direction"] == "down").sum()
                ),
            )
        )
    return pd.DataFrame([r.__dict__ for r in rows])


def _beat_pct(
    headline: pd.DataFrame,
    consensus: pd.DataFrame | None,
    ticker: str,
    fy: int,
    fq: int,
    metric: str,
) -> float | None:
    if consensus is None:
        return None
    hit = headline[headline["metric_or_topic"] == metric]
    est = consensus[
        (consensus["ticker"] == ticker)
        & (consensus["fiscal_year"] == fy)
        & (consensus["fiscal_quarter"] == fq)
        & (consensus["metric"] == metric)
    ]
    if hit.empty or est.empty:
        return None
    reported = float(hit["value"].iloc[0])
    estimate = float(est["estimate"].iloc[0])
    if estimate == 0:
        return None
    return (reported - estimate) / abs(estimate) * 100.0
