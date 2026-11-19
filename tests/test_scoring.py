"""Tests for the scoring module — accuracy, hallucination rate, bootstrap CIs."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from earnings_bench.evals.scoring import score_extraction
from earnings_bench.schemas import (
    Currency,
    Direction,
    ExtractionRecord,
    HeadlineNumber,
    ManagementCommentary,
    Period,
    SegmentResult,
    Unit,
)


def _make_record(n_headline: int, n_segments: int) -> ExtractionRecord:
    period = Period(fiscal_year=2025, fiscal_quarter=3)
    return ExtractionRecord(
        ticker="AAPL",
        period=period,
        call_date=date(2025, 10, 30),
        headline_numbers=[
            HeadlineNumber(
                metric=f"metric_{i}",
                value=Decimal(str(10 + i)),
                currency=Currency.USD,
                unit=Unit.BILLIONS,
                period=period,
                source_quote=f"Quote for metric_{i}",
            )
            for i in range(n_headline)
        ],
        segments=[
            SegmentResult(
                segment_name=f"Seg{i}",
                metric="revenue",
                value=Decimal("5"),
                yoy_change_pct=Decimal("10"),
                direction=Direction.UP,
                period=period,
                source_quote=f"Segment {i} quote",
            )
            for i in range(n_segments)
        ],
        commentary=[
            ManagementCommentary(
                topic="ai",
                sentiment=Direction.UP,
                source_quote="We are bullish on AI",
            )
        ],
    )


def test_perfect_match() -> None:
    gt = _make_record(3, 2)
    pred = _make_record(3, 2)
    result = score_extraction(pred, gt, total_cost_usd=0.10)
    assert result.n_correct == result.n_ground_truth
    assert result.accuracy == 1.0
    assert result.hallucination_rate == 0.0


def test_partial_match() -> None:
    gt = _make_record(3, 2)
    pred = _make_record(2, 2)  # missing one headline
    result = score_extraction(pred, gt, total_cost_usd=0.10)
    assert result.n_correct == result.n_ground_truth - 1
    assert 0.5 < result.accuracy < 1.0


def test_hallucination() -> None:
    gt = _make_record(2, 1)
    pred = _make_record(4, 1)  # 2 extra fabricated headlines
    result = score_extraction(pred, gt, total_cost_usd=0.10)
    assert result.hallucination_rate > 0.0


def test_no_prediction() -> None:
    gt = _make_record(3, 2)
    result = score_extraction(None, gt, total_cost_usd=0.0)
    assert result.n_predicted == 0
    assert result.accuracy == 0.0
    assert result.hallucination_rate == 0.0
    assert result.cost_per_correct == float("inf")


def test_ci_bounds_reasonable() -> None:
    gt = _make_record(5, 3)
    pred = _make_record(4, 3)
    result = score_extraction(pred, gt, total_cost_usd=0.1, bootstrap_n=200)
    lo, hi = result.accuracy_ci
    assert 0.0 <= lo <= result.accuracy <= hi <= 1.0
