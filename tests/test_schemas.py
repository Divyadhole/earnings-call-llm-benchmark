"""Unit + property-based tests on the extraction schema."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from earnings_bench.schemas import (
    Currency,
    Direction,
    ExtractionRecord,
    GuidanceRange,
    HeadlineNumber,
    ManagementCommentary,
    Period,
    SegmentResult,
    Unit,
)


def test_period_str() -> None:
    p = Period(fiscal_year=2025, fiscal_quarter=3)
    assert str(p) == "Q3 FY2025"


def test_period_quarter_bounds() -> None:
    with pytest.raises(ValidationError):
        Period(fiscal_year=2025, fiscal_quarter=5)
    with pytest.raises(ValidationError):
        Period(fiscal_year=2025, fiscal_quarter=0)


def test_ticker_uppercased() -> None:
    r = ExtractionRecord(
        ticker="aapl",
        period=Period(fiscal_year=2025, fiscal_quarter=3),
        call_date=date(2025, 10, 30),
    )
    assert r.ticker == "AAPL"


def test_metric_lowercased_and_spaces_replaced() -> None:
    h = HeadlineNumber(
        metric="Net Income",
        value=Decimal("20"),
        period=Period(fiscal_year=2025, fiscal_quarter=3),
        source_quote="x",
    )
    assert h.metric == "net_income"


def test_headline_valid() -> None:
    h = HeadlineNumber(
        metric="revenue",
        value=Decimal("94.9"),
        currency=Currency.USD,
        unit=Unit.BILLIONS,
        period=Period(fiscal_year=2025, fiscal_quarter=3),
        is_gaap=True,
        source_quote="Revenue was $94.9 billion",
    )
    assert h.value == Decimal("94.9")


def test_guidance_range() -> None:
    g = GuidanceRange(
        metric="revenue",
        low=Decimal("95"),
        high=Decimal("97"),
        unit=Unit.BILLIONS,
        currency=Currency.USD,
        target_period=Period(fiscal_year=2025, fiscal_quarter=4),
        hedge_phrases=["we expect", "approximately"],
        source_quote="We expect Q4 revenue of approximately $95-97B",
    )
    assert g.low == Decimal("95")
    assert "we expect" in g.hedge_phrases


def test_commentary_direction_enum() -> None:
    c = ManagementCommentary(
        topic="ai",
        sentiment=Direction.UP,
        speaker_role="CEO",
        source_quote="We're seeing strong momentum in AI",
    )
    assert c.sentiment == Direction.UP


def test_segment_result() -> None:
    s = SegmentResult(
        segment_name="Services",
        metric="revenue",
        value=Decimal("25"),
        yoy_change_pct=Decimal("14"),
        direction=Direction.UP,
        period=Period(fiscal_year=2025, fiscal_quarter=3),
        source_quote="Services revenue grew 14% YoY",
    )
    assert s.yoy_change_pct == Decimal("14")


@given(
    year=st.integers(min_value=2000, max_value=2100),
    quarter=st.integers(min_value=1, max_value=4),
)
def test_period_roundtrip(year: int, quarter: int) -> None:
    p = Period(fiscal_year=year, fiscal_quarter=quarter)
    assert p.fiscal_year == year
    assert p.fiscal_quarter == quarter


@given(
    ticker=st.text(
        alphabet=st.characters(min_codepoint=ord("A"), max_codepoint=ord("Z")),
        min_size=1,
        max_size=10,
    )
)
def test_ticker_arbitrary_case(ticker: str) -> None:
    r = ExtractionRecord(
        ticker=ticker.lower(),
        period=Period(fiscal_year=2025, fiscal_quarter=1),
        call_date=date(2025, 1, 1),
    )
    assert r.ticker == ticker.upper()
