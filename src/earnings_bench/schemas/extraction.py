"""Pydantic schemas for financial-fact extraction from earnings call transcripts.

These schemas define the ground-truth contract for Tier-A hand reconciliation
and are reused by every model in the benchmark (W2+) to enforce JSON validity.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Period(BaseModel):
    """A fiscal period — e.g. Q3 FY2025."""

    model_config = ConfigDict(frozen=True)

    fiscal_year: int = Field(..., ge=2000, le=2100)
    fiscal_quarter: int = Field(..., ge=1, le=4)
    period_end: date | None = None

    def __str__(self) -> str:
        return f"Q{self.fiscal_quarter} FY{self.fiscal_year}"


class Currency(StrEnum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"


class Unit(StrEnum):
    ABSOLUTE = "absolute"
    THOUSANDS = "thousands"
    MILLIONS = "millions"
    BILLIONS = "billions"


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class HeadlineNumber(BaseModel):
    """A single reported financial figure — e.g. 'Q3 revenue of $94.9B'."""

    metric: str = Field(..., description="e.g. 'revenue', 'eps_diluted', 'net_income'")
    value: Decimal
    currency: Currency | None = None
    unit: Unit = Unit.ABSOLUTE
    period: Period
    is_gaap: bool | None = None
    source_quote: str = Field(..., min_length=1, max_length=2000)

    @field_validator("metric")
    @classmethod
    def _lowercase_metric(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_")


class GuidanceRange(BaseModel):
    """Forward guidance — e.g. 'We expect Q4 revenue of $95-97B'."""

    metric: str
    low: Decimal | None = None
    high: Decimal | None = None
    point: Decimal | None = None
    currency: Currency | None = None
    unit: Unit = Unit.ABSOLUTE
    target_period: Period
    hedge_phrases: list[str] = Field(default_factory=list)
    source_quote: str

    @field_validator("metric")
    @classmethod
    def _lowercase_metric(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_")


class SegmentResult(BaseModel):
    """Per-segment result — e.g. 'Services revenue grew 14% YoY'."""

    segment_name: str
    metric: str
    value: Decimal | None = None
    yoy_change_pct: Decimal | None = None
    direction: Direction | None = None
    period: Period
    source_quote: str


class ManagementCommentary(BaseModel):
    """A classified piece of management commentary."""

    topic: str = Field(..., description="e.g. 'ai', 'china', 'supply_chain', 'margins'")
    sentiment: Direction
    speaker_role: str | None = None  # CEO, CFO, COO, …
    source_quote: str


class ExtractionRecord(BaseModel):
    """The complete extraction for one (company, period) earnings call."""

    ticker: str = Field(..., min_length=1, max_length=10)
    period: Period
    call_date: date
    headline_numbers: list[HeadlineNumber] = Field(default_factory=list)
    guidance: list[GuidanceRange] = Field(default_factory=list)
    segments: list[SegmentResult] = Field(default_factory=list)
    commentary: list[ManagementCommentary] = Field(default_factory=list)

    @field_validator("ticker")
    @classmethod
    def _uppercase_ticker(cls, v: str) -> str:
        return v.strip().upper()
