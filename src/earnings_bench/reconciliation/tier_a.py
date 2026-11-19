"""Tier-A hand reconciliation — the ground-truth backbone of the benchmark.

Tier A is a hand-reconciled subset (20 of 50 transcripts in W1, grown to 50 in W2)
where every headline number, guidance item, segment result, and classified
commentary passage has been manually extracted and verified against the original
10-Q / 10-K filing. This is what every LLM is graded against.

Workflow:
    1. Export a blank reconciliation CSV per (ticker, period) with
       `make_reconciliation_template`.
    2. Fill it in by hand while reading the transcript side-by-side with the
       10-Q filing.
    3. Validate the filled CSV with `load_reconciliation` — it round-trips through
       Pydantic to catch typos, unit mismatches, and malformed quotes.
    4. Run `dbt seed` + `dbt run` to materialize `fct_extraction_ground_truth`.
"""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

from earnings_bench.schemas import (
    ExtractionRecord,
    GuidanceRange,
    HeadlineNumber,
    ManagementCommentary,
    Period,
    SegmentResult,
)

GROUND_TRUTH_DIR = Path("data/ground_truth")
TIER_A_COLUMNS = [
    "record_type",  # headline | guidance | segment | commentary
    "ticker",
    "fiscal_year",
    "fiscal_quarter",
    "metric_or_topic",
    "value",
    "value_low",
    "value_high",
    "currency",
    "unit",
    "yoy_change_pct",
    "direction",
    "segment_name",
    "speaker_role",
    "is_gaap",
    "hedge_phrases",
    "source_quote",
]


def make_reconciliation_template(
    ticker: str,
    period: Period,
    out_dir: Path = GROUND_TRUTH_DIR,
) -> Path:
    """Write a blank reconciliation CSV for one (ticker, period) call."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}_{period.fiscal_year}_Q{period.fiscal_quarter}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TIER_A_COLUMNS)
        # One example row per record_type as scaffolding
        writer.writerow(
            [
                "headline",
                ticker,
                period.fiscal_year,
                period.fiscal_quarter,
                "revenue",
                "",
                "",
                "",
                "USD",
                "billions",
                "",
                "",
                "",
                "",
                "true",
                "",
                "",
            ]
        )
    return path


def _row_to_headline(row: dict[str, str]) -> HeadlineNumber:
    return HeadlineNumber(
        metric=row["metric_or_topic"],
        value=Decimal(row["value"]),
        currency=row["currency"] or None,
        unit=row["unit"] or "absolute",
        period=Period(
            fiscal_year=int(row["fiscal_year"]),
            fiscal_quarter=int(row["fiscal_quarter"]),
        ),
        is_gaap=_parse_bool(row.get("is_gaap")),
        source_quote=row["source_quote"],
    )


def _row_to_guidance(row: dict[str, str]) -> GuidanceRange:
    return GuidanceRange(
        metric=row["metric_or_topic"],
        low=Decimal(row["value_low"]) if row.get("value_low") else None,
        high=Decimal(row["value_high"]) if row.get("value_high") else None,
        point=Decimal(row["value"]) if row.get("value") else None,
        currency=row["currency"] or None,
        unit=row["unit"] or "absolute",
        target_period=Period(
            fiscal_year=int(row["fiscal_year"]),
            fiscal_quarter=int(row["fiscal_quarter"]),
        ),
        hedge_phrases=[p.strip() for p in (row.get("hedge_phrases") or "").split("|") if p.strip()],
        source_quote=row["source_quote"],
    )


def _row_to_segment(row: dict[str, str]) -> SegmentResult:
    return SegmentResult(
        segment_name=row["segment_name"],
        metric=row["metric_or_topic"],
        value=Decimal(row["value"]) if row.get("value") else None,
        yoy_change_pct=Decimal(row["yoy_change_pct"]) if row.get("yoy_change_pct") else None,
        direction=row.get("direction") or None,
        period=Period(
            fiscal_year=int(row["fiscal_year"]),
            fiscal_quarter=int(row["fiscal_quarter"]),
        ),
        source_quote=row["source_quote"],
    )


def _row_to_commentary(row: dict[str, str]) -> ManagementCommentary:
    return ManagementCommentary(
        topic=row["metric_or_topic"],
        sentiment=row["direction"],
        speaker_role=row.get("speaker_role") or None,
        source_quote=row["source_quote"],
    )


def _parse_bool(v: str | None) -> bool | None:
    if not v:
        return None
    return v.strip().lower() in {"true", "yes", "1", "y"}


def load_reconciliation(path: Path, ticker: str, period: Period, call_date) -> ExtractionRecord:
    """Validate a filled reconciliation CSV and return a typed ExtractionRecord."""
    headline: list[HeadlineNumber] = []
    guidance: list[GuidanceRange] = []
    segments: list[SegmentResult] = []
    commentary: list[ManagementCommentary] = []

    with path.open() as f:
        for row in csv.DictReader(f):
            # Skip rows the reviewer left blank (template scaffolding)
            if not row.get("source_quote"):
                continue
            rt = row["record_type"].strip().lower()
            if rt == "headline":
                headline.append(_row_to_headline(row))
            elif rt == "guidance":
                guidance.append(_row_to_guidance(row))
            elif rt == "segment":
                segments.append(_row_to_segment(row))
            elif rt == "commentary":
                commentary.append(_row_to_commentary(row))
            else:
                raise ValueError(f"Unknown record_type: {rt!r} in {path}")

    return ExtractionRecord(
        ticker=ticker,
        period=period,
        call_date=call_date,
        headline_numbers=headline,
        guidance=guidance,
        segments=segments,
        commentary=commentary,
    )
