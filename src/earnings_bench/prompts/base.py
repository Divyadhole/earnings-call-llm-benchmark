"""Shared prompt-strategy protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod

from earnings_bench.schemas import ExtractionRecord

SYSTEM_PROMPT = """You extract structured financial facts from earnings call transcripts.

You MUST return a JSON object conforming to the ExtractionRecord schema:
- ticker (string, uppercase)
- period: {fiscal_year: int, fiscal_quarter: int}
- call_date (ISO date)
- headline_numbers: list of {metric, value, currency, unit, period, is_gaap, source_quote}
- guidance: list of {metric, low, high, point, currency, unit, target_period, hedge_phrases, source_quote}
- segments: list of {segment_name, metric, value, yoy_change_pct, direction, period, source_quote}
- commentary: list of {topic, sentiment (up|down|flat), speaker_role, source_quote}

Hard rules:
- Every row must have a source_quote that appears VERBATIM in the transcript.
- Do not fabricate numbers. If a value is not stated, omit the row.
- Numbers stay in the units stated on the call (billions, millions, absolute).
- GAAP vs non-GAAP: fill is_gaap only if the speaker says so.
"""


class PromptStrategy(ABC):
    """A strategy that turns a transcript into an LLM request."""

    name: str

    @abstractmethod
    def build_messages(self, transcript_text: str, ticker: str) -> list[dict[str, str]]:
        """Return the messages array for the chat completion request."""

    def parse(self, raw_json: str, ticker: str, call_date) -> ExtractionRecord:
        """Parse the model's JSON response. Default impl assumes the model
        returned a valid ExtractionRecord JSON object."""
        return ExtractionRecord.model_validate_json(raw_json)
