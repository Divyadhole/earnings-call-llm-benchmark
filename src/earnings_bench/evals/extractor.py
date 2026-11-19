"""Run one (model, prompt_strategy, transcript) → ExtractionRecord."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

try:
    from litellm import completion
except ImportError:
    completion = None  # type: ignore[assignment]

from earnings_bench.models.registry import ModelSpec, cost_usd
from earnings_bench.prompts.base import PromptStrategy
from earnings_bench.schemas import ExtractionRecord


@dataclass
class ExtractionResult:
    model_slug: str
    prompt_strategy: str
    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    record: ExtractionRecord | None
    raw_response: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def run_extraction(
    model: ModelSpec,
    strategy: PromptStrategy,
    transcript_text: str,
    ticker: str,
    fiscal_year: int,
    fiscal_quarter: int,
    call_date: date,
) -> ExtractionResult:
    """Call one LLM once with one prompt strategy. Returns a typed result."""
    if completion is None:
        raise ImportError("Install the `llm` extra to get litellm")

    messages = strategy.build_messages(transcript_text, ticker)
    start = time.perf_counter()
    error: str | None = None
    record: ExtractionRecord | None = None
    raw: str = ""
    in_tok = out_tok = 0

    try:
        resp = completion(
            model=model.id,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        if usage:
            in_tok = usage.prompt_tokens
            out_tok = usage.completion_tokens
        record = strategy.parse(raw, ticker, call_date)
    except Exception as e:  # noqa: BLE001 — surface any model/parse error
        error = f"{type(e).__name__}: {e}"

    latency_ms = (time.perf_counter() - start) * 1000.0
    return ExtractionResult(
        model_slug=model.display_name,
        prompt_strategy=strategy.name,
        ticker=ticker,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        record=record,
        raw_response=raw,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost_usd(model, in_tok, out_tok),
        latency_ms=latency_ms,
        error=error,
    )
