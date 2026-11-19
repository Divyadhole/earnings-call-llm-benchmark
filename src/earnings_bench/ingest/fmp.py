"""Financial Modeling Prep transcript fetcher.

Docs: https://site.financialmodelingprep.com/developer/docs#earnings-transcripts

Free tier is ~250 requests/day, which is enough for the W2 scope (500 transcripts
amortized across a few days of pulls).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from earnings_bench.config import FMPConfig


@dataclass(frozen=True)
class FetchedTranscript:
    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    call_date: date
    raw_text: str
    source: str = "fmp"


class FMPClient:
    """Thin wrapper around the FMP earnings-transcript endpoints."""

    def __init__(self, config: FMPConfig | None = None, timeout: float = 30.0) -> None:
        self.config = config or FMPConfig.from_env()
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> FMPClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def fetch_transcript(
        self,
        ticker: str,
        fiscal_year: int,
        fiscal_quarter: int,
    ) -> FetchedTranscript | None:
        """Fetch one earnings-call transcript. Returns None if FMP has no record."""
        url = f"{self.config.base_url}/earning_call_transcript/{ticker.upper()}"
        params = {
            "year": fiscal_year,
            "quarter": fiscal_quarter,
            "apikey": self.config.api_key,
        }
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
        if not payload:
            return None
        record = payload[0] if isinstance(payload, list) else payload
        return FetchedTranscript(
            ticker=ticker.upper(),
            fiscal_year=int(record.get("year", fiscal_year)),
            fiscal_quarter=int(record.get("quarter", fiscal_quarter)),
            call_date=datetime.fromisoformat(record["date"].replace("Z", "+00:00")).date()
            if record.get("date")
            else date.today(),
            raw_text=record.get("content", "").strip(),
        )

    def fetch_batch(
        self,
        requests: list[tuple[str, int, int]],
    ) -> list[FetchedTranscript]:
        """Fetch many (ticker, year, quarter) triples. Skips missing records."""
        out: list[FetchedTranscript] = []
        for ticker, year, quarter in requests:
            t = self.fetch_transcript(ticker, year, quarter)
            if t is not None:
                out.append(t)
        return out
