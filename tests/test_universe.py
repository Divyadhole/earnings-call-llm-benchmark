"""Tests for the 50-company universe."""

from earnings_bench.ingest.universe import get_by_sector, get_universe


def test_universe_size() -> None:
    assert len(get_universe()) == 50


def test_tickers_unique() -> None:
    tickers = [c.ticker for c in get_universe()]
    assert len(tickers) == len(set(tickers)), "duplicate tickers"


def test_tickers_uppercase() -> None:
    for c in get_universe():
        assert c.ticker == c.ticker.upper()


def test_sector_coverage() -> None:
    by_sector = get_by_sector()
    # Make sure we have cross-sector coverage — at least 6 distinct sectors
    assert len(by_sector) >= 6
