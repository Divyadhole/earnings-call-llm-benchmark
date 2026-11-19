"""Command-line entry point.

Usage:
    earnings-bench universe          # print the 50-company universe
    earnings-bench template --ticker AAPL --year 2025 --quarter 3
    earnings-bench validate PATH     # round-trip a filled reconciliation CSV
"""

from __future__ import annotations

from pathlib import Path

import click

from earnings_bench.ingest.universe import get_by_sector, get_universe
from earnings_bench.reconciliation.tier_a import (
    load_reconciliation,
    make_reconciliation_template,
)
from earnings_bench.schemas import Period


@click.group()
def main() -> None:
    """Earnings Call LLM Benchmark CLI."""


@main.command()
def universe() -> None:
    """Print the 50-company universe grouped by sector."""
    by_sector = get_by_sector()
    for sector, companies in sorted(by_sector.items()):
        click.echo(f"\n{sector} ({len(companies)})")
        for c in companies:
            click.echo(f"  {c.ticker:6s}  {c.name}")
    click.echo(f"\nTotal: {len(get_universe())} companies")


@main.command()
@click.option("--ticker", required=True)
@click.option("--year", type=int, required=True)
@click.option("--quarter", type=int, required=True)
def template(ticker: str, year: int, quarter: int) -> None:
    """Emit a blank Tier-A reconciliation CSV."""
    period = Period(fiscal_year=year, fiscal_quarter=quarter)
    path = make_reconciliation_template(ticker.upper(), period)
    click.echo(f"wrote {path}")


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--ticker", required=True)
@click.option("--year", type=int, required=True)
@click.option("--quarter", type=int, required=True)
@click.option("--call-date", type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
def validate(path: Path, ticker: str, year: int, quarter: int, call_date) -> None:
    """Validate a filled reconciliation CSV through Pydantic."""
    period = Period(fiscal_year=year, fiscal_quarter=quarter)
    record = load_reconciliation(path, ticker.upper(), period, call_date.date())
    click.echo(
        f"OK — {record.ticker} {record.period}: "
        f"{len(record.headline_numbers)} headline, "
        f"{len(record.guidance)} guidance, "
        f"{len(record.segments)} segments, "
        f"{len(record.commentary)} commentary"
    )


if __name__ == "__main__":
    main()
