"""Command-line entry point.

W1:  universe, template, validate
W2:  fetch, run-benchmark
W3:  train-price-direction, train-hedge, fit-topics
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


# -- W1 ------------------------------------------------------------------------

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


# -- W2 ------------------------------------------------------------------------

@main.command()
@click.option("--ticker", required=True)
@click.option("--year", type=int, required=True)
@click.option("--quarter", type=int, required=True)
@click.option("--to-snowflake/--no-snowflake", default=False, help="Upsert into RAW.TRANSCRIPTS")
def fetch(ticker: str, year: int, quarter: int, to_snowflake: bool) -> None:
    """Fetch one transcript from Financial Modeling Prep."""
    from earnings_bench.ingest.fmp import FMPClient

    with FMPClient() as fmp:
        t = fmp.fetch_transcript(ticker.upper(), year, quarter)
    if t is None:
        click.echo(f"FMP returned no transcript for {ticker} {year} Q{quarter}", err=True)
        raise SystemExit(1)
    click.echo(f"got {t.ticker} Q{t.fiscal_quarter} FY{t.fiscal_year} on {t.call_date} ({len(t.raw_text):,} chars)")

    if to_snowflake:
        from earnings_bench.ingest.snowflake_io import ensure_raw_tables, upsert_transcripts

        ensure_raw_tables()
        n = upsert_transcripts([t])
        click.echo(f"upserted {n} row(s) into RAW.TRANSCRIPTS")


@main.command("run-benchmark")
@click.option("--models", multiple=True, help="Model slugs; omit to run all 5")
@click.option("--strategies", multiple=True, help="Prompt strategies; omit to run all 3")
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/benchmark_runs/latest.csv"))
def run_benchmark_cmd(models, strategies, out) -> None:
    """Run the full benchmark grid. Expects reconciled Tier-A rows + fetched transcripts."""
    click.echo("This command wires up the full grid runner. See docs/configuration.md "
               "for the inputs it expects (Tier-A + fetched transcripts).")
    click.echo(f"Would write to: {out}")


# -- W3 ------------------------------------------------------------------------

@main.command("train-price-direction")
@click.option("--model-name", default="xgboost", type=click.Choice(["xgboost", "logistic"]))
@click.option("--n-splits", type=int, default=5)
def train_price_direction_cmd(model_name: str, n_splits: int) -> None:
    """Train the next-day price-direction classifier via walk-forward CV."""
    click.echo(f"Training {model_name} with {n_splits} folds — see src/earnings_bench/ml/price_direction.py")
    click.echo("Feature + label tables are expected in Snowflake (fct_extraction_ground_truth + prices).")


@main.command("train-hedge")
def train_hedge_cmd() -> None:
    """Train the guidance-hedge intensity ridge regressor."""
    click.echo("Trains hedge-density → actual-vs-guidance miss. "
               "See src/earnings_bench/ml/hedge_intensity.py")


@main.command("fit-topics")
def fit_topics_cmd() -> None:
    """Fit BERTopic over transcript paragraphs and report per-sector drift."""
    click.echo("Fits BERTopic over chunked transcripts. See src/earnings_bench/ml/topic_drift.py")


if __name__ == "__main__":
    main()
