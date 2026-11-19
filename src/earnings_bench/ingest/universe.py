"""The 50-company universe for the v2 benchmark.

Selected to cover sector diversity (tech, finance, retail, healthcare, energy,
industrials, consumer) and a mix of mega-cap and mid-cap names, with consistent
quarterly reporting over the last 10 quarters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    sector: str
    cik: str | None = None


UNIVERSE: list[Company] = [
    # Technology — mega cap
    Company("AAPL", "Apple Inc.", "Technology"),
    Company("MSFT", "Microsoft Corporation", "Technology"),
    Company("GOOGL", "Alphabet Inc.", "Technology"),
    Company("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
    Company("META", "Meta Platforms Inc.", "Communication Services"),
    Company("NVDA", "NVIDIA Corporation", "Technology"),
    Company("TSLA", "Tesla Inc.", "Consumer Discretionary"),
    # Technology — enterprise + SaaS
    Company("CRM", "Salesforce Inc.", "Technology"),
    Company("ORCL", "Oracle Corporation", "Technology"),
    Company("ADBE", "Adobe Inc.", "Technology"),
    Company("NOW", "ServiceNow Inc.", "Technology"),
    Company("SNOW", "Snowflake Inc.", "Technology"),
    Company("PLTR", "Palantir Technologies", "Technology"),
    # Semiconductors
    Company("AMD", "Advanced Micro Devices", "Technology"),
    Company("INTC", "Intel Corporation", "Technology"),
    Company("AVGO", "Broadcom Inc.", "Technology"),
    Company("TSM", "Taiwan Semiconductor", "Technology"),
    # Financials
    Company("JPM", "JPMorgan Chase", "Financials"),
    Company("BAC", "Bank of America", "Financials"),
    Company("GS", "Goldman Sachs", "Financials"),
    Company("MS", "Morgan Stanley", "Financials"),
    Company("WFC", "Wells Fargo", "Financials"),
    Company("V", "Visa Inc.", "Financials"),
    Company("MA", "Mastercard Inc.", "Financials"),
    # Retail + consumer
    Company("WMT", "Walmart Inc.", "Consumer Staples"),
    Company("COST", "Costco Wholesale", "Consumer Staples"),
    Company("TGT", "Target Corporation", "Consumer Discretionary"),
    Company("HD", "Home Depot", "Consumer Discretionary"),
    Company("NKE", "Nike Inc.", "Consumer Discretionary"),
    Company("SBUX", "Starbucks Corporation", "Consumer Discretionary"),
    Company("MCD", "McDonald's Corporation", "Consumer Discretionary"),
    # Healthcare
    Company("JNJ", "Johnson & Johnson", "Healthcare"),
    Company("UNH", "UnitedHealth Group", "Healthcare"),
    Company("PFE", "Pfizer Inc.", "Healthcare"),
    Company("LLY", "Eli Lilly and Company", "Healthcare"),
    Company("ABBV", "AbbVie Inc.", "Healthcare"),
    # Energy
    Company("XOM", "Exxon Mobil", "Energy"),
    Company("CVX", "Chevron Corporation", "Energy"),
    Company("COP", "ConocoPhillips", "Energy"),
    # Industrials
    Company("CAT", "Caterpillar Inc.", "Industrials"),
    Company("BA", "Boeing Company", "Industrials"),
    Company("GE", "General Electric", "Industrials"),
    Company("LMT", "Lockheed Martin", "Industrials"),
    # Communication + media
    Company("NFLX", "Netflix Inc.", "Communication Services"),
    Company("DIS", "The Walt Disney Company", "Communication Services"),
    Company("CMCSA", "Comcast Corporation", "Communication Services"),
    # Consumer staples
    Company("KO", "Coca-Cola Company", "Consumer Staples"),
    Company("PEP", "PepsiCo Inc.", "Consumer Staples"),
    Company("PG", "Procter & Gamble", "Consumer Staples"),
    # Logistics
    Company("UPS", "United Parcel Service", "Industrials"),
]


def get_universe() -> list[Company]:
    """Return the full 50-company universe."""
    assert len(UNIVERSE) == 50, f"Universe has {len(UNIVERSE)} companies, expected 50"
    return list(UNIVERSE)


def get_by_sector() -> dict[str, list[Company]]:
    """Group the universe by sector for stratified sampling."""
    out: dict[str, list[Company]] = {}
    for c in UNIVERSE:
        out.setdefault(c.sector, []).append(c)
    return out


if __name__ == "__main__":
    universe = get_universe()
    by_sector = get_by_sector()
    print(f"Total: {len(universe)} companies")
    print(f"Sectors: {len(by_sector)}")
    for sector, companies in sorted(by_sector.items()):
        print(f"  {sector}: {len(companies)}")
