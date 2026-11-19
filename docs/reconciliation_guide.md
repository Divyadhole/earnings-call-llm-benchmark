# Tier-A Hand Reconciliation Guide

Tier A is the ground-truth backbone of the benchmark. Every LLM in Layer A is
graded against it. This guide covers the W1 goal: **20 reconciled calls by end
of weekend 1**, growing to 50 in W2.

## What gets reconciled

Each earnings call CSV contains four kinds of rows:

| `record_type` | What it captures | Example quote |
|---|---|---|
| `headline` | A single reported figure | "Q3 revenue was $94.9 billion" |
| `guidance` | Forward-looking range or point | "We expect Q4 revenue of $95-97B" |
| `segment` | Per-segment result | "Services revenue grew 14% YoY" |
| `commentary` | Topical management statement | "We're bullish on AI tailwinds" |

## Reconciliation rules

1. **Every row must have a `source_quote`** that appears verbatim in the transcript.
   Do not paraphrase. Trim leading/trailing filler but keep punctuation.
2. **Numbers are stored in the units stated on the call.** `unit = billions` for
   "$94.9 billion"; don't pre-multiply.
3. **GAAP vs non-GAAP** goes in the `is_gaap` column. If the call doesn't say,
   leave blank — do not guess.
4. **Guidance**: fill `value_low` + `value_high` for a range; `value` alone for a
   point estimate. Put every hedge phrase ("approximately", "we expect", "could")
   in `hedge_phrases`, pipe-separated.
5. **Segments**: `segment_name` matches how the company names it ("Services",
   "Data Center", "iPhone"). `metric` is usually "revenue" or "operating_income".
6. **Commentary**: `metric_or_topic` is a lowercase slug from the agreed topic
   list (`ai`, `china`, `supply_chain`, `margins`, `labor`, `macro`, `regulatory`, …).
7. **Cross-check against the 10-Q**: every headline number should match the 10-Q
   press release. If they diverge, trust the 10-Q and flag in `source_quote` with
   a `[DISAGREES WITH 10-Q]` suffix.

## Workflow

```bash
# 1. Scaffold a blank CSV
earnings-bench template --ticker AAPL --year 2025 --quarter 3

# 2. Open data/ground_truth/AAPL_2025_Q3.csv, fill it in while reading the
#    transcript and 10-Q side-by-side.

# 3. Validate — round-trips through Pydantic to catch bad units, missing quotes
earnings-bench validate data/ground_truth/AAPL_2025_Q3.csv \
    --ticker AAPL --year 2025 --quarter 3 --call-date 2025-10-30
```

## W1 reconciliation batch — 20 calls

Target a stratified sample across sectors so Tier-A isn't lopsided:

| Sector | Count | Example tickers |
|---|---|---|
| Technology | 6 | AAPL, MSFT, NVDA, GOOGL, META, AMD |
| Financials | 3 | JPM, GS, V |
| Healthcare | 2 | UNH, LLY |
| Consumer | 3 | AMZN, WMT, COST |
| Energy | 2 | XOM, CVX |
| Industrials | 2 | CAT, BA |
| Communication | 2 | NFLX, DIS |

Use the most recent quarter (e.g. Q3 FY2025) for all 20 so Tier-A anchors on a
single point in time.

## Common mistakes to avoid

- **Collapsing GAAP and non-GAAP into one row.** Two rows, two `is_gaap` values.
- **Wrong unit.** $94.9B as `value=94900000000` with `unit=absolute` — don't do
  this. `value=94.9`, `unit=billions`.
- **Missing hedge phrases.** Layer C's hedge-intensity model depends on these.
- **Making up topics.** Stick to the agreed slug list. Add new slugs via PR, not
  on the fly.
