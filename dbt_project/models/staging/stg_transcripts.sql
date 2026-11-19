{{
  config(
    materialized='view'
  )
}}

-- Standardize raw transcripts into a canonical form:
--   one row per (ticker, fiscal_year, fiscal_quarter, call_date)
-- Expectations:
--   RAW.transcripts is landed by the W1 ingest step with raw_text, call_date, ticker.
--   fiscal_year and fiscal_quarter are inferred from the call_date relative to the
--   company's fiscal calendar (most are calendar-aligned; Apple is handled below).

with source as (
    select * from {{ source('raw', 'transcripts') }}
),

renamed as (
    select
        upper(ticker)                                      as ticker,
        cast(call_date as date)                            as call_date,
        fiscal_year::int                                   as fiscal_year,
        fiscal_quarter::int                                as fiscal_quarter,
        raw_text,
        source,
        ingested_at
    from source
)

select * from renamed
