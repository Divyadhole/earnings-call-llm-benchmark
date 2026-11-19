{{
  config(
    materialized='table',
    unique_key=['ticker','fiscal_year','fiscal_quarter','record_type','metric_or_topic','segment_name','source_quote']
  )
}}

-- Tier-A hand-reconciled ground truth.
-- Loaded by `dbt seed` from data/ground_truth/*.csv (promoted to a seed per batch)
-- or landed directly into RAW.tier_a_reconciliation by the Python validator.

with src as (
    select * from {{ source('raw', 'tier_a_reconciliation') }}
)

select
    upper(ticker)                                       as ticker,
    fiscal_year::int                                    as fiscal_year,
    fiscal_quarter::int                                 as fiscal_quarter,
    lower(record_type)                                  as record_type,
    lower(metric_or_topic)                              as metric_or_topic,
    segment_name,
    cast(value          as number(20, 6))               as value,
    cast(value_low      as number(20, 6))               as value_low,
    cast(value_high     as number(20, 6))               as value_high,
    currency,
    unit,
    cast(yoy_change_pct as number(10, 4))               as yoy_change_pct,
    direction,
    speaker_role,
    is_gaap,
    hedge_phrases,
    source_quote,
    current_timestamp()                                 as loaded_at
from src
