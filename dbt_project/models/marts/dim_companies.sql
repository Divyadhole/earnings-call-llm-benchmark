{{ config(materialized='table') }}

select
    ticker,
    name,
    sector
from {{ ref('companies') }}
