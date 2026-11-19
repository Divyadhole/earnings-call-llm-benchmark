"""Snowflake IO — upsert transcripts and reconciliation rows into RAW."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from typing import Any

try:
    import snowflake.connector
except ImportError:
    snowflake = None  # type: ignore[assignment]

from earnings_bench.config import SnowflakeConfig
from earnings_bench.ingest.fmp import FetchedTranscript


@contextmanager
def get_connection(config: SnowflakeConfig | None = None):
    if snowflake is None:
        raise ImportError("Install the `dbt` or `ml` extras to get snowflake-connector-python")
    cfg = config or SnowflakeConfig.from_env()
    conn = snowflake.connector.connect(
        account=cfg.account,
        user=cfg.user,
        password=cfg.password,
        role=cfg.role,
        warehouse=cfg.warehouse,
        database=cfg.database,
        schema=cfg.schema,
    )
    try:
        yield conn
    finally:
        conn.close()


DDL_TRANSCRIPTS = """
CREATE TABLE IF NOT EXISTS RAW.TRANSCRIPTS (
    ticker           VARCHAR(10)     NOT NULL,
    fiscal_year      NUMBER(4,0)     NOT NULL,
    fiscal_quarter   NUMBER(1,0)     NOT NULL,
    call_date        DATE            NOT NULL,
    raw_text         VARIANT         NOT NULL,
    source           VARCHAR(50)     NOT NULL,
    ingested_at      TIMESTAMP_TZ    DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (ticker, fiscal_year, fiscal_quarter)
);
"""


DDL_TIER_A = """
CREATE TABLE IF NOT EXISTS RAW.TIER_A_RECONCILIATION (
    ticker             VARCHAR(10)  NOT NULL,
    fiscal_year        NUMBER(4,0)  NOT NULL,
    fiscal_quarter     NUMBER(1,0)  NOT NULL,
    record_type        VARCHAR(20)  NOT NULL,
    metric_or_topic    VARCHAR(100),
    value              NUMBER(20,6),
    value_low          NUMBER(20,6),
    value_high         NUMBER(20,6),
    currency           VARCHAR(10),
    unit               VARCHAR(20),
    yoy_change_pct     NUMBER(10,4),
    direction          VARCHAR(10),
    segment_name       VARCHAR(100),
    speaker_role       VARCHAR(50),
    is_gaap            BOOLEAN,
    hedge_phrases      VARCHAR(1000),
    source_quote       VARCHAR(2000) NOT NULL,
    loaded_at          TIMESTAMP_TZ  DEFAULT CURRENT_TIMESTAMP()
);
"""


def ensure_raw_tables(config: SnowflakeConfig | None = None) -> None:
    with get_connection(config) as conn:
        cur = conn.cursor()
        try:
            cur.execute(DDL_TRANSCRIPTS)
            cur.execute(DDL_TIER_A)
        finally:
            cur.close()


def upsert_transcripts(
    transcripts: Iterable[FetchedTranscript],
    config: SnowflakeConfig | None = None,
) -> int:
    """MERGE each transcript into RAW.TRANSCRIPTS. Returns row count."""
    rows = list(transcripts)
    if not rows:
        return 0
    merge_sql = """
    MERGE INTO RAW.TRANSCRIPTS t
    USING (SELECT %s AS ticker, %s AS fiscal_year, %s AS fiscal_quarter,
                  %s AS call_date, %s AS raw_text, %s AS source) s
    ON t.ticker = s.ticker
       AND t.fiscal_year = s.fiscal_year
       AND t.fiscal_quarter = s.fiscal_quarter
    WHEN MATCHED THEN UPDATE SET
        call_date = s.call_date,
        raw_text = PARSE_JSON(s.raw_text),
        source = s.source,
        ingested_at = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN INSERT
        (ticker, fiscal_year, fiscal_quarter, call_date, raw_text, source)
        VALUES (s.ticker, s.fiscal_year, s.fiscal_quarter, s.call_date,
                PARSE_JSON(s.raw_text), s.source)
    """
    with get_connection(config) as conn:
        cur = conn.cursor()
        try:
            for t in rows:
                payload: dict[str, Any] = {"text": t.raw_text, "fetched_at": datetime.utcnow().isoformat()}
                cur.execute(
                    merge_sql,
                    (
                        t.ticker,
                        t.fiscal_year,
                        t.fiscal_quarter,
                        t.call_date.isoformat(),
                        __import__("json").dumps(payload),
                        t.source,
                    ),
                )
            conn.commit()
        finally:
            cur.close()
    return len(rows)
