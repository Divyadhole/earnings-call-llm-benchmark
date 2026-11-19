"""Central configuration — reads from environment (via .env if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
RAW_DIR = DATA_DIR / "raw"


@dataclass(frozen=True)
class SnowflakeConfig:
    account: str
    user: str
    password: str
    role: str
    warehouse: str
    database: str
    schema: str

    @classmethod
    def from_env(cls) -> SnowflakeConfig:
        return cls(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            role=os.environ.get("SNOWFLAKE_ROLE", "TRANSFORMER"),
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            database=os.environ.get("SNOWFLAKE_DATABASE", "EARNINGS_BENCH"),
            schema=os.environ.get("SNOWFLAKE_SCHEMA", "RAW"),
        )


@dataclass(frozen=True)
class FMPConfig:
    api_key: str
    base_url: str = "https://financialmodelingprep.com/api/v3"

    @classmethod
    def from_env(cls) -> FMPConfig:
        return cls(api_key=os.environ["FMP_API_KEY"])


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    experiment_name: str

    @classmethod
    def from_env(cls) -> MLflowConfig:
        return cls(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"),
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "earnings-bench"),
        )
