# Configuration

Create a `.env` file in the repo root with the following variables.

## Snowflake

```
SNOWFLAKE_ACCOUNT=your_account_locator
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ROLE=TRANSFORMER
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=EARNINGS_BENCH
SNOWFLAKE_SCHEMA=RAW
```

## Transcript source (W2)

Financial Modeling Prep. Free tier covers ~250 requests/day.

```
FMP_API_KEY=your_fmp_api_key
TRANSCRIPT_SOURCE=fmp
```

## LLM providers (W2+)

At least one is required. LiteLLM uses whichever keys are present.

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=
```

## Tracing (W4)

```
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=earnings-bench
```

## MLflow (W3)

Defaults to a local `mlruns/` folder. Set `MLFLOW_TRACKING_URI` to point at a
remote server.

```
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=earnings-bench
```
