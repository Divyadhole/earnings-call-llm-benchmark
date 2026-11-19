# Earnings Call LLM Benchmark & Agentic Analyst

A public benchmark evaluating how reliably LLMs extract financial facts from SEC earnings call transcripts, plus a LangGraph agentic analyst and classical ML drift models on top.

## Project scope

- **500 transcripts** across **50 companies** and **10 quarters**
- **5 LLMs × 3 prompt strategies** evaluated on exact-match accuracy, hallucination rate, and cost per correct extraction
- **LangGraph agentic analyst** with RAG over transcripts and self-verification
- **3 classical ML models** — price-direction, hedge-intensity, topic drift — tracked in MLflow

## Three layers

| Layer | Focus | Stack |
|-------|-------|-------|
| **A** — Extraction benchmark | Data engineering + LLM evals | Snowflake, dbt, Pydantic, LiteLLM |
| **B** — Agentic analyst | Agent + RAG + tool use | LangGraph, LangSmith, Snowflake Cortex |
| **C** — ML modeling + drift | Classical ML + MLOps | scikit-learn, XGBoost, MLflow, Evidently |

## Repo structure

```
.
├── dbt_project/              # Snowflake warehouse models (staging → marts)
├── src/earnings_bench/       # Python package
│   ├── ingest/               # Transcript fetching (FMP) + Snowflake IO
│   ├── schemas/              # Pydantic extraction schemas
│   ├── reconciliation/       # Hand-reconciliation Tier-A workflow
│   ├── prompts/              # Zero-shot, few-shot, tool-augmented
│   ├── models/               # LLM registry (LiteLLM)
│   ├── evals/                # Extraction runner + scoring + bootstrap CIs
│   └── ml/                   # XGBoost, ridge, BERTopic + MLflow + Evidently
├── data/ground_truth/        # Tier-A reconciled CSVs
├── docs/                     # Methodology + reconciliation guide
└── tests/                    # pytest + hypothesis
```

## Getting started

```bash
# 1. Install
pip install -e ".[dev,dbt,llm,ml]"

# 2. Configure Snowflake + API keys (create .env in repo root)
# See docs/configuration.md for the required variables.

# 3. Run dbt
cd dbt_project
dbt deps
dbt seed
dbt run
dbt test
```

## License

MIT — see [LICENSE](LICENSE).
