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

## Timeline

| Weekend | Focus | Status |
|---------|-------|--------|
| **W1** | Warehouse + Tier-A hand reconciliation (20/50) | 🚧 In progress |
| W2 | Finish Tier-A, benchmark pipeline, baselines | — |
| W3 | Classical ML: price-direction, hedge-score, topic drift | — |
| W4 | LangGraph agent + RAG + agent evals | — |
| W5 | Dashboard, drift reports, launch | — |

## Repo structure

```
.
├── dbt_project/              # Snowflake warehouse models (staging → marts)
├── src/earnings_bench/       # Python package
│   ├── ingest/               # Transcript fetching
│   ├── schemas/              # Pydantic extraction schemas
│   └── reconciliation/       # Hand-reconciliation Tier-A workflow
├── data/ground_truth/        # Tier-A reconciled CSVs
├── docs/                     # Methodology + reconciliation guide
└── tests/                    # pytest + hypothesis
```

## Getting started

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Copy env + profiles
cp .env.example .env
cp dbt_project/profiles.yml.example ~/.dbt/profiles.yml
# fill in Snowflake credentials

# 3. Run dbt
cd dbt_project
dbt deps
dbt seed
dbt run
dbt test
```

## License

MIT — see [LICENSE](LICENSE).
