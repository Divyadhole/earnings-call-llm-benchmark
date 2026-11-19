# Methodology

## Pre-registered hypotheses (Layer A)

To avoid HARKing (hypothesizing after the results are known), the benchmark
commits to these hypotheses **before** running any model:

- **H1** — Models hallucinate more on hedged guidance language than on reported
  headline numbers.
- **H2** — Chain-of-thought prompting trades off latency and cost for a modest
  (but measurable) accuracy gain on segment-level extractions.
- **H3** — Tool-augmented prompts (giving the model access to the 10-Q as a
  retrieval tool) cut hallucination rate by more than half vs zero-shot.
- **H4** — Cost-per-correct extraction is dominated by output-token cost at
  Tier-A granularity, not input-token cost.

## Grading

Each model × prompt × transcript extraction is scored against Tier-A on three
axes:

| Axis | What it measures | Method |
|---|---|---|
| Accuracy | Exact numeric + categorical match | String/decimal equality with unit normalization |
| Hallucination rate | Fabricated facts not supported by any Tier-A row | Row-level precision × 100 |
| Cost per correct | $USD per correctly extracted fact | `total_api_cost / n_correct_rows` |

All metrics come with 95% bootstrap confidence intervals (1000 resamples).

## Baselines

- **Zero-shot** — JSON schema only, no examples, no tools.
- **Few-shot** — 3-shot with gold examples from a held-out Tier-A slice.
- **Tool-augmented** — model can call `retrieve_passage(topic)` against the
  transcript during extraction.

## Reproducibility

- All LLM versions pinned (model IDs + release dates).
- ML model seeds frozen; time-based CV walk-forward.
- dbt manifest committed per release tag.
- `make reproduce` from a fresh clone regenerates every dashboard figure.
