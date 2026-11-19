"""Orchestrate the full benchmark grid: models × prompts × transcripts."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from earnings_bench.evals.extractor import ExtractionResult, run_extraction
from earnings_bench.evals.scoring import ScoringResult, score_extraction
from earnings_bench.models.registry import ModelSpec, all_models
from earnings_bench.prompts import STRATEGIES
from earnings_bench.schemas import ExtractionRecord

REPORTS_DIR = Path("data/benchmark_runs")


@dataclass
class BenchmarkRow:
    model: str
    prompt_strategy: str
    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    n_ground_truth: int
    n_predicted: int
    n_correct: int
    accuracy: float
    hallucination_rate: float
    cost_per_correct: float
    cost_usd: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    error: str | None


def run_grid(
    transcripts: Iterable[tuple[str, int, int, str, date]],  # noqa: F821
    ground_truth: dict[tuple[str, int, int], ExtractionRecord],
    models: list[ModelSpec] | None = None,
    strategies: list[str] | None = None,
    out_path: Path | None = None,
) -> list[BenchmarkRow]:
    """Run every (model, strategy, transcript) combination and write a CSV."""
    models = models or all_models()
    strategy_names = strategies or list(STRATEGIES.keys())
    rows: list[BenchmarkRow] = []

    for ticker, fy, fq, text, call_date in transcripts:
        gt = ground_truth.get((ticker, fy, fq))
        if gt is None:
            continue
        for model in models:
            for sname in strategy_names:
                strategy = STRATEGIES[sname]()
                res: ExtractionResult = run_extraction(model, strategy, text, ticker, fy, fq, call_date)
                score: ScoringResult = score_extraction(res.record, gt, res.cost_usd)
                rows.append(
                    BenchmarkRow(
                        model=model.display_name,
                        prompt_strategy=sname,
                        ticker=ticker,
                        fiscal_year=fy,
                        fiscal_quarter=fq,
                        n_ground_truth=score.n_ground_truth,
                        n_predicted=score.n_predicted,
                        n_correct=score.n_correct,
                        accuracy=score.accuracy,
                        hallucination_rate=score.hallucination_rate,
                        cost_per_correct=score.cost_per_correct,
                        cost_usd=res.cost_usd,
                        latency_ms=res.latency_ms,
                        input_tokens=res.input_tokens,
                        output_tokens=res.output_tokens,
                        error=res.error,
                    )
                )

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))
    return rows
