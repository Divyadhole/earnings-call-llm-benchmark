"""Score an ExtractionResult against Tier-A ground truth.

Metrics:
  - exact_match_accuracy: fraction of ground-truth rows the model recovered
  - hallucination_rate:   fraction of model rows that have no ground-truth match
  - cost_per_correct:     USD per correctly extracted row
  - All metrics come with 95% bootstrap CIs (1000 resamples)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from earnings_bench.schemas import (
    ExtractionRecord,
    GuidanceRange,
    HeadlineNumber,
    ManagementCommentary,
    SegmentResult,
)

Row = tuple[str, str, str]  # (record_type, key, source_quote)


def _row_key(obj: HeadlineNumber | GuidanceRange | SegmentResult | ManagementCommentary) -> Row:
    """A comparable key for each ground-truth row — tolerates light phrasing diffs."""
    if isinstance(obj, HeadlineNumber):
        return ("headline", f"{obj.metric}|{_fmt_decimal(obj.value)}|{obj.unit}", obj.source_quote)
    if isinstance(obj, GuidanceRange):
        return (
            "guidance",
            f"{obj.metric}|{_fmt_decimal(obj.low)}-{_fmt_decimal(obj.high)}|{_fmt_decimal(obj.point)}",
            obj.source_quote,
        )
    if isinstance(obj, SegmentResult):
        return (
            "segment",
            f"{obj.segment_name}|{obj.metric}|{_fmt_decimal(obj.value)}|{_fmt_decimal(obj.yoy_change_pct)}",
            obj.source_quote,
        )
    if isinstance(obj, ManagementCommentary):
        return ("commentary", f"{obj.topic}|{obj.sentiment}", obj.source_quote)
    raise TypeError(f"Unknown row kind: {type(obj)}")


def _fmt_decimal(v: Decimal | None) -> str:
    return "" if v is None else f"{v.normalize():f}"


def _flatten(record: ExtractionRecord) -> set[Row]:
    rows: set[Row] = set()
    for h in record.headline_numbers:
        rows.add(_row_key(h))
    for g in record.guidance:
        rows.add(_row_key(g))
    for s in record.segments:
        rows.add(_row_key(s))
    for c in record.commentary:
        rows.add(_row_key(c))
    return rows


@dataclass
class ScoringResult:
    n_ground_truth: int
    n_predicted: int
    n_correct: int
    accuracy: float
    hallucination_rate: float
    cost_per_correct: float
    accuracy_ci: tuple[float, float]
    hallucination_ci: tuple[float, float]


def score_extraction(
    prediction: ExtractionRecord | None,
    ground_truth: ExtractionRecord,
    total_cost_usd: float,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> ScoringResult:
    """Score one prediction against one ground-truth extraction."""
    gt = _flatten(ground_truth)
    pred = _flatten(prediction) if prediction is not None else set()

    correct = gt & pred
    n_gt = len(gt)
    n_pred = len(pred)
    n_correct = len(correct)
    accuracy = n_correct / n_gt if n_gt else 0.0
    halluc = (n_pred - n_correct) / n_pred if n_pred else 0.0
    cpc = total_cost_usd / n_correct if n_correct else float("inf")

    acc_ci = _bootstrap_rate([(r in pred) for r in gt], bootstrap_n, seed)
    halluc_ci = _bootstrap_rate([(r not in gt) for r in pred], bootstrap_n, seed) if pred else (0.0, 0.0)

    return ScoringResult(
        n_ground_truth=n_gt,
        n_predicted=n_pred,
        n_correct=n_correct,
        accuracy=accuracy,
        hallucination_rate=halluc,
        cost_per_correct=cpc,
        accuracy_ci=acc_ci,
        hallucination_ci=halluc_ci,
    )


def _bootstrap_rate(observations: list[bool], n: int, seed: int) -> tuple[float, float]:
    """95% bootstrap CI for the mean of a 0/1 series."""
    if not observations:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(observations, dtype=float)
    samples = rng.choice(arr, size=(n, len(arr)), replace=True)
    rates = samples.mean(axis=1)
    return (float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975)))
