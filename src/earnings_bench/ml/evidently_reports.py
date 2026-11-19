"""Evidently AI drift reports — one per quarter.

Produces HTML reports under reports/evidently/{ref}_vs_{cur}.html that compare
feature distributions and model-prediction drift between a reference and a
current quarter.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.report import Report
except ImportError:
    ColumnMapping = Report = DataDriftPreset = TargetDriftPreset = None  # type: ignore[assignment]

REPORTS_DIR = Path("reports/evidently")


def quarterly_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_col: str | None = None,
    name: str = "quarterly_drift",
) -> Path:
    """Build an Evidently drift report comparing two quarterly snapshots."""
    if Report is None:
        raise ImportError("Install the `ml` extras: evidently")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ColumnMapping(target=target_col) if target_col else ColumnMapping()
    metrics = [DataDriftPreset()]
    if target_col and target_col in reference.columns:
        metrics.append(TargetDriftPreset())

    report = Report(metrics=metrics)
    report.run(reference_data=reference, current_data=current, column_mapping=cols)
    out = REPORTS_DIR / f"{name}.html"
    report.save_html(str(out))
    return out
