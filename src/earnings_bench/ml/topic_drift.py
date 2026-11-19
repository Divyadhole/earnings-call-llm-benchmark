"""Model 3 — Transcript-topic drift detector.

Task: cluster transcript paragraphs into topics (BERTopic + sentence-transformers)
and track topic prevalence over time per sector.

Output: a topic × quarter matrix suitable for drift charts
("AI mentions in tech earnings calls over 10 quarters").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
except ImportError:
    BERTopic = SentenceTransformer = None  # type: ignore[assignment]

from earnings_bench.ml.mlflow_utils import start_run


@dataclass
class TopicDriftResult:
    model: BERTopic
    topic_quarter_matrix: pd.DataFrame  # index: (sector, topic_id), columns: (fy, fq) → share
    topic_labels: dict[int, str]


def chunk_transcripts(transcripts: pd.DataFrame, min_words: int = 20) -> pd.DataFrame:
    """Split each raw transcript into paragraph-ish chunks for topic modeling."""
    rows = []
    for _, r in transcripts.iterrows():
        text = str(r["raw_text"])
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para.split()) >= min_words:
                rows.append(
                    {
                        "ticker": r["ticker"],
                        "sector": r.get("sector"),
                        "fiscal_year": int(r["fiscal_year"]),
                        "fiscal_quarter": int(r["fiscal_quarter"]),
                        "text": para,
                    }
                )
    return pd.DataFrame(rows)


def fit_topics(
    chunks: pd.DataFrame,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    min_topic_size: int = 10,
    seed: int = 42,
) -> TopicDriftResult:
    """Fit BERTopic and build a sector × topic × quarter prevalence matrix."""
    if BERTopic is None:
        raise ImportError("Install the `ml` + `agent` extras: bertopic, sentence-transformers")

    embedder = SentenceTransformer(embedding_model)
    embeddings = embedder.encode(chunks["text"].tolist(), show_progress_bar=False)

    topic_model = BERTopic(
        embedding_model=embedder,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(chunks["text"].tolist(), embeddings)
    chunks = chunks.assign(topic=topics)

    # Build sector × topic × (fy, fq) prevalence
    pivot = (
        chunks.groupby(["sector", "topic", "fiscal_year", "fiscal_quarter"])
        .size()
        .rename("n")
        .reset_index()
    )
    totals = (
        chunks.groupby(["sector", "fiscal_year", "fiscal_quarter"])
        .size()
        .rename("total")
        .reset_index()
    )
    merged = pivot.merge(totals, on=["sector", "fiscal_year", "fiscal_quarter"])
    merged["share"] = merged["n"] / merged["total"]

    matrix = merged.pivot_table(
        index=["sector", "topic"],
        columns=["fiscal_year", "fiscal_quarter"],
        values="share",
        fill_value=0.0,
    )

    labels = {t: " ".join(w for w, _ in topic_model.get_topic(t)[:5]) for t in set(topics) if t != -1}

    with start_run(run_name="topic_drift_bertopic", model_tag="bertopic"):
        import mlflow

        mlflow.log_param("embedding_model", embedding_model)
        mlflow.log_param("min_topic_size", min_topic_size)
        mlflow.log_metric("n_topics", int(np.max(topics) + 1) if len(topics) else 0)
        mlflow.log_metric("n_chunks", len(chunks))

    return TopicDriftResult(model=topic_model, topic_quarter_matrix=matrix, topic_labels=labels)
