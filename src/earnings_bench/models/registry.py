"""The 5 LLMs evaluated in the benchmark, pinned to specific versions.

Versions are locked for reproducibility. Any change to this file is treated as a
new benchmark release and gets its own tag.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """One benchmark entry — pinned version + per-token pricing (USD per 1M tokens)."""

    id: str                          # LiteLLM model identifier
    display_name: str
    provider: str
    input_price_per_mtok: float
    output_price_per_mtok: float
    context_window: int
    supports_tools: bool = True


REGISTRY: dict[str, ModelSpec] = {
    "gpt-4o": ModelSpec(
        id="openai/gpt-4o-2024-11-20",
        display_name="GPT-4o",
        provider="openai",
        input_price_per_mtok=2.50,
        output_price_per_mtok=10.00,
        context_window=128_000,
    ),
    "claude-sonnet": ModelSpec(
        id="anthropic/claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        provider="anthropic",
        input_price_per_mtok=3.00,
        output_price_per_mtok=15.00,
        context_window=200_000,
    ),
    "gemini-flash": ModelSpec(
        id="gemini/gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        provider="google",
        input_price_per_mtok=0.075,
        output_price_per_mtok=0.30,
        context_window=1_000_000,
    ),
    "llama-70b": ModelSpec(
        id="groq/llama-3.3-70b-versatile",
        display_name="Llama 3.3 70B",
        provider="groq",
        input_price_per_mtok=0.59,
        output_price_per_mtok=0.79,
        context_window=128_000,
    ),
    "gpt-4o-mini": ModelSpec(
        id="openai/gpt-4o-mini-2024-07-18",
        display_name="GPT-4o mini",
        provider="openai",
        input_price_per_mtok=0.15,
        output_price_per_mtok=0.60,
        context_window=128_000,
    ),
}


def all_models() -> list[ModelSpec]:
    return list(REGISTRY.values())


def get_model(slug: str) -> ModelSpec:
    try:
        return REGISTRY[slug]
    except KeyError as e:
        raise KeyError(f"Unknown model slug: {slug!r}. Options: {list(REGISTRY)}") from e


def cost_usd(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * spec.input_price_per_mtok / 1_000_000
        + output_tokens * spec.output_price_per_mtok / 1_000_000
    )
