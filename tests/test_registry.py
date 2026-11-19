"""Tests for the LLM model registry."""

import pytest

from earnings_bench.models.registry import REGISTRY, all_models, cost_usd, get_model


def test_registry_has_five_models() -> None:
    assert len(REGISTRY) == 5


def test_all_models_have_pricing() -> None:
    for m in all_models():
        assert m.input_price_per_mtok > 0
        assert m.output_price_per_mtok > 0


def test_cost_computation() -> None:
    m = get_model("gpt-4o-mini")
    # 1M input tokens at $0.15 + 1M output at $0.60 = $0.75
    c = cost_usd(m, 1_000_000, 1_000_000)
    assert abs(c - 0.75) < 1e-6


def test_unknown_model() -> None:
    with pytest.raises(KeyError):
        get_model("does-not-exist")


def test_model_ids_are_pinned() -> None:
    # Every model id must include a version/date suffix so runs are reproducible
    for m in all_models():
        assert any(char.isdigit() for char in m.id), f"{m.id} is not version-pinned"
