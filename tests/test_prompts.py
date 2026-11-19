"""Tests for prompt strategies."""

from earnings_bench.prompts import STRATEGIES


def test_strategies_registered() -> None:
    assert set(STRATEGIES) == {"zero_shot", "few_shot", "tool_augmented"}


def test_zero_shot_builds_messages() -> None:
    s = STRATEGIES["zero_shot"]()
    msgs = s.build_messages("Revenue was $94.9B", "AAPL")
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "AAPL" in msgs[-1]["content"]
    assert "Revenue was $94.9B" in msgs[-1]["content"]


def test_few_shot_includes_examples() -> None:
    s = STRATEGIES["few_shot"]()
    msgs = s.build_messages("test", "NVDA")
    roles = [m["role"] for m in msgs]
    # Expect: system, (user, assistant)*, user
    assert roles[0] == "system"
    assert "assistant" in roles
    assert roles[-1] == "user"


def test_tool_augmented_has_tool_spec() -> None:
    from earnings_bench.prompts.tool_augmented import ToolAugmentedPrompt

    tools = ToolAugmentedPrompt.tools()
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "retrieve_passage"
