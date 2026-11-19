"""Tool-augmented — the model can call retrieve_passage(topic) during extraction.

This strategy swaps the full transcript in the prompt for a smaller 10-Q excerpt,
then lets the model call a retrieval tool to pull transcript passages on demand.
Hypothesis H3: this cuts hallucination rate by >50% vs zero-shot.
"""

from __future__ import annotations

from earnings_bench.prompts.base import SYSTEM_PROMPT, PromptStrategy

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "retrieve_passage",
        "description": "Retrieve a transcript passage relevant to a topic or metric.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic or metric to search for"},
                "max_results": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
}


class ToolAugmentedPrompt(PromptStrategy):
    name = "tool_augmented"

    def build_messages(self, transcript_text: str, ticker: str) -> list[dict[str, str]]:
        # The runner handles tool calls in a loop; this just seeds the conversation.
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Extract financial facts from the {ticker} earnings call. "
                    "You have access to a `retrieve_passage` tool — call it to pull "
                    "transcript snippets by topic. Return the final JSON object only."
                ),
            },
        ]

    @staticmethod
    def tools() -> list[dict]:
        return [TOOL_SPEC]
