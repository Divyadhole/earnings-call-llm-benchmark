"""Zero-shot — JSON schema in the system prompt, no examples."""

from __future__ import annotations

from earnings_bench.prompts.base import SYSTEM_PROMPT, PromptStrategy


class ZeroShotPrompt(PromptStrategy):
    name = "zero_shot"

    def build_messages(self, transcript_text: str, ticker: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Extract all facts from the following {ticker} earnings call transcript. "
                    "Return ONLY the JSON object, no prose.\n\n"
                    f"<transcript>\n{transcript_text}\n</transcript>"
                ),
            },
        ]
