"""Few-shot — 3 gold examples from a held-out Tier-A slice."""

from __future__ import annotations

from earnings_bench.prompts.base import SYSTEM_PROMPT, PromptStrategy

# These examples come from a held-out slice NOT used in scoring.
# Swap in your own once Tier-A is populated; this is a placeholder shape.
GOLD_EXAMPLES = [
    {
        "excerpt": (
            "Revenue for the quarter was $94.9 billion, up 6% year-over-year. "
            "Services revenue grew 14% to a record $25.0 billion."
        ),
        "json": """{
  "ticker": "AAPL",
  "period": {"fiscal_year": 2024, "fiscal_quarter": 4},
  "call_date": "2024-10-31",
  "headline_numbers": [
    {"metric": "revenue", "value": "94.9", "currency": "USD", "unit": "billions",
     "period": {"fiscal_year": 2024, "fiscal_quarter": 4},
     "source_quote": "Revenue for the quarter was $94.9 billion, up 6% year-over-year."}
  ],
  "segments": [
    {"segment_name": "Services", "metric": "revenue", "value": "25.0",
     "yoy_change_pct": "14", "direction": "up",
     "period": {"fiscal_year": 2024, "fiscal_quarter": 4},
     "source_quote": "Services revenue grew 14% to a record $25.0 billion."}
  ],
  "guidance": [], "commentary": []
}""",
    },
]


class FewShotPrompt(PromptStrategy):
    name = "few_shot"

    def build_messages(self, transcript_text: str, ticker: str) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in GOLD_EXAMPLES:
            msgs.append({"role": "user", "content": f"<transcript>\n{ex['excerpt']}\n</transcript>"})
            msgs.append({"role": "assistant", "content": ex["json"]})
        msgs.append(
            {
                "role": "user",
                "content": (
                    f"Now extract all facts from this {ticker} earnings call transcript. "
                    "Return ONLY the JSON object.\n\n"
                    f"<transcript>\n{transcript_text}\n</transcript>"
                ),
            }
        )
        return msgs
