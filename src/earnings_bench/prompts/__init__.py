from earnings_bench.prompts.few_shot import FewShotPrompt
from earnings_bench.prompts.tool_augmented import ToolAugmentedPrompt
from earnings_bench.prompts.zero_shot import ZeroShotPrompt

STRATEGIES = {
    "zero_shot": ZeroShotPrompt,
    "few_shot": FewShotPrompt,
    "tool_augmented": ToolAugmentedPrompt,
}

__all__ = ["FewShotPrompt", "STRATEGIES", "ToolAugmentedPrompt", "ZeroShotPrompt"]
