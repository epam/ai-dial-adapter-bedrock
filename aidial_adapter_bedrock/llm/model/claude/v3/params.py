from typing import List, TypedDict

from anthropic import NotGiven
from anthropic.types.anthropic_beta_param import AnthropicBetaParam
from anthropic.types.beta import BetaThinkingConfigParam as ThinkingConfigParam
from anthropic.types.beta import BetaToolChoiceParam as ToolChoice
from anthropic.types.beta import BetaToolParam as ToolParam


class ClaudeParameters(TypedDict):
    """
    Subset of parameters to Anthropic Messages API request:
    https://github.com/anthropics/anthropic-sdk-python/blob/ff83982c44db0920f435916aadb37c3523083079/src/anthropic/resources/messages.py#L1827-L1847
    """

    max_tokens: int
    stop_sequences: List[str] | NotGiven
    system: str | NotGiven
    temperature: float | NotGiven
    top_p: float | NotGiven
    tools: List[ToolParam] | NotGiven
    tool_choice: ToolChoice | NotGiven
    thinking: ThinkingConfigParam | NotGiven
    betas: List[AnthropicBetaParam] | NotGiven
