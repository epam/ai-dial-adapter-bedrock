from typing import List, TypedDict

from anthropic import Omit
from anthropic.types.anthropic_beta_param import AnthropicBetaParam
from anthropic.types.beta import BetaTextBlockParam as TextBlockParam
from anthropic.types.beta import BetaThinkingConfigParam as ThinkingConfigParam
from anthropic.types.beta import BetaToolChoiceParam as ToolChoice
from anthropic.types.beta import BetaToolParam as ToolParam


class ClaudeParameters(TypedDict):
    """
    Subset of parameters to Anthropic Messages API request:
    https://github.com/anthropics/anthropic-sdk-python/blob/ff83982c44db0920f435916aadb37c3523083079/src/anthropic/resources/messages.py#L1827-L1847
    """

    max_tokens: int
    stop_sequences: List[str] | Omit
    system: str | List[TextBlockParam] | Omit
    temperature: float | Omit
    top_p: float | Omit
    tools: List[ToolParam] | Omit
    tool_choice: ToolChoice | Omit
    thinking: ThinkingConfigParam | Omit
    betas: List[AnthropicBetaParam] | Omit
