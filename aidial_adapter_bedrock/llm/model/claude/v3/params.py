from typing import List, TypedDict, Union

from anthropic import NotGiven
from anthropic.types import ToolParam
from anthropic.types.message_create_params import ToolChoice


class ClaudeParameters(TypedDict):
    """
    Subset of parameters to Anthropic Messages API request:
    https://github.com/anthropics/anthropic-sdk-python/blob/ff83982c44db0920f435916aadb37c3523083079/src/anthropic/resources/messages.py#L1827-L1847
    """

    max_tokens: int
    stop_sequences: Union[List[str], NotGiven]
    system: Union[str, NotGiven]
    temperature: Union[float, NotGiven]
    top_p: Union[float, NotGiven]
    tools: Union[List[ToolParam], NotGiven]
    tool_choice: Union[ToolChoice, NotGiven]
