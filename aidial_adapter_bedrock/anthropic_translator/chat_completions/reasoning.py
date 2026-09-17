from typing import Literal

from anthropic.types.beta.message_create_params import MessageCreateParams
from openai.types.shared import ReasoningEffort

from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


def resolve_effort(
    req: MessageCreateParams,
) -> ReasoningEffort | Literal["max"]:
    thinking = req.get("thinking")
    if thinking and thinking["type"] == "disabled":
        return "none"
    if thinking and thinking["type"] == "enabled":
        log.debug(
            "Dropping thinking.budget_tokens: no Chat Completions equivalent"
        )
    # Token budgets and qualitative effort levels have no protocol-defined
    # conversion. Leave the target's default unless effort is explicit.
    return (req.get("output_config") or {}).get("effort")
