from anthropic.types.beta.message_create_params import MessageCreateParams
from openai.types.shared import ReasoningEffort
from pydantic import TypeAdapter, ValidationError

from aidial_adapter_bedrock.anthropic_translator.chat_completions.system_turns import (
    MessageParam,
    visible_messages,
)
from aidial_adapter_bedrock.anthropic_translator.errors import (
    AnthropicErrorType,
    AnthropicHTTPError,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

_EFFORT: TypeAdapter[ReasoningEffort] = TypeAdapter(ReasoningEffort)


def resolve_effort(req: MessageCreateParams) -> ReasoningEffort:
    thinking = req.get("thinking")
    if thinking and thinking["type"] == "disabled":
        return "none"
    if thinking and thinking["type"] == "enabled":
        log.debug(
            "Dropping thinking.budget_tokens: no Chat Completions equivalent"
        )
    # Token budgets and qualitative effort levels have no protocol-defined
    # conversion. Leave the target's default unless effort is explicit.
    effort: str | None = _system_turn_effort(req) or (
        req.get("output_config") or {}
    ).get("effort")
    try:
        return _EFFORT.validate_python(effort)
    except ValidationError:
        raise AnthropicHTTPError(
            AnthropicErrorType.INVALID_REQUEST,
            f"Unsupported output_config.effort: {effort}",
        ) from None


def _system_turn_effort(req: MessageCreateParams) -> str | None:
    system_turns: list[MessageParam] = [
        message
        for message in visible_messages(req["messages"])
        if message["role"] == "system"
    ]
    if not system_turns:
        return None
    return (system_turns[-1].get("output_config") or {}).get("effort")
