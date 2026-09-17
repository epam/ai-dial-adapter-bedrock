from typing import Literal

from pydantic import TypeAdapter, ValidationError

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    MessagesRequest,
    ThinkingConfig,
)
from aidial_adapter_bedrock.anthropic_translator.errors import (
    AnthropicErrorType,
    AnthropicHTTPError,
)

OpenAIEffort = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max"
]
_EFFORT: TypeAdapter[OpenAIEffort] = TypeAdapter(OpenAIEffort)


def resolve_effort(req: MessagesRequest) -> OpenAIEffort | None:
    thinking: ThinkingConfig | None = req.thinking
    if thinking and thinking.type == "disabled":
        return "none"
    effort: str | None = req.output_config.effort if req.output_config else None
    if effort is not None:
        try:
            return _EFFORT.validate_python(effort)
        except ValidationError as error:
            raise AnthropicHTTPError(
                AnthropicErrorType.INVALID_REQUEST,
                f"Unsupported output_config.effort: {effort}",
            ) from error
    if thinking is None:
        return None
    if thinking.type == "adaptive" or thinking.budget_tokens is None:
        return "high"
    if thinking.budget_tokens <= 0:
        return "none"
    if thinking.budget_tokens <= 8000:
        return "low"
    if thinking.budget_tokens <= 24000:
        return "medium"
    return "high"
