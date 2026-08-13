"""
The DIAL-specific fields a Chat Completions response carries beyond the OpenAI
schema, and the reasoning hidden in them.

Reasoning output has no home in the Chat Completions response schema, so DIAL
carries it out-of-band on `choices[].message.custom_content`. Ignoring it makes
a thinking-enabled model appear to produce no thinking at all, while still
billing for the tokens.

The `openai` SDK keeps unrecognised fields on `model_extra` but leaves them
untyped, so they are re-parsed here. The aidial-sdk's own `CustomContent` is
deliberately not reused: its `Stage` requires a `name` and `status` that
streaming deltas omit, and it strips the stage `index` that those deltas must
be keyed off.
"""

import re
from typing import Any

from openai.types.chat.chat_completion_message import Annotation

from aidial_adapter_bedrock.utils.pydantic import ExtraAllowModel

# DIAL adapters name the reasoning stage freely ("Thinking", "Reasoning
# process", …), so it is recognised by substring rather than an exact match.
_REASONING_STAGE = re.compile(r"think|thought|reason", re.IGNORECASE)


class ClaudeMessageBlock(ExtraAllowModel):
    type: str | None = None
    thinking: str | None = None
    signature: str | None = None


class MessageState(ExtraAllowModel):
    claude_message_content: list[ClaudeMessageBlock] | None = None


class Stage(ExtraAllowModel):
    index: int | None = None
    name: str | None = None
    content: str | None = None


class CustomContent(ExtraAllowModel):
    stages: list[Stage] | None = None
    state: MessageState | None = None


class DialExtras(ExtraAllowModel):
    """Fields present on both a message and a streaming delta.

    `annotations` is declared here because the SDK's `ChoiceDelta` has no such
    field, unlike `ChatCompletionMessage`.
    """

    custom_content: CustomContent | None = None
    annotations: list[Annotation] | None = None


def parse_extras(model_extra: dict[str, Any] | None) -> DialExtras:
    """Never raises: a malformed extension must not fail the whole response."""
    try:
        return DialExtras.model_validate(model_extra or {})
    except Exception:
        return DialExtras()


def is_reasoning_stage(name: str | None) -> bool:
    return name is not None and bool(_REASONING_STAGE.search(name))


def signed_thinking(
    custom_content: CustomContent | None,
) -> ClaudeMessageBlock | None:
    """The native Anthropic thinking block DIAL echoes back, if any.

    Preferred over the stage text because it carries a real `signature`, which
    is what allows the block to be replayed across a tool-call turn.
    """
    state = custom_content.state if custom_content else None
    for block in (state.claude_message_content if state else None) or []:
        if block.type == "thinking" and block.thinking:
            return block
    return None


def stage_thinking(custom_content: CustomContent | None) -> str:
    """The reasoning stages' text. Plain text: no signature exists for it."""
    return "".join(
        stage.content or ""
        for stage in (custom_content.stages if custom_content else None) or []
        if is_reasoning_stage(stage.name)
    )
