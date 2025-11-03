from typing import List

import pydantic
from anthropic.types.beta import BetaContentBlock as ContentBlock
from anthropic.types.beta import BetaContentBlockParam as ContentBlockParam
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.message import (
    AIRegularMessage,
    AIToolCallMessage,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class MessageState(BaseModel):
    claude_message_content: List[ContentBlock]

    def to_dict(self) -> dict:
        return self.dict(
            # FIXME: a hack to exclude the private __json_buf field
            exclude={"claude_message_content": {"__all__": {"__json_buf"}}},
            # Excluding `citations: null`, since they could not be even parsed
            # currently by the Bedrock.
            exclude_none=True,
        )


def get_message_content_from_state(
    idx: int, message: AIRegularMessage | AIToolCallMessage
) -> List[ContentBlockParam] | None:
    if (cc := message.custom_content) is not None and (
        state_dict := cc.state
    ) is not None:
        try:
            state = MessageState.parse_obj(state_dict)
            return [block.to_dict() for block in state.claude_message_content]  # type: ignore
        except pydantic.ValidationError as e:
            log.error(
                f"Invalid state at the path 'messages[{idx}].custom_content.state': {e}"
            )
            return None

    return None
