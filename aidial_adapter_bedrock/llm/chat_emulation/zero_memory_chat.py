from typing import Callable, List, Tuple

from aidial_adapter_bedrock.llm.chat_emulation.history import (
    FormattedMessage,
    History,
)
from aidial_adapter_bedrock.llm.exceptions import ValidationError
from aidial_adapter_bedrock.llm.message import BaseMessage


class ZeroMemoryChatHistory(History):
    discarded_messages: int

    def trim(
        self, count_tokens: Callable[[str], int], max_prompt_tokens: int
    ) -> Tuple["ZeroMemoryChatHistory", int]:
        # Possibly, not supported operation
        return self, self.discarded_messages

    @classmethod
    def create(cls, messages: List[BaseMessage]) -> "ZeroMemoryChatHistory":
        if len(messages) == 0:
            raise ValidationError("List of messages must not be empty")

        last_message = messages[-1]
        return cls(
            messages=[
                FormattedMessage(
                    text=last_message.content, source_message=last_message
                )
            ],
            discarded_messages=len(messages) - 1,
        )
