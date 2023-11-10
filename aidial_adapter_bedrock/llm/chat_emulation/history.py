from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

from pydantic.main import BaseModel

from aidial_adapter_bedrock.llm.message import BaseMessage, SystemMessage


def is_important_message(messages: List[BaseMessage], index: int) -> bool:
    return (
        isinstance(messages[index], SystemMessage) or index == len(messages) - 1
    )


class FormattedMessage(BaseModel):
    text: str
    source_message: Optional[BaseMessage] = None
    is_important: bool = True


class History(ABC, BaseModel):
    messages: List[FormattedMessage]

    def format(self) -> str:
        return "".join(message.text for message in self.messages)

    @abstractmethod
    def trim(
        self, count_tokens: Callable[[str], int], max_prompt_tokens: int
    ) -> Tuple["History", int]:
        pass
