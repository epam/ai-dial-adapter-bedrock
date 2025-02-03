from abc import ABC, abstractmethod
from typing import Any, List, Set, Tuple

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel

from aidial_adapter_bedrock.dial_api.request import (
    ModelParameters,
    collect_text_content,
)
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages
from aidial_adapter_bedrock.utils.list_projection import ListProjection


class ChatCompletionAdapter(ABC, BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:
        pass

    async def count_prompt_tokens(
        self, params: ModelParameters, messages: List[Message]
    ) -> int:
        raise NotImplementedError

    async def count_completion_tokens(self, string: str) -> int:
        raise NotImplementedError

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: List[Message]
    ) -> DiscardedMessages | None:
        """
        The method truncates the list of messages to fit
        into the token limit set in `params.max_prompt_tokens`.

        If the limit isn't provided, then it returns None.
        Otherwise, returns the indices of _discarded_ messages which should be
        removed from the list to make the rest fit into the token limit.
        """
        raise NotImplementedError


class TextCompletionAdapter(ABC, BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    async def predict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ) -> None:
        pass

    async def count_prompt_tokens(
        self, params: ModelParameters, prompt: str
    ) -> int:
        raise NotImplementedError

    async def count_completion_tokens(self, string: str) -> int:
        raise NotImplementedError


def default_preprocess_messages(
    messages: List[Message],
) -> ListProjection[Message]:
    def _is_empty_system_message(msg: Message) -> bool:
        return (
            msg.role == Role.SYSTEM
            and collect_text_content(msg.content).strip() == ""
        )

    ret: List[Tuple[Message, Set[int]]] = []
    idx: Set[int] = set()

    for i, msg in enumerate(messages):
        idx.add(i)
        if _is_empty_system_message(msg):
            continue
        ret.append((msg, idx))
        idx = set()

    if len(ret) == 0:
        raise ValidationError("List of messages must not be empty")

    return ListProjection(ret)


def keep_last(messages: List[Any], idx: int) -> bool:
    return idx == len(messages) - 1


def keep_last_and_system_messages(messages: List[Message], idx: int) -> bool:
    return messages[idx].role == Role.SYSTEM or keep_last(messages, idx)


def trivial_partitioner(messages: List[Any]) -> List[int]:
    return [1] * len(messages)


def turn_based_partitioner(messages: List[Any]) -> List[int]:
    n = len(messages)
    return [2] * (n // 2) + [1] * (n % 2)
