from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from aidial_adapter_anthropic.adapter import ValidationError
from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import (
    ModelParameters,
    collect_text_content,
    is_system_role,
)
from aidial_sdk.chat_completion import Message

from aidial_adapter_bedrock.utils.list_projection import ListProjection


@dataclass
class TextCompletionAdapter(ABC):
    @abstractmethod
    async def predict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ) -> None:
        pass

    async def count_prompt_tokens(
        self, params: ModelParameters, prompt: str
    ) -> int:
        raise NotImplementedError()

    async def count_completion_tokens(self, string: str) -> int:
        raise NotImplementedError()


def default_preprocess_messages(
    messages: list[Message],
) -> ListProjection[Message]:
    def _is_empty_system_message(msg: Message) -> bool:
        return (
            is_system_role(msg.role)
            and collect_text_content(msg.content).strip() == ""
        )

    ret: list[tuple[Message, set[int]]] = []
    idx: set[int] = set()

    for i, msg in enumerate(messages):
        idx.add(i)
        if _is_empty_system_message(msg):
            continue
        ret.append((msg, idx))
        idx = set()

    if len(ret) == 0:
        raise ValidationError("List of messages must not be empty")

    return ListProjection(ret)


def keep_last(messages: list[Any], idx: int) -> bool:
    return idx == len(messages) - 1


def keep_last_and_system_messages(messages: list[Message], idx: int) -> bool:
    return is_system_role(messages[idx].role) or keep_last(messages, idx)


def trivial_partitioner(messages: list[Any]) -> list[int]:
    return [1] * len(messages)


def turn_based_partitioner(messages: list[Any]) -> list[int]:
    n = len(messages)
    return [2] * (n // 2) + [1] * (n % 2)
