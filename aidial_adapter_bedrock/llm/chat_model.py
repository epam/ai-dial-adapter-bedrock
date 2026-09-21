from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from aidial_adapter_anthropic.adapter import ValidationError
from aidial_adapter_anthropic.dial._message import (
    AdapterMessage,
    SystemMessage,
    is_system_role,
)
from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import AdapterRequest
from aidial_sdk.chat_completion import Message as DialMessage

from aidial_adapter_bedrock.utils.list_projection import ListProjection


@dataclass
class TextCompletionAdapter(ABC):
    @abstractmethod
    async def predict(
        self, consumer: Consumer, request: AdapterRequest, prompt: str
    ) -> None:
        pass

    async def count_prompt_tokens(
        self, request: AdapterRequest, prompt: str
    ) -> int:
        raise NotImplementedError()

    async def count_completion_tokens(self, string: str) -> int:
        raise NotImplementedError()


def to_dial_messages(
    messages: ListProjection[AdapterMessage],
) -> list[DialMessage]:
    """Unwraps the parsed messages back into the raw DIAL ones.

    `AdapterRequest` hands over messages already parsed into `AdapterMessage`,
    while the adapters below still speak raw DIAL messages. The round-trip is
    1:1, so an index into the result is an index into the projection.
    """
    return [msg.to_message() for msg in messages.raw_list]


def default_preprocess_messages(
    messages: ListProjection[AdapterMessage],
) -> ListProjection[AdapterMessage]:
    def _is_empty_system_message(msg: AdapterMessage) -> bool:
        return isinstance(msg, SystemMessage) and msg.text_content.strip() == ""

    ret: list[tuple[AdapterMessage, set[int]]] = []
    idx: set[int] = set()

    # A dropped message is attributed to the message that follows it.
    for msg, indices in messages.lst:
        idx |= indices
        if _is_empty_system_message(msg):
            continue
        ret.append((msg, idx))
        idx = set()

    if len(ret) == 0:
        raise ValidationError("List of messages must not be empty")

    return ListProjection(ret)


def keep_last(messages: list[Any], idx: int) -> bool:
    return idx == len(messages) - 1


def keep_last_and_system_messages(
    messages: list[DialMessage], idx: int
) -> bool:
    return is_system_role(messages[idx].role) or keep_last(messages, idx)


def trivial_partitioner(messages: list[Any]) -> list[int]:
    return [1] * len(messages)


def turn_based_partitioner(messages: list[Any]) -> list[int]:
    n = len(messages)
    return [2] * (n // 2) + [1] * (n % 2)
