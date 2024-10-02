from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, List, Optional

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel
from typing_extensions import override

import aidial_adapter_bedrock.utils.stream as stream_utils
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.chat_emulator import ChatEmulator
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.message import BaseMessage, SystemMessage
from aidial_adapter_bedrock.llm.tools.emulator import ToolsEmulator
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsConfig
from aidial_adapter_bedrock.llm.truncate_prompt import (
    DiscardedMessages,
    truncate_prompt,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log
from aidial_adapter_bedrock.utils.not_implemented import not_implemented
from aidial_adapter_bedrock.utils.request import (
    get_message_content_text_content,
)


def _is_empty_system_message(msg: Message) -> bool:
    return (
        msg.role == Role.SYSTEM
        and get_message_content_text_content(msg.content).strip() == ""
    )


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

    @not_implemented
    async def count_prompt_tokens(
        self, params: ModelParameters, messages: List[Message]
    ) -> int: ...

    @not_implemented
    async def count_completion_tokens(self, string: str) -> int: ...

    @not_implemented
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


class TextCompletionPrompt(BaseModel):
    text: str
    stop_sequences: List[str]
    discarded_messages: Optional[DiscardedMessages] = None


class TextCompletionAdapter(ChatCompletionAdapter):
    tools_emulator: Callable[[Optional[ToolsConfig]], ToolsEmulator]

    @abstractmethod
    async def predict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ) -> None:
        pass

    @abstractmethod
    async def truncate_and_linearize_messages(
        self, messages: List[BaseMessage], max_prompt_tokens: Optional[int]
    ) -> TextCompletionPrompt:
        pass

    def preprocess_messages(self, messages: List[Message]) -> List[Message]:
        # Skipping empty system messages
        messages = [
            msg for msg in messages if not _is_empty_system_message(msg)
        ]

        if len(messages) == 0:
            raise ValidationError("List of messages must not be empty")

        return messages

    async def get_text_completion_prompt(
        self, params: ModelParameters, messages: List[Message]
    ) -> TextCompletionPrompt:

        messages = self.preprocess_messages(messages)
        tools_emulator = self.tools_emulator(params.tool_config)
        base_messages = tools_emulator.parse_dial_messages(messages)
        tool_stop_sequences = tools_emulator.get_stop_sequences()

        prompt = await self.truncate_and_linearize_messages(
            base_messages, params.max_prompt_tokens
        )

        prompt.stop_sequences.extend(tool_stop_sequences)
        prompt.stop_sequences.extend(params.stop)

        return prompt

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:

        prompt = await self.get_text_completion_prompt(params, messages)
        params.stop = prompt.stop_sequences

        consumer.set_discarded_messages(prompt.discarded_messages)

        log.debug(f"model parameters: {params.json(exclude_none=True)}")
        log.debug(f"prompt: {prompt.text!r}")

        await self.predict(consumer, params, prompt.text)

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: List[Message]
    ) -> DiscardedMessages | None:
        prompt = await self.get_text_completion_prompt(params, messages)
        return prompt.discarded_messages


def keep_last(messages: List[Any], idx: int) -> bool:
    return idx == len(messages) - 1


def keep_last_and_system_messages(
    messages: List[BaseMessage], idx: int
) -> bool:
    return isinstance(messages[idx], SystemMessage) or keep_last(messages, idx)


def trivial_partitioner(messages: List[Any]) -> List[int]:
    return [1] * len(messages)


def turn_based_partitioner(messages: List[Any]) -> List[int]:
    n = len(messages)
    return [2] * (n // 2) + [1] * (n % 2)


class PseudoChatModel(TextCompletionAdapter):
    chat_emulator: ChatEmulator
    tokenize_string: Callable[[str], int]
    partitioner: Callable[[List[BaseMessage]], List[int]]

    async def count_prompt_tokens(
        self, params: ModelParameters, messages: List[Message]
    ) -> int:
        messages = self.preprocess_messages(messages)
        tools_emulator = self.tools_emulator(params.tool_config)
        base_messages = tools_emulator.parse_dial_messages(messages)
        return await self.tokenize_messages(base_messages)

    async def count_completion_tokens(self, string: str) -> int:
        return self.tokenize_string(string)

    async def tokenize_messages(self, messages: List[BaseMessage]) -> int:
        return self.tokenize_string(self.chat_emulator.display(messages)[0])

    @override
    async def truncate_and_linearize_messages(
        self, messages: List[BaseMessage], max_prompt_tokens: Optional[int]
    ) -> TextCompletionPrompt:
        discarded_messages, messages = await truncate_prompt(
            messages=messages,
            tokenizer=self.tokenize_messages,
            keep_message=keep_last_and_system_messages,
            partitioner=self.partitioner,
            model_limit=None,
            user_limit=max_prompt_tokens,
        )

        text, stop_sequences = self.chat_emulator.display(messages)

        if max_prompt_tokens is None:
            discarded_messages = None

        return TextCompletionPrompt(
            text=text,
            stop_sequences=stop_sequences,
            discarded_messages=discarded_messages,
        )

    @staticmethod
    def post_process_stream(
        stream: AsyncIterator[str],
        params: ModelParameters,
        emulator: ChatEmulator,
    ) -> AsyncIterator[str]:
        # Removing leading spaces
        stream = stream_utils.lstrip(stream)

        # Model may occasionally start responding with its cue.
        ai_cue = emulator.get_ai_cue()
        if ai_cue is not None:
            stream = stream_utils.remove_prefix(stream, ai_cue)
            stream = stream_utils.lstrip(stream)

        # The model may not support stop sequences, so do it manually
        if params.stop:
            stream = stream_utils.stop_at(stream, params.stop)

        # After all the post processing, the stream may become empty.
        # To avoid this, add a space to the stream.
        stream = stream_utils.ensure_not_empty(stream, " ")

        return stream
