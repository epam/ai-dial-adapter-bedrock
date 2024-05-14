from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, List, Optional

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel

import aidial_adapter_bedrock.utils.stream as stream_utils
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.chat_emulator import ChatEmulator
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.message import BaseMessage, SystemMessage
from aidial_adapter_bedrock.llm.tools.emulator import ToolsEmulator
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsConfig
from aidial_adapter_bedrock.llm.truncate_prompt import (
    TruncatePromptError,
    truncate_prompt,
)
from aidial_adapter_bedrock.utils.list import omit_by_indices
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log
from aidial_adapter_bedrock.utils.not_implemented import not_implemented


def _is_empty_system_message(msg: Message) -> bool:
    return (
        msg.role == Role.SYSTEM
        and msg.content is not None
        and msg.content.strip() == ""
    )


class ChatCompletionAdapter(ABC, BaseModel):
    tools_emulator: Callable[[Optional[ToolsConfig]], ToolsEmulator]

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
    async def truncate_prompt(
        self, params: ModelParameters, messages: List[Message]
    ) -> List[int]: ...


class TextCompletionPrompt(BaseModel):
    text: str
    stop_sequences: List[str]
    discarded_messages: Optional[List[int]] = None


class TextCompletionAdapter(ChatCompletionAdapter):

    @abstractmethod
    async def predict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ) -> None:
        pass

    @abstractmethod
    def truncate_and_linearize_messages(
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

    def get_text_completion_prompt(
        self, params: ModelParameters, messages: List[Message]
    ) -> TextCompletionPrompt:

        messages = self.preprocess_messages(messages)
        tools_emulator = self.tools_emulator(params.tool_config)
        base_messages = tools_emulator.parse_dial_messages(messages)
        tool_stop_sequences = tools_emulator.get_stop_sequences()

        prompt = self.truncate_and_linearize_messages(
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

        prompt = self.get_text_completion_prompt(params, messages)
        params.stop = prompt.stop_sequences

        if prompt.discarded_messages is not None:
            consumer.set_discarded_messages(prompt.discarded_messages)

        log.debug(f"model parameters: {params.json(exclude_none=True)}")
        log.debug(f"prompt: {prompt.text!r}")

        await self.predict(consumer, params, prompt.text)

    async def truncate_prompt(
        self, params: ModelParameters, messages: List[Message]
    ) -> List[int]:
        prompt = self.get_text_completion_prompt(params, messages)
        return prompt.discarded_messages or []


def default_keep_message(messages: List[BaseMessage], idx: int) -> bool:
    """Keep system messages and the last message."""
    return isinstance(messages[idx], SystemMessage) or idx == len(messages) - 1


def default_partitioner(messages: List[BaseMessage]) -> List[int]:
    return [1] * len(messages)


class PseudoChatModel(TextCompletionAdapter):
    chat_emulator: ChatEmulator
    tokenize_string: Callable[[str], int]
    chat_emulator: ChatEmulator
    partitioner: Callable[[List[BaseMessage]], List[int]]

    async def count_prompt_tokens(
        self, params: ModelParameters, messages: List[Message]
    ) -> int:
        messages = self.preprocess_messages(messages)
        tools_emulator = self.tools_emulator(params.tool_config)
        base_messages = tools_emulator.parse_dial_messages(messages)
        return self.tokenize_messages(base_messages)

    async def count_completion_tokens(self, string: str) -> int:
        return self.tokenize_string(string)

    def tokenize_messages(self, messages: List[BaseMessage]) -> int:
        return self.tokenize_string(self.chat_emulator.display(messages)[0])

    def truncate_and_linearize_messages(
        self, messages: List[BaseMessage], max_prompt_tokens: Optional[int]
    ) -> TextCompletionPrompt:
        truncate_result = truncate_prompt(
            messages=messages,
            tokenize_messages=self.tokenize_messages,
            keep_message=default_keep_message,
            partition_messages=self.partitioner,
            model_limit=None,
            user_limit=max_prompt_tokens,
        )

        if isinstance(truncate_result, TruncatePromptError):
            raise ValidationError(truncate_result.print())

        discarded_messages: set[int] = truncate_result

        messages = omit_by_indices(messages, truncate_result)

        text, stop_sequences = self.chat_emulator.display(messages)

        discarded_messages_list = (
            None if max_prompt_tokens is None else list(discarded_messages)
        )

        return TextCompletionPrompt(
            text=text,
            stop_sequences=stop_sequences,
            discarded_messages=discarded_messages_list,
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
