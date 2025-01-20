from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel
from typing_extensions import override

from aidial_adapter_bedrock.dial_api.request import (
    ModelParameters,
    collect_text_content,
)
from aidial_adapter_bedrock.llm.chat_emulator import ChatEmulator
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.tools.emulator import (
    ToolsEmulator,
    emulate_tools,
)
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsConfig
from aidial_adapter_bedrock.llm.truncate_prompt import (
    DiscardedMessages,
    truncate_prompt,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


def _is_empty_system_message(msg: Message) -> bool:
    return (
        msg.role == Role.SYSTEM
        and collect_text_content(msg.content).strip() == ""
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
        self, messages: List[Message], max_prompt_tokens: Optional[int]
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
        base_messages = emulate_tools(tools_emulator, messages)
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


def keep_last_and_system_messages(messages: List[Message], idx: int) -> bool:
    return messages[idx].role == Role.SYSTEM or keep_last(messages, idx)


def trivial_partitioner(messages: List[Any]) -> List[int]:
    return [1] * len(messages)


def turn_based_partitioner(messages: List[Any]) -> List[int]:
    n = len(messages)
    return [2] * (n // 2) + [1] * (n % 2)


class PseudoChatModel(TextCompletionAdapter):
    chat_emulator: ChatEmulator
    tokenize_string: Callable[[str], int]
    partitioner: Callable[[List[Message]], List[int]]

    async def count_prompt_tokens(
        self, params: ModelParameters, messages: List[Message]
    ) -> int:
        messages = self.preprocess_messages(messages)
        tools_emulator = self.tools_emulator(params.tool_config)
        messages = emulate_tools(tools_emulator, messages)
        return await self.tokenize_messages(messages)

    async def count_completion_tokens(self, string: str) -> int:
        return self.tokenize_string(string)

    async def tokenize_messages(self, messages: List[Message]) -> int:
        return self.tokenize_string(self.chat_emulator.display(messages)[0])

    @override
    async def truncate_and_linearize_messages(
        self, messages: List[Message], max_prompt_tokens: Optional[int]
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
