from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, List, Optional

from aidial_sdk.chat_completion import Message
from pydantic import BaseModel

import aidial_adapter_bedrock.utils.stream as stream_utils
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.chat_emulation.chat_emulator import ChatEmulator
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.exceptions import ValidationError
from aidial_adapter_bedrock.llm.message import (
    BaseMessage,
    SystemMessage,
    parse_message,
)
from aidial_adapter_bedrock.llm.truncate_prompt import (
    TruncatePromptError,
    truncate_prompt,
)
from aidial_adapter_bedrock.utils.list import omit_by_indices
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


def _is_empty_system_message(msg: BaseMessage) -> bool:
    return isinstance(msg, SystemMessage) and msg.content.strip() == ""


class ChatPrompt(BaseModel):
    text: str
    stop_sequences: List[str]
    discarded_messages: Optional[int] = None


class ChatModel(ABC):
    model: str

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def _prepare_prompt(
        self, messages: List[BaseMessage], max_prompt_tokens: Optional[int]
    ) -> ChatPrompt:
        pass

    @abstractmethod
    async def _apredict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ) -> None:
        pass

    def _validate_and_cleanup_messages(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        # Skipping empty system messages
        messages = [
            msg for msg in messages if not _is_empty_system_message(msg)
        ]

        if len(messages) == 0:
            raise ValidationError("List of messages must not be empty")

        return messages

    async def achat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ):
        base_messages = list(map(parse_message, messages))
        base_messages = self._validate_and_cleanup_messages(base_messages)

        chat_prompt = self._prepare_prompt(
            base_messages, params.max_prompt_tokens
        )

        params = params.add_stop_sequences(chat_prompt.stop_sequences)

        log.debug(
            f"model parameters:\n{params.json(indent=2, exclude_none=True)}"
        )
        log.debug(f"prompt:\n{chat_prompt.text}")

        await self._apredict(consumer, params, chat_prompt.text)

        if chat_prompt.discarded_messages is not None:
            consumer.set_discarded_messages(chat_prompt.discarded_messages)


def is_important_message(messages: List[BaseMessage], index: int) -> bool:
    return (
        isinstance(messages[index], SystemMessage) or index == len(messages) - 1
    )


class PseudoChatModel(ChatModel, ABC):
    chat_emulator: ChatEmulator
    tokenize: Callable[[str], int]

    def __init__(
        self,
        model: str,
        tokenize: Callable[[str], int],
        chat_emulator: ChatEmulator,
    ):
        super().__init__(model)
        self.tokenize = tokenize
        self.chat_emulator = chat_emulator

    def _tokenize(self, messages: List[BaseMessage]) -> int:
        return self.tokenize(self.chat_emulator.display(messages)[0])

    def _prepare_prompt(
        self, messages: List[BaseMessage], max_prompt_tokens: Optional[int]
    ) -> ChatPrompt:
        truncate_result = truncate_prompt(
            messages=messages,
            tokenize=self._tokenize,
            keep_message=is_important_message,
            model_limit=None,
            user_limit=max_prompt_tokens,
        )

        if isinstance(truncate_result, TruncatePromptError):
            raise ValidationError(truncate_result.print())

        discarded_messages: set[int] = truncate_result

        messages = omit_by_indices(messages, truncate_result)

        text, stop_sequences = self.chat_emulator.display(messages)

        return ChatPrompt(
            text=text,
            stop_sequences=stop_sequences,
            discarded_messages=len(discarded_messages),
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
        ai_cue = emulator.cues["ai"]
        if ai_cue is not None:
            stream = stream_utils.remove_prefix(stream, ai_cue + " ")

        # If the model doesn't support stop sequences, so do it manually
        if params.stop:
            stream = stream_utils.stop_at(stream, params.stop)

        # After all the post processing, the stream may become empty.
        # To avoid this, add a space to the stream.
        stream = stream_utils.ensure_not_empty(stream, " ")

        return stream


class Model(BaseModel):
    provider: str
    model: str

    @classmethod
    def parse(cls, model_id: str) -> "Model":
        parts = model_id.split(".")
        if len(parts) != 2:
            raise Exception(
                f"Invalid model id '{model_id}'. "
                "The model id is expected to be in format 'provider.model'"
            )
        provider, model = parts
        return cls(provider=provider, model=model)
