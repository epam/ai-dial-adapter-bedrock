from abc import ABC, abstractmethod
from typing import List, Tuple

from langchain.schema import BaseMessage
from pydantic import BaseModel

import llm.chat_emulation.claude as claude
import llm.chat_emulation.meta_chat as meta_chat
import llm.chat_emulation.zero_memory as zero_memory
from llm.chat_emulation.types import ChatEmulationType
from open_ai.types import CompletionParameters
from utils.operators import Unary, identity
from utils.text import enforce_stop_tokens


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class ChatModel(ABC):
    model_id: str
    model_params: CompletionParameters

    @abstractmethod
    def _call(self, prompt: str) -> Tuple[str, TokenUsage]:
        # TODO: Support multiple results: call the model in cycle of `self.model_params.n` iterations
        pass

    def chat(
        self,
        chat_emulation_type: ChatEmulationType,
        history: List[BaseMessage],
    ) -> Tuple[str, TokenUsage]:
        prompt, post_process = emulate_chat(
            self.model_id, chat_emulation_type, history
        )

        response, usage = self._call(prompt)

        # To support models, which doesn't have intrinsic support of stop sequences.
        if self.model_params.stop is not None:
            response = enforce_stop_tokens(response, self.model_params.stop)

        response = post_process(response)

        return response, usage


class Model(BaseModel):
    provider: str
    model: str


def parse_model_id(model_id: str) -> Model:
    parts = model_id.split(".")
    if len(parts) != 2:
        raise Exception(
            f"Invalid model id '{model_id}'. The model id is expected to be in format 'provider.model'"
        )
    provider, model = parts
    return Model(provider=provider, model=model)


def emulate_chat(
    model_id: str, emulation_type: ChatEmulationType, prompt: List[BaseMessage]
) -> Tuple[str, Unary[str]]:
    model = parse_model_id(model_id)
    if model.provider == "anthropic" and "claude" in model.model:
        return claude.emulate(prompt), identity

    match emulation_type:
        case ChatEmulationType.ZERO_MEMORY:
            return zero_memory.emulate(prompt), identity
        case ChatEmulationType.META_CHAT:
            return meta_chat.emulate(prompt)
        case _:
            raise Exception(f"Invalid emulation type: {emulation_type}")
