from abc import ABC, abstractmethod
from typing import List, Tuple

from langchain.schema import BaseMessage
from pydantic import BaseModel

import llm.chat_emulation.claude as claude
import llm.chat_emulation.meta_chat as meta_chat
import llm.chat_emulation.zero_memory as zero_memory
from llm.chat_emulation.types import ChatEmulationType
from universal_api.request import CompletionParameters
from universal_api.token_usage import TokenUsage
from utils.operators import Unary, identity
from utils.text import enforce_stop_tokens


class ResponseData(BaseModel):
    mime_type: str
    name: str
    content: str


class ModelResponse(BaseModel):
    content: str
    data: list[ResponseData]
    usage: TokenUsage


class ChatModel(ABC):
    model_id: str
    model_params: CompletionParameters

    @abstractmethod
    async def acall(self, prompt: str) -> ModelResponse:
        # TODO: Support multiple results: call the model in cycle of `self.model_params.n` iterations
        pass

    async def achat(
        self,
        chat_emulation_type: ChatEmulationType,
        history: List[BaseMessage],
    ) -> ModelResponse:
        prompt, post_process = emulate_chat(
            self.model_id, chat_emulation_type, history
        )

        response = await self.acall(prompt)

        content = post_process(
            response.content
            if self.model_params.stop is None
            # To support models, which don't have intrinsic support of stop sequences.
            else enforce_stop_tokens(response.content, self.model_params.stop)
        )

        return ModelResponse(
            content=content, data=response.data, usage=response.usage
        )


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

    if model.provider == "stability":
        return zero_memory.emulate(prompt), identity

    match emulation_type:
        case ChatEmulationType.ZERO_MEMORY:
            return zero_memory.emulate(prompt), identity
        case ChatEmulationType.META_CHAT:
            return meta_chat.emulate(prompt)
        case _:
            raise Exception(f"Invalid emulation type: {emulation_type}")
