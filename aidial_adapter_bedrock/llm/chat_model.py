from abc import ABC, abstractmethod
from typing import List, Tuple

from aidial_sdk.chat_completion import Message
from pydantic import BaseModel

import aidial_adapter_bedrock.llm.chat_emulation.claude as claude
import aidial_adapter_bedrock.llm.chat_emulation.meta_chat as meta_chat
import aidial_adapter_bedrock.llm.chat_emulation.zero_memory as zero_memory
from aidial_adapter_bedrock.llm.chat_emulation.types import ChatEmulationType
from aidial_adapter_bedrock.llm.message import BaseMessage, parse_message
from aidial_adapter_bedrock.universal_api.request import ModelParameters
from aidial_adapter_bedrock.universal_api.token_usage import TokenUsage
from aidial_adapter_bedrock.utils.operators import Unary, identity
from aidial_adapter_bedrock.utils.text import enforce_stop_tokens


class ResponseData(BaseModel):
    mime_type: str
    name: str
    content: str


class ModelResponse(BaseModel):
    content: str
    data: List[ResponseData]
    usage: TokenUsage


class ChatModel(ABC):
    model_id: str
    model_params: ModelParameters

    @abstractmethod
    async def acall(self, prompt: str) -> ModelResponse:
        # TODO: Support multiple results: call the model in cycle of `self.model_params.n` iterations
        pass

    async def achat(
        self,
        chat_emulation_type: ChatEmulationType,
        messages: List[Message],
    ) -> ModelResponse:
        prompt, post_process = emulate_chat(
            self.model_id,
            chat_emulation_type,
            list(map(parse_message, messages)),
        )

        response = await self.acall(prompt)

        content = post_process(
            enforce_stop_tokens(response.content, self.model_params.stop)
        )

        return ModelResponse(
            content=content, data=response.data, usage=response.usage
        )


class Model(BaseModel):
    provider: str
    model: str

    @classmethod
    def parse(cls, model_id: str) -> "Model":
        parts = model_id.split(".")
        if len(parts) != 2:
            raise Exception(
                f"Invalid model id '{model_id}'. The model id is expected to be in format 'provider.model'"
            )
        provider, model = parts
        return cls(provider=provider, model=model)


def emulate_chat(
    model_id: str, emulation_type: ChatEmulationType, history: List[BaseMessage]
) -> Tuple[str, Unary[str]]:
    model = Model.parse(model_id)
    if model.provider == "anthropic" and "claude" in model.model:
        return claude.emulate(history), identity

    if model.provider == "stability":
        return zero_memory.emulate(history), identity

    match emulation_type:
        case ChatEmulationType.ZERO_MEMORY:
            return zero_memory.emulate(history), identity
        case ChatEmulationType.META_CHAT:
            return meta_chat.emulate(history)
        case _:
            raise Exception(f"Invalid emulation type: {emulation_type}")