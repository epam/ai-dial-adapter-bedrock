from typing import List

from langchain.schema import BaseMessage
from pydantic import BaseModel

import llm.chat_emulation.claude as claude
import llm.chat_emulation.meta_chat as meta_chat
import llm.chat_emulation.zero_memory as zero_memory
from llm.chat_emulation.types import ChatEmulationType


class Model(BaseModel):
    provider: str
    model: str


def parse_model_id(model_id: str) -> Model:
    provider, model = model_id.split(".")
    return Model(provider=provider, model=model)


def emulate_chat(
    model_id: str, emulation_type: ChatEmulationType, prompt: List[BaseMessage]
) -> str:
    model = parse_model_id(model_id)
    if model.provider == "anthropic" and "claude" in model.model:
        return claude.emulate(prompt)

    match emulation_type:
        case ChatEmulationType.ZERO_MEMORY:
            return zero_memory.emulate(prompt)
        case ChatEmulationType.META_CHAT:
            return meta_chat.emulate(prompt)
        case _:
            raise Exception(f"Invalid emulation type: {emulation_type}")
