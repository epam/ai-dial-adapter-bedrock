import logging
from typing import List, Optional

from langchain.llms.bedrock import Bedrock
from langchain.schema import BaseMessage

from llm.chat_emulation import (
    ChatEmulationType,
    history_compression,
    meta_chat_stop,
)

log = logging.getLogger("bedrock")


def create_model(model_id: str, max_tokens: Optional[int]) -> Bedrock:
    provider = model_id.split(".")[0]

    model_kwargs = {}
    if provider == "anthropic":
        model_kwargs["max_tokens_to_sample"] = (
            max_tokens if max_tokens is not None else 500
        )

    return Bedrock(
        model_id=model_id,
        region_name="us-east-1",
        model_kwargs=model_kwargs,
    )  # type: ignore


def chat(
    model: Bedrock,
    chat_emulation_type: ChatEmulationType,
    history: List[BaseMessage],
) -> str:
    stop: Optional[List[str]] = None
    if chat_emulation_type == ChatEmulationType.META_CHAT:
        stop = [meta_chat_stop]

    prompt = history_compression(chat_emulation_type, history)
    log.debug(f"prompt:\n{prompt}")
    response = model._call(prompt, stop=stop)
    log.debug(f"response:\n{response}")
    return response


def completion(model: Bedrock, prompt: str) -> str:
    log.debug(f"prompt:\n{prompt}")
    response = model._call(prompt, stop=None)
    log.debug(f"response:\n{response}")
    return response