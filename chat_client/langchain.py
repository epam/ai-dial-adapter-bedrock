#!/usr/bin/env python3

from typing import List, Optional

from langchain.llms.bedrock import Bedrock
from langchain.schema import AIMessage, BaseMessage, HumanMessage

from chat_client.init import parse_args
from llm.chat_emulation import (
    ChatEmulationType,
    history_compression,
    meta_chat_stop,
)
from utils.init import init
from utils.printing import get_input, print_ai

init()


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

    return model._call(
        history_compression(chat_emulation_type, history), stop=stop
    )


if __name__ == "__main__":
    model_id, chat_emulation_type = parse_args()

    model = create_model(model_id=model_id, max_tokens=None)  # type: ignore

    history: List[BaseMessage] = []

    while True:
        content = get_input("> ")
        history.append(HumanMessage(content=content))

        response = chat(model, chat_emulation_type, history)
        print_ai(response.strip())
        history.append(AIMessage(content=response))
