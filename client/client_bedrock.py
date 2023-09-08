#!/usr/bin/env python3

import asyncio
from typing import List

from langchain.schema import AIMessage, BaseMessage, HumanMessage

from llm.bedrock_adapter import BedrockAdapter
from universal_api.request import CompletionParameters
from utils.cli import choose_deployment
from utils.env import get_env
from utils.init import init
from utils.printing import get_input, print_ai, print_info


async def main():
    init()

    deployment, chat_emulation_type = choose_deployment()

    model = await BedrockAdapter.create(
        model_id=deployment.get_model_id(),
        model_params=CompletionParameters(),
        region=get_env("DEFAULT_REGION"),
    )

    history: List[BaseMessage] = []

    while True:
        content = get_input("> ")
        history.append(HumanMessage(content=content))

        response = await model.achat(chat_emulation_type, history)

        print_info(response.usage.json(indent=2))

        print_ai(response.content.strip())
        history.append(AIMessage(content=response.content))


if __name__ == "__main__":
    asyncio.run(main())
