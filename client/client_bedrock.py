import asyncio
from typing import List

from aidial_sdk.chat_completion import Message, Role

from aidial_adapter_bedrock.llm.bedrock_adapter import BedrockAdapter
from aidial_adapter_bedrock.universal_api.request import ModelParameters
from aidial_adapter_bedrock.utils.env import get_env
from aidial_adapter_bedrock.utils.printing import print_ai, print_info
from client.conf import MAX_CHAT_TURN, MAX_INPUT_CHARS
from client.utils.cli import choose_deployment
from client.utils.init import init
from client.utils.input import make_input


async def main():
    location = get_env("DEFAULT_REGION")

    deployment, chat_emulation_type = choose_deployment()

    model = await BedrockAdapter.create(
        model_id=deployment.get_model_id(),
        model_params=ModelParameters(),
        region=location,
    )

    history: List[Message] = []

    chat_input = make_input()

    turn = 0
    while turn < MAX_CHAT_TURN:
        turn += 1

        content = chat_input()[:MAX_INPUT_CHARS]
        history.append(Message(role=Role.USER, content=content))

        response = await model.achat(chat_emulation_type, history)

        print_info(response.usage.json(indent=2))

        print_ai(response.content.strip())
        history.append(Message(role=Role.ASSISTANT, content=response.content))


if __name__ == "__main__":
    init()
    asyncio.run(main())
