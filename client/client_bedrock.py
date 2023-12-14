import asyncio
from typing import List

from aidial_sdk.chat_completion import Message, Role

from aidial_adapter_bedrock.dial_api.auth import get_auth
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from aidial_adapter_bedrock.llm.consumer import CollectConsumer
from aidial_adapter_bedrock.llm.model.adapter import get_bedrock_adapter
from aidial_adapter_bedrock.utils.env import get_env
from aidial_adapter_bedrock.utils.printing import print_ai, print_info
from client.conf import MAX_CHAT_TURN, MAX_INPUT_CHARS
from client.utils.cli import select_enum
from client.utils.init import init
from client.utils.input import make_input


async def main():
    location = get_env("DEFAULT_REGION")

    deployment = select_enum("Select the deployment", BedrockDeployment)

    params = ModelParameters()

    model = await get_bedrock_adapter(
        model=deployment.get_model_id(),
        region=location,
        get_auth=(lambda: get_auth(None)),
    )

    messages: List[Message] = []

    chat_input = make_input()

    turn = 0
    while turn < MAX_CHAT_TURN:
        turn += 1

        content = chat_input()[:MAX_INPUT_CHARS]
        messages.append(Message(role=Role.USER, content=content))

        response = CollectConsumer()
        await model.achat(response, params, messages)

        print_info(response.usage.json(indent=2))

        print_ai(response.content.strip())
        messages.append(Message(role=Role.ASSISTANT, content=response.content))


if __name__ == "__main__":
    init()
    asyncio.run(main())
