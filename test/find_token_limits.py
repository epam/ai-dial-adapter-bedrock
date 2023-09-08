import asyncio

from langchain.schema import HumanMessage

from llm.bedrock_adapter import BedrockAdapter
from llm.bedrock_models import BedrockDeployment
from llm.chat_emulation.types import ChatEmulationType
from universal_api.request import CompletionParameters
from utils.cli import select_enum
from utils.env import get_env
from utils.init import init
from utils.printing import print_error, print_info


async def main():
    init()

    model_id = select_enum("Select model", BedrockDeployment)

    model = await BedrockAdapter.create(
        model_id=model_id,
        model_params=CompletionParameters(max_tokens=1),
        region=get_env("DEFAULT_REGION"),
    )

    base = "a "

    min_x = 1
    max_x = 100 * 1000
    x = 1

    while True:
        prompt = x * base
        print(f"{min_x} <= {x} <= {max_x}")

        try:
            response = await model.achat(
                ChatEmulationType.ZERO_MEMORY, [HumanMessage(content=prompt)]
            )

            print_info(f"{x}: " + response.usage.json(indent=2))
            min_x = x
            next_x = (x + max_x) // 2

        except Exception as e:
            print_error(f"{x}: {str(e)}")
            max_x = x
            next_x = (min_x + x) // 2

        if next_x == x:
            break
        else:
            x = next_x


if __name__ == "__main__":
    asyncio.run(main())
