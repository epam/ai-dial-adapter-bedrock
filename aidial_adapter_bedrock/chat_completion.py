import asyncio
from typing import List, Optional

from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from aidial_adapter_bedrock.llm.consumer import ChoiceConsumer
from aidial_adapter_bedrock.llm.model.adapter import get_bedrock_adapter
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator
from aidial_adapter_bedrock.utils.log_config import app_logger as log


class BedrockChatCompletion(ChatCompletion):
    region: str

    def __init__(self, region: str):
        self.region = region

    @dial_exception_decorator
    async def chat_completion(self, request: Request, response: Response):
        params = ModelParameters.create(request)
        deployment = BedrockDeployment.from_deployment_id(request.deployment_id)
        model = await get_bedrock_adapter(
            region=self.region,
            deployment=deployment,
            headers=request.headers,
        )

        discarded_messages: Optional[List[int]] = None

        async def generate_response(usage: TokenUsage) -> None:
            nonlocal discarded_messages

            with response.create_choice() as choice:
                tools_emulator = model.tools_emulator(params.tool_config)
                consumer = ChoiceConsumer(tools_emulator, choice)
                await model.chat(consumer, params, request.messages)
                usage.accumulate(consumer.usage)
                discarded_messages = consumer.discarded_messages

        usage = TokenUsage()

        await asyncio.gather(
            *(generate_response(usage) for _ in range(request.n or 1))
        )

        log.debug(f"usage: {usage}")
        response.set_usage(usage.prompt_tokens, usage.completion_tokens)

        if discarded_messages is not None:
            response.set_discarded_messages(discarded_messages)
