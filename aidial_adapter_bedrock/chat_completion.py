import asyncio
from typing import Optional, Set

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
        model_id = BedrockDeployment.from_deployment_id(
            request.deployment_id
        ).model_id

        model = await get_bedrock_adapter(
            region=self.region,
            model=model_id,
            headers=request.headers,
        )

        async def generate_response(
            usage: TokenUsage,
            discarded_messages_set: Set[Optional[int]],
            choice_idx: int,
        ) -> None:
            with response.create_choice() as choice:
                tools_emulator = model.tools_emulator(params.tool_config)
                consumer = ChoiceConsumer(tools_emulator, choice)
                await model.achat(consumer, params, request.messages)
                usage.accumulate(consumer.usage)
                discarded_messages_set.add(consumer.discarded_messages)

        usage = TokenUsage()
        discarded_messages_set: Set[Optional[int]] = set()

        await asyncio.gather(
            *(
                generate_response(usage, discarded_messages_set, idx)
                for idx in range(request.n or 1)
            )
        )

        log.debug(f"usage: {usage}")
        response.set_usage(usage.prompt_tokens, usage.completion_tokens)

        assert (
            len(discarded_messages_set) == 1
        ), "Discarded messages count must be the same for each choice."

        discarded_messages = next(iter(discarded_messages_set))
        if discarded_messages is not None:
            response.set_discarded_messages(discarded_messages)
