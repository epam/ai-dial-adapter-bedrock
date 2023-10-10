import asyncio
from typing import List

from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from aidial_adapter_bedrock.llm.bedrock_adapter import BedrockAdapter
from aidial_adapter_bedrock.llm.chat_emulation.types import ChatEmulationType
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator
from aidial_adapter_bedrock.universal_api.request import ModelParameters
from aidial_adapter_bedrock.universal_api.token_usage import TokenUsage


class BedrockChatCompletion(ChatCompletion):
    region: str
    chat_emulation_type: ChatEmulationType

    def __init__(self, region: str, chat_emulation_type: ChatEmulationType):
        self.region = region
        self.chat_emulation_type = chat_emulation_type

    @dial_exception_decorator
    async def chat_completion(self, request: Request, response: Response):
        model = await BedrockAdapter.create(
            region=self.region,
            model_id=request.deployment_id,
            model_params=ModelParameters.create(request),
        )

        async def generate_response(idx: int) -> TokenUsage:
            model_response = await model.achat(
                self.chat_emulation_type, request.messages
            )

            with response.create_choice() as choice:
                choice.append_content(model_response.content)

                for data in model_response.data:
                    choice.add_attachment(
                        title=data.name,
                        data=data.content,
                        type=data.mime_type,
                    )

                return model_response.usage

        usages: List[TokenUsage] = await asyncio.gather(
            *(generate_response(idx) for idx in range(request.n or 1))
        )

        usage = sum(usages, TokenUsage())
        response.set_usage(usage.prompt_tokens, usage.completion_tokens)
