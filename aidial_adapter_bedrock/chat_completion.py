import asyncio
from typing import List, Optional, assert_never

from aidial_sdk.chat_completion import ChatCompletion, Request, Response
from aidial_sdk.chat_completion.request import ChatCompletionRequest
from aidial_sdk.deployment.from_request_mixin import FromRequestDeploymentMixin
from aidial_sdk.deployment.tokenize import (
    TokenizeError,
    TokenizeInputRequest,
    TokenizeInputString,
    TokenizeOutput,
    TokenizeRequest,
    TokenizeResponse,
    TokenizeSuccess,
)
from aidial_sdk.deployment.truncate_prompt import (
    TruncatePromptError,
    TruncatePromptRequest,
    TruncatePromptResponse,
    TruncatePromptResult,
    TruncatePromptSuccess,
)
from typing_extensions import override

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from aidial_adapter_bedrock.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_bedrock.llm.consumer import ChoiceConsumer
from aidial_adapter_bedrock.llm.model.adapter import get_bedrock_adapter
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator
from aidial_adapter_bedrock.utils.log_config import app_logger as log


class BedrockChatCompletion(ChatCompletion):
    region: str

    def __init__(self, region: str):
        self.region = region

    async def get_model(
        self, request: FromRequestDeploymentMixin
    ) -> ChatCompletionAdapter:
        deployment = BedrockDeployment.from_deployment_id(request.deployment_id)
        return await get_bedrock_adapter(
            region=self.region,
            deployment=deployment,
            headers=request.headers,
        )

    @dial_exception_decorator
    async def chat_completion(self, request: Request, response: Response):
        model = await self.get_model(request)
        params = ModelParameters.create(request)

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

    @override
    async def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        model = await self.get_model(request)

        outputs: List[TokenizeOutput] = []
        for input in request.inputs:
            match input:
                case TokenizeInputRequest():
                    outputs.append(
                        await self.tokenize_request(model, input.value)
                    )
                case TokenizeInputString():
                    outputs.append(
                        await self.tokenize_string(model, input.value)
                    )
                case _:
                    assert_never(input.type)
        return TokenizeResponse(outputs=outputs)

    async def tokenize_string(
        self, model: ChatCompletionAdapter, value: str
    ) -> TokenizeOutput:
        try:
            tokens = await model.count_completion_tokens(value)
            return TokenizeSuccess(token_count=tokens)
        except Exception as e:
            return TokenizeError(error=str(e))

    async def tokenize_request(
        self, model: ChatCompletionAdapter, request: ChatCompletionRequest
    ) -> TokenizeOutput:
        params = ModelParameters.create(request)

        try:
            token_count = await model.count_prompt_tokens(
                params, request.messages
            )
            return TokenizeSuccess(token_count=token_count)
        except Exception as e:
            return TokenizeError(error=str(e))

    @override
    async def truncate_prompt(
        self, request: TruncatePromptRequest
    ) -> TruncatePromptResponse:
        model = await self.get_model(request)
        outputs: List[TruncatePromptResult] = []
        for input in request.inputs:
            outputs.append(await self.truncate_prompt_request(model, input))
        return TruncatePromptResponse(outputs=outputs)

    async def truncate_prompt_request(
        self, model: ChatCompletionAdapter, request: ChatCompletionRequest
    ) -> TruncatePromptResult:
        try:
            params = ModelParameters.create(request)

            if params.max_prompt_tokens is None:
                raise ValueError("max_prompt_tokens is required")

            discarded_messages = await model.truncate_prompt(
                params, request.messages
            )
            return TruncatePromptSuccess(discarded_messages=discarded_messages)
        except Exception as e:
            return TruncatePromptError(error=str(e))
