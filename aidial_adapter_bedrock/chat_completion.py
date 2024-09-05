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
from aidial_sdk.exceptions import ResourceNotFoundError
from typing_extensions import override

from aidial_adapter_bedrock.aws_client_config import AWSClientConfigFactory
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import (
    ChatCompletionAdapter,
    TextCompletionAdapter,
)
from aidial_adapter_bedrock.llm.consumer import ChoiceConsumer
from aidial_adapter_bedrock.llm.errors import UserError, ValidationError
from aidial_adapter_bedrock.llm.model.adapter import get_bedrock_adapter
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator
from aidial_adapter_bedrock.utils.log_config import app_logger as log
from aidial_adapter_bedrock.utils.not_implemented import is_implemented


class BedrockChatCompletion(ChatCompletion):
    async def _get_model(
        self, request: FromRequestDeploymentMixin
    ) -> ChatCompletionAdapter:
        deployment = ChatCompletionDeployment.from_deployment_id(
            request.deployment_id
        )

        aws_client_config = await AWSClientConfigFactory(
            request=request,
        ).get_client_config()

        return await get_bedrock_adapter(
            deployment=deployment,
            api_key=request.api_key,
            aws_client_config=aws_client_config,
        )

    @dial_exception_decorator
    async def chat_completion(self, request: Request, response: Response):
        model = await self._get_model(request)
        params = ModelParameters.create(request)

        discarded_messages: Optional[List[int]] = None

        async def generate_response(usage: TokenUsage) -> None:
            nonlocal discarded_messages

            with response.create_choice() as choice:
                consumer = ChoiceConsumer(choice=choice)
                if isinstance(model, TextCompletionAdapter):
                    consumer.set_tools_emulator(
                        model.tools_emulator(params.tool_config)
                    )

                try:
                    await model.chat(consumer, params, request.messages)
                except UserError as e:
                    await e.report_usage(choice)
                    await response.aflush()
                    raise e

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
    @dial_exception_decorator
    async def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        model = await self._get_model(request)

        if not is_implemented(
            model.count_completion_tokens
        ) or not is_implemented(model.count_prompt_tokens):
            raise ResourceNotFoundError("The endpoint is not implemented")

        outputs: List[TokenizeOutput] = []
        for input in request.inputs:
            match input:
                case TokenizeInputRequest():
                    outputs.append(
                        await self._tokenize_request(model, input.value)
                    )
                case TokenizeInputString():
                    outputs.append(
                        await self._tokenize_string(model, input.value)
                    )
                case _:
                    assert_never(input.type)
        return TokenizeResponse(outputs=outputs)

    async def _tokenize_string(
        self, model: ChatCompletionAdapter, value: str
    ) -> TokenizeOutput:
        try:
            tokens = await model.count_completion_tokens(value)
            return TokenizeSuccess(token_count=tokens)
        except Exception as e:
            return TokenizeError(error=str(e))

    async def _tokenize_request(
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
    @dial_exception_decorator
    async def truncate_prompt(
        self, request: TruncatePromptRequest
    ) -> TruncatePromptResponse:
        model = await self._get_model(request)

        if not is_implemented(model.truncate_prompt):
            raise ResourceNotFoundError("The endpoint is not implemented")

        outputs: List[TruncatePromptResult] = []
        for input in request.inputs:
            outputs.append(await self._truncate_prompt_request(model, input))
        return TruncatePromptResponse(outputs=outputs)

    async def _truncate_prompt_request(
        self, model: ChatCompletionAdapter, request: ChatCompletionRequest
    ) -> TruncatePromptResult:
        try:
            params = ModelParameters.create(request)

            if params.max_prompt_tokens is None:
                raise ValidationError("max_prompt_tokens is required")

            discarded_messages = await model.truncate_prompt(
                params, request.messages
            )
            return TruncatePromptSuccess(discarded_messages=discarded_messages)
        except Exception as e:
            return TruncatePromptError(error=str(e))
