from typing import List, assert_never

from aidial_sdk.chat_completion import (
    ChatCompletion,
    ConfigurationRequest,
    Request,
    Response,
)
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

from aidial_adapter_bedrock.adapter_deployments import (
    AdapterChatCompletionDeployment,
)
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_bedrock.llm.consumer import ChoiceConsumer
from aidial_adapter_bedrock.llm.errors import UserError, ValidationError
from aidial_adapter_bedrock.llm.model.adapter import get_bedrock_adapter
from aidial_adapter_bedrock.server.exceptions import (
    dial_exception_decorator,
    not_implemented_handler,
)
from aidial_adapter_bedrock.upstream_config import parse_upstream_config
from aidial_adapter_bedrock.utils.log_config import app_logger as log


class BedrockChatCompletion(ChatCompletion):
    deployment: AdapterChatCompletionDeployment

    def __init__(self, deployment: AdapterChatCompletionDeployment) -> None:
        self.deployment = deployment

    async def _get_model(
        self, request: FromRequestDeploymentMixin
    ) -> ChatCompletionAdapter:
        return await get_bedrock_adapter(
            deployment=self.deployment,
            api_key=request.api_key,
            upstream_config=await parse_upstream_config(request),
        )

    @override
    @dial_exception_decorator
    @not_implemented_handler
    async def configuration(self, request: ConfigurationRequest):
        model = await self._get_model(request)
        cls = await model.configuration()
        return cls.schema()

    @dial_exception_decorator
    async def chat_completion(self, request: Request, response: Response):
        response.set_model(self.deployment.upstream_deployment_id)

        model = await self._get_model(request)
        params = ModelParameters.create(request)

        with ChoiceConsumer(response) as consumer:
            try:
                await model.chat(consumer, params, request.messages)
            except UserError as e:
                await e.report_usage(consumer.choice)
                await response.aflush()
                raise e

        log.debug(f"usage: {consumer.usage}")

    @override
    @dial_exception_decorator
    @not_implemented_handler
    async def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        model = await self._get_model(request)

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
        except NotImplementedError:
            raise
        except Exception as e:
            log.exception("Error tokenizing string")
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
        except NotImplementedError:
            raise
        except Exception as e:
            log.exception("Error tokenizing request")
            return TokenizeError(error=str(e))

    @override
    @dial_exception_decorator
    @not_implemented_handler
    async def truncate_prompt(
        self, request: TruncatePromptRequest
    ) -> TruncatePromptResponse:
        model = await self._get_model(request)

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

            discarded_messages = await model.compute_discarded_messages(
                params, request.messages
            )

            return TruncatePromptSuccess(
                discarded_messages=discarded_messages or []
            )
        except NotImplementedError:
            raise
        except Exception as e:
            log.exception("Error truncating prompt")
            return TruncatePromptError(error=str(e))
