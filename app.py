import logging.config

from aidial_sdk import (
    ChatCompletion,
    ChatCompletionRequest,
    ChatCompletionResponse,
    DIALApp,
)
from fastapi import Query, Response

from llm.bedrock_adapter import BedrockAdapter, BedrockModels
from llm.bedrock_models import BedrockDeployment
from llm.chat_emulation.types import ChatEmulationType
from server.exceptions import dial_exception_decorator
from universal_api.request import ModelParameters
from universal_api.response import ModelObject, ModelsResponse
from universal_api.token_usage import TokenUsage
from utils.env import get_env
from utils.log_config import LogConfig

logging.config.dictConfig(LogConfig().dict())

default_region = get_env("DEFAULT_REGION")
chat_emulation_type = ChatEmulationType.META_CHAT


class BedrockChatCompletion(ChatCompletion):
    @dial_exception_decorator
    async def chat_completion(
        self, request: ChatCompletionRequest, response: ChatCompletionResponse
    ):
        model = await BedrockAdapter.create(
            region=default_region,
            model_id=request.deployment_id,
            model_params=ModelParameters.create(request),
        )

        model_response = await model.achat(
            chat_emulation_type, request.messages
        )

        usage = TokenUsage()

        for _ in range(request.n or 1):
            with response.create_choice() as choice:
                choice.append_content(model_response.content)

                for data in model_response.data:
                    choice.add_attachment(
                        title=data.name,
                        data=data.content,
                        type=data.mime_type,
                    )

                usage += model_response.usage

        response.set_usage(usage.prompt_tokens, usage.completion_tokens)


app = DIALApp()


@app.get("/healthcheck")
def healthcheck():
    return Response("OK")


@app.get("/openai/models")
@dial_exception_decorator
async def models(
    region: str = Query(default=default_region, description="AWS region")
):
    bedrock_models = BedrockModels(region).models()
    models = [ModelObject(id=model["modelId"]) for model in bedrock_models]
    return ModelsResponse(data=models)


for deployment in BedrockDeployment:
    app.add_chat_completion(deployment.get_model_id(), BedrockChatCompletion())
