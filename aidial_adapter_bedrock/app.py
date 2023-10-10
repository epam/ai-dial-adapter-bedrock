import logging.config

import fastapi
from aidial_sdk import DIALApp

from aidial_adapter_bedrock.chat_completion import BedrockChatCompletion
from aidial_adapter_bedrock.llm.bedrock_adapter import BedrockModels
from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from aidial_adapter_bedrock.llm.chat_emulation.types import ChatEmulationType
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator
from aidial_adapter_bedrock.universal_api.response import (
    ModelObject,
    ModelsResponse,
)
from aidial_adapter_bedrock.utils.env import get_env
from aidial_adapter_bedrock.utils.log_config import LogConfig

logging.config.dictConfig(LogConfig().dict())

default_region = get_env("DEFAULT_REGION")
default_chat_emulation_type = ChatEmulationType.META_CHAT


app = DIALApp(description="AWS Bedrock adapter for RAIL API")


@app.get("/healthcheck")
def healthcheck():
    return fastapi.Response("OK")


@app.get("/openai/models")
@dial_exception_decorator
async def models():
    bedrock_models = BedrockModels(region=default_region).models()
    models = [ModelObject(id=model["modelId"]) for model in bedrock_models]
    return ModelsResponse(data=models)


for deployment in BedrockDeployment:
    app.add_chat_completion(
        deployment.get_model_id(),
        BedrockChatCompletion(
            region=default_region,
            chat_emulation_type=default_chat_emulation_type,
        ),
    )
