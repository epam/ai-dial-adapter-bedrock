import logging.config

import fastapi
from aidial_sdk import DIALApp
from aidial_sdk import HTTPException as DialException
from fastapi import Request
from fastapi.responses import JSONResponse

from aidial_adapter_bedrock.chat_completion import BedrockChatCompletion
from aidial_adapter_bedrock.dial_api.response import ModelObject, ModelsResponse
from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from aidial_adapter_bedrock.llm.model_listing import get_bedrock_models
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator
from aidial_adapter_bedrock.utils.env import get_env
from aidial_adapter_bedrock.utils.log_config import LogConfig
from aidial_adapter_bedrock.utils.log_config import app_logger as log

logging.config.dictConfig(LogConfig().dict())

default_region = get_env("DEFAULT_REGION")

app = DIALApp(description="AWS Bedrock adapter for DIAL API")


@app.get("/healthcheck")
def healthcheck():
    return fastapi.Response("OK")


@app.get("/openai/models")
@dial_exception_decorator
async def models():
    bedrock_models = get_bedrock_models(region=default_region)
    models = [ModelObject(id=model["modelId"]) for model in bedrock_models]
    return ModelsResponse(data=models)


for deployment in BedrockDeployment:
    app.add_chat_completion(
        deployment.get_model_id(),
        BedrockChatCompletion(region=default_region),
    )


@app.exception_handler(DialException)
async def exception_handler(request: Request, exc: DialException):
    log.exception(f"Exception: {str(exc)}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.type,
                "code": exc.code,
                "param": exc.param,
            }
        },
    )
