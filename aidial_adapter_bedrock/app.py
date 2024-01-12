import logging.config
import os

from aidial_sdk import DIALApp
from aidial_sdk import HTTPException as DialException
from aidial_sdk.telemetry.types import TelemetryConfig, TracingConfig
from fastapi import Request
from fastapi.responses import JSONResponse

from aidial_adapter_bedrock.chat_completion import BedrockChatCompletion
from aidial_adapter_bedrock.dial_api.response import ModelObject, ModelsResponse
from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator
from aidial_adapter_bedrock.utils.env import get_env
from aidial_adapter_bedrock.utils.log_config import LogConfig
from aidial_adapter_bedrock.utils.log_config import app_logger as log

logging.config.dictConfig(LogConfig().dict())

DEFAULT_REGION = get_env("DEFAULT_REGION")

OTLP_EXPORT_ENABLED: bool = (
    os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") is not None
)

SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "dial-bedrock")

app = DIALApp(
    description="AWS Bedrock adapter for DIAL API",
    add_healthcheck=True,
    telemetry_config=TelemetryConfig(
        service_name=SERVICE_NAME,
        tracing=TracingConfig(
            otlp_export=OTLP_EXPORT_ENABLED,
            logging=True,
        ),
    )
    if OTLP_EXPORT_ENABLED
    else None,
)


@app.get("/openai/models")
@dial_exception_decorator
async def models():
    return ModelsResponse(
        data=[
            ModelObject(id=deployment.deployment_id)
            for deployment in BedrockDeployment
        ]
    )


for deployment in BedrockDeployment:
    app.add_chat_completion(
        deployment.deployment_id,
        BedrockChatCompletion(region=DEFAULT_REGION),
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
