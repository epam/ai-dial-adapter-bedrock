from aidial_sdk import DIALApp
from aidial_sdk import HTTPException as DialException
from aidial_sdk.telemetry.types import TelemetryConfig
from fastapi import Request
from fastapi.responses import JSONResponse

from aidial_adapter_bedrock.chat_completion import BedrockChatCompletion
from aidial_adapter_bedrock.dial_api.response import ModelObject, ModelsResponse
from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator
from aidial_adapter_bedrock.utils.env import get_aws_default_region
from aidial_adapter_bedrock.utils.log_config import app_logger as log
from aidial_adapter_bedrock.utils.log_config import configure_loggers

AWS_DEFAULT_REGION = get_aws_default_region()

app = DIALApp(
    description="AWS Bedrock adapter for DIAL API",
    telemetry_config=TelemetryConfig(),
    add_healthcheck=True,
)

# NOTE: configuring logger after the DIAL telemetry is initialized,
# because it may have configured the root logger on its own via
# logging=True configuration.
configure_loggers()


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
        BedrockChatCompletion(region=AWS_DEFAULT_REGION),
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
