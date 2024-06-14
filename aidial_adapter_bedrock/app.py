import json
from typing import Optional

from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import TelemetryConfig
from fastapi import Body, Header, Path

from aidial_adapter_bedrock.chat_completion import BedrockChatCompletion
from aidial_adapter_bedrock.deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_bedrock.dial_api.request import (
    EmbeddingsRequest,
    EmbeddingsType,
)
from aidial_adapter_bedrock.dial_api.response import (
    ModelObject,
    ModelsResponse,
    make_embeddings_response,
)
from aidial_adapter_bedrock.llm.model.adapter import get_embeddings_model
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator
from aidial_adapter_bedrock.utils.env import get_aws_default_region
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log
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
            for deployment in ChatCompletionDeployment
        ]
    )


for deployment in ChatCompletionDeployment:
    app.add_chat_completion(
        deployment.deployment_id,
        BedrockChatCompletion(region=AWS_DEFAULT_REGION),
    )


@app.post("/openai/deployments/{deployment}/embeddings")
@dial_exception_decorator
async def embeddings(
    embeddings_type: EmbeddingsType = Header(
        alias="X-DIAL-Type", default=EmbeddingsType.SYMMETRIC
    ),
    embeddings_instruction: Optional[str] = Header(
        alias="X-DIAL-Instruction", default=None
    ),
    deployment: EmbeddingsDeployment = Path(...),
    request: dict = Body(..., examples=[EmbeddingsRequest.example()]),
):
    log.debug(f"request: {json.dumps(request)}")

    model = await get_embeddings_model(
        deployment=deployment, region=AWS_DEFAULT_REGION
    )

    response = await model.embeddings(
        request, embeddings_instruction, embeddings_type
    )

    return make_embeddings_response(deployment, response)
