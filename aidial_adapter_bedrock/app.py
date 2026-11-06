from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from aidial_adapter_anthropic.passthrough import mount_anthropic_api
from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import TelemetryConfig
from aidial_sdk.translators.messages_to_chat_completions import (
    mount_translator,
)
from fastapi import FastAPI, Request

from aidial_adapter_bedrock.anthropic_passthrough import (
    get_anthropic_client,
)
from aidial_adapter_bedrock.bedrock import (
    create_anthropic_client,
    create_boto_client,
    get_anthropic_http_client,
    get_dial_client_pool,
)
from aidial_adapter_bedrock.chat_completion import BedrockChatCompletion
from aidial_adapter_bedrock.dial_api.response import ModelObject, ModelsResponse
from aidial_adapter_bedrock.embeddings import BedrockEmbeddings
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator
from aidial_adapter_bedrock.upstream_config import get_sts_client
from aidial_adapter_bedrock.utils.adapter_deployments import (
    get_static_deployments,
)
from aidial_adapter_bedrock.utils.log_config import configure_loggers


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await create_anthropic_client.clear()
        await create_boto_client.clear()
        await get_sts_client.clear()
        await get_anthropic_http_client.clear()
        await get_dial_client_pool.clear()


app = DIALApp(
    description="AWS Bedrock adapter for DIAL API",
    telemetry_config=TelemetryConfig(),
    add_healthcheck=True,
    lifespan=lifespan,
)

# NOTE: configuring logger after the DIAL telemetry is initialized,
# because it may have configured the root logger on its own.
configure_loggers()

deployments = get_static_deployments()


@app.get("/openai/models")
@dial_exception_decorator
async def models():
    return ModelsResponse(
        data=list(map(ModelObject.chat_completions, deployments.chat))
        + list(map(ModelObject.embeddings, deployments.embeddings))
    )


app.add_chat_completion("{deployment_id}", BedrockChatCompletion())
app.add_embeddings("{deployment_id}", BedrockEmbeddings())


async def get_dial_client(_: Request) -> httpx.AsyncClient:
    return get_anthropic_http_client()


mount_anthropic_api(app, get_anthropic_client, path="/anthropic")
mount_translator(app, get_dial_client, path="/openaiChatCompletions")
