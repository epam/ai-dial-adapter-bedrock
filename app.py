import logging.config

from fastapi import Body, FastAPI, Path, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from llm.bedrock_adapter import BedrockAdapter, BedrockModels
from llm.bedrock_models import BedrockDeployment
from llm.chat_emulation.types import ChatEmulationType
from server.exceptions import OpenAIException, open_ai_exception_decorator
from universal_api.request import ChatCompletionRequest
from universal_api.response import ModelObject, ModelsResponse, make_response
from utils.env import get_env
from utils.log_config import LogConfig
from utils.log_config import app_logger as log

logging.config.dictConfig(LogConfig().dict())

app = FastAPI(
    description="Bedrock adapter for OpenAI Chat API",
    version="0.0.1",
)

# CORS

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints


default_region = get_env("DEFAULT_REGION")


@app.get("/healthcheck")
def healthcheck():
    return Response("OK")


@app.get("/openai/models")
@open_ai_exception_decorator
async def models(
    region: str = Query(default=default_region, description="AWS region")
) -> ModelsResponse:
    return get_models(region)


def get_models(region: str) -> ModelsResponse:
    bedrock_models = BedrockModels(region).models()
    models = [ModelObject(id=model["modelId"]) for model in bedrock_models]
    return ModelsResponse(data=models)


@app.post("/openai/deployments/{deployment}/chat/completions")
@open_ai_exception_decorator
async def chat_completions(
    deployment: BedrockDeployment = Path(...),
    chat_emulation_type: ChatEmulationType = Query(
        default=ChatEmulationType.META_CHAT,
        description="The chat emulation type for models which only support completion mode",
    ),
    region: str = Query(default=default_region, description="AWS region"),
    query: ChatCompletionRequest = Body(...),
):
    model_id = deployment.get_model_id()
    model = await BedrockAdapter.create(
        region=region, model_id=model_id, model_params=query
    )
    response = await model.achat(chat_emulation_type, query.messages)
    log.debug(f"response:\n{response}")

    return make_response(
        bool(query.stream), deployment, "chat.completion", response
    )


@app.exception_handler(OpenAIException)
async def exception_handler(request: Request, exc: OpenAIException):
    log.exception(f"Exception: {str(exc)}")
    return JSONResponse(
        status_code=exc.status_code, content={"error": exc.error}
    )
