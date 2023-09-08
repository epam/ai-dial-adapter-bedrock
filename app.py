import logging.config

from fastapi import Body, FastAPI, Path, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm.bedrock_adapter import BedrockAdapter, BedrockModels
from llm.bedrock_models import BedrockDeployment
from llm.chat_emulation.types import ChatEmulationType
from server.exceptions import OpenAIException, open_ai_exception_decorator
from universal_api.request import ChatCompletionQuery
from universal_api.response import make_response
from utils.env import get_env
from utils.log_config import LogConfig
from utils.log_config import app_logger as log

logging.config.dictConfig(LogConfig().dict())  # type: ignore

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


class ModelDescription(BaseModel):
    id: str
    object: str


default_region = get_env("DEFAULT_REGION")


@app.get("/healthcheck")
def healthcheck():
    return Response("OK")


@app.get("/openai/models")
@open_ai_exception_decorator
async def models(
    region: str = Query(default=default_region, description="AWS region")
):
    bedrock_models = BedrockModels(region).models()

    models = [
        ModelDescription(id=model["modelId"], object="model").dict()
        for model in bedrock_models
    ]

    return {"object": "list", "data": models}


@app.post("/openai/deployments/{deployment}/chat/completions")
@open_ai_exception_decorator
async def chat_completions(
    deployment: BedrockDeployment = Path(...),
    chat_emulation_type: ChatEmulationType = Query(
        default=ChatEmulationType.META_CHAT,
        description="The chat emulation type for models which only support completion mode",
    ),
    region: str = Query(default=default_region, description="AWS region"),
    query: ChatCompletionQuery = Body(...),
):
    model_id = deployment.get_model_id()
    model = await BedrockAdapter.create(
        region=region, model_id=model_id, model_params=query
    )
    messages = [message.to_base_message() for message in query.messages]
    response = await model.achat(chat_emulation_type, messages)
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
