import logging

from fastapi import Body, FastAPI, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm.bedrock_custom import BedrockCustom, BedrockModels
from llm.chat_emulation.types import ChatEmulationType
from server.exceptions import OpenAIException, error_handling_decorator
from universal_api.request import ChatCompletionQuery, CompletionQuery
from universal_api.response import make_response
from utils.log_config import LogConfig

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


default_region = "us-east-1"


@app.get("/openai/models")
@error_handling_decorator
async def models(
    region: str = Query(default=default_region, description="AWS region")
):
    bedrock_models = BedrockModels(region).models()

    models = [
        ModelDescription(id=model["modelId"], object="model").dict()
        for model in bedrock_models
    ]

    return {"object": "list", "data": models}


@app.post("/openai/deployments/{model_id}/chat/completions")
@error_handling_decorator
async def chat_completions(
    model_id: str = Path(...),
    chat_emulation_type: ChatEmulationType = Query(
        default=ChatEmulationType.META_CHAT,
        description="The chat emulation type for models which only support completion mode",
    ),
    region: str = Query(default=default_region, description="AWS region"),
    query: ChatCompletionQuery = Body(...),
):
    model = await BedrockCustom.create(
        region=region, model_id=model_id, model_params=query
    )
    messages = [message.to_base_message() for message in query.messages]
    response = await model.achat(chat_emulation_type, messages)

    streaming = query.stream or False
    return make_response(streaming, model_id, "chat.completion", response)


@app.post("/openai/deployments/{model_id}/completions")
@error_handling_decorator
async def completions(
    model_id: str = Path(...),
    region: str = Query(default=default_region, description="AWS region"),
    query: CompletionQuery = Body(...),
):
    model = await BedrockCustom.create(
        region=region, model_id=model_id, model_params=query
    )
    response = await model.acall(query.prompt)

    streaming = query.stream or False
    return make_response(streaming, model_id, "text_completion", response)


@app.exception_handler(OpenAIException)
async def open_ai_exception_handler(request: Request, exc: OpenAIException):
    return JSONResponse(
        status_code=exc.status_code, content={"error": exc.error}
    )
