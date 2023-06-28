#!/usr/bin/env python3

import logging

import uvicorn
from fastapi import Body, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm.bedrock_custom import BedrockCustom, BedrockModels
from llm.chat_emulation.types import ChatEmulationType
from open_ai.response import make_response
from open_ai.types import ChatCompletionQuery, CompletionQuery
from server.exceptions import OpenAIException, error_handling_decorator
from utils.args import get_host_port_args
from utils.init import init
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


@app.get("/models")
@error_handling_decorator
def models():
    bedrock_models = BedrockModels().models()

    models = [
        ModelDescription(id=model["modelId"], object="model").dict()
        for model in bedrock_models
    ]

    return {"object": "list", "data": models}


@app.post("/chat/completions")
@error_handling_decorator
def chat_completions(
    chat_emulation_type: ChatEmulationType = Query(
        default=ChatEmulationType.META_CHAT,
        description="The chat emulation type for models which only support completion mode",
    ),
    query: ChatCompletionQuery = Body(...),
):
    model_id = query.model
    model = BedrockCustom(model_id=model_id, model_params=query)
    messages = [message.to_base_message() for message in query.messages]
    response = model.chat(chat_emulation_type, messages)

    streaming = query.stream or False
    return make_response(streaming, model_id, "chat.completion", response)


@app.post("/completions")
@error_handling_decorator
def completions(
    query: CompletionQuery = Body(...),
):
    model_id = query.model
    model = BedrockCustom(model_id=model_id, model_params=query)
    response = model._call(query.prompt)

    streaming = query.stream or False
    return make_response(streaming, model_id, "text_completion", response)


@app.exception_handler(OpenAIException)
async def open_ai_exception_handler(request: Request, exc: OpenAIException):
    return JSONResponse(
        status_code=exc.status_code, content={"error": exc.error}
    )


if __name__ == "__main__":
    init()
    host, port = get_host_port_args()
    uvicorn.run(app, host=host, port=port)
