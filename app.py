#!/usr/bin/env python3

import logging
import time
import uuid

import uvicorn
from fastapi import Body, FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chat_client.langchain import chat, completion, create_model
from llm.bedrock import BedrockModels
from llm.chat_emulation import ChatEmulationType
from open_ai_api.types import ChatCompletionQuery, CompletionQuery
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


def wrap_message(object: str, content: str) -> dict:
    response_id = str(uuid.uuid4())
    timestamp = time.time()

    return {
        "id": response_id,
        "object": object,
        "created": timestamp,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
        },
    }


@app.post("/{chat_emulation_type}/chat/completions")
def chat_completions(
    chat_emulation_type: ChatEmulationType = Path(
        description="The chat emulation type for models which only support completion mode",
    ),
    query: ChatCompletionQuery = Body(...),
):
    model = create_model(model_id=query.model, max_tokens=query.max_tokens)
    messages = [message.to_base_message() for message in query.messages]
    response = chat(model, chat_emulation_type, messages)

    return wrap_message("chat.completion", response)


@app.post("/completions")
def completions(
    query: CompletionQuery = Body(...),
):
    model = create_model(model_id=query.model, max_tokens=query.max_tokens)
    response = completion(model, query.prompt)

    return wrap_message("text_completion", response)


class ModelDescription(BaseModel):
    id: str
    object: str


@app.get("/models")
def models():
    bedrock_models = BedrockModels().models()

    models = [
        ModelDescription(id=model["modelId"], object="model").dict()
        for model in bedrock_models
    ]

    return {"object": "list", "data": models}


if __name__ == "__main__":
    init()
    host, port = get_host_port_args()
    uvicorn.run(app, host=host, port=port)
