#!/usr/bin/env python3

import json
import logging
import time
import uuid
from typing import Generator, List, Tuple

import uvicorn
from fastapi import Body, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from llm.bedrock_custom import BedrockModels
from llm.bedrock_langchain import TokenUsage, chat, completion, create_model
from llm.chat_emulation.types import ChatEmulationType
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


def generate_event_stream(
    name: str, id: str, timestamp: int, gen: Generator[dict, None, None]
):
    def event_stream():
        for item in gen:
            if item == {}:
                choice = {"index": 0, "delta": {}, "finish_reason": "stop"}
                message = wrap_streaming_chunk(name, id, timestamp, choice)
                yield message
                yield "data: [DONE]\n\n"
                break

            choice = {"index": 0, "delta": item, "finish_reason": None}
            message = wrap_streaming_chunk(name, id, timestamp, choice)
            yield message

    return StreamingResponse(
        event_stream(), headers={"Content-Type": "text/event-stream"}
    )


def wrap_streaming_chunk(name: str, id: str, timestamp: int, choice: dict):
    return (
        "data: "
        + json.dumps(
            {
                "id": id,
                "object": name,
                "created": timestamp,
                "choices": [choice],
            }
        )
        + "\n\n"
    )


def wrap_single_message(
    name: str, id: str, timestamp: int, chunk: dict, usage: TokenUsage
) -> dict:
    return {
        "id": id,
        "object": name,
        "created": timestamp,
        "choices": [
            {
                "index": 0,
                "message": chunk,
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def make_response(name: str, streaming: bool, resp: Tuple[str, TokenUsage]):
    id = str(uuid.uuid4())
    timestamp = int(time.time())
    content, usage = resp

    if streaming:
        chunks: List[dict] = [{"role": "assistant"}, {"content": content}, {}]
        return generate_event_stream(
            name + ".chunk", id, timestamp, (c for c in chunks)
        )
    else:
        chunk = {
            "role": "assistant",
            "content": content,
        }
        return wrap_single_message(name, id, timestamp, chunk, usage)


@app.post("/chat/completions")
def chat_completions(
    chat_emulation_type: ChatEmulationType = Query(
        default=ChatEmulationType.META_CHAT,
        description="The chat emulation type for models which only support completion mode",
    ),
    query: ChatCompletionQuery = Body(...),
):
    model = create_model(model_id=query.model, max_tokens=query.max_tokens)
    messages = [message.to_base_message() for message in query.messages]
    response = chat(model, chat_emulation_type, messages)

    streaming = query.stream or False
    return make_response("chat.completion", streaming, response)


@app.post("/completions")
def completions(
    query: CompletionQuery = Body(...),
):
    model = create_model(model_id=query.model, max_tokens=query.max_tokens)
    response = completion(model, query.prompt)

    streaming = query.stream or False
    return make_response("text_completion", streaming, response)


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
