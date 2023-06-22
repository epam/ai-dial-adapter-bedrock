#!/usr/bin/env python3

import time
import uuid

import uvicorn
from fastapi import Body, FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware

from chat_client.langchain import chat, completion, create_model
from llm.chat_emulation import ChatEmulationType
from open_ai_api.types import ChatCompletionQuery, CompletionQuery
from utils.args import get_host_port_args
from utils.init import init

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


def wrap_message(
    object: str, response_id: str, timestamp: float, content: str
) -> dict:
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

    response_id = str(uuid.uuid4())
    timestamp = time.time()
    return wrap_message("chat.completion", response_id, timestamp, response)


@app.post("/completions")
def completions(
    query: CompletionQuery = Body(...),
):
    model = create_model(model_id=query.model, max_tokens=query.max_tokens)
    response = completion(model, query.prompt)

    response_id = str(uuid.uuid4())
    timestamp = time.time()
    return wrap_message("text_completion", response_id, timestamp, response)


if __name__ == "__main__":
    init()
    host, port = get_host_port_args()
    uvicorn.run(app, host=host, port=port)
