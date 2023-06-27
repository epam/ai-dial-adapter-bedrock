import json
import time
import uuid
from typing import Generator, List, Tuple

from starlette.responses import StreamingResponse

from llm.bedrock_langchain import TokenUsage


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
        "usage": usage.to_dict(),
    }


def make_response(name: str, streaming: bool, resp: Tuple[str, TokenUsage]):
    id = str(uuid.uuid4())
    timestamp = int(time.time())
    content, usage = resp

    if streaming:
        # TODO: add token usage!
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
