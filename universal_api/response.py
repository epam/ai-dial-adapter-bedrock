import json
import time
import uuid
from typing import Generator, List

from pydantic import BaseModel
from starlette.responses import StreamingResponse

from llm.chat_model import ResponseData, ModelResponse
from universal_api.token_usage import TokenUsage


class ResponseParameters(BaseModel):
    created: int
    id: str
    model: str
    object: str


def generate_event_stream(
    params: ResponseParameters,
    gen: Generator[dict, None, None],
    usage: TokenUsage,
):
    def event_stream():
        for item in gen:
            if item == {}:
                choice = {"index": 0, "delta": {}, "finish_reason": "stop"}

                # Adding usage to the last chunk.
                # OpenAI itself leaves this field undefined/null, but we provide meaningful usage.
                yield wrap_streaming_chunk(
                    params, {"choices": [choice], "usage": usage.to_dict()}
                )

                yield "data: [DONE]\n\n"
                break

            choice = {"index": 0, "delta": item, "finish_reason": None}
            yield wrap_streaming_chunk(params, {"choices": [choice]})

    return StreamingResponse(
        event_stream(), headers={"Content-Type": "text/event-stream"}
    )


def wrap_streaming_chunk(params: ResponseParameters, payload: dict):
    return (
        "data: "
        + json.dumps(
            {
                **params.dict(),
                **payload,
            }
        )
        + "\n\n"
    )


def wrap_single_message(
    params: ResponseParameters,
    chunk: dict,
    usage: TokenUsage,
) -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": chunk,
                "finish_reason": "stop",
            }
        ],
        **params.dict(),
        "usage": usage.to_dict(),
    }


def make_response(
    streaming: bool, model_id: str, name: str, resp: ModelResponse
):
    id = str(uuid.uuid4())
    timestamp = int(time.time())

    if streaming:
        params = ResponseParameters(
            model=model_id, id=id, created=timestamp, object=name + ".chunk"
        )
        chunks: List[dict] = [{"role": "assistant"}, {"content": resp.content} | make_attachments(resp.data), {}]
        return generate_event_stream(params, (c for c in chunks), resp.usage)
    else:
        params = ResponseParameters(
            model=model_id, id=id, created=timestamp, object=name
        )
        chunk = {
            "role": "assistant",
            "content": resp.content,
        } | make_attachments(resp.data)
        return wrap_single_message(params, chunk, resp.usage)


def make_attachments(data: list[ResponseData]):
    return {} if len(data) == 0 else {
        "custom_content": {
            "attachments": [
                {
                    "index": index,
                    "type": d.mime_type,
                    "title": d.name,
                    "data": d.content
                } for index, d in enumerate(data)
            ]
        }
    }
