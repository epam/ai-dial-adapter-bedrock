import json
import re
from typing import Any, Callable, List, Optional

import httpx
from aidial_sdk.utils.merge_chunks import (
    cleanup_indices,
    merge_chat_completion_chunks,
)
from openai import AsyncAzureOpenAI, AsyncStream
from openai._types import NOT_GIVEN
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartParam,
)
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as ToolFunction,
)
from openai.types.chat.completion_create_params import Function
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel

from aidial_adapter_bedrock.utils.resource import Resource
from tests.utils.json import match_objects


def sys(content: str) -> ChatCompletionSystemMessageParam:
    return {"role": "system", "content": content}


def ai(content: str) -> ChatCompletionAssistantMessageParam:
    return {"role": "assistant", "content": content}


def ai_function(
    function_call: ToolFunction,
) -> ChatCompletionAssistantMessageParam:
    return {"role": "assistant", "function_call": function_call}


def ai_tools(
    tool_calls: List[ChatCompletionMessageToolCallParam],
) -> ChatCompletionAssistantMessageParam:
    return {"role": "assistant", "tool_calls": tool_calls}


def user(
    content: str | List[ChatCompletionContentPartParam],
) -> ChatCompletionUserMessageParam:
    return {"role": "user", "content": content}


def user_with_attachment_data(
    content: str, resource: Resource
) -> ChatCompletionUserMessageParam:
    return {
        "role": "user",
        "content": content,
        "custom_content": {  # type: ignore
            "attachments": [
                {"type": resource.type, "data": resource.data_base64}
            ]
        },
    }


def user_with_image_content_part(
    content: str, resource: Resource
) -> ChatCompletionUserMessageParam:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": content},
            {"type": "image_url", "image_url": {"url": resource.to_data_url()}},
        ],
    }


def user_with_attachment_url(
    content: str, resource: Resource
) -> ChatCompletionUserMessageParam:
    return {
        "role": "user",
        "content": content,
        "custom_content": {  # type: ignore
            "attachments": [
                {
                    "type": resource.type,
                    "url": resource.to_data_url(),
                }
            ]
        },
    }


def user_with_image_url(
    content: str, image: Resource
) -> ChatCompletionUserMessageParam:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": content},
            {
                "type": "image_url",
                "image_url": {"url": image.to_data_url()},
            },
        ],
    }


def function_request(name: str, args: Any) -> ToolFunction:
    return {"name": name, "arguments": json.dumps(args)}


def tool_request(
    id: str, name: str, args: Any
) -> ChatCompletionMessageToolCallParam:
    return {
        "id": id,
        "type": "function",
        "function": function_request(name, args),
    }


def function_response(
    name: str, content: str
) -> ChatCompletionFunctionMessageParam:
    return {"role": "function", "name": name, "content": content}


def tool_response(id: str, content: str) -> ChatCompletionToolMessageParam:
    return {"role": "tool", "tool_call_id": id, "content": content}


def function_to_tool(function: FunctionDefinition) -> ChatCompletionToolParam:
    return {"type": "function", "function": function}


def sanitize_test_name(name: str) -> str:
    name = "".join(
        c if (c.isalnum() or c in "/:") else "_" for c in name.lower()
    )
    return re.sub("_+", "_", name)


class ChatCompletionResult(BaseModel):
    response: ChatCompletion

    @property
    def message(self) -> ChatCompletionMessage:
        return self.response.choices[0].message

    @property
    def content(self) -> str:
        return self.message.content or ""

    @property
    def contents(self) -> List[str]:
        return [
            choice.message.content or "" for choice in self.response.choices
        ]

    @property
    def usage(self) -> CompletionUsage | None:
        return self.response.usage

    @property
    def function_call(self) -> FunctionCall | None:
        return self.message.function_call

    @property
    def tool_calls(self) -> List[ChatCompletionMessageToolCall] | None:
        return self.message.tool_calls

    def content_contains_all(self, matches: List[Any]) -> bool:
        return all(
            str(match).lower() in self.content.lower() for match in matches
        )


def for_all_choices(
    predicate: Callable[[str], bool], n: int = 1
) -> Callable[[ChatCompletionResult], bool]:
    def f(resp: ChatCompletionResult) -> bool:
        contents = resp.contents
        assert (
            len(contents) == n
        ), f"Expected {n} candidates, got {len(contents)}"
        return all(predicate(content) for content in contents)

    return f


async def chat_completion(
    client: AsyncAzureOpenAI,
    messages: List[ChatCompletionMessageParam],
    stream: bool,
    stop: Optional[List[str]],
    max_tokens: Optional[int],
    n: Optional[int],
    functions: List[Function] | None,
    tools: List[ChatCompletionToolParam] | None,
    temperature: float = 0.0,
) -> ChatCompletionResult:
    async def get_response() -> ChatCompletion:
        response = await client.chat.completions.create(
            model="dummy-model",
            messages=messages,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            function_call="auto" if functions is not None else NOT_GIVEN,
            functions=functions or NOT_GIVEN,
            tool_choice="auto" if tools is not None else NOT_GIVEN,
            tools=tools or NOT_GIVEN,
        )

        if isinstance(response, AsyncStream):
            chunks: List[dict] = []
            async for chunk in response:
                chunks.append(chunk.dict())

            response_dict = merge_chat_completion_chunks(*chunks)

            for choice in response_dict["choices"]:
                choice["message"] = cleanup_indices(choice["delta"])
                del choice["delta"]

            response_dict["object"] = "chat.completion"

            return ChatCompletion.parse_obj(response_dict)
        else:
            return response

    response = await get_response()
    return ChatCompletionResult(response=response)


async def truncate_prompt(
    client: httpx.AsyncClient,
    model: str,
    messages: List[ChatCompletionMessageParam],
    max_prompt_tokens: Optional[int],
) -> dict:
    request: dict = {"messages": messages}
    if max_prompt_tokens is not None:
        request["max_prompt_tokens"] = max_prompt_tokens

    request = {"inputs": [request]}

    response = await client.post(
        url=f"/openai/deployments/{model}/truncate_prompt",
        json=request,
        headers={"api-key": "dummy"},
    )

    response.raise_for_status()
    return response.json()


async def tokenize(
    client: httpx.AsyncClient,
    model: str,
    messages: List[ChatCompletionMessageParam],
) -> dict:

    request = {
        "inputs": [
            {
                "type": "request",
                "value": {"messages": messages},
            }
        ]
    }

    response = await client.post(
        url=f"/openai/deployments/{model}/tokenize",
        json=request,
        headers={"api-key": "dummy"},
    )

    response.raise_for_status()
    return response.json()


GET_WEATHER_FUNCTION: Function = {
    "name": "get_current_weather",
    "description": "Get the current weather",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use. Infer this from the users location.",
            },
        },
        "required": ["location", "format"],
    },
}


def is_valid_function_call(
    call: FunctionCall | None, expected_name: str, expected_args: Any
) -> bool:
    assert call is not None
    assert call.name == expected_name
    obj = json.loads(call.arguments)
    match_objects(expected_args, obj)
    return True


def is_valid_tool_call(
    calls: List[ChatCompletionMessageToolCall] | None,
    tool_call_idx: int,
    check_tool_id: Callable[[str], bool],
    expected_name: str,
    expected_args: dict,
) -> bool:
    assert calls is not None

    call = calls[tool_call_idx]

    function = call.function
    assert check_tool_id(call.id)
    assert expected_name == function.name

    actual_args = json.loads(function.arguments)
    match_objects(expected_args, actual_args)
    return True
