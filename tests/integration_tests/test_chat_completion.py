import json
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping

import pytest
from openai import APIError, BadRequestError, UnprocessableEntityError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.completion_create_params import Function
from pydantic import BaseModel

from aidial_adapter_bedrock.aws_client_config import (
    AWSClientConfigFactory,
    UpstreamConfig,
)
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from tests.utils.json import match_objects
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    GET_WEATHER_TOOL,
    ChatCompletionResult,
    ai,
    ai_function,
    ai_tools,
    chat_completion,
    function_request,
    function_response,
    sanitize_test_name,
    sys,
    tool_request,
    tool_response,
    user,
)


def is_valid_function_call(
    call: FunctionCall | None, expected_name: str, expected_args: Any
) -> bool:
    assert call is not None
    assert call.name == expected_name
    obj = json.loads(call.arguments)
    match_objects(expected_args, obj)
    return True


def is_valid_tool_calls(
    calls: List[ChatCompletionMessageToolCall] | None,
    check_tool_id: Callable[[str], bool],
    expected_name: str,
    expected_args: Any,
) -> bool:
    assert calls is not None
    assert len(calls) == 1
    call = calls[0]

    function = call.function
    assert check_tool_id(call.id)
    assert expected_name == function.name

    actual_args = json.loads(function.arguments)
    match_objects(expected_args, actual_args)
    return True


class ExpectedException(BaseModel):
    type: type[APIError]
    message: str
    status_code: int | None = None


def expected_success(*args, **kwargs):
    return True


@dataclass
class TestCase:
    __test__ = False

    name: str
    region: str
    deployment: ChatCompletionDeployment
    streaming: bool

    messages: List[ChatCompletionMessageParam]

    expected: Callable[[ChatCompletionResult], bool] | ExpectedException

    max_tokens: int | None
    stop: List[str] | None

    n: int | None

    functions: List[Function] | None
    tools: List[ChatCompletionToolParam] | None

    def get_id(self):
        max_tokens_str = f"maxt={self.max_tokens}" if self.max_tokens else ""
        stop_sequence_str = f"stop={self.stop}" if self.stop else ""
        n_str = f"n={self.n}" if self.n else ""
        return sanitize_test_name(
            f"{self.deployment.value} {self.streaming} {max_tokens_str} "
            f"{stop_sequence_str} {n_str} {self.name}"
        )


_EAST = "us-east-1"
_WEST = "us-west-2"

chat_deployments: Mapping[ChatCompletionDeployment, str] = {
    ChatCompletionDeployment.AMAZON_TITAN_TG1_LARGE: _WEST,
    ChatCompletionDeployment.AI21_J2_GRANDE_INSTRUCT: _EAST,
    ChatCompletionDeployment.AI21_J2_JUMBO_INSTRUCT: _EAST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_INSTANT_V1: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2_1: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_US: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_US: _WEST,
    ChatCompletionDeployment.META_LLAMA2_13B_CHAT_V1: _WEST,
    ChatCompletionDeployment.META_LLAMA2_70B_CHAT_V1: _WEST,
    ChatCompletionDeployment.META_LLAMA3_8B_INSTRUCT_V1: _WEST,
    ChatCompletionDeployment.META_LLAMA3_70B_INSTRUCT_V1: _WEST,
    ChatCompletionDeployment.META_LLAMA3_1_405B_INSTRUCT_V1: _WEST,
    ChatCompletionDeployment.META_LLAMA3_1_70B_INSTRUCT_V1: _WEST,
    ChatCompletionDeployment.META_LLAMA3_1_8B_INSTRUCT_V1: _WEST,
    ChatCompletionDeployment.COHERE_COMMAND_TEXT_V14: _WEST,
    ChatCompletionDeployment.COHERE_COMMAND_LIGHT_TEXT_V14: _WEST,
}


def supports_tools(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2_1,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_US,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_EU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_US,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_EU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_US,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_EU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS_US,
    ]


def are_tools_emulated(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2_1,
    ]


def is_llama3(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.META_LLAMA3_8B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_70B_INSTRUCT_V1,
    ]


def is_cohere(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.COHERE_COMMAND_LIGHT_TEXT_V14,
        ChatCompletionDeployment.COHERE_COMMAND_TEXT_V14,
    ]


def is_claude3(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_US,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_EU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_US,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_EU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_US,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_EU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS_US,
    ]


def is_ai21(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.AI21_J2_GRANDE_INSTRUCT,
        ChatCompletionDeployment.AI21_J2_JUMBO_INSTRUCT,
    ]


cohere_invalid_request_error = ExpectedException(
    type=BadRequestError,
    message="Invalid parameter combination",
    status_code=400,
)


def get_test_cases(
    deployment: ChatCompletionDeployment, region: str, streaming: bool
) -> List[TestCase]:
    test_cases: List[TestCase] = []

    def streaming_error(exc: ExpectedException) -> ExpectedException:
        if streaming:
            return ExpectedException(
                type=APIError,
                message="An error occurred during streaming",
            )
        else:
            return exc

    def test_case(
        name: str,
        messages: List[ChatCompletionMessageParam],
        expected: (
            Callable[[ChatCompletionResult], bool] | ExpectedException
        ) = expected_success,
        n: int | None = None,
        max_tokens: int | None = None,
        stop: List[str] | None = None,
        functions: List[Function] | None = None,
        tools: List[ChatCompletionToolParam] | None = None,
    ) -> None:
        test_cases.append(
            TestCase(
                name,
                region,
                deployment,
                streaming,
                messages,
                expected,
                max_tokens,
                stop,
                n,
                functions,
                tools,
            )
        )

    def dial_recall_expected(r: ChatCompletionResult):
        content = r.content.lower()
        success = "anton" in content
        # Amazon Titan and Cohere performances have degraded recently
        if deployment in [
            ChatCompletionDeployment.AMAZON_TITAN_TG1_LARGE,
            ChatCompletionDeployment.COHERE_COMMAND_TEXT_V14,
        ]:
            return not success
        return success

    test_case(
        name="dialog recall",
        messages=[
            user("my name is Anton"),
            ai("nice to meet you"),
            user("what's my name?"),
        ],
        max_tokens=32,
        expected=dial_recall_expected,
    )

    test_case(
        name="2+3=5",
        messages=[user("compute (2+3)")],
        expected=lambda s: "5" in s.content,
    )

    test_case(
        name="empty system message",
        messages=[sys(""), user("compute (2+4)")],
        expected=lambda s: "6" in s.content,
    )

    query = 'Reply with "Hello"'
    if deployment == ChatCompletionDeployment.ANTHROPIC_CLAUDE_INSTANT_V1:
        query = 'Print "Hello"'

    test_case(
        name="hello",
        messages=[user(query)],
        expected=lambda s: "hello" in s.content.lower()
        or "hi" in s.content.lower(),
    )

    test_case(
        name="empty dialog",
        max_tokens=1,
        messages=[],
        expected=ExpectedException(
            type=UnprocessableEntityError,
            message="List of messages must not be empty",
            status_code=422,
        ),
    )

    expected_empty_message_error = expected_success
    if is_claude3(deployment):
        expected_empty_message_error = streaming_error(
            ExpectedException(
                type=BadRequestError,
                message="messages: text content blocks must be non-empty",
                status_code=400,
            )
        )
    elif is_cohere(deployment):
        expected_empty_message_error = streaming_error(
            cohere_invalid_request_error
        )

    test_case(
        name="empty user message",
        max_tokens=1,
        messages=[user("")],
        expected=expected_empty_message_error,
    )

    expected_whitespace_message = expected_success
    if is_claude3(deployment):
        expected_whitespace_message = streaming_error(
            ExpectedException(
                type=BadRequestError,
                message="messages: text content blocks must contain non-whitespace text",
                status_code=400,
            )
        )
    elif is_cohere(deployment):
        expected_whitespace_message = streaming_error(
            cohere_invalid_request_error
        )

    test_case(
        name="single space user message",
        max_tokens=1,
        messages=[user(" ")],
        expected=expected_whitespace_message,
    )

    test_case(
        name="pinocchio in one token",
        max_tokens=1,
        messages=[user("tell me the full story of Pinocchio")],
        expected=lambda s: len(s.content.split()) <= 1,
    )

    # ai21 models do not support more than one stop word
    if is_ai21(deployment):
        stop = ["John"]
    else:
        stop = ["John", "john"]

    test_case(
        name="stop sequence",
        stop=stop,
        messages=[user('Reply with "John"')],
        expected=lambda s: "John" not in s.content.lower(),
    )

    if is_llama3(deployment):
        test_case(
            name="out of turn",
            messages=[ai("hello"), user("what's 7+5?")],
            expected=lambda s: "12" in s.content.lower(),
        )

        test_case(
            name="many system",
            messages=[
                sys("act as a helpful assistant"),
                sys("act as a calculator"),
                user("2+5=?"),
            ],
            expected=lambda s: "7" in s.content.lower(),
        )

    if supports_tools(deployment):
        query = "What's the temperature in Glasgow in celsius?"

        function_args_checker = {
            "location": lambda s: "glasgow" in s.lower(),
            "format": "celsius",
        }

        function_args = {"location": "Glasgow", "format": "celsius"}

        name = GET_WEATHER_FUNCTION["name"]

        # Functions
        test_case(
            name="weather function",
            messages=[user(query)],
            functions=[GET_WEATHER_FUNCTION],
            expected=lambda s: is_valid_function_call(
                s.function_call, name, function_args_checker
            ),
        )

        function_req = ai_function(function_request(name, function_args))
        function_resp = function_response(name, "15 celsius")

        test_case(
            name="weather function followup",
            messages=[user(query), function_req, function_resp],
            functions=[GET_WEATHER_FUNCTION],
            expected=lambda s: "15" in s.content.lower(),
        )

        # Tools
        tool_call_id = f"{name}_1"

        def check_tool_call_id(id: str):
            return (
                tool_call_id == id if are_tools_emulated(deployment) else True
            )

        test_case(
            name="weather tool",
            messages=[user(query)],
            tools=[GET_WEATHER_TOOL],
            expected=lambda s: is_valid_tool_calls(
                s.tool_calls, check_tool_call_id, name, function_args_checker
            ),
        )

        tool_req = ai_tools([tool_request(tool_call_id, name, function_args)])
        tool_resp = tool_response(tool_call_id, "15 celsius")

        test_case(
            name="weather tool followup",
            messages=[user(query), tool_req, tool_resp],
            tools=[GET_WEATHER_TOOL],
            expected=lambda s: "15" in s.content.lower(),
        )

    return test_cases


def get_extra_headers(region: str) -> Mapping[str, str]:
    return {
        AWSClientConfigFactory.UPSTREAM_CONFIG_HEADER_NAME: UpstreamConfig(
            region=region
        ).json()
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    [
        test
        for deployment, region in chat_deployments.items()
        for streaming in [False, True]
        for test in get_test_cases(deployment, region, streaming)
    ],
    ids=lambda test: test.get_id(),
)
async def test_chat_completion_openai(get_openai_client, test: TestCase):
    client = get_openai_client(
        test.deployment.value, get_extra_headers(test.region)
    )

    async def run_chat_completion() -> ChatCompletionResult:
        return await chat_completion(
            client,
            test.messages,
            test.streaming,
            test.stop,
            test.max_tokens,
            test.n,
            test.functions,
            test.tools,
        )

    if isinstance(test.expected, ExpectedException):
        with pytest.raises(Exception) as exc_info:
            await run_chat_completion()

        actual_exc = exc_info.value

        assert isinstance(actual_exc, test.expected.type)
        actual_status_code = getattr(actual_exc, "status_code", None)
        assert actual_status_code == test.expected.status_code
        assert re.search(test.expected.message, str(actual_exc))
    else:
        actual_output = await run_chat_completion()
        assert test.expected(
            actual_output
        ), f"Failed output test, actual output: {actual_output}"
