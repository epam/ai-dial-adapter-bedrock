import re
from dataclasses import dataclass
from typing import Callable, List, Mapping

import openai
import pytest
from openai import APIError, BadRequestError, UnprocessableEntityError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import Function
from pydantic import BaseModel

from aidial_adapter_bedrock.aws_client_config import (
    AWSClientConfigFactory,
    UpstreamConfig,
)
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from tests.integration_tests.constants import SAMPLE_DOG_RESOURCE
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    ChatCompletionResult,
    ai,
    ai_function,
    ai_tools,
    chat_completion,
    function_request,
    function_response,
    function_to_tool,
    is_valid_function_call,
    is_valid_tool_call,
    sanitize_test_name,
    sys,
    tool_request,
    tool_response,
    user,
    user_with_attachment_data,
    user_with_attachment_url,
    user_with_image_url,
)


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
    temperature: float = 0.0

    def get_id(self):
        max_tokens_str = f"maxt={self.max_tokens}" if self.max_tokens else ""
        stop_sequence_str = f"stop={self.stop}" if self.stop else ""
        n_str = f"n={self.n}" if self.n else ""
        temperature_str = f"temp={self.temperature}" if self.temperature else ""
        return sanitize_test_name(
            f"{self.deployment.value} {self.streaming} {max_tokens_str} "
            f"{stop_sequence_str} {n_str} {temperature_str} {self.name}"
        )


_EAST = "us-east-1"
_WEST = "us-west-2"

chat_deployments: Mapping[ChatCompletionDeployment, str] = {
    ChatCompletionDeployment.AMAZON_TITAN_TG1_LARGE: _WEST,
    ChatCompletionDeployment.AI21_J2_GRANDE_INSTRUCT: _EAST,
    ChatCompletionDeployment.AI21_J2_JUMBO_INSTRUCT: _EAST,
    ChatCompletionDeployment.AI21_J2_MID_V1: _EAST,
    ChatCompletionDeployment.AI21_J2_ULTRA_V1: _EAST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_INSTANT_V1: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2_1: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_US: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_US: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2_US: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU: _WEST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU_US: _WEST,
    ChatCompletionDeployment.META_LLAMA3_8B_INSTRUCT_V1: _WEST,
    ChatCompletionDeployment.META_LLAMA3_70B_INSTRUCT_V1: _WEST,
    ChatCompletionDeployment.META_LLAMA3_1_8B_INSTRUCT_V1: _WEST,
    ChatCompletionDeployment.META_LLAMA3_1_70B_INSTRUCT_V1: _WEST,
    ChatCompletionDeployment.META_LLAMA3_1_405B_INSTRUCT_V1: _WEST,
    # Llama 3.2 1B is too unstable in responses for integration tests
    # Sometimes it cannot calculate 2+2
    # ChatCompletionDeployment.META_LLAMA3_2_1B_INSTRUCT_V1: _WEST,
    ChatCompletionDeployment.META_LLAMA3_2_3B_INSTRUCT_V1: _WEST,
    ChatCompletionDeployment.META_LLAMA3_2_11B_INSTRUCT_V1: _WEST,
    ChatCompletionDeployment.META_LLAMA3_2_90B_INSTRUCT_V1: _WEST,
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
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2_US,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_US,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_EU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU_US,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS_US,
        ChatCompletionDeployment.META_LLAMA3_1_70B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_1_405B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_2_90B_INSTRUCT_V1,
    ]


def supports_parallel_tool_calls(deployment: ChatCompletionDeployment) -> bool:
    return deployment not in [
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2_US,
        ChatCompletionDeployment.META_LLAMA3_1_70B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_1_405B_INSTRUCT_V1,
    ] and supports_tools(deployment)


def is_llama3(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.META_LLAMA3_8B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_70B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_1_8B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_1_70B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_1_405B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_2_1B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_2_3B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_2_11B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_2_90B_INSTRUCT_V1,
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
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2_US,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_US,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_EU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU_US,
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


def is_vision_model(deployment: ChatCompletionDeployment) -> bool:
    allowed_models = [
        ChatCompletionDeployment.META_LLAMA3_2_11B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_2_90B_INSTRUCT_V1,
    ]

    # Claude 3.5 Haiku was launched as a text-only model
    # https://assets.anthropic.com/m/61e7d27f8c8f5919/original/Claude-3-Model-Card.pdf
    excluded_models = {
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU_US,
    }

    is_allowed_model = is_claude3(deployment) or deployment in allowed_models
    is_excluded_model = deployment in excluded_models

    return is_allowed_model and not is_excluded_model


def are_tools_emulated(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2_1,
    ]


def get_test_cases(
    deployment: ChatCompletionDeployment, region: str, streaming: bool
) -> List[TestCase]:
    test_cases: List[TestCase] = []

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
        temperature: float = 0.0,
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
                temperature,
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
        expected_empty_message_error = ExpectedException(
            type=(
                openai.InternalServerError
                if streaming
                else openai.BadRequestError
            ),
            message="messages: text content blocks must be non-empty",
            status_code=500 if streaming else 400,
        )
    elif is_cohere(deployment):
        expected_empty_message_error = cohere_invalid_request_error
    elif is_llama3(deployment):
        expected_empty_message_error = ExpectedException(
            type=BadRequestError,
            message="Add text to the text field, and try again.",
            status_code=400,
        )

    test_case(
        name="empty user message",
        max_tokens=1,
        messages=[user("")],
        expected=expected_empty_message_error,
    )

    expected_whitespace_message = expected_success
    if is_claude3(deployment):
        expected_whitespace_message = ExpectedException(
            type=(
                openai.InternalServerError
                if streaming
                else openai.BadRequestError
            ),
            message="messages: text content blocks must contain non-whitespace text",
            status_code=500 if streaming else 400,
        )
    elif is_cohere(deployment):
        expected_whitespace_message = cohere_invalid_request_error
    elif is_llama3(deployment):
        expected_whitespace_message = ExpectedException(
            type=BadRequestError,
            message="Add text to the text field, and try again.",
            status_code=400,
        )

    test_case(
        name="single space user message",
        max_tokens=1,
        messages=[user(" ")],
        expected=expected_whitespace_message,
    )

    if is_vision_model(deployment):
        content = "describe the image"
        for idx, user_message in enumerate(
            [
                user_with_attachment_data(content, SAMPLE_DOG_RESOURCE),
                user_with_attachment_url(content, SAMPLE_DOG_RESOURCE),
                user_with_image_url(content, SAMPLE_DOG_RESOURCE),
            ]
        ):
            test_case(
                name=f"describe image {idx}",
                max_tokens=100,
                messages=[sys("be a helpful assistant"), user_message],  # type: ignore
                expected=lambda s: "dog" in s.content.lower(),
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
            name="out_of_turn",
            messages=[ai("hello"), user("what's 7+5?")],
            expected=(
                ExpectedException(
                    type=BadRequestError,
                    message="A conversation must start with a user message",
                    status_code=400,
                )
            ),
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

    city_config = (
        [[("Glasgow", 15)], [("Glasgow", 15), ("London", 20)]]
        if supports_parallel_tool_calls(deployment)
        else [[("Glasgow", 15)]]
    )

    if supports_tools(deployment):

        for cities in city_config:
            function = GET_WEATHER_FUNCTION
            tool = function_to_tool(function)
            fun_name = function["name"]

            city_names = [name for name, _ in cities]
            city_temps = [temp for _, temp in cities]

            query = f"What's the temperature in {' and in '.join(city_names)} in celsius?"

            init_messages = [
                user("2+3=?"),
                ai("5"),
                user(query),
            ]
            # Llama 3 works badly with system messages along tools
            if not is_llama3(deployment):
                init_messages.insert(0, sys("act as a helpful assistant"))

            def create_fun_args(city: str):
                return {
                    "location": city,
                    "format": "celsius",
                }

            def check_fun_args(city: str):
                return {
                    "location": lambda s: city.lower() in s.lower(),
                    "format": "celsius",
                }

            test_name_suffix = " ".join(city_names)

            # Functions
            test_case(
                name=f"weather function {test_name_suffix}",
                messages=init_messages,
                functions=[function],
                expected=lambda s, n=city_names[0]: is_valid_function_call(
                    s.function_call, fun_name, check_fun_args(n)
                ),
                temperature=1 if is_llama3(deployment) else 0.0,
            )

            function_req = ai_function(
                function_request(fun_name, create_fun_args(city_names[0]))
            )
            function_resp = function_response(
                fun_name, f"{city_temps[0]} celsius"
            )

            if len(cities) == 1:
                test_case(
                    name=f"weather function followup {test_name_suffix}",
                    messages=[
                        *init_messages,
                        function_req,
                        function_resp,
                    ],
                    functions=[function],
                    expected=lambda s, t=city_temps[0]: s.content_contains_all(
                        [t]
                    ),
                    temperature=1 if is_llama3(deployment) else 0.0,
                )
            else:
                test_case(
                    name=f"weather function followup {test_name_suffix}",
                    messages=[
                        *init_messages,
                        function_req,
                        function_resp,
                    ],
                    functions=[function],
                    expected=lambda s, n=city_names[1]: is_valid_function_call(
                        s.function_call, fun_name, check_fun_args(n)
                    ),
                    temperature=1 if is_llama3(deployment) else 0.0,
                )

            # Tools
            def create_tool_call_id(idx: int):
                return f"{fun_name}_{idx+1}"

            def check_tool_call_id(idx: int):
                def _check(id: str) -> bool:
                    return (
                        f"{fun_name}_{idx+1}" == id
                        if are_tools_emulated(deployment)
                        else True
                    )

                return _check

            expected_city_names = (
                city_names[:1] if are_tools_emulated(deployment) else city_names
            )

            test_case(
                name=f"weather tool {test_name_suffix}",
                messages=init_messages,
                tools=[tool],
                expected=lambda s, n=expected_city_names: all(
                    is_valid_tool_call(
                        s.tool_calls,
                        idx,
                        check_tool_call_id(idx),
                        fun_name,
                        check_fun_args(n[idx]),
                    )
                    for idx in range(len(n))
                ),
                temperature=1 if is_llama3(deployment) else 0.0,
            )

            tool_reqs = ai_tools(
                [
                    tool_request(
                        create_tool_call_id(idx),
                        fun_name,
                        create_fun_args(name),
                    )
                    for idx, (name, _) in enumerate(cities)
                ]
            )
            tool_resps = [
                tool_response(create_tool_call_id(idx), f"{temp} celsius")
                for idx, (_, temp) in enumerate(cities)
            ]

            test_case(
                name=f"weather tool followup {test_name_suffix}",
                messages=[*init_messages, tool_reqs, *tool_resps],
                tools=[tool],
                expected=lambda s, t=city_temps: s.content_contains_all(t),
                temperature=1 if is_llama3(deployment) else 0.0,
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
            test.temperature,
        )

    if isinstance(test.expected, ExpectedException):
        with pytest.raises(Exception) as exc_info:
            await run_chat_completion()

        actual_exc = exc_info.value

        assert isinstance(
            actual_exc, test.expected.type
        ), f"Actual exception type ({type(actual_exc)}) doesn't match the expected one ({test.expected.type})"
        actual_status_code = getattr(actual_exc, "status_code", None)
        assert actual_status_code == test.expected.status_code
        assert re.search(test.expected.message, str(actual_exc))
    else:
        actual_output = await run_chat_completion()
        assert test.expected(
            actual_output
        ), f"Failed output test, actual output: {actual_output}"
