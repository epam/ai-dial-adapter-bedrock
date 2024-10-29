import re
from dataclasses import dataclass
from typing import Callable, List

import pytest
from openai import APIError, BadRequestError, UnprocessableEntityError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import Function
from pydantic import BaseModel

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from aidial_adapter_bedrock.utils.resource import Resource
from tests.conftest import TEST_SERVER_URL
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
    get_client,
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


chat_deployments = [
    ChatCompletionDeployment.AMAZON_TITAN_TG1_LARGE,
    ChatCompletionDeployment.AI21_J2_GRANDE_INSTRUCT,
    ChatCompletionDeployment.AI21_J2_JUMBO_INSTRUCT,
    ChatCompletionDeployment.AI21_J2_MID_V1,
    ChatCompletionDeployment.AI21_J2_ULTRA_V1,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_INSTANT_V1,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2_1,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_US,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_US,
    ChatCompletionDeployment.META_LLAMA2_13B_CHAT_V1,
    ChatCompletionDeployment.META_LLAMA2_70B_CHAT_V1,
    ChatCompletionDeployment.META_LLAMA3_8B_INSTRUCT_V1,
    ChatCompletionDeployment.META_LLAMA3_70B_INSTRUCT_V1,
    ChatCompletionDeployment.META_LLAMA3_1_405B_INSTRUCT_V1,
    ChatCompletionDeployment.META_LLAMA3_1_70B_INSTRUCT_V1,
    ChatCompletionDeployment.META_LLAMA3_1_8B_INSTRUCT_V1,
    ChatCompletionDeployment.COHERE_COMMAND_TEXT_V14,
]


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


def is_llama3(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.META_LLAMA3_8B_INSTRUCT_V1,
        ChatCompletionDeployment.META_LLAMA3_70B_INSTRUCT_V1,
    ]


def is_claude3(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_US,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_EU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_US,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_EU,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS,
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS_US,
    ]


def is_vision_model(deployment: ChatCompletionDeployment) -> bool:
    return is_claude3(deployment)


def are_tools_emulated(deployment: ChatCompletionDeployment) -> bool:
    return deployment in [
        ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2_1,
    ]


blue_pic = Resource.from_base64(
    type="image/png",
    data_base64="iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAIAAADZSiLoAAAAF0lEQVR4nGNkYPjPwMDAwMDAxAADCBYAG10BBdmz9y8AAAAASUVORK5CYII=",
)


def get_test_cases(
    deployment: ChatCompletionDeployment, streaming: bool
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

    test_case(
        name="dialog recall",
        messages=[
            user("my name is Anton"),
            ai("nice to meet you"),
            user("what's my name?"),
        ],
        expected=lambda s: "anton" in s.content.lower(),
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

    test_case(
        name="empty user message",
        max_tokens=1,
        messages=[user("")],
        expected=(
            streaming_error(
                ExpectedException(
                    type=BadRequestError,
                    message="messages: text content blocks must be non-empty",
                    status_code=400,
                )
            )
            if is_claude3(deployment)
            else expected_success
        ),
    )

    test_case(
        name="single space user message",
        max_tokens=1,
        messages=[user(" ")],
        expected=(
            streaming_error(
                ExpectedException(
                    type=BadRequestError,
                    message="messages: text content blocks must contain non-whitespace text",
                    status_code=400,
                )
            )
            if is_claude3(deployment)
            else expected_success
        ),
    )

    if is_vision_model(deployment):
        content = "describe the image"
        for idx, user_message in enumerate(
            [
                user_with_attachment_data(content, blue_pic),
                user_with_attachment_url(content, blue_pic),
                user_with_image_url(content, blue_pic),
            ]
        ):
            test_case(
                name=f"describe image {idx}",
                max_tokens=100,
                messages=[sys("be a helpful assistant"), user_message],  # type: ignore
                expected=lambda s: "blue" in s.content.lower(),
            )

    test_case(
        name="pinocchio in one token",
        max_tokens=1,
        messages=[user("tell me the full story of Pinocchio")],
        expected=lambda s: len(s.content.split()) <= 1,
    )

    # ai21 models do not support more than one stop word
    if "ai21" in deployment.model_id:
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

        for cities in [[("Glasgow", 15)], [("Glasgow", 15), ("London", 20)]]:
            function = GET_WEATHER_FUNCTION
            tool = function_to_tool(function)
            fun_name = function["name"]

            city_names = [name for name, _ in cities]
            city_temps = [temp for _, temp in cities]

            query = f"What's the temperature in {' and in '.join(city_names)} in celsius?"

            init_messages = [
                sys("act as a helpful assistant"),
                user("2+3=?"),
                ai("5"),
                user(query),
            ]

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
            )

    return test_cases


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    [
        test
        for deployment in chat_deployments
        for streaming in [False, True]
        for test in get_test_cases(deployment, streaming)
    ],
    ids=lambda test: test.get_id(),
)
async def test_chat_completion_openai(server, test: TestCase):
    client = get_client(TEST_SERVER_URL, test.deployment.value)

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
