import contextlib
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Mapping, Unpack

import openai
import pytest
from openai import (
    APIError,
    AsyncAzureOpenAI,
    BadRequestError,
    UnprocessableEntityError,
)
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import Function
from pydantic import BaseModel

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment as D
from aidial_adapter_bedrock.utils.region_deployment import RegionDeployment
from tests.integration_tests.constants import SAMPLE_DOG_RESOURCE
from tests.unit_tests.test_configuration import (
    deployments_supporting_optimized_latency,
)
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    ChatCompletionArgs,
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
from tests.utils.selector import Selector, pred


class ExpectedException(BaseModel):
    type: type[APIError]
    message: str
    display_message: str | None = None
    status_code: int | None = None


@contextlib.asynccontextmanager
async def expected_exception(
    cls: type[APIError],
    message: str,
    display_message: str | None = None,
    status_code: int | None = None,
):
    try:
        yield
    except Exception as e:
        assert isinstance(
            e, cls
        ), f"Actual exception type ({type(e)}) doesn't match the expected one ({cls})"
        actual_status_code = getattr(e, "status_code", None)
        assert actual_status_code == status_code
        assert re.search(message, str(e))
        assert (e.body or {}).get("display_message") == display_message  # type: ignore


def expected_success(*args, **kwargs):
    return True


Deployment = RegionDeployment[D]


@dataclass
class TestCase:
    __test__ = False

    name: str
    region: str
    deployment: Deployment
    streaming: bool

    messages: List[ChatCompletionMessageParam]

    expected: Callable[[ChatCompletionResult], bool] | ExpectedException

    max_tokens: int | None
    stop: List[str] | None

    n: int | None

    functions: List[Function] | None
    tools: List[ChatCompletionToolParam] | None
    tool_choice: ChatCompletionToolChoiceOptionParam | None
    temperature: float = 0.0

    def get_id(self):
        maxt = f"maxt:{self.max_tokens}" if self.max_tokens else None
        stop = f"stop:{self.stop}" if self.stop else None
        n = f"n:{self.n}" if self.n else None
        temp = f"temp:{self.temperature}" if self.temperature else None
        tools = None
        if self.tools is not None:
            tools = "tools"
            if self.tool_choice is not None:
                if isinstance(self.tool_choice, str):
                    tools += f":{self.tool_choice}"
                else:
                    tools += ":forced"

        return sanitize_test_name(
            "/".join(
                str(part)
                for part in [
                    self.deployment.value,
                    self.streaming,
                    maxt,
                    stop,
                    n,
                    temp,
                    tools,
                    self.name,
                ]
                if part is not None
            )
        )


_WEST = "us-west-2"
_EAST_1 = "us-east-1"
_EAST_2 = "us-east-2"


chat_deployments: Mapping[Deployment, str] = {
    D.AMAZON_TITAN_TG1_LARGE: _WEST,
    D.AI21_J2_GRANDE_INSTRUCT: _EAST_1,
    D.AI21_J2_JUMBO_INSTRUCT: _EAST_1,
    D.AI21_J2_MID_V1: _EAST_1,
    D.AI21_J2_ULTRA_V1: _EAST_1,
    D.AI21_JAMBA_1_5_LARGE_V1: _EAST_1,
    D.AI21_JAMBA_1_5_MINI_V1: _EAST_1,
    D.ANTHROPIC_CLAUDE_INSTANT_V1: _WEST,
    D.ANTHROPIC_CLAUDE_V2: _WEST,
    D.ANTHROPIC_CLAUDE_V2_1: _WEST,
    D.ANTHROPIC_CLAUDE_V3_SONNET.US: _WEST,
    D.ANTHROPIC_CLAUDE_V3_5_SONNET.US: _WEST,
    D.ANTHROPIC_CLAUDE_V3_5_SONNET_V2.US: _WEST,
    D.ANTHROPIC_CLAUDE_V3_5_HAIKU.US: _WEST,
    D.ANTHROPIC_CLAUDE_V3_7_SONNET.US: _EAST_1,
    D.ANTHROPIC_CLAUDE_V4_SONNET.US: _EAST_1,
    D.ANTHROPIC_CLAUDE_V4_OPUS.US: _EAST_1,
    D.META_LLAMA3_8B_INSTRUCT_V1: _WEST,
    D.META_LLAMA3_70B_INSTRUCT_V1: _WEST,
    D.META_LLAMA3_1_8B_INSTRUCT_V1: _WEST,
    D.META_LLAMA3_1_70B_INSTRUCT_V1.US: _WEST,
    D.META_LLAMA3_1_405B_INSTRUCT_V1.US: _EAST_2,
    # Llama 3.2 1B is too unstable in responses for integration tests
    # Sometimes it cannot calculate 2+2
    # D.META_LLAMA3_2_1B_INSTRUCT_V1.US: _WEST_2,
    D.META_LLAMA3_2_3B_INSTRUCT_V1.US: _WEST,
    D.META_LLAMA3_2_11B_INSTRUCT_V1.US: _WEST,
    D.META_LLAMA3_2_90B_INSTRUCT_V1.US: _WEST,
    D.META_LLAMA3_3_70B_INSTRUCT_V1: _EAST_2,
    D.COHERE_COMMAND_TEXT_V14: _WEST,
    D.COHERE_COMMAND_LIGHT_TEXT_V14: _WEST,
    D.COHERE_COMMAND_R_V1: _WEST,
    D.COHERE_COMMAND_R_PLUS_V1: _WEST,
    D.AMAZON_NOVA_MICRO: _EAST_1,
    D.AMAZON_NOVA_PRO.US: _EAST_1,
    D.AMAZON_NOVA_LITE: _EAST_1,
    D.DEEPSEEK_R1_V2.US: _EAST_1,
}


def is_retired_model(deployment: D) -> bool:
    return deployment in {
        D.AI21_J2_GRANDE_INSTRUCT,
        D.AI21_J2_JUMBO_INSTRUCT,
        D.AI21_J2_MID_V1,
        D.AI21_J2_ULTRA_V1,
        # FIXME: add it
        # D.STABILITY_STABLE_DIFFUSION_XL, _WEST
        # D.STABILITY_STABLE_DIFFUSION_XL_V1, _WEST
    }


def select(p: Selector[D], xs: List[Deployment]) -> List[Deployment]:
    return [x for x in xs if p(x.origin)]


deployments = list(chat_deployments.keys())
alive_deployments = select(~pred(is_retired_model), deployments)


@pytest.fixture
def get_deployment_region() -> Mapping[Deployment, str]:
    return chat_deployments


def supports_tools(deployment: D) -> bool:
    return is_claude_3_or_4(deployment) or deployment in [
        D.ANTHROPIC_CLAUDE_V2_1,
        D.META_LLAMA3_1_70B_INSTRUCT_V1,
        D.META_LLAMA3_1_405B_INSTRUCT_V1,
        D.META_LLAMA3_2_90B_INSTRUCT_V1,
        D.META_LLAMA3_3_70B_INSTRUCT_V1,
        # Technically, Nova Micro supports tools, but it's unstable
        # D.AMAZON_NOVA_MICRO,
        D.AMAZON_NOVA_PRO,
        D.AMAZON_NOVA_LITE,
        D.AMAZON_NOVA_MICRO,
        # DeepSeek via Converse API doesn't support tools even though
        # tool support is claimed in the official documentation:
        # https://api-docs.deepseek.com/guides/function_calling
        # D.DEEPSEEK_R1_V2,
        D.AI21_JAMBA_1_5_LARGE_V1,
        # Mini is very bad with tools
        # D.AI21_JAMBA_1_5_MINI_V1,
        D.COHERE_COMMAND_R_V1,
        D.COHERE_COMMAND_R_PLUS_V1,
    ]


def supports_forced_tool_choice(deployment: D) -> bool:
    return supports_tools(deployment) and is_claude_3_or_4(deployment)


def supports_parallel_tool_calls(deployment: D) -> bool:
    return (
        deployment
        not in [
            D.ANTHROPIC_CLAUDE_V3_5_SONNET_V2,
            D.ANTHROPIC_CLAUDE_V3_7_SONNET,
            D.ANTHROPIC_CLAUDE_V4_OPUS,
            D.ANTHROPIC_CLAUDE_V4_SONNET,
            D.META_LLAMA3_1_70B_INSTRUCT_V1,
            D.META_LLAMA3_1_405B_INSTRUCT_V1,
            D.META_LLAMA3_3_70B_INSTRUCT_V1,
            D.AI21_JAMBA_1_5_LARGE_V1,
            D.AI21_JAMBA_1_5_MINI_V1,
        ]
        and not is_nova(deployment)
        and supports_tools(deployment)
    )


def is_llama3(deployment: D) -> bool:
    return deployment in [
        D.META_LLAMA3_8B_INSTRUCT_V1,
        D.META_LLAMA3_70B_INSTRUCT_V1,
        D.META_LLAMA3_1_8B_INSTRUCT_V1,
        D.META_LLAMA3_1_70B_INSTRUCT_V1,
        D.META_LLAMA3_1_405B_INSTRUCT_V1,
        D.META_LLAMA3_2_1B_INSTRUCT_V1,
        D.META_LLAMA3_2_3B_INSTRUCT_V1,
        D.META_LLAMA3_2_11B_INSTRUCT_V1,
        D.META_LLAMA3_2_90B_INSTRUCT_V1,
        D.META_LLAMA3_3_70B_INSTRUCT_V1,
    ]


def is_cohere(deployment: D) -> bool:
    return deployment in [
        D.COHERE_COMMAND_LIGHT_TEXT_V14,
        D.COHERE_COMMAND_TEXT_V14,
    ]


def is_cohere_command_plus(deployment: D) -> bool:
    return deployment in [
        D.COHERE_COMMAND_R_V1,
        D.COHERE_COMMAND_R_PLUS_V1,
    ]


def is_claude_3_or_4(deployment: D) -> bool:
    return deployment in [
        D.ANTHROPIC_CLAUDE_V3_SONNET,
        D.ANTHROPIC_CLAUDE_V3_5_SONNET,
        D.ANTHROPIC_CLAUDE_V3_5_SONNET_V2,
        D.ANTHROPIC_CLAUDE_V3_HAIKU,
        D.ANTHROPIC_CLAUDE_V3_5_HAIKU,
        D.ANTHROPIC_CLAUDE_V3_OPUS,
        D.ANTHROPIC_CLAUDE_V3_7_SONNET,
        D.ANTHROPIC_CLAUDE_V4_SONNET,
        D.ANTHROPIC_CLAUDE_V4_OPUS,
    ]


def is_nova(deployment: D) -> bool:
    return deployment in [
        D.AMAZON_NOVA_MICRO,
        D.AMAZON_NOVA_PRO,
        D.AMAZON_NOVA_LITE,
    ]


def is_reasoning_model(deployment: D) -> bool:
    return deployment in [
        D.DEEPSEEK_R1_V2,
    ]


def is_deepseek(deployment: D) -> bool:
    return deployment in [
        D.DEEPSEEK_R1_V2,
    ]


def is_ai21(deployment: D) -> bool:
    return deployment in [
        D.AI21_J2_GRANDE_INSTRUCT,
        D.AI21_J2_JUMBO_INSTRUCT,
        D.AI21_JAMBA_1_5_MINI_V1,
        D.AI21_JAMBA_1_5_LARGE_V1,
    ]


cohere_invalid_request_error = ExpectedException(
    type=BadRequestError,
    message="Invalid parameter combination",
    status_code=400,
)


def is_vision_model(deployment: D) -> bool:
    allowed_models = [
        D.META_LLAMA3_2_11B_INSTRUCT_V1,
        D.META_LLAMA3_2_90B_INSTRUCT_V1,
        D.AMAZON_NOVA_PRO,
        D.AMAZON_NOVA_LITE,
    ]

    # Claude 3.5 Haiku was launched as a text-only model
    # https://assets.anthropic.com/m/61e7d27f8c8f5919/original/Claude-3-Model-Card.pdf
    excluded_models = {
        D.ANTHROPIC_CLAUDE_V3_5_HAIKU,
    }

    is_allowed_model = (
        is_claude_3_or_4(deployment) or deployment in allowed_models
    )
    is_excluded_model = deployment in excluded_models

    return is_allowed_model and not is_excluded_model


def are_tools_emulated(deployment: D) -> bool:
    return deployment in [D.ANTHROPIC_CLAUDE_V2_1]


@pytest.fixture
def deployment(request) -> Deployment:
    return request.param


@pytest.fixture(params=[True, False], ids=lambda b: "stream" if b else "block")
def stream(request) -> bool:
    return request.param


@pytest.fixture
def openai_client(
    deployment: Deployment, get_deployment_region, get_openai_client
):
    region = get_deployment_region.get(deployment)
    if region is None:
        raise ValueError(
            f"{deployment.value!r} is missing from the region mapping"
        )
    return get_openai_client(deployment.value, region=region)


Chat = Callable[..., Awaitable[ChatCompletionResult]]


@pytest.fixture
def chat(openai_client: AsyncAzureOpenAI, stream: bool):
    async def _inner(
        **kwargs: Unpack[ChatCompletionArgs],
    ) -> ChatCompletionResult:
        return await chat_completion(openai_client, stream=stream, **kwargs)

    return _inner


def display_deployment(dep: Deployment):
    return sanitize_test_name(dep.value)


@pytest.mark.parametrize(
    "deployment", select(is_retired_model, deployments), ids=display_deployment
)
async def test_retired_models(chat: Chat):
    async with expected_exception(
        cls=openai.NotFoundError,
        status_code=404,
        message="This model version has reached the end of its life. Please refer to the AWS documentation for more details.",
        display_message="This model version has reached the end of its life",
    ):
        await chat(messages=[user("test")], max_tokens=1)


@pytest.mark.parametrize(
    "deployment", alive_deployments, ids=display_deployment
)
async def test_dialog_recall(deployment: Deployment, chat: Chat):
    response = await chat(
        messages=[
            user("Remember Paris city. Just say hello"),
            ai("Hello"),
            user("What city did I mention earlier?"),
        ],
        # It could take hundreds of tokens for a reasoning model
        # to come up with an answer to a simple question like this.
        max_tokens=32 if not is_reasoning_model(deployment.origin) else 512,
    )
    assert "paris" in response.content.lower()


@pytest.mark.parametrize(
    "deployment", alive_deployments, ids=display_deployment
)
async def test_model_field(deployment: Deployment, chat: Chat):
    response = await chat(messages=[user("test")], max_tokens=1)
    assert deployment.value == response.response.model


@pytest.mark.parametrize(
    "deployment", alive_deployments, ids=display_deployment
)
async def test_2_plus_3(chat: Chat):
    response = await chat(messages=[user("compute (2+3)")])
    assert "5" in response.content


@pytest.mark.parametrize(
    "deployment", alive_deployments, ids=display_deployment
)
async def test_empty_system_message(chat: Chat):
    response = await chat(messages=[sys(""), user("compute (2+4)")])
    assert "6" in response.content


@pytest.mark.parametrize(
    "deployment", alive_deployments, ids=display_deployment
)
async def test_multiple_candidates(deployment: Deployment, chat: Chat):
    response = await chat(
        # It could take hundreds of tokens for a reasoning model
        # to come up with an answer to a simple question like this.
        max_tokens=10 if not is_reasoning_model(deployment.origin) else 512,
        n=5,
        messages=[user("2+7=?. Reply with a single number")],
    )
    assert len(response.contents) == 5
    for content in response.contents:
        assert "9" in content


@pytest.mark.parametrize(
    "deployment", alive_deployments, ids=display_deployment
)
async def test_hello(deployment: Deployment, chat: Chat):
    query = 'Reply with "Hello"'
    if deployment.origin == D.ANTHROPIC_CLAUDE_INSTANT_V1:
        query = 'Print "Hello"'

    response = await chat(messages=[user(query)])
    content = response.content.lower()
    assert "hello" in content or "hi" in content


@pytest.mark.parametrize(
    "deployment", alive_deployments, ids=display_deployment
)
async def test_empty_dialog(chat: Chat):
    async with expected_exception(
        status_code=422,
        cls=UnprocessableEntityError,
        message="List of messages must not be empty",
    ):
        await chat(max_tokens=1, messages=[])


@pytest.mark.parametrize(
    "deployment", alive_deployments, ids=display_deployment
)
@pytest.mark.parametrize(
    "is_empty", [True, False], ids=lambda b: "empty" if b else "non-empty"
)
async def test_empty_user_message(
    deployment: Deployment, is_empty: bool, chat: Chat
):
    origin = deployment.origin

    if is_claude_3_or_4(origin):
        if is_empty:
            message = "messages: text content blocks must be non-empty"
        else:
            message = (
                "messages: text content blocks must contain non-whitespace text"
            )
    elif is_cohere(origin):
        message = "Invalid parameter combination"
    elif is_llama3(origin) or is_nova(origin):
        message = "Add text to the text field, and try again."
    elif (
        is_deepseek(origin) or is_ai21(origin) or is_cohere_command_plus(origin)
    ):
        message = "The text field in the ContentBlock object at messages.0.content.0 is blank. Add text to the text field, and try again."
    else:
        message = None

    async def _run():
        await chat(max_tokens=1, messages=[user("" if is_empty else " ")])

    if message is not None:
        async with expected_exception(
            status_code=400, cls=BadRequestError, message=message
        ):
            await _run()
    else:
        await _run()


@pytest.mark.parametrize(
    "deployment",
    select(pred(is_vision_model), alive_deployments),
    ids=display_deployment,
)
@pytest.mark.parametrize(
    "message_factory",
    [
        user_with_attachment_data,
        user_with_attachment_url,
        user_with_image_url,
    ],
    ids=[
        "attachment_data",
        "attachment_data_url",
        "content_part_image_url",
    ],
)
async def test_vision(chat: Chat, message_factory):
    user_message = message_factory("describe the image", SAMPLE_DOG_RESOURCE)
    response = await chat(
        max_tokens=100, messages=[sys("be a helpful assistant"), user_message]
    )
    assert "dog" in response.content.lower()


@pytest.mark.parametrize(
    "deployment", alive_deployments, ids=display_deployment
)
async def test_finish_reason_length(chat: Chat):
    response = await chat(
        max_tokens=1,
        messages=[user("tell me the full story of Pinocchio")],
    )
    assert len(response.content.split()) <= 1
    assert response.usage is not None
    assert response.usage.completion_tokens == 1
    assert response.finish_reasons == ["length"]


@pytest.mark.parametrize(
    "deployment", alive_deployments, ids=display_deployment
)
async def test_stop_sequence(chat: Chat):
    response = await chat(
        stop=["John", "john"],
        messages=[user('Reply with "John"')],
    )
    assert "john" not in response.content.lower()


@pytest.mark.parametrize(
    "deployment",
    select(pred(is_llama3), alive_deployments),
    ids=display_deployment,
)
async def test_llama_out_of_turn_dialog(chat: Chat):
    async with expected_exception(
        cls=BadRequestError,
        message="A conversation must start with a user message",
        status_code=400,
    ):
        await chat(
            messages=[ai("hello"), user("what's 7+5?")],
        )


@pytest.mark.parametrize(
    "deployment",
    select(pred(is_llama3), alive_deployments),
    ids=display_deployment,
)
async def test_llama_many_system_messages(chat: Chat):
    response = await chat(
        messages=[
            sys("act as a helpful assistant"),
            sys("act as a calculator"),
            user("2+5=?"),
        ],
    )
    assert "7" in response.content


def get_test_cases(
    deployment: Deployment, region: str, streaming: bool
) -> List[TestCase]:
    origin = deployment.origin

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
        tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
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
                tool_choice,
                temperature,
            )
        )

    city_config = (
        [[("Glasgow", 15)], [("Glasgow", 15), ("London", 20)]]
        if supports_parallel_tool_calls(origin)
        else [[("Glasgow", 15)]]
    )

    if supports_tools(origin):

        def _success_check(s):
            return "4" in s.content.lower()

        tool_choice_none_expected: (
            ExpectedException | Callable[[ChatCompletionResult], bool] | None
        )

        if origin in [
            D.ANTHROPIC_CLAUDE_V4_OPUS,
            D.ANTHROPIC_CLAUDE_V4_SONNET,
            D.ANTHROPIC_CLAUDE_V3_7_SONNET,
        ]:
            tool_choice_none_expected = _success_check
        elif "claude-3" in origin.value:
            tool_choice_none_expected = ExpectedException(
                type=BadRequestError,
                message="none is not a valid enum value, please reformat your input and try again",
                status_code=400,
            )
        elif "claude-v2" in origin.value:
            tool_choice_none_expected = None
        else:
            tool_choice_none_expected = ExpectedException(
                type=UnprocessableEntityError,
                message="tool_choice=none isn't supported by Converse API",
                status_code=422,
            )

        if tool_choice_none_expected:
            test_case(
                name="tool_choice=none + existing tool calls",
                messages=[
                    user("What's the weather in Glasgow?"),
                    ai_tools(
                        [
                            tool_request(
                                "tool_1",
                                "get_weather",
                                {"location": "Glasgow", "unit": "celsius"},
                            )
                        ]
                    ),
                    tool_response("tool_1", "20 degrees"),
                    ai("It's 20 degrees"),
                    user("2+2=?"),
                ],
                tools=[function_to_tool(GET_WEATHER_FUNCTION)],
                tool_choice="none",
                expected=tool_choice_none_expected,
            )

        if supports_forced_tool_choice(origin):
            test_case(
                name="tool_choice=function",
                messages=[user("Glasgow is a city in Scotland. What's 2+2?")],
                tools=[function_to_tool(GET_WEATHER_FUNCTION)],
                tool_choice={
                    "type": "function",
                    "function": {"name": GET_WEATHER_FUNCTION["name"]},
                },
                expected=lambda s: is_valid_tool_call(
                    s.tool_calls,
                    0,
                    lambda _: True,
                    GET_WEATHER_FUNCTION["name"],
                    {
                        "location": lambda s: "Glasgow" in s,
                        "unit": "celsius",
                    },
                ),
            )

        for cities in city_config:
            function = GET_WEATHER_FUNCTION
            tool = function_to_tool(function)
            fun_name = function["name"]

            city_names = [name for name, _ in cities]
            city_temps = [temp for _, temp in cities]

            query = f"Tell me what's the temperature in {' and in '.join(city_names)} in celsius?"

            init_messages = [
                user("2+3=?"),
                ai("5"),
                user(query),
            ]

            # Llama 3 works badly with system messages along tools
            if not is_llama3(origin):
                init_messages.insert(0, sys("act as a helpful assistant"))

            def create_fun_args(city: str):
                return {
                    "location": city,
                    "unit": "celsius",
                }

            def check_fun_args(city: str):
                return {
                    "location": lambda s: city.lower() in s.lower(),
                    "unit": "celsius",
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
                        if are_tools_emulated(origin)
                        else True
                    )

                return _check

            expected_city_names = (
                city_names[:1] if are_tools_emulated(origin) else city_names
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
async def test_chat_completion(get_openai_client, test: TestCase):
    deployment_id = test.deployment.value
    client: openai.AsyncAzureOpenAI = get_openai_client(
        deployment_id, region=test.region
    )

    async def run_chat_completion() -> ChatCompletionResult:
        configuration = {}
        low_latency_regions = (
            deployments_supporting_optimized_latency.get(test.deployment.origin)
            or []
        )
        if test.region in low_latency_regions:
            configuration["performanceConfig"] = {"latency": "optimized"}

        return await chat_completion(
            client,
            messages=test.messages,
            stream=test.streaming,
            stop=test.stop,
            max_tokens=test.max_tokens,
            n=test.n,
            functions=test.functions,
            tools=test.tools,
            tool_choice=test.tool_choice,
            temperature=test.temperature,
            configuration=configuration,
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
        assert (actual_exc.body or {}).get("display_message") == test.expected.display_message  # type: ignore
    else:
        actual_output = await run_chat_completion()
        assert test.expected(
            actual_output
        ), f"Failed output test, actual output: {actual_output}"
