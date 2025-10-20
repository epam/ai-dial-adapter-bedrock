import json
from typing import Awaitable, Callable, List, Mapping, Unpack

import openai
import pytest
from openai import AsyncAzureOpenAI, BadRequestError, UnprocessableEntityError

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment as D
from aidial_adapter_bedrock.utils.region_deployment import RegionDeployment
from tests.integration_tests.constants import DOG_PICTURE, DOG_PICTURE_CONTENT
from tests.unit_tests.test_configuration import (
    deployments_supporting_optimized_latency,
)
from tests.utils.exception import ExpectedException, expected_exception
from tests.utils.json import match_objects
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    ChatCompletionArgs,
    ChatCompletionResult,
    ai,
    ai_tools,
    chat_completion,
    function_to_tool,
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
from tests.utils.tools import ToolCallTest

Deployment = RegionDeployment[D]

_WEST = "us-west-2"
_EAST_1 = "us-east-1"
_EAST_2 = "us-east-2"
_DEPLOYMENT_TO_REGION: Mapping[Deployment, str] = {
    D.AMAZON_TITAN_TG1_LARGE: _WEST,
    D.AI21_J2_GRANDE_INSTRUCT: _EAST_1,
    D.AI21_J2_JUMBO_INSTRUCT: _EAST_1,
    D.AI21_J2_MID_V1: _EAST_1,
    D.AI21_J2_ULTRA_V1: _EAST_1,
    D.AI21_JAMBA_1_5_LARGE_V1: _EAST_1,
    D.AI21_JAMBA_1_5_MINI_V1: _EAST_1,
    D.ANTHROPIC_CLAUDE_V3_SONNET.US: _WEST,
    D.ANTHROPIC_CLAUDE_V3_5_SONNET.US: _WEST,
    D.ANTHROPIC_CLAUDE_V3_5_SONNET_V2.US: _WEST,
    D.ANTHROPIC_CLAUDE_V3_5_HAIKU.US: _WEST,
    D.ANTHROPIC_CLAUDE_V3_7_SONNET.US: _EAST_1,
    D.ANTHROPIC_CLAUDE_V4_SONNET.US: _EAST_1,
    D.ANTHROPIC_CLAUDE_V4_OPUS.US: _EAST_1,
    D.ANTHROPIC_CLAUDE_V4_5_SONNET.US: _EAST_1,
    D.ANTHROPIC_CLAUDE_V4_5_HAIKU.US: _EAST_1,
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
    D.COHERE_COMMAND_R_V1: _WEST,
    D.COHERE_COMMAND_R_PLUS_V1: _WEST,
    D.AMAZON_NOVA_MICRO: _EAST_1,
    D.AMAZON_NOVA_PRO.US: _EAST_1,
    D.AMAZON_NOVA_LITE: _EAST_1,
    D.DEEPSEEK_R1_V2.US: _EAST_1,
    D.STABILITY_STABLE_DIFFUSION_XL: _WEST,
    D.STABILITY_STABLE_DIFFUSION_XL_V1: _WEST,
}


def is_retired_model(deployment: D) -> bool:
    # Keep at least one model on the list to test how the adapter handles retired models in streaming and non-streaming modes
    return deployment in {
        D.AI21_J2_GRANDE_INSTRUCT,
        D.AI21_J2_JUMBO_INSTRUCT,
        D.AI21_J2_MID_V1,
        D.AI21_J2_ULTRA_V1,
        D.STABILITY_STABLE_DIFFUSION_XL,
        D.STABILITY_STABLE_DIFFUSION_XL_V1,
    }


def is_claude(deployment: D) -> bool:
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
        D.ANTHROPIC_CLAUDE_V4_5_HAIKU,
        D.ANTHROPIC_CLAUDE_V4_5_SONNET,
    ]


def is_vision_model(deployment: D) -> bool:
    return deployment in [
        D.META_LLAMA3_2_11B_INSTRUCT_V1,
        D.META_LLAMA3_2_90B_INSTRUCT_V1,
        D.AMAZON_NOVA_PRO,
        D.AMAZON_NOVA_LITE,
    ] or (
        is_claude(deployment)
        # Claude 3.5 Haiku was launched as a text-only model
        # https://assets.anthropic.com/m/61e7d27f8c8f5919/original/Claude-3-Model-Card.pdf
        and deployment != D.ANTHROPIC_CLAUDE_V3_5_HAIKU
    )


def select(p: Selector[D], xs: List[Deployment]) -> List[Deployment]:
    return [x for x in xs if p(x.origin)]


all_deployments = list(_DEPLOYMENT_TO_REGION.keys())
deployments = select(~pred(is_retired_model), all_deployments)
retired_deployments = select(pred(is_retired_model), all_deployments)
vision_deployments = select(pred(is_vision_model), deployments)
vision_deployments_not_llama3_2_90b = select(
    lambda d: d.origin != D.META_LLAMA3_2_90B_INSTRUCT_V1, vision_deployments
)


def supports_tools(deployment: D) -> bool:
    return is_claude(deployment) or deployment in [
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
    return supports_tools(deployment) and is_claude(deployment)


def supports_parallel_tool_calls(deployment: D) -> bool:
    return deployment not in [
        D.ANTHROPIC_CLAUDE_V3_5_HAIKU,
        D.ANTHROPIC_CLAUDE_V3_5_SONNET_V2,
        D.ANTHROPIC_CLAUDE_V3_7_SONNET,
        D.ANTHROPIC_CLAUDE_V3_SONNET,
        D.ANTHROPIC_CLAUDE_V4_5_HAIKU,
        D.ANTHROPIC_CLAUDE_V4_5_SONNET,
        D.META_LLAMA3_3_70B_INSTRUCT_V1,
        D.AI21_JAMBA_1_5_MINI_V1,
        D.AMAZON_NOVA_MICRO,
    ] and supports_tools(deployment)


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


def is_cohere_command_plus(deployment: D) -> bool:
    return deployment in [
        D.COHERE_COMMAND_R_V1,
        D.COHERE_COMMAND_R_PLUS_V1,
    ]


def is_nova(deployment: D) -> bool:
    return deployment in [
        D.AMAZON_NOVA_MICRO,
        D.AMAZON_NOVA_PRO,
        D.AMAZON_NOVA_LITE,
    ]


def is_reasoning_model(deployment: D) -> bool:
    return deployment in [D.DEEPSEEK_R1_V2]


def is_deepseek(deployment: D) -> bool:
    return deployment in [D.DEEPSEEK_R1_V2]


def is_ai21(deployment: D) -> bool:
    return deployment in [
        D.AI21_J2_GRANDE_INSTRUCT,
        D.AI21_J2_JUMBO_INSTRUCT,
        D.AI21_JAMBA_1_5_MINI_V1,
        D.AI21_JAMBA_1_5_LARGE_V1,
    ]


@pytest.fixture
def deployment(request) -> Deployment:
    return request.param


@pytest.fixture
def region(deployment: Deployment) -> str:
    region = _DEPLOYMENT_TO_REGION.get(deployment)
    if region is None:
        raise ValueError(
            f"{deployment.value!r} is missing from the region mapping"
        )
    return region


@pytest.fixture(params=[True, False], ids=lambda b: "stream" if b else "block")
def stream(request) -> bool:
    return request.param


@pytest.fixture(
    params=[
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
def create_message_with_image(request) -> Callable:
    return request.param


@pytest.fixture
def openai_client(deployment: Deployment, region: str, get_openai_client):
    return get_openai_client(deployment.value, region=region)


Chat = Callable[..., Awaitable[ChatCompletionResult]]


@pytest.fixture(
    params=[True, False], ids=lambda b: "optimized" if b else "standard"
)
def optimized_latency(request, deployment: Deployment, region: str) -> bool:
    optimized_latency = request.param

    opt_latency_regions = (
        deployments_supporting_optimized_latency.get(deployment.origin) or []
    )

    supports_optimized_latency = region in opt_latency_regions
    if not supports_optimized_latency and optimized_latency:
        pytest.skip(
            "The deployment and/or region doesn't support the optimized latency mode"
        )

    return optimized_latency


@pytest.fixture
def chat(
    optimized_latency: bool, openai_client: AsyncAzureOpenAI, stream: bool
):
    async def _inner(
        **kwargs: Unpack[ChatCompletionArgs],
    ) -> ChatCompletionResult:
        kwargs["configuration"] = kwargs.get("configuration") or {}
        if optimized_latency:
            kwargs["configuration"]["performanceConfig"] = {
                "latency": "optimized"
            }

        return await chat_completion(openai_client, stream=stream, **kwargs)

    return _inner


def display_deployment(dep: Deployment):
    return sanitize_test_name(dep.value)


@pytest.mark.parametrize(
    "deployment", retired_deployments, ids=display_deployment
)
async def test_retired_models(chat: Chat):
    async with expected_exception(
        cls=openai.NotFoundError,
        status_code=404,
        message="This model version has reached the end of its life. Please refer to the AWS documentation for more details.",
        display_message="This model version has reached the end of its life",
    ):
        await chat(messages=[user("test")], max_tokens=1)


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
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


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_model_field(deployment: Deployment, chat: Chat):
    response = await chat(messages=[user("test")], max_tokens=1)
    assert deployment.value == response.response.model


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_2_plus_3(chat: Chat):
    response = await chat(messages=[user("compute (2+3)")])
    assert "5" in response.content


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_text_content_parts_in_assistant_message(
    deployment: D, chat: Chat
):
    response = await chat(
        messages=[
            user("compute (2+3) and (5+8)"),
            ai(
                [
                    {"type": "text", "text": "5"},
                    {"type": "text", "text": "13"},
                ]
            ),
            user("compute (11+22). Reply with a number."),
        ],
        max_tokens=10 if not is_reasoning_model(deployment.origin) else 512,
    )
    assert "33" in response.content


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_empty_system_message(chat: Chat):
    response = await chat(messages=[sys(""), user("compute (2+4)")])
    assert "6" in response.content


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_multiple_candidates(deployment: Deployment, chat: Chat):
    response = await chat(
        # It could take hundreds of tokens for a reasoning model
        # to come up with an answer to a simple question like this.
        max_tokens=10 if not is_reasoning_model(deployment.origin) else 1024,
        n=5,
        messages=[user("2+7=?. Reply with a single number")],
    )
    assert len(response.contents) == 5
    for content in response.contents:
        assert "9" in content


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_hello(chat: Chat):
    query = 'Reply with "Hello"'
    response = await chat(messages=[user(query)])
    content = response.content.lower()
    assert "hello" in content or "hi" in content


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_empty_dialog(chat: Chat):
    async with expected_exception(
        status_code=422,
        cls=UnprocessableEntityError,
        message="List of messages must not be empty",
    ):
        await chat(max_tokens=1, messages=[])


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
@pytest.mark.parametrize(
    "is_empty", [True, False], ids=lambda b: "empty" if b else "non-empty"
)
async def test_empty_user_message(
    deployment: Deployment, optimized_latency: bool, is_empty: bool, chat: Chat
):
    origin = deployment.origin

    if is_claude(origin) and not optimized_latency:
        if is_empty:
            message = "messages: text content blocks must be non-empty"
        else:
            message = (
                "messages: text content blocks must contain non-whitespace text"
            )
    elif is_llama3(origin) or is_nova(origin):
        message = "Add text to the text field, and try again."
    elif (
        is_deepseek(origin)
        or is_ai21(origin)
        or is_cohere_command_plus(origin)
        or (is_claude(origin) and optimized_latency)
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
    "deployment", vision_deployments, ids=display_deployment
)
async def test_vision_single_turn_with_text_part(
    deployment: D, chat: Chat, create_message_with_image
):
    messages = [create_message_with_image("describe the image", DOG_PICTURE)]
    await _run_test_vision(deployment, chat, messages, DOG_PICTURE_CONTENT)


@pytest.mark.parametrize(
    "deployment", vision_deployments_not_llama3_2_90b, ids=display_deployment
)
async def test_vision_single_turn_with_empty_text_part(
    deployment: D, chat: Chat, create_message_with_image
):
    messages = [create_message_with_image("", DOG_PICTURE)]
    await _run_test_vision(deployment, chat, messages, DOG_PICTURE_CONTENT)


@pytest.mark.parametrize(
    "deployment", vision_deployments_not_llama3_2_90b, ids=display_deployment
)
async def test_vision_single_turn_without_text_part(deployment: D, chat: Chat):
    messages = [user_with_image_url(None, DOG_PICTURE)]
    await _run_test_vision(deployment, chat, messages, DOG_PICTURE_CONTENT)


@pytest.mark.parametrize(
    "deployment", vision_deployments_not_llama3_2_90b, ids=display_deployment
)
async def test_vision_two_turns(
    deployment: D, chat: Chat, create_message_with_image
):
    user_message = create_message_with_image("", DOG_PICTURE)
    messages = [
        sys("describe an image when you receive it"),
        user("2+3=?"),
        ai("5"),
        user_message,
    ]
    await _run_test_vision(deployment, chat, messages, DOG_PICTURE_CONTENT)


@pytest.mark.parametrize(
    "deployment", vision_deployments_not_llama3_2_90b, ids=display_deployment
)
async def test_vision_single_turn_with_system(
    deployment: D, chat: Chat, create_message_with_image
):
    user_message = create_message_with_image(None, DOG_PICTURE)
    messages = [sys("describe the image"), user_message]
    await _run_test_vision(deployment, chat, messages, DOG_PICTURE_CONTENT)


async def _run_test_vision(
    deployment: D,
    chat: Chat,
    messages,
    expected: str | List[str] | ExpectedException,
):
    async def _run():
        return await chat(max_tokens=100, messages=messages)

    if isinstance(expected, ExpectedException):
        async with expected_exception(expected):
            await _run()
    else:
        response = await _run()
        substrings = [expected] if isinstance(expected, str) else expected
        assert any(s in response.content.lower() for s in substrings)


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_finish_reason_length(chat: Chat):
    response = await chat(
        max_tokens=1,
        messages=[user("tell me the full story of Pinocchio")],
    )
    assert len(response.content.split()) <= 1
    assert response.usage is not None
    assert response.usage.completion_tokens == 1
    assert response.finish_reasons == ["length"]


@pytest.mark.parametrize("deployment", deployments, ids=display_deployment)
async def test_stop_sequence(chat: Chat):
    stop = ["cat", "dog", "fish"]
    response = await chat(
        stop=stop,
        messages=[user('Reply with "cat dog fish"')],
    )
    content = response.content.lower()
    assert not all(w in content for w in stop)


@pytest.mark.parametrize(
    "deployment",
    select(pred(is_llama3), deployments),
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
    select(pred(is_llama3), deployments),
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


@pytest.mark.parametrize(
    "deployment",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
async def test_tool_choice_none(
    deployment: D, optimized_latency: bool, chat: Chat
):
    origin = deployment.origin

    if (
        origin
        in [
            D.ANTHROPIC_CLAUDE_V4_OPUS,
            D.ANTHROPIC_CLAUDE_V4_SONNET,
            D.ANTHROPIC_CLAUDE_V3_7_SONNET,
        ]
        and not optimized_latency
    ):
        exc = None
    elif is_claude(origin) and not optimized_latency:
        exc = ExpectedException(
            type=BadRequestError,
            message="(none is not a valid enum value, please reformat your input and try again|tool_choice: Input tag 'none' found using 'type' does not match any of the expected tags)",
            status_code=400,
        )
    else:
        exc = ExpectedException(
            type=UnprocessableEntityError,
            message="tool_choice=none isn't supported by Converse API",
            status_code=422,
        )

    async def _run():
        return await chat(
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
        )

    if exc:
        async with expected_exception(exc):
            await _run()
    else:
        response = await _run()
        assert "4" in response.content


@pytest.mark.parametrize(
    "deployment",
    select(pred(supports_forced_tool_choice), deployments),
    ids=display_deployment,
)
async def test_forced_tool_choice(chat: Chat):
    func_name = GET_WEATHER_FUNCTION["name"]

    response = await chat(
        messages=[user("Glasgow is a city in Scotland. What's 2+2?")],
        tools=[function_to_tool(GET_WEATHER_FUNCTION)],
        tool_choice={
            "type": "function",
            "function": {"name": func_name},
        },
    )

    tool_calls = response.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1

    function = tool_calls[0].function
    assert function.name == func_name

    match_objects(
        {
            "location": lambda s: "Glasgow" in s,
            "unit": "celsius",
        },
        json.loads(function.arguments),
    )


@pytest.mark.parametrize(
    "deployment",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
@pytest.mark.parametrize(
    "test", [ToolCallTest(1), ToolCallTest(2)], ids=lambda x: x.get_id()
)
async def test_function_call(
    deployment: Deployment, test: ToolCallTest, chat: Chat
):
    origin = deployment.origin

    response = await chat(
        messages=test.messages(not is_llama3(origin)),
        functions=test.functions,
    )

    function_call = response.function_call
    assert function_call is not None
    assert function_call.name == test.function_name

    function_args = json.loads(function_call.arguments)
    assert match_objects(test.expected_function_args(0), function_args)


@pytest.mark.parametrize(
    "deployment",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
@pytest.mark.parametrize("test", [ToolCallTest(1)], ids=lambda x: x.get_id())
async def test_function_response(
    deployment: Deployment, test: ToolCallTest, chat: Chat
):
    origin = deployment.origin
    messages = [
        *test.messages(not is_llama3(origin)),
        test.function_request(0),
        test.function_response(0),
    ]

    response = await chat(messages=messages, functions=test.functions)

    assert str(test.city_temps[0]) in response.content


@pytest.mark.parametrize(
    "deployment",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
@pytest.mark.parametrize(
    "test", [ToolCallTest(1), ToolCallTest(2)], ids=lambda x: x.get_id()
)
async def test_tool_call(
    deployment: Deployment, test: ToolCallTest, chat: Chat
):
    origin = deployment.origin

    response = await chat(
        messages=test.messages(not is_llama3(origin)),
        tools=test.tools,
    )

    tool_calls = response.tool_calls
    assert tool_calls is not None

    expected_calls = test.targets if supports_parallel_tool_calls(origin) else 1

    assert (
        len(tool_calls) >= expected_calls
    ), f"Number of tools calls: actual ({len(tool_calls)}), expected ({expected_calls})"

    for idx, tool_call in enumerate(tool_calls):
        function_call = tool_call.function
        assert function_call.name == test.function_name

        function_args = json.loads(function_call.arguments)
        assert match_objects(test.expected_function_args(idx), function_args)


@pytest.mark.parametrize(
    "deployment",
    select(pred(supports_tools), deployments),
    ids=display_deployment,
)
@pytest.mark.parametrize(
    "test", [ToolCallTest(1), ToolCallTest(2)], ids=lambda x: x.get_id()
)
async def test_tool_response(
    deployment: Deployment, test: ToolCallTest, chat: Chat
):
    origin = deployment.origin

    messages = [
        *test.messages(not is_llama3(origin)),
        test.tool_request(),
        *test.tool_responses(),
    ]

    response = await chat(messages=messages, tools=test.tools)

    for temp in test.city_temps:
        assert str(temp) in response.content
