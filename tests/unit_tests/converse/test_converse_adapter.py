from dataclasses import dataclass
from typing import List

import pytest
from aidial_sdk.chat_completion.request import (
    Function,
    FunctionCall,
    ImageURL,
    Message,
    MessageContentImagePart,
    MessageContentTextPart,
    Role,
    ToolCall,
)

from aidial_adapter_bedrock.aws_client_config import AWSClientConfig
from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.converse.adapter import ConverseAdapter
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseImagePart,
    ConverseImagePartConfig,
    ConverseImageSource,
    ConverseMessage,
    ConverseRequestWrapper,
    ConverseRole,
    ConverseTextPart,
    ConverseToolResultPart,
    ConverseToolUseConfig,
    ConverseToolUsePart,
    InferenceConfig,
)
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsConfig
from aidial_adapter_bedrock.utils.list_projection import ListProjection
from tests.integration_tests.constants import BLUE_PNG_PICTURE


async def _input_tokenizer_factory(_deployment, _params):
    async def _test_tokenizer(_messages) -> int:
        return 100

    return _test_tokenizer


@dataclass
class ExpectedException:
    type: type[Exception]
    message: str


@dataclass
class TestCase:
    __test__ = False
    name: str
    messages: List[Message]
    params: ModelParameters
    expected_output: ConverseRequestWrapper | None = None
    expected_error: ExpectedException | None = None


default_inference_config = InferenceConfig(stopSequences=[])
TEST_CASES = [
    TestCase(
        name="plain_message",
        messages=[Message(role=Role.USER, content="Hello, world!")],
        params=ModelParameters(tool_config=None),
        expected_output=ConverseRequestWrapper(
            inferenceConfig=default_inference_config,
            messages=ListProjection(
                list=[
                    (
                        ConverseMessage(
                            role=ConverseRole.USER,
                            content=[ConverseTextPart(text="Hello, world!")],
                        ),
                        {0},
                    )
                ]
            ),
        ),
    ),
    TestCase(
        name="system_message",
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Hello!"),
        ],
        params=ModelParameters(tool_config=None),
        expected_output=ConverseRequestWrapper(
            inferenceConfig=default_inference_config,
            system=[ConverseTextPart(text="You are a helpful assistant.")],
            messages=ListProjection(
                list=[
                    (
                        ConverseMessage(
                            role=ConverseRole.USER,
                            content=[ConverseTextPart(text="Hello!")],
                        ),
                        {1},
                    )
                ]
            ),
        ),
    ),
    TestCase(
        name="system_message_after_user",
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Hello!"),
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        ],
        params=ModelParameters(tool_config=None),
        expected_error=ExpectedException(
            type=ValidationError,
            message="A system message can only follow another system message",
        ),
    ),
    TestCase(
        name="tools_convert",
        messages=[
            Message(role=Role.USER, content="What's the weather?"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall(
                        index=0,
                        id="call_123",
                        type="function",
                        function=FunctionCall(
                            name="get_weather",
                            arguments='{"location": "London"}',
                        ),
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                content='{"temperature": "20C"}',
                tool_call_id="call_123",
            ),
        ],
        params=ModelParameters(
            tool_config=ToolsConfig(
                functions=[
                    Function(
                        name="get_weather",
                        description="Get the weather",
                        parameters={"type": "object", "properties": {}},
                    )
                ],
                required=True,
                tool_ids=None,
            )
        ),
        expected_output=ConverseRequestWrapper(
            inferenceConfig=default_inference_config,
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": "get_weather",
                            "description": "Get the weather",
                            "inputSchema": {
                                "json": {"properties": {}, "type": "object"}
                            },
                        }
                    }
                ],
                "toolChoice": {"any": {}},
            },
            messages=ListProjection(
                list=[
                    (
                        ConverseMessage(
                            role=ConverseRole.USER,
                            content=[
                                ConverseTextPart(text="What's the weather?")
                            ],
                        ),
                        {0},
                    ),
                    (
                        ConverseMessage(
                            role=ConverseRole.ASSISTANT,
                            content=[
                                ConverseToolUsePart(
                                    toolUse=ConverseToolUseConfig(
                                        toolUseId="call_123",
                                        name="get_weather",
                                        input={"location": "London"},
                                    )
                                )
                            ],
                        ),
                        {1},
                    ),
                    (
                        ConverseMessage(
                            role=ConverseRole.USER,
                            content=[
                                ConverseToolResultPart(
                                    toolResult={
                                        "toolUseId": "call_123",
                                        "content": [
                                            {"json": {"temperature": "20C"}}
                                        ],
                                        "status": "success",
                                    }
                                )
                            ],
                        ),
                        {2},
                    ),
                ]
            ),
        ),
    ),
    TestCase(
        name="content_parts",
        messages=[
            Message(
                role=Role.USER,
                content=[
                    MessageContentTextPart(type="text", text="Hello!"),
                    MessageContentImagePart(
                        type="image_url",
                        image_url=ImageURL(url=BLUE_PNG_PICTURE.to_data_url()),
                    ),
                ],
            )
        ],
        params=ModelParameters(tool_config=None),
        expected_output=ConverseRequestWrapper(
            inferenceConfig=default_inference_config,
            messages=ListProjection(
                list=[
                    (
                        ConverseMessage(
                            role=ConverseRole.USER,
                            content=[
                                ConverseTextPart(text="Hello!"),
                                ConverseImagePart(
                                    image=ConverseImagePartConfig(
                                        format="png",
                                        source=ConverseImageSource(
                                            bytes=BLUE_PNG_PICTURE.data
                                        ),
                                    )
                                ),
                            ],
                        ),
                        {0},
                    )
                ]
            ),
        ),
    ),
    TestCase(
        name="shrink_messages",
        messages=[
            Message(role=Role.USER, content="Say hello."),
            Message(role=Role.USER, content="And have a good day."),
            Message(
                role=Role.ASSISTANT,
                content="Hello",
            ),
            Message(
                role=Role.ASSISTANT,
                content=[
                    MessageContentTextPart(type="text", text="Have a nice"),
                    MessageContentTextPart(type="text", text="day!"),
                ],
            ),
        ],
        params=ModelParameters(temperature=10),
        expected_output=ConverseRequestWrapper(
            inferenceConfig=InferenceConfig(temperature=10, stopSequences=[]),
            messages=ListProjection(
                list=[
                    (
                        ConverseMessage(
                            role=ConverseRole.USER,
                            content=[
                                ConverseTextPart(text="Say hello."),
                                ConverseTextPart(text="And have a good day."),
                            ],
                        ),
                        {0, 1},
                    ),
                    (
                        ConverseMessage(
                            role=ConverseRole.ASSISTANT,
                            content=[
                                ConverseTextPart(text="Hello"),
                                ConverseTextPart(text="Have a nice"),
                                ConverseTextPart(text="day!"),
                            ],
                        ),
                        {2, 3},
                    ),
                ]
            ),
        ),
    ),
]


@pytest.mark.parametrize(
    "test_case", TEST_CASES, ids=lambda test_case: test_case.name
)
@pytest.mark.asyncio
async def test_converse_adapter(
    test_case: TestCase,
):
    adapter = ConverseAdapter(
        deployment="test",
        bedrock=await Bedrock.acreate(AWSClientConfig(region="us-east-1")),
        tokenize_text=lambda x: len(x),
        input_tokenizer_factory=_input_tokenizer_factory,  # type: ignore
        support_tools=True,
        storage=None,
    )
    construct_coro = adapter.construct_converse_params(
        messages=test_case.messages,
        params=test_case.params,
    )

    if test_case.expected_error is not None:
        with pytest.raises(test_case.expected_error.type) as exc_info:
            converse_request = await construct_coro
        assert hasattr(exc_info.value, "message")
        error_message = getattr(exc_info.value, "message")
        assert isinstance(error_message, str)
        assert error_message == test_case.expected_error.message
    else:
        converse_request = await construct_coro
        assert converse_request == test_case.expected_output
