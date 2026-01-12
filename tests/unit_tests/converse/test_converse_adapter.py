from dataclasses import dataclass, field
from typing import Any, List

import pytest
from aidial_adapter_anthropic.dial_api.request import ModelParameters
from aidial_adapter_anthropic.llm.errors import UserError, ValidationError
from aidial_adapter_anthropic.llm.tools.tools_config import (
    ToolsConfig,
    ToolsMode,
)
from aidial_sdk.chat_completion.request import (
    Attachment,
    CacheBreakpoint,
    CustomContent,
    Function,
    FunctionCall,
    FunctionChoice,
    ImageURL,
    Message,
    MessageContentImagePart,
    MessageContentTextPart,
    MessageCustomFields,
    Role,
    Tool,
    ToolCall,
    ToolChoice,
    ToolCustomFields,
)

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.llm.converse.adapter import ConverseAdapter
from aidial_adapter_bedrock.llm.converse.constants import (
    CONVERSE_DOCUMENT_TYPE_TO_MIME,
    CONVERSE_IMAGE_TYPE_TO_MIME,
    DOCUMENT_MIME_TO_CONVERSE_TYPE,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseCachePoint,
    ConverseCachePointPart,
    ConverseDocumentPart,
    ConverseDocumentPartConfig,
    ConverseDocumentType,
    ConverseImagePart,
    ConverseImagePartConfig,
    ConverseImageType,
    ConverseMessage,
    ConverseRequestWrapper,
    ConverseRole,
    ConverseSource,
    ConverseTextPart,
    ConverseToolResultPart,
    ConverseToolSpec,
    ConverseToolUseConfig,
    ConverseToolUsePart,
    InferenceConfig,
)
from aidial_adapter_bedrock.upstream_config import CloudUpstreamConfig
from aidial_adapter_bedrock.utils.list_projection import ListProjection
from tests.integration_tests.constants import (
    BLUE_PNG_PICTURE,
    SAMPLE_DOCUMENT_RESOURCE,
)


@dataclass(frozen=True)
class UndefinedValue(str):
    """Sentinel object for values that should be ignored in comparisons."""

    def __eq__(self, other: Any) -> bool:
        return True

    def __repr__(self) -> str:
        return "UNDEFINED"


UNDEFINED = UndefinedValue()

DIAL_MESSAGE_CACHE_POINT = MessageCustomFields(
    cache_breakpoint=CacheBreakpoint(expire_at=None)
)
DIAL_TOOL_CACHE_POINT = ToolCustomFields(
    cache_breakpoint=CacheBreakpoint(expire_at=None)
)

CONVERSE_CACHE_POINT_PART = ConverseCachePointPart(
    cachePoint=ConverseCachePoint(type="default")
)

DIAL_WEATHER_FUNCTION = Function(
    name="get_weather",
    description="Get the weather",
    parameters={"type": "object", "properties": {}},
)

CONVERSE_WEATHER_TOOL_SPEC = ConverseToolSpec(
    toolSpec={
        "name": "get_weather",
        "description": "Get the weather",
        "inputSchema": {"json": {"properties": {}, "type": "object"}},
    }
)


@dataclass
class ExpectedException:
    type: type[Exception]
    message: str


@dataclass
class TestCase:
    __test__ = False
    name: str
    messages: List[Message]
    supported_image_types: list[ConverseImageType] = field(
        default_factory=ConverseImageType.all
    )
    supported_document_types: list[ConverseDocumentType] = field(
        default_factory=ConverseDocumentType.all
    )
    params: ModelParameters = field(default_factory=ModelParameters)
    expected_output: ConverseRequestWrapper | None = None
    expected_error: ExpectedException | None = None

    async def get_converse_adapter(self):
        client = await Bedrock.acreate(CloudUpstreamConfig(region="us-east-1"))
        return ConverseAdapter(
            deployment="test",
            bedrock=client,
            support_tools=True,
            storage=None,
            supported_image_types=self.supported_image_types,
            supported_document_types=self.supported_document_types,
            ensure_non_empty_tool_descriptions=False,
        )


def _create_document_test_cases() -> List[TestCase]:
    return [
        TestCase(
            name=f"attachment_document_{converse_type}",
            messages=[
                Message(
                    role=Role.USER,
                    content="tell me about this document",
                    custom_content=CustomContent(
                        attachments=[
                            Attachment(
                                type=mime_type,
                                data=SAMPLE_DOCUMENT_RESOURCE.data_base64,
                            )
                        ]
                    ),
                )
            ],
            expected_output=ConverseRequestWrapper(
                messages=ListProjection(
                    list=[
                        (
                            ConverseMessage(
                                role=ConverseRole.USER,
                                content=[
                                    ConverseTextPart(
                                        text="tell me about this document"
                                    ),
                                    ConverseDocumentPart(
                                        document=ConverseDocumentPartConfig(
                                            name=UNDEFINED,
                                            format=converse_type,
                                            source=ConverseSource(
                                                bytes=SAMPLE_DOCUMENT_RESOURCE.data
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
        )
        for mime_type, converse_type in DOCUMENT_MIME_TO_CONVERSE_TYPE.items()
    ]


def _create_unsupported_multi_modal_type_test_cases() -> List[TestCase]:
    test_cases = []

    # Fully unknown type

    test_cases.append(
        TestCase(
            name="unsupported_multi_modal_type_unknown",
            messages=[
                Message(
                    role=Role.USER,
                    content="Describe this attachment",
                    custom_content=CustomContent(
                        attachments=[
                            Attachment(
                                type="some/unknown-type",
                                data=SAMPLE_DOCUMENT_RESOURCE.data_base64,
                            )
                        ]
                    ),
                )
            ],
            expected_error=ExpectedException(
                type=UserError,
                message=(
                    "Unsupported attachment type: some/unknown-type"
                    "\nSupported image types: "
                    + ", ".join([t.value for t in ConverseImageType.all()])
                    + "\nSupported document types: "
                    + ", ".join([t.value for t in ConverseDocumentType.all()])
                ),
            ),
        )
    )
    for converse_type in ConverseImageType.all() + ConverseDocumentType.all():
        supported_image_types = [
            t for t in ConverseImageType.all() if t != converse_type
        ]
        supported_document_types = [
            t for t in ConverseDocumentType.all() if t != converse_type
        ]
        if converse_type in ConverseImageType:
            mime_type = CONVERSE_IMAGE_TYPE_TO_MIME[converse_type]  # type: ignore
        else:
            mime_type = CONVERSE_DOCUMENT_TYPE_TO_MIME[converse_type]  # type: ignore

        error_message = (
            "Unsupported attachment type: "
            + mime_type
            + "\nSupported image types: "
            + ", ".join([t.value for t in supported_image_types])
            + "\nSupported document types: "
            + ", ".join([t.value for t in supported_document_types])
        )

        test_cases.append(
            TestCase(
                name=f"unsupported_multi_modal_type_{converse_type.value}",
                supported_image_types=supported_image_types,
                supported_document_types=supported_document_types,
                messages=[
                    Message(
                        role=Role.USER,
                        content="Describe this attachment",
                        custom_content=CustomContent(
                            attachments=[
                                Attachment(
                                    type=mime_type,
                                    data=SAMPLE_DOCUMENT_RESOURCE.data_base64,
                                )
                            ]
                        ),
                    )
                ],
                expected_error=ExpectedException(
                    type=UserError,
                    message=error_message,
                ),
            )
        )
    return test_cases


TEST_CASES = [
    TestCase(
        name="plain_message",
        messages=[Message(role=Role.USER, content="Hello, world!")],
        expected_output=ConverseRequestWrapper(
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
        expected_output=ConverseRequestWrapper(
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
        expected_error=ExpectedException(
            type=ValidationError,
            message="A system message can only follow system or developer message",
        ),
    ),
    TestCase(
        name="multiple_system_messages",
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.SYSTEM, content="You are also very friendly."),
            Message(role=Role.USER, content="Hello!"),
        ],
        expected_output=ConverseRequestWrapper(
            system=[
                ConverseTextPart(text="You are a helpful assistant."),
                ConverseTextPart(text="You are also very friendly."),
            ],
            messages=ListProjection(
                list=[
                    (
                        ConverseMessage(
                            role=ConverseRole.USER,
                            content=[ConverseTextPart(text="Hello!")],
                        ),
                        {2},
                    )
                ]
            ),
        ),
    ),
    TestCase(
        name="system_message_multiple_parts",
        messages=[
            Message(
                role=Role.SYSTEM,
                content=[
                    MessageContentTextPart(
                        type="text", text="You are a helpful assistant."
                    ),
                    MessageContentTextPart(
                        type="text", text="You are also very friendly."
                    ),
                ],
            ),
            Message(role=Role.USER, content="Hello!"),
        ],
        expected_output=ConverseRequestWrapper(
            system=[
                ConverseTextPart(text="You are a helpful assistant."),
                ConverseTextPart(text="You are also very friendly."),
            ],
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
        name="system_messages_with_cache_breakpoint",
        messages=[
            Message(
                role=Role.SYSTEM,
                content=[
                    MessageContentTextPart(
                        type="text", text="You are a helpful assistant."
                    ),
                    MessageContentTextPart(
                        type="text", text="You are also very friendly."
                    ),
                ],
                custom_fields=DIAL_MESSAGE_CACHE_POINT,
            ),
            Message(role=Role.USER, content="Hello!"),
        ],
        expected_output=ConverseRequestWrapper(
            system=[
                ConverseTextPart(text="You are a helpful assistant."),
                ConverseTextPart(text="You are also very friendly."),
                CONVERSE_CACHE_POINT_PART,
            ],
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
        name="user_message_with_cache_breakpoint",
        messages=[
            Message(
                role=Role.USER,
                content="hello",
                custom_fields=DIAL_MESSAGE_CACHE_POINT,
            ),
        ],
        expected_output=ConverseRequestWrapper(
            messages=ListProjection(
                list=[
                    (
                        ConverseMessage(
                            role=ConverseRole.USER,
                            content=[
                                ConverseTextPart(text="hello"),
                                CONVERSE_CACHE_POINT_PART,
                            ],
                        ),
                        {0},
                    )
                ]
            ),
        ),
    ),
    TestCase(
        name="tools_with_cache_breakpoint",
        messages=[
            Message(role=Role.USER, content="hello"),
        ],
        params=ModelParameters(
            tool_config=ToolsConfig(
                tools=[
                    Tool(
                        type="function",
                        function=DIAL_WEATHER_FUNCTION,
                        custom_fields=DIAL_TOOL_CACHE_POINT,
                    )
                ],
                tool_choice="required",
                tools_mode=ToolsMode.TOOLS,
                tool_ids={},
            )
        ),
        expected_output=ConverseRequestWrapper(
            toolConfig={
                "tools": [
                    CONVERSE_WEATHER_TOOL_SPEC,
                    CONVERSE_CACHE_POINT_PART,
                ],
                "toolChoice": {"any": {}},
            },
            messages=ListProjection(
                list=[
                    (
                        ConverseMessage(
                            role=ConverseRole.USER,
                            content=[ConverseTextPart(text="hello")],
                        ),
                        {0},
                    )
                ]
            ),
        ),
    ),
    TestCase(
        name="system_message_with_forbidden_image",
        messages=[
            Message(
                role=Role.SYSTEM,
                content=[
                    MessageContentTextPart(
                        type="text", text="You are a helpful assistant."
                    ),
                    MessageContentImagePart(
                        type="image_url",
                        image_url=ImageURL(url=BLUE_PNG_PICTURE.to_data_url()),
                    ),
                ],
            ),
            Message(role=Role.USER, content="Hello!"),
        ],
        expected_error=ExpectedException(
            type=ValidationError,
            message="System messages cannot contain images",
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
                tools=[
                    Tool(
                        type="function",
                        function=DIAL_WEATHER_FUNCTION,
                    )
                ],
                tool_choice=ToolChoice(
                    type="function", function=FunctionChoice(name="get_weather")
                ),
                tools_mode=ToolsMode.TOOLS,
                tool_ids={},
            )
        ),
        expected_output=ConverseRequestWrapper(
            toolConfig={
                "tools": [CONVERSE_WEATHER_TOOL_SPEC],
                "toolChoice": {"tool": {"name": "get_weather"}},
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
        expected_output=ConverseRequestWrapper(
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
                                        source=ConverseSource(
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
            inferenceConfig=InferenceConfig(temperature=10),
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
    *(_create_document_test_cases()),
    *(_create_unsupported_multi_modal_type_test_cases()),
]


@pytest.mark.parametrize(
    "test_case", TEST_CASES, ids=lambda test_case: test_case.name
)
async def test_converse_adapter(test_case: TestCase):
    adapter = await test_case.get_converse_adapter()
    construct_coro = adapter.construct_converse_params(
        messages=test_case.messages, params=test_case.params
    )

    if (err := test_case.expected_error) is not None:
        with pytest.raises(err.type) as e:
            await construct_coro
        message = getattr(e.value, "message", None) or getattr(
            e.value, "error_message", None
        )
        assert message == err.message
    else:
        converse_request = await construct_coro
        assert converse_request == test_case.expected_output


@pytest.mark.parametrize(
    "test_case", TEST_CASES, ids=lambda test_case: test_case.name
)
async def test_converse_prompt_tokenizer(test_case: TestCase):
    adapter = await test_case.get_converse_adapter()
    construct_coro = adapter.count_prompt_tokens(
        messages=test_case.messages, params=test_case.params
    )

    if (err := test_case.expected_error) is not None:
        with pytest.raises(err.type) as e:
            await construct_coro
        message = getattr(e.value, "message", None) or getattr(
            e.value, "error_message", None
        )
        assert message == err.message

    else:
        prompt_tokens = await construct_coro
        assert prompt_tokens > 0
