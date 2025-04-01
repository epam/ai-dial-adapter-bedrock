from dataclasses import dataclass, field
from typing import Any, List

import pytest
from aidial_sdk.chat_completion.request import (
    Attachment,
    CustomContent,
    Function,
    FunctionCall,
    ImageURL,
    Message,
    MessageContentImagePart,
    MessageContentTextPart,
    Role,
    Tool,
    ToolCall,
)

from aidial_adapter_bedrock.aws_client_config import AWSClientConfig
from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.converse.adapter import ConverseAdapter
from aidial_adapter_bedrock.llm.converse.constants import (
    CONVERSE_DOCUMENT_TYPE_TO_MIME,
    CONVERSE_IMAGE_TYPE_TO_MIME,
    DOCUMENT_MIME_TO_CONVERSE_TYPE,
)
from aidial_adapter_bedrock.llm.converse.types import (
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
    ConverseToolUseConfig,
    ConverseToolUsePart,
    InferenceConfig,
)
from aidial_adapter_bedrock.llm.errors import UserError, ValidationError
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsConfig
from aidial_adapter_bedrock.utils.list_projection import ListProjection
from tests.integration_tests.constants import (
    BLUE_PNG_PICTURE,
    SAMPLE_DOCUMENT_RESOURCE,
)


async def _input_tokenizer_factory(_deployment, _params):
    async def _test_tokenizer(_messages) -> int:
        return 100

    return _test_tokenizer


@dataclass(frozen=True)
class UndefinedValue(str):
    """Sentinel object for values that should be ignored in comparisons."""

    def __eq__(self, other: Any) -> bool:
        return True

    def __repr__(self) -> str:
        return "UNDEFINED"


UNDEFINED = UndefinedValue()


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


default_inference_config = InferenceConfig(stopSequences=[])


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
                inferenceConfig=default_inference_config,
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
        params=ModelParameters(tool_config=None),
        expected_output=ConverseRequestWrapper(
            inferenceConfig=default_inference_config,
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
        params=ModelParameters(tool_config=None),
        expected_output=ConverseRequestWrapper(
            inferenceConfig=default_inference_config,
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
        params=ModelParameters(tool_config=None),
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
                tools=[
                    Tool(
                        type="function",
                        function=Function(
                            name="get_weather",
                            description="Get the weather",
                            parameters={"type": "object", "properties": {}},
                        ),
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
    *(_create_document_test_cases()),
    *(_create_unsupported_multi_modal_type_test_cases()),
]


@pytest.mark.parametrize(
    "test_case", TEST_CASES, ids=lambda test_case: test_case.name
)
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
        supported_image_types=test_case.supported_image_types,
        supported_document_types=test_case.supported_document_types,
    )
    construct_coro = adapter.construct_converse_params(
        messages=test_case.messages,
        params=test_case.params,
    )

    if test_case.expected_error is not None:
        with pytest.raises(test_case.expected_error.type) as exc_info:
            converse_request = await construct_coro
        assert hasattr(exc_info.value, "message") or hasattr(
            exc_info.value, "error_message"
        )
        error_message = getattr(exc_info.value, "message", None) or getattr(
            exc_info.value, "error_message"
        )
        assert isinstance(error_message, str)
        assert error_message == test_case.expected_error.message
    else:
        converse_request = await construct_coro
        assert converse_request == test_case.expected_output
