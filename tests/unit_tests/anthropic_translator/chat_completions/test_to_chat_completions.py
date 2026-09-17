import json
from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest
from aidial_sdk.chat_completion.request import (
    FunctionChoice,
    ImageURL,
    InputFile,
    MessageContentFilePart,
    MessageContentImagePart,
    MessageContentPart,
    MessageContentTextPart,
    MessageCustomFields,
    Role,
    StaticTool,
    ToolCall,
    ToolChoice,
)
from aidial_sdk.chat_completion.request import Message as SdkMessage
from aidial_sdk.chat_completion.request import (
    Tool as SdkTool,
)

from aidial_adapter_bedrock.anthropic_translator.chat_completions.to_chat_completions import (
    CoreChatCompletionRequest,
    to_chat_completions_request,
)
from aidial_adapter_bedrock.anthropic_translator.errors import (
    AnthropicErrorType,
    AnthropicHTTPError,
)
from aidial_adapter_bedrock.anthropic_translator.request import validate_request
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)
from tests.unit_tests.anthropic_translator.helpers import (
    DEPLOYMENT,
    convert,
    user,
)

LONG_MCP_NAME: str = "mcp__" + "s" * 60 + "__do_the_thing"


@pytest.mark.parametrize("max_tokens", [100, 999999])
def test_minimal_request(max_tokens: int) -> None:
    result: CoreChatCompletionRequest = convert(
        {**user("hello"), "max_tokens": max_tokens}
    )
    assert result.model_dump(mode="json", exclude_none=True) == {
        "model": DEPLOYMENT,
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": max_tokens,
        "stream": False,
    }


def test_missing_max_tokens_raises_400() -> None:
    with pytest.raises(AnthropicHTTPError) as exc:
        to_chat_completions_request(
            validate_request({"model": DEPLOYMENT, **user("hi")}),
            DEPLOYMENT,
            ToolNameAliases(),
        )
    assert exc.value.status_code == 400
    assert exc.value.message == "'max_tokens' is required"


def test_system_string() -> None:
    result: CoreChatCompletionRequest = convert(
        {"system": "be nice", **user("hi")}
    )
    assert result.messages[0].role == Role.SYSTEM
    assert result.messages[0].content == "be nice"


def test_system_blocks_joined_and_cache_control_stripped() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            "system": [
                {
                    "type": "text",
                    "text": "a",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "b"},
            ],
            **user("hi"),
        }
    )
    assert result.messages[0].role == Role.SYSTEM
    assert result.messages[0].content == "a\n\nb"

    assert "cache_control" not in result.model_dump_json(exclude_none=True)

    assert result.messages[0].custom_fields is not None
    assert result.messages[0].custom_fields.cache_breakpoint is not None


def test_all_three_system_sources_merge_into_one_leading_message() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            "system": "be nice",
            "messages": [
                {"role": "system", "content": "hook context"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "mid_conv_system",
                            "content": [{"type": "text", "text": "injected"}],
                        },
                        {"type": "text", "text": "hi"},
                    ],
                },
            ],
        }
    )
    system_messages: list[SdkMessage] = [
        m for m in result.messages if m.role == Role.SYSTEM
    ]
    assert len(system_messages) == 1
    assert system_messages[0].content == "be nice\n\nhook context\n\ninjected"
    assert system_messages[0] is result.messages[0]


def test_system_sources_merge_in_client_order_not_grouped_by_kind() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            "system": "TOP",
            "messages": [
                {"role": "system", "content": "A"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "mid_conv_system",
                            "content": [{"type": "text", "text": "B"}],
                        },
                        {"type": "text", "text": "hi"},
                    ],
                },
                {"role": "system", "content": "C"},
            ],
        }
    )
    assert result.messages[0].content == "TOP\n\nA\n\nB\n\nC"


def test_mid_conv_system_on_an_assistant_message_is_merged_not_warned() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "mid_conv_system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "from the assistant turn",
                                }
                            ],
                        },
                        {"type": "text", "text": "hello"},
                    ],
                },
            ]
        }
    )
    assert result.messages[0].role == Role.SYSTEM
    assert result.messages[0].content == "from the assistant turn"
    assert result.messages[2].content == "hello"


def test_mid_conv_system_inside_a_system_role_message() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "outer"},
                        {
                            "type": "mid_conv_system",
                            "content": [{"type": "text", "text": "inner"}],
                        },
                    ],
                },
                {"role": "user", "content": "hi"},
            ]
        }
    )
    assert result.messages[0].content == "outer\n\ninner"


def test_mid_conv_system_cache_control_marks_the_merged_message() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "mid_conv_system",
                            "content": [{"type": "text", "text": "ctx"}],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        }
    )
    assert result.messages[0].custom_fields is not None
    assert result.messages[0].custom_fields.cache_breakpoint is not None


def test_unsupported_system_role_content_block_is_dropped() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "keep me"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/image.png",
                            },
                        },
                    ],
                },
                {"role": "user", "content": "hi"},
            ]
        }
    )
    system_messages: list[SdkMessage] = [
        m for m in result.messages if m.role == Role.SYSTEM
    ]
    assert len(system_messages) == 1
    assert system_messages[0].content == "keep me"


def test_base64_image() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "QUJD",
                    },
                }
            ]
        )
    )
    content: str | list[MessageContentPart] | None = result.messages[0].content
    assert isinstance(content, list)
    assert content[0] == MessageContentImagePart(
        type="image_url",
        image_url=ImageURL(url="data:image/jpeg;base64,QUJD"),
    )


def test_url_image() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://x/y.png"},
                }
            ]
        )
    )
    content: str | list[MessageContentPart] | None = result.messages[0].content
    assert isinstance(content, list)
    assert content[0] == MessageContentImagePart(
        type="image_url", image_url=ImageURL(url="https://x/y.png")
    )


def test_pdf_document_base64() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "document",
                    "title": "report.pdf",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "UERG",
                    },
                }
            ]
        )
    )
    content: str | list[MessageContentPart] | None = result.messages[0].content
    assert isinstance(content, list)
    assert content[0] == MessageContentFilePart(
        type="file",
        file=InputFile(
            filename="report.pdf",
            file_data="data:application/pdf;base64,UERG",
        ),
    )


def test_document_url_has_no_equivalent_and_is_dropped() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "document",
                    "source": {"type": "url", "url": "https://x/y.pdf"},
                }
            ]
        )
    )
    assert result.messages == []


def test_document_text_source_becomes_text_part() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "inline document text",
                    },
                }
            ]
        )
    )
    content: str | list[MessageContentPart] | None = result.messages[0].content
    assert isinstance(content, list)
    assert content[0] == MessageContentTextPart(
        type="text", text="inline document text"
    )


def test_unsupported_image_source_type_is_dropped() -> None:
    result: CoreChatCompletionRequest = convert(
        user([{"type": "image", "source": {"type": "file", "file_id": "f1"}}])
    )

    assert result.messages == []


def test_assistant_tool_use_and_text_combine_into_one_message() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            "messages": [
                {"role": "user", "content": "search"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "let me look"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "search",
                            "input": {"q": "cats"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "found cats",
                        },
                        {"type": "text", "text": "thanks"},
                    ],
                },
            ]
        }
    )
    messages: list[SdkMessage] = result.messages
    assert messages[1].role == Role.ASSISTANT
    assert messages[1].content == "let me look"
    tool_calls: list[ToolCall] | None = messages[1].tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "toolu_1"
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "search"
    assert tool_calls[0].function.arguments == json.dumps({"q": "cats"})

    assert messages[2].role == Role.TOOL
    assert messages[2].tool_call_id == "toolu_1"
    assert messages[2].content == "found cats"
    assert messages[3].role == Role.USER
    assert messages[3].content == [
        MessageContentTextPart(type="text", text="thanks")
    ]


def test_tool_result_only_turn_emits_no_user_message() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "ok",
                }
            ]
        )
    )
    assert len(result.messages) == 1
    assert result.messages[0].role == Role.TOOL
    assert result.messages[0].tool_call_id == "toolu_1"
    assert result.messages[0].content == "ok"


def test_tool_result_is_error_prefixes_output() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "t",
                    "content": "boom",
                    "is_error": True,
                }
            ]
        )
    )
    assert result.messages[0].content == "Error: boom"


def test_tool_result_image_becomes_user_image_url() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "t",
                    "content": [
                        {"type": "text", "text": "see"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "AA",
                            },
                        },
                    ],
                }
            ]
        )
    )
    assert result.messages[0].role == Role.TOOL
    assert result.messages[0].content == "see"
    assert result.messages[1].content == [
        MessageContentImagePart(
            type="image_url",
            image_url=ImageURL(url="data:image/png;base64,AA"),
        )
    ]


def test_unknown_role_raises_400() -> None:
    with pytest.raises(AnthropicHTTPError) as exc:
        convert({"messages": [{"role": "developer", "content": "hi"}]})
    assert exc.value.status_code == 400
    assert exc.value.error_type is AnthropicErrorType.INVALID_REQUEST
    assert "role" in exc.value.message


def test_custom_tool_mapping() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            **user("hi"),
            "tools": [
                {
                    "name": "get_weather",
                    "description": "weather",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        }
    )
    tools: list[SdkTool | StaticTool] | None = result.tools
    assert tools is not None
    tool: SdkTool | StaticTool = tools[0]
    assert isinstance(tool, SdkTool)
    assert tool.type == "function"
    assert tool.function.name == "get_weather"
    assert tool.function.parameters == {"type": "object", "properties": {}}

    assert tool.function.strict is False
    assert tool.function.description == "weather"
    assert tool.custom_fields is None


def test_schema_key_is_stripped_from_tool_parameters() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            **user("hi"),
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {"a": {"type": "string"}},
                    },
                }
            ],
        }
    )
    tools: list[SdkTool | StaticTool] | None = result.tools
    assert tools is not None
    tool: SdkTool | StaticTool = tools[0]
    assert isinstance(tool, SdkTool)
    assert tool.function.parameters == {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }


@pytest.mark.parametrize(
    "tool", [{"name": "get_weather"}, {"input_schema": {"type": "object"}}]
)
def test_incomplete_custom_tool_is_rejected(tool: dict[str, object]) -> None:
    with pytest.raises(AnthropicHTTPError) as exc:
        convert({**user("hi"), "tools": [tool]})
    assert exc.value.status_code == 400


def test_web_search_and_other_server_tools_all_dropped() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            **user("hi"),
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250124", "name": "str_replace_editor"},
                {
                    "type": "computer_20250124",
                    "name": "computer",
                    "display_width_px": 100,
                    "display_height_px": 100,
                },
                {"type": "code_execution_20250522", "name": "code_execution"},
            ],
        }
    )
    assert result.tools is None


@pytest.mark.parametrize(
    "tool_choice, expected",
    [
        ({"type": "auto"}, "auto"),
        ({"type": "any"}, "required"),
        ({"type": "none"}, "none"),
        (
            {"type": "tool", "name": "search"},
            ToolChoice(type="function", function=FunctionChoice(name="search")),
        ),
    ],
)
def test_tool_choice_matrix(
    tool_choice: dict[str, object], expected: str | ToolChoice | None
) -> None:
    result: CoreChatCompletionRequest = convert(
        {**user("hi"), "tool_choice": tool_choice}
    )
    assert result.tool_choice == expected


def test_disable_parallel_tool_use_inverts() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            **user("hi"),
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        }
    )
    assert result.tool_choice == "auto"
    assert result.parallel_tool_calls is False


def test_a_long_mcp_name_is_aliased_identically_at_all_three_sites() -> None:
    aliases: ToolNameAliases = ToolNameAliases()
    result: CoreChatCompletionRequest = convert(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": LONG_MCP_NAME,
                            "input": {},
                        }
                    ],
                },
            ],
            "tools": [
                {"name": LONG_MCP_NAME, "input_schema": {"type": "object"}}
            ],
            "tool_choice": {"type": "tool", "name": LONG_MCP_NAME},
        },
        aliases=aliases,
    )
    tools: list[SdkTool | StaticTool] | None = result.tools
    assert tools is not None
    tool: SdkTool | StaticTool = tools[0]
    assert isinstance(tool, SdkTool)
    alias: str = tool.function.name

    assert alias != LONG_MCP_NAME
    assert len(alias) <= 64
    tool_choice: Literal["auto", "none", "required"] | ToolChoice | None = (
        result.tool_choice
    )
    assert isinstance(tool_choice, ToolChoice)
    assert tool_choice.function.name == alias
    tool_calls: list[ToolCall] | None = result.messages[1].tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.name == alias

    assert aliases.to_client(alias) == LONG_MCP_NAME


def test_no_cache_control_omits_cache_breakpoint() -> None:
    result: CoreChatCompletionRequest = convert(
        {"system": "be nice", **user("hi")}
    )
    assert result.messages[0].custom_fields is None


def test_user_message_cache_control_marks_message() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        )
    )
    assert result.messages[0].custom_fields is not None
    assert result.messages[0].custom_fields.cache_breakpoint is not None


def test_assistant_message_cache_control_marks_message() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ]
        }
    )
    assert result.messages[1].custom_fields is not None
    assert result.messages[1].custom_fields.cache_breakpoint is not None


def test_tool_result_turn_cache_control_marks_all_split_messages() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "t",
                    "content": "found cats",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "thanks"},
            ]
        )
    )
    assert result.messages[0].role == Role.TOOL
    assert result.messages[0].custom_fields is not None
    assert result.messages[1].role == Role.USER
    assert result.messages[1].custom_fields is not None


def test_a_block_ttl_reaches_the_marker_as_an_absolute_instant() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ]
        )
    )
    custom_fields: MessageCustomFields | None = result.messages[0].custom_fields
    assert custom_fields is not None
    assert custom_fields.cache_breakpoint is not None
    assert custom_fields.cache_breakpoint.expire_at is not None


def test_the_longest_ttl_of_the_merged_system_sources_wins() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            "system": [
                {
                    "type": "text",
                    "text": "a",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                }
            ],
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "b",
                            "cache_control": {"type": "ephemeral", "ttl": "1h"},
                        }
                    ],
                },
                {"role": "user", "content": "hi"},
            ],
        }
    )
    custom_fields: MessageCustomFields | None = result.messages[0].custom_fields
    assert custom_fields is not None
    assert custom_fields.cache_breakpoint is not None
    expire_at: str | None = custom_fields.cache_breakpoint.expire_at
    assert expire_at is not None
    remaining: timedelta = datetime.fromisoformat(expire_at) - datetime.now(UTC)
    assert remaining.total_seconds() == pytest.approx(3600, abs=5)


def test_cache_control_nested_inside_a_tool_result_is_not_seen() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "t",
                    "content": [
                        {
                            "type": "text",
                            "text": "found cats",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        )
    )
    assert result.messages[0].custom_fields is None


def test_tool_definition_cache_control_marks_tool() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            **user("hi"),
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    )
    tools: list[SdkTool | StaticTool] | None = result.tools
    assert tools is not None
    tool: SdkTool | StaticTool = tools[0]
    assert isinstance(tool, SdkTool)
    assert tool.custom_fields is not None
    assert tool.custom_fields.cache_breakpoint is not None


def test_temperature_and_top_p_pass_through() -> None:
    result: CoreChatCompletionRequest = convert(
        {**user("hi"), "temperature": 0.5, "top_p": 0.9}
    )
    assert result.temperature == 0.5
    assert result.top_p == 0.9


def test_top_k_and_thinking_history_are_dropped() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "secret",
                            "signature": "s",
                        },
                        {
                            "type": "redacted_thinking",
                            "data": "also secret",
                        },
                        {"type": "text", "text": "answer"},
                    ],
                }
            ],
            "top_k": 5,
        }
    )
    dumped: str = json.dumps(result.model_dump(mode="json", exclude_none=True))
    assert "top_k" not in dumped
    assert "secret" not in dumped
    assert result.messages[0].role == Role.ASSISTANT
    assert result.messages[0].content == "answer"


def test_fields_with_no_chat_completions_counterpart_are_dropped() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            **user("hi"),
            "mcp_servers": [
                {
                    "type": "url",
                    "name": "server",
                    "url": "https://example.com/mcp",
                }
            ],
            "container": {"id": "container_1"},
            "inference_geo": "eu",
            "context_management": {
                "edits": [{"type": "clear_tool_uses_20250919"}]
            },
            "top_k": 40,
            "cache_control": {"type": "ephemeral"},
        }
    )
    dumped: str = json.dumps(result.model_dump(mode="json", exclude_none=True))
    for dropped in (
        "mcp_servers",
        "container",
        "inference_geo",
        "context_management",
        "top_k",
        "cache_control",
    ):
        assert dropped not in dumped

    assert result.messages[0].content == "hi"


@pytest.mark.parametrize("user_id", ["u1", "u" * 65])
def test_user_id_is_forwarded_as_user(user_id: str) -> None:
    result: CoreChatCompletionRequest = convert(
        {**user("hi"), "metadata": {"user_id": user_id}}
    )
    assert result.user == user_id


@pytest.mark.parametrize(
    "tier, expected",
    [("auto", "auto"), ("standard_only", "default")],
)
def test_service_tier_is_a_closed_table(
    tier: str, expected: str | None
) -> None:
    result: CoreChatCompletionRequest = convert(
        {**user("hi"), "service_tier": tier}
    )
    assert result.service_tier == expected


def test_output_config_format_json_schema_converts() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    result: CoreChatCompletionRequest = convert(
        {
            **user("hi"),
            "output_config": {
                "format": {"type": "json_schema", "schema": schema}
            },
        }
    )
    assert result.response_format is not None
    assert result.response_format.type == "json_schema"
    assert result.response_format.json_schema.name == "response"
    assert result.response_format.json_schema.schema_ == schema

    assert result.response_format.json_schema.strict is True


@pytest.mark.parametrize(
    "output_format", [{"type": "text"}, {"type": "json_schema"}]
)
def test_invalid_output_config_format_is_rejected(
    output_format: dict[str, object],
) -> None:
    with pytest.raises(AnthropicHTTPError) as exc:
        convert({**user("hi"), "output_config": {"format": output_format}})
    assert exc.value.status_code == 400


def test_empty_output_schema_is_dropped() -> None:
    result = convert(
        {
            **user("hi"),
            "output_config": {"format": {"type": "json_schema", "schema": {}}},
        }
    )
    assert result.response_format is None


def test_citations_enabled_sets_custom_fields() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "UERG",
                    },
                    "citations": {"enabled": True},
                }
            ]
        )
    )
    assert result.custom_fields is not None
    assert result.custom_fields.configuration == {"enable_citations": True}


def test_stop_sequences_map_to_stop() -> None:
    result: CoreChatCompletionRequest = convert(
        {**user("hi"), "stop_sequences": ["STOP"]}
    )
    assert result.stop == ["STOP"]


@pytest.mark.parametrize(
    "suffix",
    [
        [],
        [{"role": "assistant", "content": "prefill"}],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call",
                        "content": "done",
                    }
                ],
            }
        ],
    ],
)
def test_cache_shorthand_marks_last_surviving_user(
    suffix: list[dict[str, object]],
) -> None:
    result: CoreChatCompletionRequest = convert(
        {
            "messages": [*user("hi")["messages"], *suffix],
            "cache_control": {"type": "ephemeral"},
        }
    )
    marked: list[SdkMessage] = [
        m for m in result.messages if m.custom_fields is not None
    ]
    assert len(marked) == 1
    assert marked[0].role == "user"
    assert marked[0].content == "hi"


def test_cache_shorthand_without_user_has_no_marker() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            "messages": [{"role": "assistant", "content": "hi"}],
            "cache_control": {"type": "ephemeral"},
        }
    )
    assert result.messages[0].custom_fields is None


def test_cache_shorthand_preserves_longer_block_expiry() -> None:
    result: CoreChatCompletionRequest = convert(
        {
            **user(
                [
                    {
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ]
            ),
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        }
    )
    fields: MessageCustomFields | None = result.messages[0].custom_fields
    assert (
        fields and fields.cache_breakpoint and fields.cache_breakpoint.expire_at
    )
    seconds: float = (
        datetime.fromisoformat(fields.cache_breakpoint.expire_at)
        - datetime.now(UTC)
    ).total_seconds()
    assert seconds == pytest.approx(3600, abs=2)


def test_null_block_cache_control_does_not_mark() -> None:
    result: CoreChatCompletionRequest = convert(
        user(
            [
                {
                    "type": "text",
                    "text": "hi",
                    "cache_control": None,
                }
            ]
        )
    )
    assert result.messages[0].custom_fields is None
