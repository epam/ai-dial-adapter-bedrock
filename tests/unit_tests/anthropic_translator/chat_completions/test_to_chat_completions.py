import json

import pytest
from aidial_sdk.chat_completion.request import (
    ChatCompletionRequest,
    FunctionChoice,
    ImageURL,
    InputFile,
    MessageContentFilePart,
    MessageContentImagePart,
    MessageContentTextPart,
    ReasoningEffort,
    Role,
    ToolChoice,
)
from aidial_sdk.chat_completion.request import (
    Tool as SdkTool,
)

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    MessagesRequest,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.to_chat_completions import (
    to_chat_completions_request,
)
from aidial_adapter_bedrock.anthropic_translator.errors import (
    AnthropicHTTPError,
)


def convert(body: dict, model: str = "gpt-5.5") -> ChatCompletionRequest:
    body = {"max_tokens": 100, **body}
    return to_chat_completions_request(
        MessagesRequest.model_validate(body), model
    )


def test_minimal_string_message():
    result = convert({"messages": [{"role": "user", "content": "hello"}]})
    assert result.model == "gpt-5.5"
    assert result.max_completion_tokens == 100
    # `stream` is set by the transport (`app.py`), not by the translator, so
    # it's excluded here the same way `app.py` excludes it before dumping.
    dumped = result.model_dump(
        mode="json", exclude_none=True, exclude={"stream"}
    )
    assert "store" not in dumped
    assert "stream" not in dumped
    assert result.messages[0].role == Role.USER
    assert result.messages[0].content == "hello"


def test_system_string():
    result = convert(
        {"system": "be nice", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert result.messages[0].role == Role.SYSTEM
    assert result.messages[0].content == "be nice"


def test_system_blocks_joined_and_cache_control_stripped():
    result = convert(
        {
            "system": [
                {
                    "type": "text",
                    "text": "a",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "b"},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert result.messages[0].role == Role.SYSTEM
    assert result.messages[0].content == "a\n\nb"
    # The raw Anthropic `cache_control` value is never forwarded as-is...
    assert "cache_control" not in result.model_dump_json(exclude_none=True)
    # ...but its presence sets DIAL's own cache-breakpoint marker.
    assert result.messages[0].custom_fields is not None
    assert result.messages[0].custom_fields.cache_breakpoint is not None


def test_top_level_and_mid_conversation_system_merged_into_one_message():
    # Unlike Responses, some Chat-Completions-backed deployments 400 on more
    # than one system message, so every system-origin text is merged into a
    # single leading message.
    result = convert(
        {
            "system": "be nice",
            "messages": [
                {"role": "system", "content": "hook context"},
                {"role": "user", "content": "hi"},
            ],
        }
    )
    system_messages = [m for m in result.messages if m.role == Role.SYSTEM]
    assert len(system_messages) == 1
    assert system_messages[0].content == "be nice\n\nhook context"


def test_base64_image():
    result = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "QUJD",
                            },
                        }
                    ],
                }
            ]
        }
    )
    content = result.messages[0].content
    assert isinstance(content, list)
    part = content[0]
    assert part == MessageContentImagePart(
        type="image_url",
        image_url=ImageURL(url="data:image/jpeg;base64,QUJD"),
    )


def test_url_image():
    result = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://x/y.png",
                            },
                        }
                    ],
                }
            ]
        }
    )
    content = result.messages[0].content
    assert isinstance(content, list)
    assert content[0] == MessageContentImagePart(
        type="image_url", image_url=ImageURL(url="https://x/y.png")
    )


def test_pdf_document_base64():
    result = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "title": "report.pdf",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "UERG",
                            },
                        }
                    ],
                }
            ]
        }
    )
    content = result.messages[0].content
    assert isinstance(content, list)
    assert content[0] == MessageContentFilePart(
        type="file",
        file=InputFile(
            filename="report.pdf",
            file_data="data:application/pdf;base64,UERG",
        ),
    )


def test_document_url_has_no_equivalent_and_is_dropped():
    # Unlike Responses, Chat Completions' `file` part has no remote-URL form.
    result = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "url",
                                "url": "https://x/y.pdf",
                            },
                        }
                    ],
                }
            ]
        }
    )
    assert result.messages == []


def test_document_text_source_becomes_text_part():
    result = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "inline document text",
                            },
                        }
                    ],
                }
            ]
        }
    )
    content = result.messages[0].content
    assert isinstance(content, list)
    assert content[0] == MessageContentTextPart(
        type="text", text="inline document text"
    )


def test_unsupported_image_source_type_is_dropped():
    result = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "file", "file_id": "f1"},
                        }
                    ],
                }
            ]
        }
    )
    # The only content block was an unsupported image source, so no parts
    # survive and the whole user turn produces no message.
    assert result.messages == []


def test_mcp_servers_and_container_are_dropped_without_error():
    # Neither field has a Chat Completions equivalent; the translator must
    # accept and silently ignore them rather than erroring on unknown fields.
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": [{"type": "url", "url": "https://example.com/mcp"}],
            "container": {"id": "container_1"},
        }
    )
    assert len(result.messages) == 1
    assert result.messages[0].content == "hi"


def test_unsupported_system_role_content_block_is_dropped():
    result = convert(
        {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "keep me"},
                        {"type": "bogus_block"},
                    ],
                },
                {"role": "user", "content": "hi"},
            ]
        }
    )
    system_messages = [m for m in result.messages if m.role == Role.SYSTEM]
    assert len(system_messages) == 1
    assert system_messages[0].content == "keep me"


def test_custom_tool_without_name_is_dropped():
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"input_schema": {"type": "object", "properties": {}}}],
        }
    )
    assert result.tools is None


def test_assistant_tool_use_and_text_combine_into_one_message():
    result = convert(
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
    messages = result.messages
    assert messages[1].role == Role.ASSISTANT
    assert messages[1].content == "let me look"
    tool_calls = messages[1].tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call.id == "toolu_1"
    assert tool_call.type == "function"
    assert tool_call.function.name == "search"
    assert tool_call.function.arguments == json.dumps({"q": "cats"})
    # tool result BEFORE the residual user message.
    assert messages[2].role == Role.TOOL
    assert messages[2].tool_call_id == "toolu_1"
    assert messages[2].content == "found cats"
    assert messages[3].role == Role.USER
    assert messages[3].content == [
        MessageContentTextPart(type="text", text="thanks")
    ]


def test_tool_result_only_turn_emits_no_user_message():
    result = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "ok",
                        }
                    ],
                }
            ]
        }
    )
    assert len(result.messages) == 1
    assert result.messages[0].role == Role.TOOL
    assert result.messages[0].tool_call_id == "toolu_1"
    assert result.messages[0].content == "ok"


def test_tool_result_is_error_prefixes_output():
    result = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t",
                            "content": "boom",
                            "is_error": True,
                        }
                    ],
                }
            ]
        }
    )
    assert result.messages[0].content == "Error: boom"


def test_tool_result_image_becomes_user_image_url():
    result = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
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
                    ],
                }
            ]
        }
    )
    assert result.messages[0].role == Role.TOOL
    assert result.messages[0].tool_call_id == "t"
    assert result.messages[0].content == "see"
    assert result.messages[1].content == [
        MessageContentImagePart(
            type="image_url",
            image_url=ImageURL(url="data:image/png;base64,AA"),
        )
    ]


def test_custom_tool_mapping():
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "weather",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        }
    )
    tools = result.tools
    assert tools is not None
    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool, SdkTool)
    assert tool.type == "function"
    assert tool.function.name == "get_weather"
    assert tool.function.parameters == {"type": "object", "properties": {}}
    assert tool.function.strict is False
    assert tool.function.description == "weather"


def test_web_search_and_other_server_tools_all_dropped():
    # Unlike Responses (which maps `web_search`), Chat Completions has no
    # server-tool equivalent at all.
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "bash_20250124", "name": "bash"},
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
            {"type": "tool", "name": "f"},
            ToolChoice(type="function", function=FunctionChoice(name="f")),
        ),
    ],
)
def test_tool_choice_matrix(tool_choice, expected):
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": tool_choice,
        }
    )
    assert result.tool_choice == expected


def test_disable_parallel_tool_use_inverts():
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        }
    )
    assert result.tool_choice == "auto"
    assert result.parallel_tool_calls is False


@pytest.mark.parametrize("budget", [1, 4096, 16384, 999999])
def test_thinking_alone_does_not_set_reasoning(budget):
    # `thinking` bounds a token budget with no OpenAI equivalent; on its own
    # (no output_config.effort) it must NOT produce a reasoning effort.
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": budget},
        }
    )
    assert result.reasoning_effort is None


def test_thinking_disabled_omits_reasoning_effort():
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "disabled"},
        }
    )
    assert result.reasoning_effort is None


@pytest.mark.parametrize(
    "anthropic_effort, expected",
    [
        ("low", ReasoningEffort.LOW),
        ("medium", ReasoningEffort.MEDIUM),
        ("high", ReasoningEffort.HIGH),
        # Chat Completions' ReasoningEffort has no `xhigh`; `max`/`xhigh`
        # clamp to `high`.
        ("max", ReasoningEffort.HIGH),
        ("xhigh", ReasoningEffort.HIGH),
    ],
)
def test_output_config_effort_maps_and_clamps(anthropic_effort, expected):
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": anthropic_effort},
        }
    )
    assert result.reasoning_effort == expected


def test_output_config_effort_wins_over_thinking():
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "low"},
            "thinking": {"type": "enabled", "budget_tokens": 1000},
        }
    )
    assert result.reasoning_effort == ReasoningEffort.LOW


def test_no_reasoning_signal_omits_reasoning_effort():
    result = convert({"messages": [{"role": "user", "content": "hi"}]})
    assert result.reasoning_effort is None


def test_output_config_format_json_schema_converts():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {
                "format": {"type": "json_schema", "schema": schema}
            },
        }
    )
    assert result.response_format is not None
    assert result.response_format.type == "json_schema"
    assert result.response_format.json_schema.name == "response"
    assert result.response_format.json_schema.schema_ == schema
    assert result.response_format.json_schema.strict is False


def test_output_config_format_missing_drops_response_format():
    result = convert({"messages": [{"role": "user", "content": "hi"}]})
    assert result.response_format is None


def test_output_config_format_unsupported_type_dropped():
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"format": {"type": "text"}},
        }
    )
    assert result.response_format is None


def test_output_config_format_missing_schema_dropped():
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"format": {"type": "json_schema"}},
        }
    )
    assert result.response_format is None


def test_stop_sequences_mapped_to_stop():
    # Unlike Responses (which drops it), Chat Completions supports it
    # directly.
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["STOP"],
        }
    )
    assert result.stop == ["STOP"]


def test_dropped_and_kept_fields():
    result = convert(
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
                        {"type": "text", "text": "answer"},
                    ],
                }
            ],
            "top_k": 5,
            "temperature": 0.5,
            "top_p": 0.9,
            "metadata": {"user_id": "u1"},
        }
    )
    dumped = result.model_dump(mode="json", exclude_none=True)
    dumped_str = json.dumps(dumped)
    assert "top_k" not in dumped_str
    assert "secret" not in dumped_str  # thinking history dropped
    assert result.temperature == 0.5
    assert result.top_p == 0.9
    # safety_identifier is disabled (the vertexai-adapter doesn't forward it).
    assert "safety_identifier" not in dumped
    # assistant message keeps only the text block
    assert result.messages[0].role == Role.ASSISTANT
    assert result.messages[0].content == "answer"


def test_user_id_never_forwarded_as_safety_identifier():
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"user_id": "u" * 65},
        }
    )
    dumped = result.model_dump(mode="json", exclude_none=True)
    assert "safety_identifier" not in dumped


def test_citations_enabled_sets_custom_fields():
    result = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "UERG",
                            },
                            "citations": {"enabled": True},
                        }
                    ],
                }
            ]
        }
    )
    assert result.custom_fields is not None
    assert result.custom_fields.configuration == {"enable_citations": True}


def test_no_citations_omits_custom_fields():
    result = convert({"messages": [{"role": "user", "content": "hi"}]})
    assert result.custom_fields is None


def test_mid_conversation_system_cache_control_marks_merged_message():
    result = convert(
        {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "hook context",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": "hi"},
            ]
        }
    )
    system_message = result.messages[0]
    assert system_message.role == Role.SYSTEM
    assert system_message.custom_fields is not None
    assert system_message.custom_fields.cache_breakpoint is not None


def test_no_cache_control_omits_cache_breakpoint():
    result = convert(
        {
            "system": "be nice",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert result.messages[0].custom_fields is None


def test_user_message_cache_control_marks_message():
    result = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        }
    )
    assert result.messages[0].custom_fields is not None
    assert result.messages[0].custom_fields.cache_breakpoint is not None


def test_assistant_message_cache_control_marks_message():
    result = convert(
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


def test_tool_result_turn_cache_control_marks_all_split_messages():
    # A `tool_result`-bearing turn splits into a `tool`-role message plus a
    # residual `user`-role message; `cache_control` anywhere in the turn
    # marks BOTH resulting messages, not just the block that carried it.
    result = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t",
                            "content": "found cats",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": "thanks"},
                    ],
                }
            ]
        }
    )
    assert result.messages[0].role == Role.TOOL
    assert result.messages[0].custom_fields is not None
    assert result.messages[0].custom_fields.cache_breakpoint is not None
    assert result.messages[1].role == Role.USER
    assert result.messages[1].custom_fields is not None
    assert result.messages[1].custom_fields.cache_breakpoint is not None


def test_tool_definition_cache_control_marks_tool():
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": {"type": "object", "properties": {}},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    )
    tools = result.tools
    assert tools is not None
    tool = tools[0]
    assert isinstance(tool, SdkTool)
    assert tool.custom_fields is not None
    assert tool.custom_fields.cache_breakpoint is not None


def test_tool_without_cache_control_omits_custom_fields():
    result = convert(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        }
    )
    tools = result.tools
    assert tools is not None
    tool = tools[0]
    assert isinstance(tool, SdkTool)
    assert tool.custom_fields is None


def test_unknown_role_raises_400():
    with pytest.raises(AnthropicHTTPError) as exc:
        convert({"messages": [{"role": "developer", "content": "hi"}]})
    assert exc.value.status_code == 400
    assert exc.value.type == "invalid_request_error"
    assert exc.value.message == "Unknown message role: 'developer'"
