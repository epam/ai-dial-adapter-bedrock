import json
from datetime import UTC, datetime

import pytest
from aidial_sdk.chat_completion.request import (
    FunctionChoice,
    ImageURL,
    InputFile,
    MessageContentFilePart,
    MessageContentImagePart,
    MessageContentTextPart,
    Role,
    ToolChoice,
)
from aidial_sdk.chat_completion.request import (
    Tool as SdkTool,
)

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    MessagesRequest,
)
from aidial_adapter_bedrock.anthropic_translator.capabilities import (
    UNRESOLVED_PROFILE,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.to_chat_completions import (
    CoreChatCompletionRequest,
    to_chat_completions_request,
)
from aidial_adapter_bedrock.anthropic_translator.errors import (
    AnthropicHTTPError,
)
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)
from tests.unit_tests.anthropic_translator.helpers import make_profile

# A deployment that accepts `stop`; the `gpt-5.` family is exercised
# separately because it does not.
DEPLOYMENT = "gpt-4o"

LONG_MCP_NAME = "mcp__" + "s" * 60 + "__do_the_thing"


def convert_with_aliases(
    body: dict, model: str = DEPLOYMENT, profile=None
) -> tuple[CoreChatCompletionRequest, ToolNameAliases]:
    return to_chat_completions_request(
        MessagesRequest.model_validate({"max_tokens": 100, **body}),
        model,
        profile if profile is not None else make_profile(),
    )


def convert(
    body: dict, model: str = DEPLOYMENT, profile=None
) -> CoreChatCompletionRequest:
    return convert_with_aliases(body, model, profile)[0]


def user(content) -> dict:
    return {"messages": [{"role": "user", "content": content}]}


def test_minimal_string_message():
    result = convert(user("hello"))
    assert result.model == DEPLOYMENT
    # `stream` is set by the transport (`app.py`), not by the translator, so
    # it's excluded here the same way `app.py` excludes it before dumping.
    dumped = result.model_dump(
        mode="json", exclude_none=True, exclude={"stream"}
    )
    # Strict adapters reject unrecognised top-level fields.
    assert "store" not in dumped
    assert "stream" not in dumped
    assert result.messages[0].role == Role.USER
    assert result.messages[0].content == "hello"


def test_missing_max_tokens_raises_400():
    with pytest.raises(AnthropicHTTPError) as exc:
        to_chat_completions_request(
            MessagesRequest.model_validate(user("hi")),
            DEPLOYMENT,
            make_profile(),
        )
    assert exc.value.status_code == 400
    assert exc.value.message == "'max_tokens' is required"


# --- output cap --------------------------------------------------------------


def test_max_tokens_uses_the_older_spelling_by_default():
    result = convert(user("hi"))
    assert result.max_tokens == 100
    assert result.max_completion_tokens is None


def test_max_completion_tokens_is_used_when_advertised():
    result = convert(
        user("hi"), profile=make_profile(max_completion_tokens_supported=True)
    )
    assert result.max_completion_tokens == 100
    assert result.max_tokens is None


def test_max_tokens_is_forwarded_verbatim():
    # The features header carries no limits, so there is no deployment ceiling
    # to clamp against; a cap above it surfaces as the upstream's own error.
    result = convert({"max_tokens": 999999, **user("hi")})
    assert result.max_tokens == 999999


# --- §5.2 system consolidation -----------------------------------------------


def test_system_string():
    result = convert({"system": "be nice", **user("hi")})
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
            **user("hi"),
        }
    )
    assert result.messages[0].role == Role.SYSTEM
    assert result.messages[0].content == "a\n\nb"
    # The raw Anthropic `cache_control` value is never forwarded as-is...
    assert "cache_control" not in result.model_dump_json(exclude_none=True)
    # ...but its presence sets DIAL's own cache-breakpoint marker.
    assert result.messages[0].custom_fields is not None
    assert result.messages[0].custom_fields.cache_breakpoint is not None


def test_all_three_system_sources_merge_into_one_leading_message():
    result = convert(
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
    system_messages = [m for m in result.messages if m.role == Role.SYSTEM]
    assert len(system_messages) == 1
    assert system_messages[0].content == "be nice\n\nhook context\n\ninjected"
    assert system_messages[0] is result.messages[0]


def test_system_sources_merge_in_client_order_not_grouped_by_kind():
    # The top-level `system` always leads; everything after it follows the
    # order the client sent it in, walking `messages[]` once.
    result = convert(
        {
            "system": "TOP",
            "messages": [
                {"role": "system", "content": "A"},
                {
                    "role": "user",
                    "content": [
                        {"type": "mid_conv_system", "content": "B"},
                        {"type": "text", "text": "hi"},
                    ],
                },
                {"role": "system", "content": "C"},
            ],
        }
    )
    assert result.messages[0].content == "TOP\n\nA\n\nB\n\nC"


def test_mid_conv_system_on_an_assistant_message_is_merged_not_warned():
    result = convert(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "mid_conv_system",
                            "content": "from the assistant turn",
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


def test_mid_conv_system_inside_a_system_role_message():
    result = convert(
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


def test_mid_conv_system_cache_control_marks_the_merged_message():
    result = convert(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "mid_conv_system",
                            "content": "ctx",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        }
    )
    assert result.messages[0].custom_fields is not None
    assert result.messages[0].custom_fields.cache_breakpoint is not None


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


def test_no_system_text_emits_no_system_message():
    result = convert(user("hi"))
    assert all(m.role != Role.SYSTEM for m in result.messages)


# --- §5.3 content blocks -----------------------------------------------------


def test_base64_image():
    result = convert(
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
    content = result.messages[0].content
    assert isinstance(content, list)
    assert content[0] == MessageContentImagePart(
        type="image_url",
        image_url=ImageURL(url="data:image/jpeg;base64,QUJD"),
    )


def test_url_image():
    result = convert(
        user(
            [
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://x/y.png"},
                }
            ]
        )
    )
    content = result.messages[0].content
    assert isinstance(content, list)
    assert content[0] == MessageContentImagePart(
        type="image_url", image_url=ImageURL(url="https://x/y.png")
    )


def test_pdf_document_base64():
    result = convert(
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


def test_document_text_source_becomes_text_part():
    result = convert(
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
    content = result.messages[0].content
    assert isinstance(content, list)
    assert content[0] == MessageContentTextPart(
        type="text", text="inline document text"
    )


def test_unsupported_image_source_type_is_dropped():
    result = convert(
        user([{"type": "image", "source": {"type": "file", "file_id": "f1"}}])
    )
    # The only content block was an unsupported image source, so no parts
    # survive and the whole user turn produces no message.
    assert result.messages == []


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
    assert tool_calls[0].id == "toolu_1"
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "search"
    assert tool_calls[0].function.arguments == json.dumps({"q": "cats"})
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


def test_tool_result_is_error_prefixes_output():
    result = convert(
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


def test_tool_result_image_becomes_user_image_url():
    result = convert(
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


def test_unknown_role_raises_400():
    with pytest.raises(AnthropicHTTPError) as exc:
        convert({"messages": [{"role": "developer", "content": "hi"}]})
    assert exc.value.status_code == 400
    assert exc.value.type == "invalid_request_error"
    assert exc.value.message == "Unknown message role: 'developer'"


# --- §7 tools ----------------------------------------------------------------


def test_custom_tool_mapping():
    result = convert(
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
    tools = result.tools
    assert tools is not None
    tool = tools[0]
    assert isinstance(tool, SdkTool)
    assert tool.type == "function"
    assert tool.function.name == "get_weather"
    assert tool.function.parameters == {"type": "object", "properties": {}}
    # Anthropic tool schemas routinely fail OpenAI's strict-mode requirements.
    assert tool.function.strict is False
    assert tool.function.description == "weather"


def test_schema_key_is_stripped_from_tool_parameters():
    # `$schema` is legal JSON Schema but strict adapters reject the whole
    # request over it.
    result = convert(
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
    tools = result.tools
    assert tools is not None
    tool = tools[0]
    assert isinstance(tool, SdkTool)
    assert tool.function.parameters == {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }


def test_a_tool_without_an_input_schema_gets_an_empty_object():
    result = convert({**user("hi"), "tools": [{"name": "get_weather"}]})
    tools = result.tools
    assert tools is not None
    tool = tools[0]
    assert isinstance(tool, SdkTool)
    assert tool.function.parameters == {"type": "object", "properties": {}}


def test_custom_tool_without_name_is_dropped():
    result = convert(
        {
            **user("hi"),
            "tools": [{"input_schema": {"type": "object", "properties": {}}}],
        }
    )
    assert result.tools is None


def test_web_search_and_other_server_tools_all_dropped():
    # Server tools carry no `input_schema`, so forcing them through the
    # function shape would produce a malformed definition.
    result = convert(
        {
            **user("hi"),
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250124", "name": "str_replace"},
                {"type": "computer_20250124", "name": "computer"},
                {"type": "code_execution_20250522", "name": "code"},
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
        ({"type": "bogus"}, None),
    ],
)
def test_tool_choice_matrix(tool_choice, expected):
    result = convert({**user("hi"), "tool_choice": tool_choice})
    assert result.tool_choice == expected


def test_disable_parallel_tool_use_inverts():
    result = convert(
        {
            **user("hi"),
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        }
    )
    assert result.tool_choice == "auto"
    assert result.parallel_tool_calls is False


# --- §7.3 tool-name aliasing -------------------------------------------------


def test_a_long_mcp_name_is_aliased_identically_at_all_three_sites():
    result, aliases = convert_with_aliases(
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
            "tools": [{"name": LONG_MCP_NAME}],
            "tool_choice": {"type": "tool", "name": LONG_MCP_NAME},
        }
    )
    tools = result.tools
    assert tools is not None
    tool = tools[0]
    assert isinstance(tool, SdkTool)
    alias = tool.function.name

    assert alias != LONG_MCP_NAME
    assert len(alias) <= 64
    tool_choice = result.tool_choice
    assert isinstance(tool_choice, ToolChoice)
    assert tool_choice.function.name == alias
    tool_calls = result.messages[1].tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.name == alias
    # The client only ever sees the name it sent.
    assert aliases.to_client(alias) == LONG_MCP_NAME


def test_a_conforming_tool_name_is_never_aliased():
    result, aliases = convert_with_aliases(
        {**user("hi"), "tools": [{"name": "get_weather"}]}
    )
    tools = result.tools
    assert tools is not None
    tool = tools[0]
    assert isinstance(tool, SdkTool)
    assert tool.function.name == "get_weather"
    assert aliases.to_client("get_weather") == "get_weather"


# --- §8 prompt caching -------------------------------------------------------


def test_no_cache_control_omits_cache_breakpoint():
    result = convert({"system": "be nice", **user("hi")})
    assert result.messages[0].custom_fields is None


def test_user_message_cache_control_marks_message():
    result = convert(
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


def test_a_block_ttl_reaches_the_marker_as_an_absolute_instant():
    result = convert(
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
    custom_fields = result.messages[0].custom_fields
    assert custom_fields is not None
    assert custom_fields.cache_breakpoint is not None
    assert custom_fields.cache_breakpoint.expire_at is not None


def test_the_longest_ttl_of_the_merged_system_sources_wins():
    # The merged system message draws from several sources but carries one
    # breakpoint, so the maximum is taken: it can only keep content cached
    # longer than one source asked for.
    result = convert(
        {
            "system": [
                {
                    "type": "text",
                    "text": "a",
                    "cache_control": {"ttl": "5m"},
                }
            ],
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "b",
                            "cache_control": {"ttl": "1h"},
                        }
                    ],
                },
                {"role": "user", "content": "hi"},
            ],
        }
    )
    custom_fields = result.messages[0].custom_fields
    assert custom_fields is not None
    assert custom_fields.cache_breakpoint is not None
    expire_at = custom_fields.cache_breakpoint.expire_at
    assert expire_at is not None
    remaining = datetime.fromisoformat(expire_at) - datetime.now(UTC)
    assert remaining.total_seconds() == pytest.approx(3600, abs=5)


def test_cache_control_nested_inside_a_tool_result_is_not_seen():
    # Marking is per-message, and there is no object to hang a nested one on.
    result = convert(
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


def test_tool_definition_cache_control_marks_tool():
    result = convert(
        {
            **user("hi"),
            "tools": [
                {"name": "get_weather", "cache_control": {"type": "ephemeral"}}
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
    result = convert({**user("hi"), "tools": [{"name": "get_weather"}]})
    tools = result.tools
    assert tools is not None
    tool = tools[0]
    assert isinstance(tool, SdkTool)
    assert tool.custom_fields is None


def test_cache_markers_are_omitted_when_the_deployment_does_not_cache():
    # An auto-caching upstream forwards message fields raw, which makes a
    # marker useless at best and a schema violation at worst.
    body = {
        "system": [
            {
                "type": "text",
                "text": "a",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        **user(
            [
                {
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
        "tools": [
            {"name": "get_weather", "cache_control": {"type": "ephemeral"}}
        ],
    }
    result = convert(body, profile=make_profile(cache_supported=False))
    assert all(m.custom_fields is None for m in result.messages)
    tools = result.tools
    assert tools is not None
    tool = tools[0]
    assert isinstance(tool, SdkTool)
    assert tool.custom_fields is None


def test_top_level_cache_control_is_dropped():
    # "Mark the last cacheable block" has no faithful translation once blocks
    # are flattened into messages.
    result = convert({**user("hi"), "cache_control": {"type": "ephemeral"}})
    assert "cache_control" not in result.model_dump_json(exclude_none=True)


# --- the capability profile as a whole ---------------------------------------


def test_an_unresolved_profile_emits_no_capability_gated_field():
    # Unknown is not unsupported. `temperature` is the one gate that points the
    # other way: dropping it on a guess silently changes generation.
    result = convert(
        {
            **user(
                [
                    {
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
            "temperature": 0.5,
            "output_config": {"effort": "high"},
            "tools": [
                {"name": "get_weather", "cache_control": {"type": "ephemeral"}}
            ],
        },
        profile=UNRESOLVED_PROFILE,
    )
    assert result.temperature == 0.5
    assert result.reasoning_effort is None
    assert result.custom_fields is None
    assert all(m.custom_fields is None for m in result.messages)
    assert result.tools is not None
    tool = result.tools[0]
    assert isinstance(tool, SdkTool)
    assert tool.custom_fields is None
    # The output cap still travels, under the older spelling.
    assert result.max_tokens == 100
    assert result.max_completion_tokens is None


# --- §5.1 remaining top-level fields -----------------------------------------


def test_temperature_and_top_p_pass_through():
    result = convert({**user("hi"), "temperature": 0.5, "top_p": 0.9})
    assert result.temperature == 0.5
    assert result.top_p == 0.9


def test_temperature_is_dropped_when_unsupported():
    result = convert(
        {**user("hi"), "temperature": 0.5},
        profile=make_profile(temperature_supported=False),
    )
    assert result.temperature is None


def test_top_k_and_thinking_history_are_dropped():
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
    dumped = json.dumps(result.model_dump(mode="json", exclude_none=True))
    assert "top_k" not in dumped
    assert "secret" not in dumped
    assert result.messages[0].role == Role.ASSISTANT
    assert result.messages[0].content == "answer"


def test_fields_with_no_chat_completions_counterpart_are_dropped():
    result = convert(
        {
            **user("hi"),
            "mcp_servers": [{"type": "url", "url": "https://example.com/mcp"}],
            "container": {"id": "container_1"},
            "inference_geo": "eu",
            "context_management": {
                "edits": [{"type": "clear_tool_uses_20250919"}]
            },
            "top_k": 40,
            "cache_control": {"type": "ephemeral"},
        }
    )
    dumped = json.dumps(result.model_dump(mode="json", exclude_none=True))
    for dropped in (
        "mcp_servers",
        "container",
        "inference_geo",
        "context_management",
        "top_k",
        "cache_control",
    ):
        assert dropped not in dumped
    # Dropped, not rejected: Claude Code sends fields this translator cannot
    # honour on every request.
    assert result.messages[0].content == "hi"


def test_user_id_is_forwarded_as_user():
    result = convert({**user("hi"), "metadata": {"user_id": "u1"}})
    assert result.user == "u1"


def test_an_over_long_user_id_is_dropped_not_truncated():
    # A silently shortened abuse-detection identifier is worse than none, and
    # Claude Code sends a JSON blob well over the limit.
    result = convert({**user("hi"), "metadata": {"user_id": "u" * 65}})
    assert result.user is None


@pytest.mark.parametrize(
    "tier, expected",
    [("auto", "auto"), ("standard_only", "default"), ("flex", None)],
)
def test_service_tier_is_a_closed_table(tier, expected):
    result = convert({**user("hi"), "service_tier": tier})
    assert result.service_tier == expected


def test_output_config_format_json_schema_converts():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    result = convert(
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
    # Anthropic's structured output has no non-strict mode.
    assert result.response_format.json_schema.strict is True


def test_output_config_format_missing_drops_response_format():
    assert convert(user("hi")).response_format is None


@pytest.mark.parametrize(
    "output_format", [{"type": "text"}, {"type": "json_schema"}]
)
def test_unusable_output_config_format_is_dropped(output_format):
    result = convert({**user("hi"), "output_config": {"format": output_format}})
    assert result.response_format is None


def test_citations_enabled_sets_custom_fields():
    result = convert(
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


def test_no_citations_omits_custom_fields():
    assert convert(user("hi")).custom_fields is None


# --- §9 stop sequences -------------------------------------------------------


def test_stop_sequences_map_to_stop():
    result = convert({**user("hi"), "stop_sequences": ["STOP"]})
    assert result.stop == ["STOP"]


def test_stop_is_omitted_for_a_deployment_that_rejects_it():
    result = convert(
        {**user("hi"), "stop_sequences": ["STOP"]}, model="gpt-5.5"
    )
    assert result.stop is None
