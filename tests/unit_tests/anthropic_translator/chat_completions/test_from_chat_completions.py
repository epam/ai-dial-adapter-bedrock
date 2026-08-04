from typing import Any, Literal

import pytest
from anthropic.types import (
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    WebSearchToolResultBlock,
)
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
    Custom,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

from aidial_adapter_bedrock.anthropic_translator.chat_completions.from_chat_completions import (
    from_chat_completions,
    stop_reason,
)
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]

LONG_MCP_NAME = "mcp__" + "s" * 60 + "__do_the_thing"


def message(**fields: Any) -> ChatCompletionMessage:
    return ChatCompletionMessage.model_validate(
        {"role": "assistant", "content": None, **fields}
    )


def response(
    msg: ChatCompletionMessage | None = None,
    finish_reason: FinishReason = "stop",
    model: str = "gpt-5.5",
    usage: CompletionUsage | None = None,
) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl_abc",
        object="chat.completion",
        created=0,
        model=model,
        choices=[
            Choice(
                index=0,
                message=msg if msg is not None else message(),
                finish_reason=finish_reason,
            )
        ],
        usage=usage
        or CompletionUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        ),
    )


def translate(
    completion: ChatCompletion,
    aliases: ToolNameAliases | None = None,
    stop_sequences: list[str] | None = None,
):
    return from_chat_completions(
        completion,
        "requested-model",
        aliases or ToolNameAliases(),
        stop_sequences or [],
    )


def tool_call(
    name: str = "search", arguments: str = '{"q": "cats"}', id: str = "toolu_1"
) -> ChatCompletionMessageFunctionToolCall:
    return ChatCompletionMessageFunctionToolCall(
        id=id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def test_text_output():
    msg = translate(response(message(content="hello")))
    assert msg.id == "chatcmpl_abc"
    assert msg.type == "message"
    assert msg.role == "assistant"
    assert msg.model == "gpt-5.5"
    assert msg.content == [TextBlock(type="text", text="hello")]
    assert msg.stop_reason == "end_turn"
    assert msg.stop_sequence is None


def test_model_falls_back_to_requested():
    # `model` is a required `str` on the real `ChatCompletion`, so the
    # "missing" case is an empty string (falsy), not `None`.
    assert translate(response(model="")).model == "requested-model"


def test_only_first_choice_translated():
    completion = response(message(content="first"))
    completion.choices.append(
        Choice(
            index=1,
            message=message(content="second"),
            finish_reason="stop",
        )
    )
    assert translate(completion).content == [
        TextBlock(type="text", text="first")
    ]


# --- §10.4 zero-block guard --------------------------------------------------


@pytest.mark.parametrize(
    "completion",
    [
        response(message(content="")),
        response(message(content=None)),
        ChatCompletion(
            id="c",
            object="chat.completion",
            created=0,
            model="gpt-5.5",
            choices=[],
        ),
    ],
)
def test_an_empty_completion_yields_one_empty_text_block(completion):
    # Anthropic messages always carry at least one block, and SDKs index
    # `content[0]` unconditionally.
    assert translate(completion).content == [TextBlock(type="text", text="")]


# --- §10.1 reasoning ---------------------------------------------------------


def test_signed_thinking_from_state_leads_the_content():
    msg = translate(
        response(
            message(
                content="the answer",
                custom_content={
                    "state": {
                        "claude_message_content": [
                            {
                                "type": "thinking",
                                "thinking": "let me think",
                                "signature": "sig-abc",
                            }
                        ]
                    }
                },
            )
        )
    )
    assert msg.content[0] == ThinkingBlock(
        type="thinking", thinking="let me think", signature="sig-abc"
    )
    assert msg.content[1] == TextBlock(type="text", text="the answer")


def test_thinking_from_reasoning_stages_has_an_empty_signature():
    msg = translate(
        response(
            message(
                content="answer",
                custom_content={
                    "stages": [
                        {"index": 0, "name": "Thinking", "content": "a"},
                        {"index": 1, "name": "Searching", "content": "ignored"},
                        {"index": 2, "name": "Reasoning", "content": "b"},
                    ]
                },
            )
        )
    )
    # An empty signature round-trips and cannot be mistaken for a real one.
    assert msg.content[0] == ThinkingBlock(
        type="thinking", thinking="ab", signature=""
    )


def test_state_thinking_is_preferred_over_stages():
    msg = translate(
        response(
            message(
                custom_content={
                    "stages": [
                        {"index": 0, "name": "Thinking", "content": "stage"}
                    ],
                    "state": {
                        "claude_message_content": [
                            {
                                "type": "thinking",
                                "thinking": "signed",
                                "signature": "s",
                            }
                        ]
                    },
                },
            )
        )
    )
    assert msg.content[0] == ThinkingBlock(
        type="thinking", thinking="signed", signature="s"
    )


def test_custom_content_without_reasoning_adds_no_thinking_block():
    msg = translate(
        response(
            message(
                content="hi",
                custom_content={
                    "stages": [
                        {"index": 0, "name": "Searching", "content": "x"}
                    ]
                },
            )
        )
    )
    assert msg.content == [TextBlock(type="text", text="hi")]


def test_malformed_custom_content_is_ignored():
    msg = translate(response(message(content="hi", custom_content="nonsense")))
    assert msg.content == [TextBlock(type="text", text="hi")]


# --- §10.2 web search citations ----------------------------------------------


def test_url_citations_become_a_server_tool_use_pair():
    msg = translate(
        response(
            message(
                content="answer",
                annotations=[
                    {
                        "type": "url_citation",
                        "url_citation": {
                            "url": "https://example.com",
                            "title": "Example",
                            "start_index": 0,
                            "end_index": 1,
                        },
                    }
                ],
            )
        )
    )
    use, result, text = msg.content
    assert isinstance(use, ServerToolUseBlock)
    assert isinstance(result, WebSearchToolResultBlock)
    assert use.name == "web_search"
    assert use.id.startswith("srvtoolu_")
    # The pair shares one identifier.
    assert result.tool_use_id == use.id
    assert isinstance(result.content, list)
    assert result.content[0].url == "https://example.com"
    assert result.content[0].title == "Example"
    assert text == TextBlock(type="text", text="answer")


# --- §10.3 tool calls and stop reason ----------------------------------------


def test_tool_call_becomes_tool_use():
    msg = translate(
        response(message(tool_calls=[tool_call()]), finish_reason="tool_calls")
    )
    block = msg.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.id == "toolu_1"
    assert block.name == "search"
    assert block.input == {"q": "cats"}
    assert msg.stop_reason == "tool_use"


def test_an_aliased_tool_name_is_restored():
    aliases = ToolNameAliases()
    alias = aliases.to_upstream(LONG_MCP_NAME)
    msg = translate(
        response(message(tool_calls=[tool_call(name=alias)])), aliases
    )
    block = msg.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.name == LONG_MCP_NAME


def test_tool_calls_win_the_stop_reason_over_the_finish_reason():
    msg = translate(
        response(message(tool_calls=[tool_call()]), finish_reason="stop")
    )
    assert msg.stop_reason == "tool_use"


@pytest.mark.parametrize("arguments", ["[1, 2, 3]", "{not json", ""])
def test_unusable_tool_arguments_become_an_empty_input(arguments):
    # A malformed tool call must not take down the whole response.
    msg = translate(
        response(message(tool_calls=[tool_call(arguments=arguments)]))
    )
    block = msg.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.input == {}


def test_a_non_function_tool_call_is_skipped():
    msg = translate(
        response(
            message(
                content="hi",
                tool_calls=[
                    ChatCompletionMessageCustomToolCall(
                        id="x", type="custom", custom=Custom(name="f", input="")
                    )
                ],
            )
        )
    )
    assert msg.content == [TextBlock(type="text", text="hi")]
    assert msg.stop_reason == "end_turn"


def test_a_tool_call_without_a_name_is_skipped():
    msg = translate(
        response(message(content="hi", tool_calls=[tool_call(name="")]))
    )
    assert msg.content == [TextBlock(type="text", text="hi")]


def test_refusal_sets_stop_reason_refusal():
    msg = translate(response(message(refusal="no")))
    assert msg.content == [TextBlock(type="text", text="no")]
    assert msg.stop_reason == "refusal"


@pytest.mark.parametrize(
    "finish_reason, expected",
    [
        ("length", "max_tokens"),
        ("content_filter", "refusal"),
        ("stop", "end_turn"),
    ],
)
def test_stop_reason_mapping(finish_reason, expected):
    msg = translate(response(message(content="x"), finish_reason=finish_reason))
    # Never null: clients choke on it.
    assert msg.stop_reason == expected


@pytest.mark.parametrize("finish_reason", [None, "bogus"])
def test_an_unrecognised_finish_reason_falls_back_to_end_turn(finish_reason):
    # `Choice.finish_reason` is a closed literal upstream, so the unrecognised
    # and absent cases are exercised on the mapping itself.
    assert (
        stop_reason(finish_reason, None, saw_tool_use=False, saw_refusal=False)
        == "end_turn"
    )


# --- §9 stop-sequence emulation ----------------------------------------------


def test_an_emulated_stop_sequence_truncates_and_wins_the_stop_reason():
    msg = translate(
        response(message(content="keep STOP drop")), stop_sequences=["STOP"]
    )
    assert msg.content == [TextBlock(type="text", text="keep ")]
    assert msg.stop_reason == "stop_sequence"
    assert msg.stop_sequence == "STOP"


def test_a_stop_sequence_beats_a_tool_call():
    msg = translate(
        response(
            message(content="keep STOP", tool_calls=[tool_call()]),
            finish_reason="tool_calls",
        ),
        stop_sequences=["STOP"],
    )
    assert msg.stop_reason == "stop_sequence"


def test_no_stop_sequences_leaves_the_text_alone():
    msg = translate(response(message(content="keep STOP drop")))
    assert msg.content == [TextBlock(type="text", text="keep STOP drop")]
    assert msg.stop_sequence is None


# --- §12 usage ---------------------------------------------------------------


def test_cached_tokens_are_subtracted_from_input_tokens():
    # Anthropic reports cache reads outside `input_tokens` while OpenAI counts
    # them inside `prompt_tokens`; forwarding it verbatim double-counts them.
    msg = translate(
        response(
            usage=CompletionUsage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=30),
            )
        )
    )
    assert msg.usage.input_tokens == 70
    assert msg.usage.output_tokens == 20
    assert msg.usage.cache_read_input_tokens == 30
    assert msg.usage.cache_creation_input_tokens == 0


def test_usage_without_cache_details():
    msg = translate(
        response(
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
        )
    )
    assert msg.usage.input_tokens == 10
    assert msg.usage.cache_read_input_tokens == 0


def test_input_tokens_are_floored_at_zero():
    msg = translate(
        response(
            usage=CompletionUsage(
                prompt_tokens=5,
                completion_tokens=1,
                total_tokens=6,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=9),
            )
        )
    )
    assert msg.usage.input_tokens == 0


def test_missing_usage_is_zeroed_not_an_error():
    completion = response()
    completion.usage = None
    usage = translate(completion).usage
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


# --- §10 block order ---------------------------------------------------------


def test_full_block_order():
    msg = translate(
        response(
            message(
                content="text",
                refusal="refused",
                custom_content={
                    "stages": [{"index": 0, "name": "Thinking", "content": "t"}]
                },
                annotations=[
                    {
                        "type": "url_citation",
                        "url_citation": {
                            "url": "https://e.com",
                            "title": "E",
                            "start_index": 0,
                            "end_index": 1,
                        },
                    }
                ],
                tool_calls=[tool_call()],
            )
        )
    )
    assert [block.type for block in msg.content] == [
        "thinking",
        "server_tool_use",
        "web_search_tool_result",
        "text",
        "text",
        "tool_use",
    ]
