from typing import Literal

import pytest
from anthropic.types import TextBlock, ToolUseBlock
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
)

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]


def _response(
    message: ChatCompletionMessage | None = None,
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
                message=message
                or ChatCompletionMessage(role="assistant", content=None),
                finish_reason=finish_reason,
            )
        ],
        usage=usage
        or CompletionUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        ),
    )


def test_text_output():
    msg = from_chat_completions(
        _response(
            message=ChatCompletionMessage(role="assistant", content="hello")
        ),
        "requested-model",
    )
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
    msg = from_chat_completions(_response(model=""), "requested-model")
    assert msg.model == "requested-model"


def test_function_call_sets_tool_use_stop_reason():
    msg = from_chat_completions(
        _response(
            message=ChatCompletionMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatCompletionMessageFunctionToolCall(
                        id="toolu_1",
                        type="function",
                        function=Function(
                            name="search", arguments='{"q": "cats"}'
                        ),
                    )
                ],
            ),
            finish_reason="tool_calls",
        ),
        "m",
    )
    tool_use = msg.content[0]
    assert isinstance(tool_use, ToolUseBlock)
    assert tool_use.type == "tool_use"
    assert tool_use.id == "toolu_1"
    assert tool_use.name == "search"
    assert tool_use.input == {"q": "cats"}
    assert msg.stop_reason == "tool_use"


def test_no_choices_returns_empty_content():
    response = ChatCompletion(
        id="chatcmpl_none",
        object="chat.completion",
        created=0,
        model="gpt-5.5",
        choices=[],
        usage=CompletionUsage(
            prompt_tokens=1, completion_tokens=0, total_tokens=1
        ),
    )
    msg = from_chat_completions(response, "requested-model")
    assert msg.content == []
    assert msg.stop_reason == "end_turn"


def test_tool_call_arguments_valid_json_non_dict_becomes_empty_input():
    # `arguments` parses as valid JSON but isn't a JSON object; the anthropic
    # `input` field must still come out as `{}`, not the parsed list.
    msg = from_chat_completions(
        _response(
            message=ChatCompletionMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatCompletionMessageFunctionToolCall(
                        id="t",
                        type="function",
                        function=Function(name="f", arguments="[1, 2, 3]"),
                    )
                ],
            ),
            finish_reason="tool_calls",
        ),
        "m",
    )
    tool_use = msg.content[0]
    assert isinstance(tool_use, ToolUseBlock)
    assert tool_use.input == {}


def test_malformed_arguments_become_empty_input():
    msg = from_chat_completions(
        _response(
            message=ChatCompletionMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatCompletionMessageFunctionToolCall(
                        id="t",
                        type="function",
                        function=Function(name="f", arguments="{not json"),
                    )
                ],
            ),
            finish_reason="tool_calls",
        ),
        "m",
    )
    tool_use = msg.content[0]
    assert isinstance(tool_use, ToolUseBlock)
    assert tool_use.input == {}


def test_non_function_tool_call_raises():
    # This translator only ever offers the model `type: "function"` tools,
    # so a `"custom"` tool call can only mean a broken upstream response.
    with pytest.raises(ValueError, match="Unsupported tool call type"):
        from_chat_completions(
            _response(
                message=ChatCompletionMessage(
                    role="assistant",
                    content="hi",
                    tool_calls=[
                        ChatCompletionMessageCustomToolCall(
                            id="x",
                            type="custom",
                            custom=Custom(name="f", input="{}"),
                        )
                    ],
                )
            ),
            "m",
        )


def test_tool_call_with_missing_name_raises():
    # `Function.name` is a required `str`, so the "missing name" case is an
    # empty string (falsy), not `None`.
    with pytest.raises(ValueError, match="missing its required id or name"):
        from_chat_completions(
            _response(
                message=ChatCompletionMessage(
                    role="assistant",
                    content="hi",
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id="t",
                            type="function",
                            function=Function(name="", arguments="{}"),
                        )
                    ],
                )
            ),
            "m",
        )


def test_refusal_sets_stop_reason_refusal():
    msg = from_chat_completions(
        _response(
            message=ChatCompletionMessage(
                role="assistant", content=None, refusal="no"
            )
        ),
        "m",
    )
    assert msg.content == [TextBlock(type="text", text="no")]
    assert msg.stop_reason == "refusal"


def test_stop_reason_max_tokens():
    msg = from_chat_completions(
        _response(
            message=ChatCompletionMessage(role="assistant", content="cut"),
            finish_reason="length",
        ),
        "m",
    )
    assert msg.stop_reason == "max_tokens"


def test_stop_reason_content_filter():
    msg = from_chat_completions(
        _response(
            message=ChatCompletionMessage(role="assistant", content="blocked"),
            finish_reason="content_filter",
        ),
        "m",
    )
    assert msg.stop_reason == "refusal"


def test_usage_mapping_with_cached_tokens():
    msg = from_chat_completions(
        _response(
            usage=CompletionUsage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=30),
            )
        ),
        "m",
    )
    usage = msg.usage
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.cache_read_input_tokens == 30
    assert usage.cache_creation_input_tokens == 0


def test_only_first_choice_translated():
    response = _response(
        message=ChatCompletionMessage(role="assistant", content="first")
    )
    response.choices.append(
        Choice(
            index=1,
            message=ChatCompletionMessage(role="assistant", content="second"),
            finish_reason="stop",
        )
    )
    msg = from_chat_completions(response, "m")
    assert msg.content == [TextBlock(type="text", text="first")]
