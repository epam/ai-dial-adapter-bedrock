import json
from typing import Any, AsyncIterator, Dict, assert_never

from aidial_sdk.chat_completion import FinishReason as DialFinishReason
from aidial_sdk.chat_completion import FunctionCall as DialFunctionCall
from aidial_sdk.chat_completion import ToolCall as DialToolCall
from aidial_sdk.exceptions import RuntimeServerError

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.converse.constants import (
    CONVERSE_TO_DIAL_FINISH_REASON,
)
from aidial_adapter_bedrock.llm.converse.types import ConverseStopReason
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsMode


def to_dial_finish_reason(
    converse_stop_reason: ConverseStopReason,
) -> DialFinishReason:
    if converse_stop_reason not in CONVERSE_TO_DIAL_FINISH_REASON.keys():
        raise RuntimeServerError(
            f"Unsupported converse stop reason: {converse_stop_reason}"
        )
    return CONVERSE_TO_DIAL_FINISH_REASON[converse_stop_reason]


async def process_streaming(
    params: ModelParameters,
    stream: AsyncIterator[Any],
    consumer: Consumer,
) -> None:
    current_tool_use = None

    async for event in stream:
        if (content_block_start := event.get("contentBlockStart")) and (
            tool_use := content_block_start.get("start", {}).get("toolUse")
        ):
            if current_tool_use is not None:
                raise ValueError("Tool use already started")
            current_tool_use = {"input": ""} | tool_use

        elif content_block := event.get("contentBlockDelta"):
            delta = content_block.get("delta", {})

            if message := delta.get("text"):
                consumer.append_content(message)

            if "toolUse" in delta:
                if current_tool_use is None:
                    raise ValueError("Received tool delta before start block")
                else:
                    current_tool_use["input"] += delta["toolUse"].get(
                        "input", ""
                    )

        elif event.get("contentBlockStop"):
            if current_tool_use:

                match params.tools_mode:
                    case ToolsMode.TOOLS:
                        consumer.create_function_tool_call(
                            call=DialToolCall(
                                type="function",
                                id=current_tool_use["toolUseId"],
                                index=None,
                                function=DialFunctionCall(
                                    name=current_tool_use["name"],
                                    arguments=current_tool_use["input"],
                                ),
                            )
                        )
                    case ToolsMode.FUNCTIONS:
                        # ignoring multiple function calls in one response
                        if not consumer.has_function_call:
                            consumer.create_function_call(
                                call=DialFunctionCall(
                                    name=current_tool_use["name"],
                                    arguments=current_tool_use["input"],
                                )
                            )
                    case None:
                        raise RuntimeError(
                            "Tool use received without tools mode"
                        )
                    case _:
                        assert_never(params.tools_mode)
                current_tool_use = None

        elif (message_stop := event.get("messageStop")) and (
            stop_reason := message_stop.get("stopReason")
        ):
            consumer.close_content(to_dial_finish_reason(stop_reason))


def process_non_streaming(
    params: ModelParameters,
    response: Dict[str, Any],
    consumer: Consumer,
) -> None:
    message = response["output"]["message"]
    for content_block in message.get("content", []):
        if "text" in content_block:
            consumer.append_content(content_block["text"])
        if "toolUse" in content_block:
            match params.tools_mode:
                case ToolsMode.TOOLS:
                    consumer.create_function_tool_call(
                        call=DialToolCall(
                            type="function",
                            id=content_block["toolUse"]["toolUseId"],
                            index=None,
                            function=DialFunctionCall(
                                name=content_block["toolUse"]["name"],
                                arguments=json.dumps(
                                    content_block["toolUse"]["input"]
                                ),
                            ),
                        )
                    )
                case ToolsMode.FUNCTIONS:
                    # ignoring multiple function calls in one response
                    if not consumer.has_function_call:
                        consumer.create_function_call(
                            call=DialFunctionCall(
                                name=content_block["toolUse"]["name"],
                                arguments=json.dumps(
                                    content_block["toolUse"]["input"]
                                ),
                            )
                        )
                case None:
                    raise RuntimeError("Tool use received without tools mode")
                case _:
                    assert_never(params.tools_mode)

    if usage := response.get("usage"):
        consumer.add_usage(
            TokenUsage(
                prompt_tokens=usage.get("inputTokens", 0),
                completion_tokens=usage.get("outputTokens", 0),
            )
        )

    if stop_reason := response.get("stopReason"):
        consumer.close_content(to_dial_finish_reason(stop_reason))
