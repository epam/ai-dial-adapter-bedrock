import json
from logging import DEBUG
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
from aidial_adapter_bedrock.utils.json import json_dumps_short
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


def to_dial_finish_reason(
    converse_stop_reason: ConverseStopReason,
) -> DialFinishReason:
    if converse_stop_reason not in CONVERSE_TO_DIAL_FINISH_REASON.keys():
        raise RuntimeServerError(
            f"Unsupported converse stop reason: {converse_stop_reason}"
        )
    return CONVERSE_TO_DIAL_FINISH_REASON[converse_stop_reason]


def to_dial_usage(
    converse_usage: Dict[str, Any],
) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=converse_usage.get("inputTokens") or 0,
        completion_tokens=converse_usage.get("outputTokens") or 0,
        cache_read_input_tokens=converse_usage.get("cacheReadInputTokens") or 0,
        cache_write_input_tokens=converse_usage.get("cacheWriteInputTokens")
        or 0,
    )


async def process_streaming(
    params: ModelParameters,
    stream: AsyncIterator[Any],
    consumer: Consumer,
) -> None:
    current_tool_use = None

    thinking_stage = consumer.create_stage("Thinking")

    async for event in stream:
        if log.isEnabledFor(DEBUG):
            log.debug(f"response event: {json_dumps_short(event)}")

        if (metadata := event.get("metadata")) and (
            usage := metadata.get("usage")
        ):
            consumer.add_usage(to_dial_usage(usage))

        if (content_block_start := event.get("contentBlockStart")) and (
            tool_use := content_block_start.get("start", {}).get("toolUse")
        ):
            if current_tool_use is not None:
                raise ValueError("Tool use already started")
            current_tool_use = {"input": ""} | tool_use

        elif (content_block := event.get("contentBlockDelta")) and (
            delta := content_block.get("delta")
        ):

            if message := delta.get("text"):
                consumer.append_content(message)

            if tool_use := delta.get("toolUse"):
                if current_tool_use is None:
                    raise ValueError("Received tool delta before start block")
                else:
                    current_tool_use["input"] += tool_use.get("input") or ""

            # NOTE: reasoningContent.(redactedContent, signature) aren't yet supported.
            # They are only relevant for Claude 3.7 that we call via anthropic sdk anyway.
            if (reasoning_content := delta.get("reasoningContent")) and (
                text := reasoning_content.get("text")
            ):
                thinking_stage.append_content(text)

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

    thinking_stage.close()


def process_non_streaming(
    params: ModelParameters,
    response: Dict[str, Any],
    consumer: Consumer,
) -> None:
    if log.isEnabledFor(DEBUG):
        log.debug(f"response: {json_dumps_short(response)}")

    thinking_stage = consumer.create_stage("Thinking")

    message = response["output"]["message"]
    for content_block in message.get("content") or []:
        if text := content_block.get("text"):
            consumer.append_content(text)

        # NOTE: reasoningContent.readactedContent and reasoningContent.reasoningText.signature
        # are ignored since they are only relevant for Claude 3.7
        if reasoning_content := content_block.get("reasoningContent"):
            if reasoning_text := reasoning_content.get("reasoningText"):
                if text := reasoning_text.get("text"):
                    thinking_stage.append_content(text)

        if tool_use := content_block.get("toolUse"):
            match params.tools_mode:
                case ToolsMode.TOOLS:
                    consumer.create_function_tool_call(
                        call=DialToolCall(
                            type="function",
                            id=tool_use["toolUseId"],
                            index=None,
                            function=DialFunctionCall(
                                name=tool_use["name"],
                                arguments=json.dumps(tool_use["input"]),
                            ),
                        )
                    )
                case ToolsMode.FUNCTIONS:
                    # ignoring multiple function calls in one response
                    if not consumer.has_function_call:
                        consumer.create_function_call(
                            call=DialFunctionCall(
                                name=tool_use["name"],
                                arguments=json.dumps(tool_use["input"]),
                            )
                        )
                case None:
                    raise RuntimeError("Tool use received without tools mode")
                case _:
                    assert_never(params.tools_mode)

    thinking_stage.close()

    if usage := response.get("usage"):
        consumer.add_usage(to_dial_usage(usage))

    if stop_reason := response.get("stopReason"):
        consumer.close_content(to_dial_finish_reason(stop_reason))
