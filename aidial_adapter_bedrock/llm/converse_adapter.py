import json
from typing import Any, AsyncGenerator, Dict, List, Optional, assert_never

from aidial_sdk.chat_completion import FinishReason as DialFinishReason
from aidial_sdk.chat_completion import FunctionCall as DialFunctionCall
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role as DialRole
from aidial_sdk.chat_completion import ToolCall as DialToolCall

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters, ToolsConfig
from aidial_adapter_bedrock.dial_api.resource import (
    AttachmentResource,
    URLResource,
)
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.utils.json import remove_nones


class ConverseChatCompletionAdapter(ChatCompletionAdapter):
    model_id: str
    bedrock: Bedrock
    storage: FileStorage | None
    """Flag, either model support tool calls in streaming mode"""
    is_streaming_support_tool_calls: bool
    """Flag, whether to drop back to non-streaming mode when tool calls are detected"""
    fix_streaming_tool_calls: bool

    def to_dial_finish_reason(self, stop_reason: str) -> DialFinishReason:
        bedrock_to_dial_finish_reason = {
            "end_turn": DialFinishReason.STOP,
            "tool_use": DialFinishReason.TOOL_CALLS,
            "max_tokens": DialFinishReason.LENGTH,
            "stop_sequence": DialFinishReason.STOP,
            "guardrail_intervened": DialFinishReason.CONTENT_FILTER,
            "content_filtered": DialFinishReason.CONTENT_FILTER,
        }
        if stop_reason not in bedrock_to_dial_finish_reason:
            raise ValueError(f"Unsupported stop reason: {stop_reason}")

        return bedrock_to_dial_finish_reason[stop_reason]

    @classmethod
    def create(
        cls,
        bedrock: Bedrock,
        model_id: str,
        dial_api_key: str,
        is_streaming_support_tool_calls: bool,
        fix_streaming_tool_calls: bool,
    ):
        return cls(
            bedrock=bedrock,
            model_id=model_id,
            storage=create_file_storage(dial_api_key),
            is_streaming_support_tool_calls=is_streaming_support_tool_calls,
            fix_streaming_tool_calls=fix_streaming_tool_calls,
        )

    def to_converse_role(self, role: DialRole) -> str:
        """
        Converse API accepts only 'user' and 'assistant' roles
        """
        if role in (DialRole.USER, DialRole.TOOL, DialRole.FUNCTION):
            return "user"
        elif role == DialRole.ASSISTANT:
            return "assistant"
        else:
            raise ValueError(f"Unsupported role: {role}")

    def to_converse_system_prompt(self, message: DialMessage) -> Dict[str, Any]:
        if message.role != DialRole.SYSTEM:
            raise ValueError("System message is required")
        if not isinstance(message.content, str):
            raise ValueError("System message content must be a plain string")
        return {"text": message.content}

    async def to_converse_message(self, message: DialMessage) -> Dict[str, Any]:
        content = []

        if isinstance(message.content, str):
            content.append({"text": message.content})
        elif isinstance(message.content, list):
            for part in message.content:
                if part.type == "text":
                    content.append({"text": part.text})
                elif part.type == "image_url":
                    content.append(
                        {
                            "image": {
                                "source": {
                                    "bytes": await URLResource(
                                        url=part.image_url.url
                                    ).download(self.storage)
                                },
                            }
                        }
                    )

        if message.custom_content and message.custom_content.attachments:
            for attachment in message.custom_content.attachments:
                content.append(
                    {
                        "image": {
                            "source": {
                                "bytes": await AttachmentResource(
                                    attachment=attachment,
                                ).download(self.storage)
                            },
                        }
                    }
                )

        bedrock_message = {"role": message.role, "content": content}

        if message.function_call:
            bedrock_message["toolUse"] = self.to_converse_tool_call(
                message.function_call
            )

        if message.role == "function":
            bedrock_message["toolResult"] = self.to_converse_tool_result(
                message
            )

        return bedrock_message

    def to_converse_tool_call(
        self, function_call: DialFunctionCall
    ) -> Dict[str, Any]:
        return {
            "toolUseId": function_call.name,  # Using function name as tool ID
            "name": function_call.name,
            "input": function_call.arguments,
        }

    def to_converse_tool_result(self, message: DialMessage) -> Dict[str, Any]:
        return {
            "toolUseId": message.name,  # Assuming name is used as the tool ID
            "content": [{"text": message.content}],
            "status": "success",  # Assuming success, you might want to add logic to determine this
        }

    def to_converse_tools(self, tools_config: ToolsConfig) -> Dict[str, Any]:
        tools = []
        for function in tools_config.functions:
            tool = {
                "toolSpec": {
                    "name": function.name,
                    "description": function.description or "",
                    "inputSchema": {"json": function.parameters or {}},
                }
            }
            tools.append(tool)

        return {
            "tools": tools,
        }

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[DialMessage],
    ) -> None:

        converse_params = {
            "messages": [
                await self.to_converse_message(msg)
                for msg in messages
                if msg.role != DialRole.SYSTEM
            ],
            "system": [
                self.to_converse_system_prompt(msg)
                for msg in messages
                if msg.role == DialRole.SYSTEM
            ],
            "inferenceConfig": remove_nones(
                {
                    "temperature": params.temperature,
                    "topP": params.top_p,
                    "maxTokens": params.max_tokens,
                    "stopSequences": params.stop,
                }
            ),
        }

        is_stream = params.stream

        if params.tool_config:
            converse_params["toolConfig"] = self.to_converse_tools(
                params.tool_config
            )
            if is_stream and not self.is_streaming_support_tool_calls:
                if self.fix_streaming_tool_calls:
                    is_stream = False
                else:
                    raise ValueError(
                        "Streaming is not supported for tool calls"
                    )

        if is_stream:
            await self.process_streaming(
                stream=(
                    await self.bedrock.aconverse_streaming(
                        self.model_id, **converse_params
                    )
                ),
                consumer=consumer,
            )
        else:
            self.process_non_streaming_response(
                response=await self.bedrock.aconverse_non_streaming(
                    self.model_id, **converse_params
                ),
                consumer=consumer,
            )

    async def process_streaming(
        self, stream: AsyncGenerator[Any, Any], consumer: Consumer
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

                if message := delta.get("message"):
                    consumer.append_content(message)

                if "toolUse" in delta:
                    if current_tool_use is None:
                        raise ValueError(
                            "Received tool delta before start block"
                        )
                    else:
                        current_tool_use["input"] += delta["toolUse"].get(
                            "input", ""
                        )

            elif event.get("contentBlockStop"):
                if current_tool_use:
                    consumer.create_function_tool_call(
                        tool_call=DialToolCall(
                            type="function",
                            id=current_tool_use["toolUseId"],
                            index=None,
                            function=DialFunctionCall(
                                name=current_tool_use["name"],
                                arguments=current_tool_use["input"],
                            ),
                        )
                    )
                    current_tool_use = None

            elif (message_stop := event.get("messageStop")) and (
                stop_reason := message_stop.get("stopReason")
            ):
                # TODO: convert bedrock stop reason to dial stop reason
                consumer.close_content(self.to_dial_finish_reason(stop_reason))

    def process_non_streaming_response(
        self, response: Dict[str, Any], consumer: Consumer
    ) -> None:
        message = response["output"]["message"]
        for content_block in message.get("content", []):
            if "text" in content_block:
                consumer.append_content(content_block["text"])
            if "toolUse" in content_block:
                consumer.create_function_tool_call(
                    tool_call=DialToolCall(
                        type="function",
                        id=content_block["toolUse"]["toolUseId"],
                        index=None,
                        function=DialFunctionCall(
                            name=content_block["toolUse"]["toolUseId"],
                            arguments=json.dumps(
                                content_block["toolUse"]["input"]
                            ),
                        ),
                    )
                )

        if usage := response.get("usage"):
            consumer.add_usage(
                TokenUsage(
                    prompt_tokens=usage.get("inputTokens", 0),
                    completion_tokens=usage.get("outputTokens", 0),
                )
            )

        if stop_reason := response.get("stopReason"):
            # TODO: convert bedrock stop reason to dial stop reason
            consumer.close_content(self.to_dial_finish_reason(stop_reason))
