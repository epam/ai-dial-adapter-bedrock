import json
from typing import Any, AsyncGenerator, Dict, List, Set, Tuple

from aidial_sdk.chat_completion import FinishReason as DialFinishReason
from aidial_sdk.chat_completion import FunctionCall as DialFunctionCall
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role as DialRole
from aidial_sdk.chat_completion import ToolCall as DialToolCall

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.converse.input import (
    get_converse_system_prompt,
    to_converse_finish_reason,
    to_converse_message,
    to_converse_tools,
    to_dial_finish_reason,
)
from aidial_adapter_bedrock.llm.converse.types import ConverseMessage
from aidial_adapter_bedrock.utils.json import remove_nones
from aidial_adapter_bedrock.utils.list import group_by
from aidial_adapter_bedrock.utils.list_projection import ListProjection


class ConverseChatCompletionAdapter(ChatCompletionAdapter):
    model_id: str
    bedrock: Bedrock
    storage: FileStorage | None

    @classmethod
    def create(
        cls,
        bedrock: Bedrock,
        model_id: str,
        dial_api_key: str,
    ):
        return cls(
            bedrock=bedrock,
            model_id=model_id,
            storage=create_file_storage(dial_api_key),
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
                consumer.close_content(to_dial_finish_reason(stop_reason))

    def _process_non_streaming(
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
                            name=content_block["toolUse"]["name"],
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
            consumer.close_content(to_dial_finish_reason(stop_reason))

    async def process_messages(
        self, messages: List[DialMessage]
    ) -> ListProjection[ConverseMessage]:
        def _merge(
            a: Tuple[ConverseMessage, Set[int]],
            b: Tuple[ConverseMessage, Set[int]],
        ) -> Tuple[ConverseMessage, Set[int]]:
            (msg1, set1), (msg2, set2) = a, b

            content1 = msg1["content"]
            content2 = msg2["content"]

            return {
                "role": msg1["role"],
                "content": list(content1) + list(content2),
            }, set1 | set2

        converted: List[Tuple[ConverseMessage, Set[int]]] = [
            (await to_converse_message(msg, self.storage), set([idx]))
            for idx, msg in enumerate(messages)
            if msg.role != DialRole.SYSTEM
        ]

        # Merge messages with same roles, to preserve turn-based user/assistant turns
        return ListProjection(
            group_by(
                lst=converted,
                key=lambda msg: msg[0]["role"],
                init=lambda msg: msg,
                merge=_merge,
            )
        )

    async def construct_converse_params(
        self,
        messages: List[DialMessage],
        params: ModelParameters,
    ) -> Dict[str, Any]:
        system_message = get_converse_system_prompt(messages)
        return remove_nones(
            {
                "system": [system_message] if system_message else None,
                "messages": (await self.process_messages(messages)).raw_list,
                "inferenceConfig": remove_nones(
                    {
                        "temperature": params.temperature,
                        "topP": params.top_p,
                        "maxTokens": params.max_tokens,
                        "stopSequences": [
                            to_converse_finish_reason(
                                DialFinishReason(finish_reason)
                            ).value
                            for finish_reason in params.stop
                        ],
                    }
                ),
                "toolConfig": (
                    to_converse_tools(params.tool_config)
                    if params.tool_config
                    else None
                ),
            }
        )

    def is_stream(self, params: ModelParameters) -> bool:
        return params.stream

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[DialMessage],
    ) -> None:

        converse_params = await self.construct_converse_params(messages, params)

        if self.is_stream(params):
            await self.process_streaming(
                stream=(
                    await self.bedrock.aconverse_streaming(
                        self.model_id, **converse_params
                    )
                ),
                consumer=consumer,
            )
        else:
            self._process_non_streaming(
                response=await self.bedrock.aconverse_non_streaming(
                    self.model_id, **converse_params
                ),
                consumer=consumer,
            )


class ConverseToolStreamingAdapter(ConverseChatCompletionAdapter):
    """
    Some adapter, like LLama 3.2, does support tool calls, but not in streaming mode.
    So we need to drop back to non-streaming mode when tool calls are detected.
    """

    def is_stream(self, params: ModelParameters) -> bool:
        if params.tool_config:
            return False
        return super().is_stream(params)
