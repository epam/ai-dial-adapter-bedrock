"""
Legacy tools support for Claude models:
https://docs.anthropic.com/claude/docs/legacy-tool-use
"""

from typing import List, Optional

from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIFunctionCallMessage,
    AIRegularMessage,
    AIToolCallMessage,
    BaseMessage,
    HumanFunctionResultMessage,
    HumanRegularMessage,
    HumanToolResultMessage,
    SystemMessage,
    ToolMessage,
)
from aidial_adapter_bedrock.llm.tools.call_recognizer import CallRecognizer
from aidial_adapter_bedrock.llm.tools.claude_protocol import (
    FUNC_END_TAG,
    FUNC_START_TAG,
    get_system_message,
    parse_call,
    print_function_call,
    print_function_call_result,
    print_tool_calls,
    print_tool_declarations,
)
from aidial_adapter_bedrock.llm.tools.emulator import ToolsEmulator
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsConfig


def convert_to_base_message(
    tool_config: Optional[ToolsConfig], msg: ToolMessage
) -> BaseMessage:

    match msg:
        case HumanToolResultMessage(id=id, content=content):
            if tool_config is None:
                raise ValidationError(
                    "Tool message is used, but tools are not declared"
                )
            name = tool_config.get_tool_name(id)
            return HumanRegularMessage(
                content=print_function_call_result(name=name, content=content)
            )

        case HumanFunctionResultMessage(name=name, content=content):
            return HumanRegularMessage(
                content=print_function_call_result(name=name, content=content)
            )

        case AIToolCallMessage(calls=calls):
            return AIRegularMessage(content=print_tool_calls(calls))

        case AIFunctionCallMessage(call=call):
            return AIRegularMessage(content=print_function_call(call))


class Claude2_1_ToolsEmulator(ToolsEmulator):
    call_recognizer: CallRecognizer

    class Config:
        arbitrary_types_allowed = True

    @property
    def _tool_declarations(self) -> Optional[str]:
        return self.tool_config and print_tool_declarations(
            self.tool_config.functions
        )

    def add_tool_declarations(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        if self._tool_declarations is None:
            return messages

        system_message = get_system_message(self._tool_declarations)

        # Concat with the user system message
        if len(messages) > 0 and isinstance(messages[0], SystemMessage):
            system_message += "\n" + messages[0].text_content
            messages = messages[1:]

        return [SystemMessage(content=system_message), *messages]

    def get_stop_sequences(self) -> List[str]:
        return [] if self._tool_declarations is None else [FUNC_END_TAG]

    def convert_to_base_messages(
        self, messages: List[BaseMessage | ToolMessage]
    ) -> List[BaseMessage]:
        return [
            (
                message
                if isinstance(message, BaseMessage)
                else convert_to_base_message(self.tool_config, message)
            )
            for message in messages
        ]

    def recognize_call(
        self, content: str | None
    ) -> str | AIToolCallMessage | AIFunctionCallMessage | None:
        return (
            self.call_recognizer.consume_chunk(content)
            if self.tool_config
            else content
        )


def legacy_tools_emulator(
    tool_config: Optional[ToolsConfig],
) -> ToolsEmulator:
    return Claude2_1_ToolsEmulator(
        tool_config=tool_config,
        call_recognizer=CallRecognizer(
            start_tag=FUNC_START_TAG,
            call_parser=lambda text: parse_call(
                tool_config, text + FUNC_END_TAG
            ),
        ),
    )
