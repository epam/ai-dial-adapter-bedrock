from typing import Dict, List, Optional, Tuple

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
from aidial_adapter_bedrock.llm.tools.tool_config import ToolConfig, ToolsMode
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


def convert_to_base_message(
    tool_config: Optional[ToolConfig],
    id_to_name: Dict[str, str],
    msg: ToolMessage,
) -> BaseMessage:
    mode: Optional[ToolsMode] = tool_config and tool_config.mode
    match msg:
        case HumanToolResultMessage(id=id, content=content):
            assert (
                mode is None or mode == ToolsMode.TOOLS
            ), f"Received tool result in '{mode.value}' mode"
            name = id_to_name.get(id)
            if name is None:
                name = "_unknown_"
                log.warning(
                    f"Unable to find tool name for id '{id}', assuming '{name}' name"
                )

            return HumanRegularMessage(
                content=print_function_call_result(name=name, content=content)
            )
        case HumanFunctionResultMessage(name=name, content=content):
            assert (
                mode is None or mode == ToolsMode.FUNCTIONS
            ), f"Received function result in '{mode.value}' mode"
            return HumanRegularMessage(
                content=print_function_call_result(name=name, content=content)
            )
        case AIToolCallMessage(calls=calls):
            assert (
                mode is None or mode == ToolsMode.TOOLS
            ), f"Received tool call in '{mode.value}' mode"
            for call in calls:
                id_to_name[call.id] = call.function.name
            return AIRegularMessage(content=print_tool_calls(calls))
        case AIFunctionCallMessage(call=call):
            assert (
                mode is None or mode == ToolsMode.FUNCTIONS
            ), f"Received function call in '{mode.value}' mode"
            return AIRegularMessage(content=print_function_call(call))


class Claude2_1_ToolsEmulator(ToolsEmulator):
    call_recognizer: CallRecognizer

    class Config:
        arbitrary_types_allowed = True

    @property
    def _tool_declarations(self) -> Optional[str]:
        return self.tool_config and print_tool_declarations(
            self.tool_config.tools
        )

    def add_tool_declarations(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[BaseMessage], List[str]]:
        if self._tool_declarations is None:
            return messages, []

        system_message = get_system_message(self._tool_declarations)

        # Concat with the user system message
        if len(messages) > 0 and isinstance(messages[0], SystemMessage):
            system_message += "\n" + messages[0].content
            messages = messages[1:]

        return [SystemMessage(content=system_message), *messages], [
            FUNC_END_TAG
        ]

    def convert_to_base_messages(
        self, messages: List[BaseMessage | ToolMessage]
    ) -> List[BaseMessage]:
        id_to_name: Dict[str, str] = {}
        return [
            message
            if isinstance(message, BaseMessage)
            else convert_to_base_message(self.tool_config, id_to_name, message)
            for message in messages
        ]

    def recognize_call(
        self, content: str | None
    ) -> str | AIToolCallMessage | AIFunctionCallMessage | None:
        return self.call_recognizer.consume_chunk(content)


def claude_v2_1_tools_emulator(
    tool_config: Optional[ToolConfig],
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
