from enum import Enum
from typing import Dict, List, Literal, Self

from aidial_sdk.chat_completion import (
    Function,
    FunctionChoice,
    Message,
    Role,
    Tool,
    ToolChoice,
)
from aidial_sdk.chat_completion.request import (
    AzureChatCompletionRequest,
    StaticTool,
)
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.utils.log_config import app_logger as log


class ToolsMode(Enum):
    TOOLS = "TOOLS"
    FUNCTIONS = "FUNCTIONS"
    """
    Functions are deprecated instrument that came before tools
    """


class ToolsConfig(BaseModel):
    tools: List[Tool]
    """
    List of functions/tools.
    """

    tools_mode: ToolsMode

    tool_choice: Literal["auto", "none", "required"] | ToolChoice

    tool_ids: Dict[str, str]
    """
    Mapping from tool call IDs to corresponding tool names.
    Empty when there are no tool calls in the messages.
    """

    def not_supported(self) -> None:
        if self.tools:
            if self.tools_mode == ToolsMode.TOOLS:
                raise ValidationError("The tools aren't supported")
            else:
                raise ValidationError("The functions aren't supported")

    def create_fresh_tool_call_id(self, tool_name: str) -> str:
        idx = 1
        while True:
            id = f"{tool_name}_{idx}"
            if id not in self.tool_ids:
                self.tool_ids[id] = tool_name
                return id
            idx += 1

    def get_tool_name(self, tool_call_id: str) -> str:
        tool_name = self.tool_ids.get(tool_call_id)
        if tool_name is None:
            raise ValidationError(f"Tool call ID not found: {self.tool_ids}")
        return tool_name

    @staticmethod
    def _function_call_to_tool_choice(
        function_call: Literal["auto", "none"] | FunctionChoice | None,
    ) -> Literal["auto", "none", "required"] | ToolChoice | None:
        match function_call:
            case FunctionChoice():
                return ToolChoice(type="function", function=function_call)
            case _:
                return function_call

    @staticmethod
    def _get_tool_from_function(tool: Function | Tool | StaticTool) -> Tool:
        if isinstance(tool, StaticTool):
            raise ValidationError("Static tools aren't supported")
        if isinstance(tool, Function):
            return Tool(type="function", function=tool)
        else:
            return tool

    @staticmethod
    def _get_tools_from_functions(
        tools: List[Function] | List[Tool | StaticTool],
    ) -> List[Tool]:
        return [ToolsConfig._get_tool_from_function(tool) for tool in tools]

    @classmethod
    def from_request(cls, request: AzureChatCompletionRequest) -> Self | None:
        validate_messages(request)

        tool_ids = _collect_tool_ids(request.messages)

        if request.functions is not None:
            tools_mode = ToolsMode.FUNCTIONS
            tools = cls._get_tools_from_functions(request.functions)
            tool_choice = cls._function_call_to_tool_choice(
                request.function_call
            )
        elif request.tools is not None:
            tools_mode = ToolsMode.TOOLS
            tools = cls._get_tools_from_functions(request.tools)
            tool_choice = request.tool_choice
        elif tool_ids:
            tools_mode = ToolsMode.TOOLS
            tools = []
            tool_choice = None
        else:
            return None

        return cls(
            tools=tools,
            tools_mode=tools_mode,
            tool_choice=tool_choice or "auto",
            tool_ids=tool_ids,
        )


def validate_messages(request: AzureChatCompletionRequest) -> None:
    decl_tools = request.tools is not None
    decl_functions = request.functions is not None

    if decl_functions and decl_tools:
        raise ValidationError("Both functions and tools are not allowed")

    def warn(msg: str):
        log.warning(
            f"The request is incomplete: {msg}. The model may misbehave."
        )

    tool_defs_are_missing = (
        "the request is missing tool definitions in the 'tools' field"
    )
    func_defs_are_missing = (
        "the request is missing function definitions in the 'functions' field"
    )

    for idx, message in enumerate(request.messages):
        if (
            message.role == Role.ASSISTANT
            and message.tool_calls is not None
            and not decl_tools
        ):
            warn(
                f"'messages[{idx}]' is an Assistant message with a tool call, but {tool_defs_are_missing}"
            )
        if (
            message.role == Role.ASSISTANT
            and message.function_call is not None
            and not decl_functions
        ):
            warn(
                f"'messages[{idx}]' is an Assistant messages with a function call, but {func_defs_are_missing}"
            )
        if message.role == Role.FUNCTION and not decl_functions:
            warn(
                f"'messages[{idx}]' is a Function message, but {func_defs_are_missing}"
            )
        if message.role == Role.TOOL and not decl_tools:
            warn(
                f"'messages[{idx}]' is a Tool message, but {tool_defs_are_missing}"
            )


def _collect_tool_ids(messages: List[Message]) -> Dict[str, str]:
    ret: Dict[str, str] = {}

    for message in messages:
        if message.role == Role.ASSISTANT and message.tool_calls is not None:
            for tool_call in message.tool_calls:
                ret[tool_call.id] = tool_call.function.name

    return ret
