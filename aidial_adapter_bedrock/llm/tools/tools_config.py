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

    tool_choice: Literal["auto", "none", "required"] | ToolChoice

    tool_ids: Dict[str, str] | None
    """
    Mapping from tool call IDs to corresponding tool names.
    None means that functions are used, not tools.
    """

    @property
    def tools_mode(self) -> ToolsMode:
        if self.tool_ids is not None:
            return ToolsMode.TOOLS
        else:
            return ToolsMode.FUNCTIONS

    def not_supported(self) -> None:
        if self.tools:
            if self.tools_mode == ToolsMode.TOOLS:
                raise ValidationError("The tools aren't supported")
            else:
                raise ValidationError("The functions aren't supported")

    def create_fresh_tool_call_id(self, tool_name: str) -> str:
        if self.tool_ids is None:
            raise ValidationError("Function are used, but requested tool id")

        idx = 1
        while True:
            id = f"{tool_name}_{idx}"
            if id not in self.tool_ids:
                self.tool_ids[id] = tool_name
                return id
            idx += 1

    def get_tool_name(self, tool_call_id: str) -> str:
        if self.tool_ids is None:
            raise ValidationError("Function are used, but requested tool name")

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

    @classmethod
    def from_request(cls, request: AzureChatCompletionRequest) -> Self | None:
        validate_messages(request)

        if request.functions is not None:
            tools = [
                ToolsConfig._get_tool_from_function(tool)
                for tool in request.functions
            ]
            tool_choice = ToolsConfig._function_call_to_tool_choice(
                request.function_call
            )
            tool_ids = None
        elif request.tools is not None:
            tools = [
                ToolsConfig._get_tool_from_function(tool)
                for tool in request.tools
            ]
            tool_choice = request.tool_choice
            tool_ids = _collect_tool_ids(request.messages)
        else:
            return None

        return cls(
            tools=tools,
            tool_choice=tool_choice or "auto",
            tool_ids=tool_ids,
        )


def validate_messages(request: AzureChatCompletionRequest) -> None:
    decl_tools = request.tools is not None
    decl_functions = request.functions is not None

    if decl_functions and decl_tools:
        raise ValidationError("Both functions and tools are not allowed")

    for message in request.messages:
        if message.role == Role.ASSISTANT:
            use_tools = message.tool_calls is not None
            if use_tools and not decl_tools:
                raise ValidationError(
                    "Assistant message uses tools, but tools are not declared"
                )

            use_functions = message.function_call is not None
            if use_functions and not decl_functions:
                raise ValidationError(
                    "Assistant message uses functions, but functions are not declared"
                )
        if message.role == Role.FUNCTION:
            if not decl_functions:
                raise ValidationError(
                    "Function message is used, but functions are not declared"
                )
        if message.role == Role.TOOL:
            if not decl_tools:
                raise ValidationError(
                    "Tool message is used, but tools are not declared"
                )


def _collect_tool_ids(messages: List[Message]) -> Dict[str, str]:
    ret: Dict[str, str] = {}

    for message in messages:
        if message.role == Role.ASSISTANT and message.tool_calls is not None:
            for tool_call in message.tool_calls:
                ret[tool_call.id] = tool_call.function.name

    return ret
