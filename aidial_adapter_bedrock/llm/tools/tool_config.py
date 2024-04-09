from enum import Enum
from typing import List, Optional

from aidial_sdk.chat_completion import (
    Function,
    FunctionChoice,
    Tool,
    ToolChoice,
)
from aidial_sdk.chat_completion.request import ChatCompletionRequest
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.exceptions import ValidationError


class ToolsMode(str, Enum):
    TOOLS = "tools"
    FUNCTIONS = "functions"


def _fun_to_tool(fun: Function) -> Tool:
    return Tool(type="function", function=fun)


class ToolConfig(BaseModel):
    mode: ToolsMode
    tools: List[Tool]

    @classmethod
    def from_request(
        cls, request: ChatCompletionRequest
    ) -> Optional["ToolConfig"]:
        mode: ToolsMode = ToolsMode.TOOLS
        tools: List[Tool] = []
        selected_function: Optional[str] = None

        if request.functions is not None and len(request.functions) > 0:
            choice = request.function_call
            if choice == "none":
                return None

            if isinstance(choice, FunctionChoice):
                selected_function = choice.name

            mode = ToolsMode.FUNCTIONS
            tools = [_fun_to_tool(fun) for fun in request.functions]

        elif request.tools is not None and len(request.tools) > 0:
            choice = request.tool_choice
            if choice == "none":
                return None

            if isinstance(choice, ToolChoice):
                selected_function = choice.function.name

            mode = ToolsMode.TOOLS
            tools = request.tools
        else:
            return None

        if selected_function is not None:
            tools = [
                tool
                for tool in tools
                if tool.function.name == selected_function
            ]

            if len(tools) == 0:
                raise ValidationError(
                    f"Unable to find tool with name '{selected_function}'"
                )

        return cls(mode=mode, tools=tools)
