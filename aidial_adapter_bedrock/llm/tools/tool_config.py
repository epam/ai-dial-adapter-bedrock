from enum import Enum
from typing import List, Literal, Optional, Union

from aidial_sdk.chat_completion import (
    Function,
    FunctionChoice,
    Request,
    Tool,
    ToolChoice,
)
from pydantic import BaseModel


class ToolsMode(str, Enum):
    TOOLS = "tools"
    FUNCTIONS = "functions"


def _fun_choice_to_tool_choice(choice: FunctionChoice) -> ToolChoice:
    return ToolChoice(type="function", function=choice)


def _fun_to_tool(fun: Function) -> Tool:
    return Tool(type="function", function=fun)


class ToolConfig(BaseModel):
    mode: ToolsMode
    tools: List[Tool]
    choice: Union[Literal["auto"], ToolChoice]

    @classmethod
    def from_request(cls, request: Request) -> Optional["ToolConfig"]:
        if request.functions is not None:
            fun_choice = request.function_call
            if fun_choice is None or fun_choice == "none":
                return None
            tool_choice = (
                "auto"
                if fun_choice == "auto"
                else _fun_choice_to_tool_choice(fun_choice)
            )
            return cls(
                mode=ToolsMode.FUNCTIONS,
                tools=[_fun_to_tool(fun) for fun in request.functions],
                choice=tool_choice,
            )

        if request.tools is not None:
            choice = request.tool_choice
            if choice is None or choice == "none":
                return None
            return cls(
                mode=ToolsMode.TOOLS,
                tools=request.tools,
                choice=choice,
            )

        return None
