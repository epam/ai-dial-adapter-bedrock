from typing import Dict, List, Literal, Self, Tuple, assert_never

from aidial_sdk.chat_completion import (
    Function,
    FunctionChoice,
    Message,
    Role,
    ToolChoice,
)
from aidial_sdk.chat_completion.request import AzureChatCompletionRequest
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.errors import ValidationError


class ToolsConfig(BaseModel):
    functions: List[Function]
    """
    List of functions/tools.
    """

    required: bool
    """
    True forces the model to call one of the available functions.
    False allows the model to pick between generating a message or
    calling one or more tools/functions.
    """

    tool_ids: Dict[str, str] | None
    """
    Mapping from tool call IDs to corresponding tool names.
    None means that functions are used, not tools.
    """

    @property
    def is_tool(self) -> bool:
        return self.tool_ids is not None

    def not_supported(self) -> None:
        if self.functions:
            if self.is_tool:
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
    def filter_functions(
        function_call: Literal["auto", "none"] | FunctionChoice,
        functions: List[Function],
    ) -> Tuple[bool, List[Function]]:
        match function_call:
            case "none":
                return False, []
            case "auto":
                return False, functions
            case FunctionChoice(name=name):
                new_functions = [
                    func for func in functions if func.name == name
                ]
                if not new_functions:
                    raise ValidationError(
                        f"Function {name!r} is not on the list of available functions"
                    )
                return True, new_functions
            case _:
                assert_never(function_call)

    @staticmethod
    def tool_choice_to_function_call(
        tool_choice: Literal["auto", "none"] | ToolChoice | None,
    ) -> Literal["auto", "none"] | FunctionChoice | None:
        match tool_choice:
            case ToolChoice(function=FunctionChoice(name=name)):
                return FunctionChoice(name=name)
            case _:
                return tool_choice

    @classmethod
    def from_request(cls, request: AzureChatCompletionRequest) -> Self | None:
        validate_messages(request)

        if request.functions is not None:
            functions = request.functions
            function_call = request.function_call
            tool_ids = None

        elif request.tools is not None:
            functions = [tool.function for tool in request.tools]
            function_call = ToolsConfig.tool_choice_to_function_call(
                request.tool_choice
            )
            tool_ids = collect_tool_ids(request.messages)

        else:
            functions = []
            function_call = None
            tool_ids = None

        if function_call is None:
            function_call = "auto" if functions else "none"

        required, selected = ToolsConfig.filter_functions(
            function_call, functions
        )

        if selected == []:
            return None

        return cls(functions=selected, required=required, tool_ids=tool_ids)


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


def collect_tool_ids(messages: List[Message]) -> Dict[str, str]:
    ret: Dict[str, str] = {}

    for message in messages:
        if message.role == Role.ASSISTANT and message.tool_calls is not None:
            for tool_call in message.tool_calls:
                ret[tool_call.id] = tool_call.function.name

    return ret
