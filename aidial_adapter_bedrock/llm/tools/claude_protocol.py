import json
from typing import Dict, List, Literal, Optional

from aidial_sdk.chat_completion import FunctionCall, Tool, ToolCall
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.message import (
    AIFunctionCallMessage,
    AIToolCallMessage,
)
from aidial_adapter_bedrock.llm.tools.tool_config import ToolConfig, ToolsMode
from aidial_adapter_bedrock.utils.pydnatic import ExtraForbidModel
from aidial_adapter_bedrock.utils.xml import parse_xml, tag, tag_nl

FUNC_TAG_NAME = "function_calls"
FUNC_START_TAG = f"<{FUNC_TAG_NAME}>"
FUNC_END_TAG = f"</{FUNC_TAG_NAME}>"


def get_system_message(tool_declarations: str) -> str:
    return f"""
In this environment you have access to a set of tools you can use to answer the user's question.

You may call them like this. Only invoke one function at a time and wait for the results before invoking another function:
<function_calls>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
...
</parameters>
</invoke>
</function_calls>

Avoid showing the function calls and respective results to the user.

Here are the tools available:
{tool_declarations}
""".strip()


class ToolParameterProperties(ExtraForbidModel):
    type: str
    description: Optional[str]


class ToolParameters(BaseModel):
    type: Literal["object"]
    properties: Dict[str, ToolParameterProperties]
    required: Optional[List[str]]


def _print_tool_parameter(name: str, props: ToolParameterProperties) -> str:
    return tag_nl(
        "parameter",
        [
            tag("name", name),
            tag("type", props.type),
            tag("description", props.description),
        ],
    )


def _print_tool_parameters(parameters: ToolParameters) -> str:
    return tag_nl(
        "parameters",
        [
            _print_tool_parameter(name, props)
            for name, props in parameters.properties.items()
        ],
    )


def _print_tool_declaration(tool: Tool) -> str:
    return tag_nl(
        "tool_description",
        [
            tag("tool_name", tool.function.name),
            tag("description", tool.function.description),
            _print_tool_parameters(
                ToolParameters.parse_obj(tool.function.parameters)
            ),
        ],
    )


def print_tool_declarations(tools: List[Tool]) -> str:
    return tag_nl("tools", [_print_tool_declaration(tool) for tool in tools])


def _print_function_call_parameters(parameters: dict) -> str:
    return tag_nl(
        "parameters",
        [tag(name, value) for name, value in parameters.items()],
    )


def print_tool_calls(calls: List[ToolCall]) -> str:
    return tag_nl(
        FUNC_TAG_NAME,
        [print_function_call(call.function) for call in calls],
    )


def print_function_call(call: FunctionCall) -> str:
    try:
        arguments = json.loads(call.arguments)
    except Exception:
        raise Exception(
            "Unable to parse function call arguments: it's not a valid JSON"
        )

    return tag_nl(
        FUNC_TAG_NAME,
        tag_nl(
            "invoke",
            [
                tag("tool_name", call.name),
                _print_function_call_parameters(arguments),
            ],
        ),
    )


def _parse_function_call(text: str) -> FunctionCall:
    start_index = text.find(FUNC_START_TAG)
    if start_index == -1:
        raise Exception(
            f"Unable to parse function call, missing '{FUNC_TAG_NAME}' tag"
        )

    try:
        dict = parse_xml(text[start_index:])
        invocation = dict[FUNC_TAG_NAME]["invoke"]

        tool_name = invocation["tool_name"]
        parameters = invocation["parameters"]
    except Exception:
        raise Exception("Unable to parse function call")

    return FunctionCall(name=tool_name, arguments=json.dumps(parameters))


def parse_call(
    config: Optional[ToolConfig], text: str
) -> AIToolCallMessage | AIFunctionCallMessage | None:
    if config is None:
        return None

    call = _parse_function_call(text)
    match config.mode:
        case ToolsMode.TOOLS:
            return AIToolCallMessage(
                calls=[ToolCall(id=call.name, type="function", function=call)]
            )
        case ToolsMode.FUNCTIONS:
            return AIFunctionCallMessage(call=call)


def print_function_call_result(name: str, content: str) -> str:
    return tag_nl(
        "function_results",
        [
            tag_nl(
                "result",
                [
                    tag("tool_name", name),
                    tag_nl("stdout", content),
                ],
            )
        ],
    )
