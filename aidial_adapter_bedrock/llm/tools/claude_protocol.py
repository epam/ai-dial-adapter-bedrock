import json
from typing import Dict, List, Literal, Optional

from aidial_sdk.chat_completion import Function, FunctionCall, ToolCall
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.message import (
    AIFunctionCallMessage,
    AIToolCallMessage,
)
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsConfig, ToolsMode
from aidial_adapter_bedrock.utils.pydantic import ExtraForbidModel
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
    default: Optional[str] = None
    items: Optional["ToolParameterProperties"] = None
    enum: Optional[List[str]] = None
    # The title is allowed according to the JSON Schema, but not used
    title: Optional[str] = None


class ToolParameters(BaseModel):
    type: Literal["object"]
    properties: Dict[str, ToolParameterProperties]
    required: Optional[List[str]]


def _print_tool_parameter_properties(
    props: ToolParameterProperties,
) -> list[str | None]:
    return [
        tag("type", props.type),
        tag_nl(
            "items",
            (
                _print_tool_parameter_properties(props.items)
                if props.items
                else None
            ),
        ),
        tag("enum", ", ".join(props.enum) if props.enum else None),
        tag("description", props.description),
        tag("default", props.default),
    ]


def _print_tool_parameter(name: str, props: ToolParameterProperties) -> str:
    return tag_nl(
        "parameter",
        [tag("name", name)] + _print_tool_parameter_properties(props),
    )


def _print_tool_parameters(parameters: ToolParameters) -> str:
    return tag_nl(
        "parameters",
        [
            _print_tool_parameter(name, props)
            for name, props in parameters.properties.items()
        ],
    )


def _print_tool_declaration(function: Function) -> str:
    return tag_nl(
        "tool_description",
        [
            tag("tool_name", function.name),
            tag("description", function.description),
            _print_tool_parameters(
                ToolParameters.parse_obj(function.parameters)
            ),
        ],
    )


def print_tool_declarations(functions: List[Function]) -> str:
    return tag_nl(
        "tools", [_print_tool_declaration(function) for function in functions]
    )


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
            f"Unable to parse function call, missing {FUNC_TAG_NAME!r} tag"
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
    config: Optional[ToolsConfig], text: str
) -> AIToolCallMessage | AIFunctionCallMessage | None:
    if config is None:
        return None

    call = _parse_function_call(text)
    if config.tools_mode == ToolsMode.TOOLS:
        id = config.create_fresh_tool_call_id(call.name)
        tool_call = ToolCall(index=0, id=id, type="function", function=call)
        return AIToolCallMessage(calls=[tool_call])
    else:
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
