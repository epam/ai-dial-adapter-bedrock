import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from aidial_sdk.chat_completion import FunctionCall, Tool, ToolCall
from defusedxml import ElementTree

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
from aidial_adapter_bedrock.llm.tools.base import ToolConfig, ToolsMode
from aidial_adapter_bedrock.llm.tools.call_recognizer import CallRecognizer
from aidial_adapter_bedrock.llm.tools.emulator import ToolsEmulator
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

Arg = None | str | List[str | None]


def _arg_to_str(arg: Arg) -> str | None:
    if isinstance(arg, list):
        return "\n".join([x for x in arg if x is not None])
    return arg


def _tag(name: str, arg: Arg) -> str | None:
    content = _arg_to_str(arg)
    return content and f"<{name}>{content}</{name}>"


def _tag_nl(name: str, arg: Arg) -> str | None:
    content = _arg_to_str(arg)
    return content and f"<{name}>\n{content}\n</{name}>"


def _format_parameters(parameters: dict) -> str | None:
    """
    Converts parameters to XML string:
    <parameters>
    <parameter>
    <name>latitude</name>
    <type>string</type>
    <description>The latitude coordinate as a string</description>
    </parameter> <parameter>
    <name>longitude</name>
    <type>string</type>
    <description>The longitude coordinate as a string</description>
    </parameter>
    </parameters>
    """

    return _tag_nl(
        "parameters",
        [
            _tag_nl(
                "parameter",
                [
                    _tag("name", name),
                    _tag("type", body["type"]),
                    _tag("description", body.get("description")),
                ],
            )
            for name, body in parameters["properties"].items()
        ],
    )


def _format_tool(tool: Tool) -> str | None:
    """
    Converts tool to XML string:
    <tool_description>
    <tool_name>get_weather</tool_name>
    <description>
    Returns weather data for a given latitude and longitude. </description>
    <parameters>
    <parameter>
    <name>latitude</name>
    <type>string</type>
    <description>The latitude coordinate as a string</description>
    </parameter> <parameter>
    <name>longitude</name>
    <type>string</type>
    <description>The longitude coordinate as a string</description>
    </parameter>
    </parameters>
    </tool_description>
    """

    description = tool.function.description

    return _tag_nl(
        "tool_description",
        [
            _tag("tool_name", tool.function.name),
            description and _tag("description", description),
            _format_parameters(tool.function.parameters),
        ],
    )


def _format_tools(tools: List[Tool]) -> str | None:
    return _tag_nl("tools", [_format_tool(tool) for tool in tools])


_system_message_template = """
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
{tools_string}
"""


def etree_to_dict(t) -> dict[str, Any]:
    d = {t.tag: {}}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(("@" + k, v) for k, v in t.attrib.items())
    if t.text and t.text.strip():
        if children or t.attrib:
            d[t.tag]["#text"] = t.text
        else:
            d[t.tag] = t.text
    return d


def _print_function_parameters(parameters: dict) -> str:
    return (
        _tag_nl(
            "parameters",
            [_tag(name, value) for name, value in parameters.items()],
        )
        or ""
    )


def _print_tool_call(call: ToolCall) -> str:
    return _print_function_call(call.function)


def _print_tool_calls(calls: List[ToolCall]) -> str:
    return (
        _tag_nl(
            "function_calls",
            [_print_tool_call(call) for call in calls],
        )
        or ""
    )


def _print_function_call(call: FunctionCall) -> str:
    """
    Prints function call in the following format:
    <function_calls>
    <invoke>
    <tool_name>get_lat_long</tool_name>
    <parameters>
    <place>London</place>
    </parameters>
    </invoke>
    """
    return (
        _tag_nl(
            "function_calls",
            _tag_nl(
                "invoke",
                [
                    _tag("tool_name", call.name),
                    _print_function_parameters(json.loads(call.arguments)),
                ],
            ),
        )
        or ""
    )


def _parse_function_call(text: str) -> FunctionCall:
    """
    Parses function call string:
    <function_calls>
    <invoke>
    <tool_name>get_lat_long</tool_name>
    <parameters>
    <place>London</place>
    </parameters>
    </invoke>
    """
    skip_n = len("<function_calls>")
    start_index = text.find("<function_calls>")
    if start_index == -1:
        raise Exception(
            "Unable to parse function call, missing 'function_calls' tag"
        )

    extracted_text = text[start_index + skip_n :]

    xml = ElementTree.fromstring(extracted_text)
    tool_name_element = xml.find("tool_name")
    if tool_name_element is None:
        raise Exception(
            "Unable to parse function call, invalid XML or missing 'tool_name' tag"
        )

    tool_name = tool_name_element.text.strip()
    parameters_xml = xml.find("parameters")
    if parameters_xml is None:
        raise Exception(
            "Unable to parse function call, invalid XML or missing 'parameters' tag"
        )

    param_dict = etree_to_dict(parameters_xml)
    arguments = param_dict["parameters"]

    return FunctionCall(
        name=tool_name, arguments=json.dumps(arguments, indent=2)
    )


def _parse_call(
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


def _print_function_call_result(name: str, content: str) -> str:
    return (
        _tag_nl(
            "function_results",
            [
                _tag_nl(
                    "result",
                    [
                        _tag("tool_name", name),
                        _tag_nl("stdout", content),
                    ],
                )
            ],
        )
        or ""
    )


class Claude2_1_ToolsEmulator(ToolsEmulator):
    call_recognizer: CallRecognizer

    class Config:
        arbitrary_types_allowed = True

    @property
    def _tool_declarations(self) -> Optional[str]:
        return self.tool_config and _format_tools(self.tool_config.tools)

    def add_tool_declarations(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[BaseMessage], List[str]]:
        if self._tool_declarations is None:
            return messages, []

        system_message = _system_message_template.format(
            tools_string=self._tool_declarations
        )

        return [SystemMessage(content=system_message), *messages], [
            "</function_calls>"
        ]

    def convert_to_base_messages(
        self, messages: List[BaseMessage | ToolMessage]
    ) -> List[BaseMessage]:
        id_to_name: Dict[str, str] = {}
        return [
            message
            if isinstance(message, BaseMessage)
            else self._to_base_message(id_to_name, message)
            for message in messages
        ]

    def _to_base_message(
        self, id_to_name: Dict[str, str], msg: ToolMessage
    ) -> BaseMessage:
        mode: Optional[ToolsMode] = self.tool_config and self.tool_config.mode
        match msg:
            case HumanToolResultMessage(id=id, content=content):
                assert (
                    mode is None or mode == ToolsMode.TOOLS
                ), f"Received tool result in '{mode.value}' mode"
                name = id_to_name.get(id)
                if name is None:
                    log.warning(
                        f"Unable to find tool name for id '{id}', assuming '_unknown' name"
                    )
                    name = "unknown"

                return HumanRegularMessage(
                    content=_print_function_call_result(
                        name=name, content=content
                    )
                )
            case HumanFunctionResultMessage(name=name, content=content):
                assert (
                    mode is None or mode == ToolsMode.FUNCTIONS
                ), f"Received function result in '{mode.value}' mode"
                return HumanRegularMessage(
                    content=_print_function_call_result(
                        name=name, content=content
                    )
                )
            case AIToolCallMessage(calls=calls):
                assert (
                    mode is None or mode == ToolsMode.TOOLS
                ), f"Received tool call in '{mode.value}' mode"
                for call in calls:
                    id_to_name[call.id] = call.function.name
                return AIRegularMessage(content=_print_tool_calls(calls))
            case AIFunctionCallMessage(call=call):
                assert (
                    mode is None or mode == ToolsMode.FUNCTIONS
                ), f"Received function call in '{mode.value}' mode"
                return AIRegularMessage(content=_print_function_call(call))

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
            init_buffer=30,
            start_tag="<function_calls>",
            call_parser=lambda text: _parse_call(tool_config, text),
        ),
    )
