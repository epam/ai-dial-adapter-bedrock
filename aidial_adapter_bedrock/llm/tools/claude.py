import json
from collections import defaultdict
from typing import Any, List, Optional, Tuple

from aidial_sdk.chat_completion import FunctionCall, Tool
from defusedxml import ElementTree

from aidial_adapter_bedrock.llm.message import BaseMessage, SystemMessage
from aidial_adapter_bedrock.llm.tools.emulator import ToolsEmulator

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


def parse_function_call(text: str) -> FunctionCall:
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


class Claude2_1_ToolsEmulator(ToolsEmulator):
    @property
    def tools_string(self) -> Optional[str]:
        return _format_tools(self.tool_config.tools)

    def add_tool_declarations(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[BaseMessage], List[str]]:
        if self.tools_string is None:
            return messages, []

        system_message = _system_message_template.format(
            tools_string=self.tools_string
        )

        return [SystemMessage(content=system_message), *messages], [
            "</function_calls>"
        ]
