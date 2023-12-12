from typing import List, Optional

from aidial_sdk.chat_completion import Tool

from aidial_adapter_bedrock.llm.message import BaseMessage, SystemMessage
from aidial_adapter_bedrock.llm.tools_emulation.emulator import ToolsEmulator

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
                    _tag("name", param.name),
                    _tag("type", param.type),
                    _tag("description", param.description),
                ],
            )
            for param in parameters
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


class Claude2_1_ToolsEmulator(ToolsEmulator):
    @property
    def tools_string(self) -> Optional[str]:
        return self.tool_config and _format_tools(self.tool_config.tools)

    def transform_messages(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        if self.tools_string is None:
            return messages

        system_message = _system_message_template.format(
            tools_string=self.tools_string
        )
        return [SystemMessage(content=system_message), *messages]
