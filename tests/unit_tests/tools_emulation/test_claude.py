import json

from aidial_sdk.chat_completion import Function, FunctionCall

from aidial_adapter_bedrock.llm.tools.claude_protocol import (
    _parse_function_call,
    print_function_call,
    print_tool_declarations,
)
from aidial_adapter_bedrock.utils.xml import parse_xml

TOOL_ARITY_2 = Function(
    name="func_arity_2",
    description="desc",
    parameters={
        "type": "object",
        "properties": {
            "param1": {"type": "type1", "description": "desc1"},
            "param2": {"type": "type2"},
        },
    },
)

TOOL_ARITY_0 = Function(
    name="func_arity_0",
    description="desc",
    parameters={"type": "object", "properties": {}},
)

TOOL_ENUM_PARAM = Function(
    name="func_enum_param",
    description="tool with enum parameter",
    parameters={
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "enum": ["value1", "value2", "value3"],
            }
        },
    },
)

TOOL_ARRAY_PARAM = Function(
    name="func_array_param",
    description="tool with array parameter",
    parameters={
        "type": "object",
        "properties": {
            "param1": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
    },
)


FUNCTION_CALL = FunctionCall(
    name="name",
    arguments=json.dumps(
        {
            "arg_name1": "arg_value1",
            "arg_name2": "arg_value2",
            "arg_name3": "arg_value3",
        }
    ),
)

FUNCTION_CALL_STR = """
<function_calls>
<invoke>
<tool_name>name</tool_name>
<parameters>
<arg_name1>arg_value1</arg_name1>
<arg_name2>arg_value2</arg_name2>
<arg_name3>arg_value3</arg_name3>
</parameters>
</invoke>
</function_calls>
""".strip()


def test_print_tool_decls():
    assert (
        print_tool_declarations(
            [TOOL_ARITY_2, TOOL_ARITY_0, TOOL_ENUM_PARAM, TOOL_ARRAY_PARAM]
        )
        == """
<tools>
<tool_description>
<tool_name>func_arity_2</tool_name>
<description>desc</description>
<parameters>
<parameter>
<name>param1</name>
<type>type1</type>
<description>desc1</description>
</parameter>
<parameter>
<name>param2</name>
<type>type2</type>
</parameter>
</parameters>
</tool_description>
<tool_description>
<tool_name>func_arity_0</tool_name>
<description>desc</description>
<parameters>
</parameters>
</tool_description>
<tool_description>
<tool_name>func_enum_param</tool_name>
<description>tool with enum parameter</description>
<parameters>
<parameter>
<name>param1</name>
<type>string</type>
<enum>value1, value2, value3</enum>
</parameter>
</parameters>
</tool_description>
<tool_description>
<tool_name>func_array_param</tool_name>
<description>tool with array parameter</description>
<parameters>
<parameter>
<name>param1</name>
<type>array</type>
<items>
<type>string</type>
</items>
</parameter>
</parameters>
</tool_description>
</tools>
""".strip()
    )


def test_print_function_call():
    assert print_function_call(FUNCTION_CALL) == FUNCTION_CALL_STR


def test_parse_function_call():
    assert _parse_function_call(FUNCTION_CALL_STR) == FUNCTION_CALL


def test_parse_xml():
    assert parse_xml(FUNCTION_CALL_STR) == {
        "function_calls": {
            "invoke": {
                "parameters": {
                    "arg_name1": "arg_value1",
                    "arg_name2": "arg_value2",
                    "arg_name3": "arg_value3",
                },
                "tool_name": "name",
            }
        }
    }
