import pytest

from aidial_adapter_bedrock.anthropic_translator.errors import (
    AnthropicHTTPError,
)
from aidial_adapter_bedrock.anthropic_translator.request import validate_request
from tests.unit_tests.anthropic_translator.helpers import convert, user


@pytest.mark.parametrize(
    "body",
    [
        {"messages": {}},
        {"messages": ""},
        {"messages": [{"role": "user", "content": {}}]},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "source": "invalid"}],
                }
            ]
        },
        {"tools": [{"name": "missing_schema"}]},
        {"thinking": {"type": "enabled"}},
        {"thinking": {"type": "enabled", "budget_tokens": True}},
        {"output_config": {"effort": "turbo"}},
        {"tool_choice": {"type": "bogus"}},
        {"service_tier": "flex"},
        {"cache_control": {}},
    ],
)
def test_sdk_rejects_invalid_request_fields(body: dict[str, object]) -> None:
    with pytest.raises(AnthropicHTTPError) as error:
        validate_request(
            {"model": "foobar", "max_tokens": 100, **user("hi"), **body}
        )
    assert error.value.status_code == 400


def test_tool_schema_keywords_survive_sdk_validation() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "additionalProperties": False,
        "$defs": {"name": {"type": "string"}},
        "oneOf": [{"required": ["name"]}],
    }
    result = convert(
        {**user("hi"), "tools": [{"name": "test_tool", "input_schema": schema}]}
    )
    assert result.model_dump()["tools"][0]["function"]["parameters"] == schema


def test_validation_checks_later_nested_blocks() -> None:
    with pytest.raises(AnthropicHTTPError) as error:
        validate_request(
            {
                "model": "foobar",
                "max_tokens": 100,
                **user(
                    [
                        {"type": "text", "text": "valid"},
                        {"type": "text", "text": 42},
                    ]
                ),
            }
        )
    assert error.value.status_code == 400
    assert "messages.0.content" in error.value.message
