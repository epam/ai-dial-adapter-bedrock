import json

import httpx
import openai
import pytest
from pydantic import BaseModel, ValidationError

from aidial_adapter_bedrock.anthropic_translator.errors import (
    anthropic_error_from_upstream,
    format_validation_error,
    status_to_anthropic_type,
    translator_error_handler,
)


def _status_error(
    status: int, body: object | None, message: str
) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://core/openai/v1/responses")
    return openai.APIStatusError(
        message, response=httpx.Response(status, request=request), body=body
    )


@pytest.mark.parametrize(
    "status, expected",
    [
        (400, "invalid_request_error"),
        (422, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (413, "request_too_large"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (502, "api_error"),
        (503, "overloaded_error"),
        (529, "overloaded_error"),
    ],
)
def test_status_to_anthropic_type(status, expected):
    assert status_to_anthropic_type(status) == expected


def test_error_message_from_openai_body():
    response = anthropic_error_from_upstream(
        _status_error(
            400, {"error": {"message": "bad field"}}, "Error code: 400"
        )
    )
    assert response.status_code == 400
    body = json.loads(bytes(response.body))
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"] == "bad field"


def test_error_message_from_raw_text():
    response = anthropic_error_from_upstream(
        _status_error(500, "Internal boom", "Internal boom")
    )
    body = json.loads(bytes(response.body))
    assert body["error"]["message"] == "Internal boom"


def test_error_message_falls_back_to_sdk_message_when_body_is_empty():
    response = anthropic_error_from_upstream(
        _status_error(500, "", "Error code: 500")
    )
    body = json.loads(bytes(response.body))
    assert body["error"]["message"] == "Error code: 500"


def test_error_message_prefers_string_error_field():
    # `error` itself can be a bare string rather than `{"message": ...}`.
    response = anthropic_error_from_upstream(
        _status_error(400, {"error": "field is required"}, "Error code: 400")
    )
    body = json.loads(bytes(response.body))
    assert body["error"]["message"] == "field is required"


def test_error_message_falls_back_to_top_level_message_field():
    # No `error` key at all; some upstreams put the message at the top level.
    response = anthropic_error_from_upstream(
        _status_error(
            400,
            {"message": "top-level message, no 'error' key"},
            "Error code: 400",
        )
    )
    body = json.loads(bytes(response.body))
    assert body["error"]["message"] == "top-level message, no 'error' key"


class _FourRequiredFieldsModel(BaseModel):
    a: int
    b: int
    c: int
    d: int


def test_format_validation_error_caps_at_three_entries():
    with pytest.raises(ValidationError) as exc:
        _FourRequiredFieldsModel.model_validate({})
    message = format_validation_error(exc.value)
    # Only the first 3 of the 4 missing-field errors are rendered.
    assert message.count("; ") == 2
    assert "a: " in message
    assert "b: " in message
    assert "c: " in message
    assert "d: " not in message


async def test_translator_error_handler_maps_connection_error_to_502():
    @translator_error_handler
    async def _handler():
        raise openai.APIConnectionError(
            message="Connection refused",
            request=httpx.Request("POST", "http://core/openai/v1/responses"),
        )

    response = await _handler()

    assert response.status_code == 502
    body = json.loads(bytes(response.body))
    assert body["error"]["type"] == "api_error"
    assert "Connection refused" in body["error"]["message"]


async def test_translator_error_handler_maps_unexpected_exception_to_500():
    @translator_error_handler
    async def _handler():
        raise RuntimeError("totally unexpected")

    response = await _handler()

    assert response.status_code == 500
    body = json.loads(bytes(response.body))
    assert body["error"]["type"] == "api_error"
    assert body["error"]["message"] == "totally unexpected"


class _StrictIntModel(BaseModel):
    x: int


async def test_pydantic_validation_error_reshaped_to_400_not_500():
    # A translator builds outbound requests via real pydantic models (e.g.
    # `ChatCompletionRequest`); a well-formed Anthropic input can still fail a
    # constraint the model enforces that the inbound `MessagesRequest` doesn't
    # (e.g. `stop` capped at 4 entries). That's the client's own input and the
    # client can fix it, so it must come back as a 400, not fall through to
    # the bare `except Exception` 500 branch.
    @translator_error_handler
    async def _handler():
        _StrictIntModel.model_validate({"x": "not-an-int"})
        raise AssertionError("model_validate should have raised")

    response = await _handler()

    assert response.status_code == 400
    body = json.loads(bytes(response.body))
    assert body["error"]["type"] == "invalid_request_error"
    assert "x" in body["error"]["message"]
