"""
Error mapping for the Anthropic-facing translators.

Everything a translator returns must be Anthropic-shaped: both the HTTP error
body and the terminal SSE `error` event share the same inner object:

    {"type": "error", "error": {"type": "<anthropic type>", "message": "<text>"}}

This module deliberately does NOT reuse `dial_exception_decorator`, which
produces OpenAI-shaped DIAL errors.
"""

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

import openai
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

# Anthropic error `type` values, named once so the spelling can't drift.
INVALID_REQUEST_ERROR = "invalid_request_error"
API_ERROR = "api_error"


class AnthropicHTTPError(Exception):
    """An error that must surface to the client as an Anthropic error body with
    a specific HTTP status code."""

    def __init__(self, status_code: int, type: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.type = type
        self.message = message


def status_to_anthropic_type(status_code: int) -> str:
    if status_code in (400, 422):
        return INVALID_REQUEST_ERROR
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 413:
        return "request_too_large"
    if status_code == 429:
        return "rate_limit_error"
    # 503/529 are "overloaded"; every other 5xx (and anything unexpected) maps
    # to a generic api_error.
    if status_code in (503, 529):
        return "overloaded_error"
    return API_ERROR


def anthropic_error_object(type: str, message: str) -> dict:
    return {"type": "error", "error": {"type": type, "message": message}}


def anthropic_error_response(
    status_code: int, type: str, message: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=anthropic_error_object(type, message),
    )


def _extract_error_message(e: openai.APIStatusError) -> str:
    """Pull `error.message` out of an OpenAI-shaped error body.

    `e.body` is whatever the SDK could make of the error response: the decoded
    JSON, the raw text, or None. `e.message` is the SDK's own rendering, used
    as the fallback.
    """
    body: object | None = e.body
    if isinstance(body, dict):
        error: Any = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str) and error:
            return error
        if isinstance(body.get("message"), str):
            return body["message"]

    if isinstance(body, str) and body.strip():
        return body.strip()

    return e.message


def anthropic_error_from_upstream(e: openai.APIStatusError) -> JSONResponse:
    """Translate a non-2xx response from DIAL Core into an Anthropic error."""
    type: str = status_to_anthropic_type(e.status_code)
    return anthropic_error_response(
        e.status_code, type, _extract_error_message(e)
    )


def format_validation_error(e: ValidationError) -> str:
    """Render a pydantic `ValidationError` as a short, human-readable message."""
    parts: list[str] = []
    for err in e.errors()[:3]:
        loc: str = ".".join(str(x) for x in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return "; ".join(parts) or "Invalid request body"


def translator_error_handler(
    func: Callable[..., Awaitable[Response]],
) -> Callable[..., Awaitable[Response]]:
    """Catch-all that guarantees an Anthropic-shaped error body for any failure
    escaping a translator endpoint implementation."""

    @wraps(func)
    async def wrapper(*args, **kwargs) -> Response:
        try:
            return await func(*args, **kwargs)
        except AnthropicHTTPError as e:
            return anthropic_error_response(e.status_code, e.type, e.message)
        except openai.APIStatusError as e:
            log.warning("DIAL Core returned status %s", e.status_code)
            return anthropic_error_from_upstream(e)
        except openai.APIConnectionError as e:
            log.exception("Failed to reach DIAL Core from the translator")
            return anthropic_error_response(
                502, API_ERROR, f"Failed to reach DIAL Core: {e}"
            )
        except ValidationError as e:
            # Raised when a translator builds an outbound request that
            # violates a constraint the inbound model didn't enforce (e.g.
            # `stop` exceeding the SDK's 4-item cap). The client's own input
            # caused this, so it's a 400, not a 500.
            return anthropic_error_response(
                400, INVALID_REQUEST_ERROR, format_validation_error(e)
            )
        except Exception as e:
            log.exception("Unexpected error in the Anthropic translator")
            return anthropic_error_response(500, API_ERROR, str(e))

    return wrapper
