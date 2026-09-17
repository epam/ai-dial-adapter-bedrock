from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
from functools import wraps
from typing import Literal, ParamSpec

import openai
from fastapi.responses import JSONResponse, Response
from typing_extensions import TypedDict

from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

INTERNAL_ERROR_MESSAGE: str = "Internal server error"
CONNECTION_ERROR_MESSAGE: str = "Failed to reach DIAL Core"
P = ParamSpec("P")


class AnthropicErrorType(Enum):
    INVALID_REQUEST = "invalid_request_error", 400, 422
    AUTHENTICATION = "authentication_error", 401
    PERMISSION = "permission_error", 403
    NOT_FOUND = "not_found_error", 404
    REQUEST_TOO_LARGE = "request_too_large", 413
    RATE_LIMIT = "rate_limit_error", 429
    API = "api_error", 500, 502
    OVERLOADED = "overloaded_error", 529, 503

    def __init__(
        self, code: str, status_code: int, *alternate_status_codes: int
    ) -> None:
        self.code: str = code
        self.status_codes: tuple[int, ...] = (
            status_code,
            *alternate_status_codes,
        )

    @property
    def status_code(self) -> int:
        return self.status_codes[0]

    @classmethod
    def from_status_code(cls, status_code: int) -> AnthropicErrorType:
        for error_type in cls:
            if status_code in error_type.status_codes:
                return error_type
        return cls.API


class ErrorDetail(TypedDict):
    type: str
    message: str


class ErrorResponse(TypedDict):
    type: Literal["error"]
    error: ErrorDetail


class AnthropicHTTPError(Exception):
    def __init__(self, error_type: AnthropicErrorType, message: str) -> None:
        super().__init__(message)
        self.error_type: AnthropicErrorType = error_type
        self.message: str = message

    @property
    def status_code(self) -> int:
        return self.error_type.status_code


def anthropic_error_object(
    error_type: AnthropicErrorType, message: str
) -> ErrorResponse:
    return {
        "type": "error",
        "error": {"type": error_type.code, "message": message},
    }


def anthropic_error_response(
    error_type: AnthropicErrorType,
    message: str,
    *,
    status_code: int | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=error_type.status_code
        if status_code is None
        else status_code,
        content=anthropic_error_object(error_type, message),
    )


def _extract_error_message(e: openai.APIStatusError) -> str:
    body: object | None = e.body
    if isinstance(body, dict):
        error: object = body.get("error")
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
    error_type: AnthropicErrorType = AnthropicErrorType.from_status_code(
        e.status_code
    )
    return anthropic_error_response(
        error_type, _extract_error_message(e), status_code=e.status_code
    )


def translator_error_handler(
    func: Callable[P, Awaitable[Response]],
) -> Callable[P, Awaitable[Response]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Response:
        try:
            return await func(*args, **kwargs)
        except AnthropicHTTPError as e:
            return anthropic_error_response(e.error_type, e.message)
        except openai.APIStatusError as e:
            log.warning("DIAL Core returned status %s", e.status_code)
            return anthropic_error_from_upstream(e)
        except openai.APIConnectionError:
            log.exception("Failed to reach DIAL Core from the translator")
            return anthropic_error_response(
                AnthropicErrorType.API,
                CONNECTION_ERROR_MESSAGE,
                status_code=502,
            )
        except Exception:
            log.exception("Unexpected error in the Anthropic translator")
            return anthropic_error_response(
                AnthropicErrorType.API, INTERNAL_ERROR_MESSAGE
            )

    return wrapper
