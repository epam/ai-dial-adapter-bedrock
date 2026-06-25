"""
The kinds of service exceptions which bedrock invocation may throw:

https://github.com/boto/botocore/blob/1.31.57/botocore/data/bedrock-runtime/2023-09-30/service-2.json#L46-L57

The service exceptions have the following inheritance hierarchy:

- ValidationException (botocore.errorfactory)
    - ClientError (botocore.exceptions)
        - Exception (builtins)
            - BaseException (builtins)
                - object (builtins)

The recommended way to discriminate service exceptions is
to access `response` field of a ClientError instance:

https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html#parsing-error-responses-and-catching-exceptions-from-aws-services
"""

import json
import re
from enum import Enum
from functools import wraps
from typing import assert_never

import anthropic
import fastapi
from aidial_adapter_anthropic.adapter import UserError, ValidationError
from aidial_sdk.exceptions import (
    DeploymentNotFoundError,
    InternalServerError,
    InvalidRequestError,
    ResourceNotFoundError,
)
from aidial_sdk.exceptions import HTTPException as DialException
from botocore.exceptions import ClientError as BotocoreClientError

from aidial_adapter_bedrock.utils.log_config import app_logger as log


def _get_exception_type(status_code: int) -> str | None:
    if status_code in {400, 422}:
        return "invalid_request_error"
    if status_code == 500:
        return "internal_server_error"
    return None


def _get_anthropic_error_message(e: anthropic.APIStatusError) -> str:
    if isinstance(body := e.body, dict) and isinstance(
        (msg := body.get("message")), str
    ):
        return msg
    return e.message


def _anthropic_error_message_to_response(
    e: anthropic.APIStatusError,
) -> fastapi.Response:
    return fastapi.Response(
        content=e.response.content,
        status_code=e.status_code,
        headers=_copy_anthropic_headers(e),
    )


def _parse_anthropic_streaming_error(text: str) -> DialException | None:
    # Unfortunately, anthropic SDK obscures the original error message:
    # https://github.com/anthropics/anthropic-sdk-python/blob/8b244157a7d03766bec645b0e1dc213c6d462165/src/anthropic/lib/bedrock/_stream_decoder.py#L57-L58
    # So we have to parse it manually.

    prefix = "Bad response code, expected 200: "
    if not text.startswith(prefix):
        return None
    text = text.removeprefix(prefix)

    code_pattern = re.search(r"'status_code':\s*(\d+)", text)
    message_pattern = re.search(r"\"message\":\s*\"(.*?)\"", text)

    code = int(code_pattern.group(1)) if code_pattern else None
    message = str(message_pattern.group(1)) if message_pattern else None

    if code and message:
        message = message.replace("\\'", "'")
        return _create_error(code, message)
    return None


def _copy_headers(
    e: anthropic.APIStatusError, keys: list[str]
) -> dict[str, str] | None:
    headers = e.response.headers
    copied = {
        key: value for key in keys if (value := headers.get(key)) is not None
    }
    return copied or None


def _copy_anthropic_headers(
    e: anthropic.APIStatusError,
) -> dict[str, str] | None:
    return _copy_headers(e, ["Retry-After", "Retry-After-Ms"])


def _create_error(
    status_code: int, message: str, headers: dict[str, str] | None = None
) -> DialException:
    return DialException(
        status_code=status_code,
        type=_get_exception_type(status_code),
        message=message,
        headers=headers,
    )


class BedrockExceptionCode(Enum):
    """
    See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithResponseStream.html#API_runtime_InvokeModelWithResponseStream_ResponseSyntax
    for the types of exceptions
    """

    INTERNAL_SERVER = "internalServerException"
    MODEL_STREAM_ERROR = "modelStreamErrorException"
    MODEL_TIMEOUT = "modelTimeoutException"
    SERVER_UNAVAILABLE = "serviceUnavailableException"
    THROTTLING = "throttlingException"
    VALIDATION = "validationException"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        return NotImplemented

    def get_status_code(self) -> int:
        match self:
            case BedrockExceptionCode.INTERNAL_SERVER:
                return 500
            case BedrockExceptionCode.MODEL_STREAM_ERROR:
                return 424
            case BedrockExceptionCode.MODEL_TIMEOUT:
                return 408
            case BedrockExceptionCode.SERVER_UNAVAILABLE:
                return 503
            case BedrockExceptionCode.THROTTLING:
                return 429
            case BedrockExceptionCode.VALIDATION:
                return 400
            case _:
                assert_never(self)


def _get_meta_status_code(response: dict) -> int | None:
    code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if isinstance(code, int):
        return code
    return None


def _get_response_error_code(response: dict) -> int | None:
    code = response.get("Error", {}).get("Code")
    try:
        return BedrockExceptionCode(code).get_status_code()
    except Exception:
        return None


def _get_status_code(response: dict) -> int:
    return (
        _get_response_error_code(response)
        or _get_meta_status_code(response)
        or 500
    )


def _get_end_of_life_error(
    response: dict, status_code: int
) -> DialException | None:
    eol_message = "This model version has reached the end of its life"
    if (
        status_code == 404
        and (message := response.get("message"))
        and eol_message in message
    ):
        return DeploymentNotFoundError(
            message=message, display_message=eol_message, type=None
        )
    return None


def _get_content_filter_error(response: dict) -> DialException | None:
    if (
        message := response.get("message")
    ) and "One or more prompts contains filtered words" in message:
        return InvalidRequestError(message=message, code="content_filter")
    return None


def to_dial_exception(e: Exception) -> DialException:
    if (
        isinstance(e, BotocoreClientError)
        and hasattr(e, "response")
        and isinstance(e.response, dict)
    ):
        response = e.response
        log.debug(
            f"botocore.exceptions.ClientError.response: {json.dumps(response)}"
        )

        if error := _get_content_filter_error(response):
            return error

        status_code = _get_status_code(response)
        if error := _get_end_of_life_error(response, status_code):
            return error

        return _create_error(status_code, str(e))

    if isinstance(e, anthropic.APIStatusError):
        if error := _get_end_of_life_error(e.response.json(), e.status_code):
            return error

        message = _get_anthropic_error_message(e)

        # We want to save Retry-After header if it's present:
        # https://platform.claude.com/docs/en/api/rate-limits#tier-1
        headers = _copy_anthropic_headers(e)
        return _create_error(e.status_code, message, headers)

    if isinstance(e, ValidationError):
        return e.to_dial_exception()

    if isinstance(e, UserError):
        return e.to_dial_exception()

    if isinstance(e, DialException):
        return e

    if isinstance(e, ValueError):
        exc = _parse_anthropic_streaming_error(str(e))
        if exc is not None:
            return exc

    return InternalServerError(str(e))


def dial_exception_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            dial_exception = to_dial_exception(e)
            log.exception(
                f"Caught exception: {type(e).__module__}.{type(e).__name__}. "
                f"The exception converted to the dial exception: {dial_exception!r}."
            )
            raise dial_exception from e

    return wrapper


def anthropic_exception_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs) -> fastapi.Response:
        try:
            return await func(*args, **kwargs)
        except anthropic.APIStatusError as e:
            return _anthropic_error_message_to_response(e)
        except Exception as e:
            dial_exception = to_dial_exception(e)
            log.exception(
                f"Caught exception: {type(e).__module__}.{type(e).__name__}. "
                f"The exception converted to the dial exception: {dial_exception!r}."
            )
            return dial_exception.to_fastapi_response()

    return wrapper


def not_implemented_handler(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except NotImplementedError:
            raise ResourceNotFoundError(
                "The endpoint is not implemented"
            ) from None

    return wrapper
