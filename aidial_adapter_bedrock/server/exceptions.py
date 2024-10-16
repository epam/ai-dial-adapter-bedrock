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
from enum import Enum
from functools import wraps

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InternalServerError, InvalidRequestError
from anthropic import APIStatusError
from botocore.exceptions import ClientError

from aidial_adapter_bedrock.llm.errors import UserError, ValidationError
from aidial_adapter_bedrock.utils.log_config import app_logger as log


def create_error(status_code: int, message: str) -> DialException:
    return (
        InvalidRequestError(message)
        if status_code < 500
        else InternalServerError(message)
    )


class BedrockExceptionCode(Enum):
    """
    See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithResponseStream.html#API_runtime_InvokeModelWithResponseStream_ResponseSyntax
    for the types of exceptions
    """

    THROTTLING = "throttlingException"
    MODEL_TIMEOUT = "modelTimeoutException"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        return NotImplemented


def _get_meta_status_code(response: dict) -> int | None:
    code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if isinstance(code, int):
        return code
    return None


def _get_response_error_code(response: dict) -> int | None:
    code = response.get("Error", {}).get("Code")

    if isinstance(code, str):
        match code:
            case BedrockExceptionCode.THROTTLING:
                return 429
            case BedrockExceptionCode.MODEL_TIMEOUT:
                return 408
            case _:
                pass
    return None


def _get_content_filter_error(response: dict) -> DialException | None:
    if (
        message := response.get("message")
    ) and "One or more prompts contains filtered words" in message:
        return InvalidRequestError(message=message, code="content_filter")
    return None


def to_dial_exception(e: Exception) -> DialException:
    if (
        isinstance(e, ClientError)
        and hasattr(e, "response")
        and isinstance(e.response, dict)
    ):
        response = e.response
        log.debug(
            f"botocore.exceptions.ClientError.response: {json.dumps(response)}"
        )

        if error := _get_content_filter_error(response):
            return error

        status_code = (
            _get_response_error_code(response)
            or _get_meta_status_code(response)
            or 500
        )

        return create_error(status_code, str(e))

    if isinstance(e, APIStatusError):
        return create_error(e.status_code, e.message)

    if isinstance(e, ValidationError):
        return e.to_dial_exception()

    if isinstance(e, UserError):
        return e.to_dial_exception()

    if isinstance(e, DialException):
        return e

    return InternalServerError(str(e))


def dial_exception_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            log.exception(
                f"caught exception: {type(e).__module__}.{type(e).__name__}"
            )
            raise to_dial_exception(e) from e

    return wrapper
