from functools import wraps

from aidial_sdk import HTTPException as DialException
from anthropic import APIStatusError
from botocore.exceptions import ClientError

from aidial_adapter_bedrock.llm.exceptions import UserError, ValidationError
from aidial_adapter_bedrock.utils.log_config import app_logger as log


# TODO: Catch the rate limit exception and dress it like OpenAI rate limit exception: botocore.errorfactory.ThrottlingException: An error occurred (ThrottlingException) when calling the InvokeModel operation (reached max retries: 4): Too many requests, please wait before trying again. You have sent too many requests.  Wait before trying again.
def to_dial_exception(e: Exception) -> DialException:
    if isinstance(e, ClientError):
        if "The security token included in the request is invalid" in str(e):
            return DialException(
                status_code=401,
                type="invalid_request_error",
                message=f"Invalid Authentication: {str(e)}",
                code="invalid_api_key",
                param=None,
            )
        elif "ModelTimeoutException" in str(e):
            return DialException(
                status_code=502,
                type="timeout",
                message="Request timed out",
                code=None,
                param=None,
            )

    if isinstance(e, APIStatusError):
        return DialException(
            status_code=e.status_code,
            type="invalid_request_error",
            message=e.message,
            code=None,
            param=None,
        )

    if isinstance(e, ValidationError):
        return e.to_dial_exception()

    if isinstance(e, UserError):
        return e.to_dial_exception()

    if isinstance(e, DialException):
        return e

    return DialException(
        status_code=500,
        type="internal_server_error",
        message=str(e),
        code=None,
        param=None,
    )


def dial_exception_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            log.exception(e)
            raise to_dial_exception(e) from e

    return wrapper
