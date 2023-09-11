from functools import wraps
from typing import Literal, Optional, TypedDict

from botocore.exceptions import ClientError

from llm.exceptions import ValidationError


class OpenAIError(TypedDict):
    type: Literal["invalid_request_error", "internal_server_error"] | str
    message: str
    param: Optional[str]
    code: Optional[str]


class OpenAIException(Exception):
    def __init__(self, status_code: int, error: OpenAIError):
        self.status_code = status_code
        self.error = error

    def __str__(self) -> str:
        return f"OpenAIException(status_code={self.status_code}, error={self.error})"


# TODO: Catch the rate limit exception and dress it like OpenAI rate limit exception: botocore.errorfactory.ThrottlingException: An error occurred (ThrottlingException) when calling the InvokeModel operation (reached max retries: 4): Too many requests, please wait before trying again. You have sent too many requests.  Wait before trying again.
def to_open_ai_exception(e: Exception) -> OpenAIException:
    if isinstance(e, ClientError):
        if "The security token included in the request is invalid" in str(e):
            return OpenAIException(
                status_code=401,
                error={
                    "type": "invalid_request_error",
                    "message": f"Invalid Authentication: {str(e)}",
                    "code": "invalid_api_key",
                    "param": None,
                },
            )
        elif "ModelTimeoutException" in str(e):
            return OpenAIException(
                status_code=502,
                error={
                    "type": "timeout",
                    "message": "Request timed out",
                    "code": None,
                    "param": None,
                },
            )

    if isinstance(e, ValidationError):
        return OpenAIException(
            status_code=422,
            error={
                "type": "invalid_request_error",
                "message": e.message,
                "code": "invalid_argument",
                "param": None,
            },
        )

    return OpenAIException(
        status_code=500,
        error={
            "type": "internal_server_error",
            "message": str(e),
            "code": None,
            "param": None,
        },
    )


def open_ai_exception_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            raise to_open_ai_exception(e)

    return wrapper
