from functools import wraps
from typing import Literal, Optional, TypedDict

from botocore.exceptions import ClientError


class OpenAIError(TypedDict):
    type: Literal["invalid_request_error", "internal_server_error"] | str
    message: str
    param: Optional[str]
    code: Optional[str]


class OpenAIException(Exception):
    def __init__(self, status_code: int, error: OpenAIError):
        self.status_code = status_code
        self.error = error


def error_handling_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ClientError as e:
            if "The security token included in the request is invalid" in str(
                e
            ):
                raise OpenAIException(
                    status_code=401,
                    error={
                        "type": "invalid_request_error",
                        "message": f"Invalid Authentication: {str(e)}",
                        "code": "invalid_api_key",
                        "param": None,
                    },
                )
            raise e
        except Exception as e:
            raise OpenAIException(
                status_code=500,
                error={
                    "type": "internal_server_error",
                    "message": str(e),
                    "code": None,
                    "param": None,
                },
            )

    return wrapper
