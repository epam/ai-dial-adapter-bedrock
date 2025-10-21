import contextlib
import re
from typing import overload

from openai import APIError
from pydantic import BaseModel


class ExpectedException(BaseModel):
    type: type[APIError]
    message: str
    display_message: str | None = None
    status_code: int | None = None


@overload
async def expected_exception(
    exception: ExpectedException,
): ...


@overload
async def expected_exception(
    cls: type[APIError],
    message: str,
    display_message: str | None = None,
    status_code: int | None = None,
): ...


@contextlib.asynccontextmanager
async def expected_exception(
    cls: type[APIError] | ExpectedException,
    message: str | None = None,
    display_message: str | None = None,
    status_code: int | None = None,
):
    try:
        yield
    except Exception as e:
        if isinstance(cls, ExpectedException):
            message = cls.message
            display_message = cls.display_message
            status_code = cls.status_code
            cls = cls.type

        assert message is not None

        assert isinstance(
            e, cls
        ), f"Actual exception type ({type(e)}) doesn't match the expected one ({cls})"
        actual_status_code = getattr(e, "status_code", None)
        assert actual_status_code == status_code
        assert re.search(message, str(e))
        assert (e.body or {}).get("display_message") == display_message  # type: ignore
    else:
        assert False, f"The test didn't raise the expected exception {cls}"
