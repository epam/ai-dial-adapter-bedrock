import json
from enum import Enum
from typing import Any

from pydantic import BaseModel


def remove_nones(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def json_dumps_short(
    obj: Any,
    *,
    string_limit: int = 100,
    list_len_limit: int = 10,
    **kwargs,
) -> str:
    return json.dumps(
        _truncate_lists(
            _truncate_strings(to_dict(obj), string_limit), list_len_limit
        ),
        default=str,
        **kwargs,
    )


def json_dumps(obj: Any) -> str:
    return json.dumps(to_dict(obj))


def to_dict(obj: Any) -> Any:
    rec = to_dict

    if isinstance(obj, bytes):
        return f"<bytes>({len(obj):_} B)"

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, dict):
        return {key: rec(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [rec(element) for element in obj]

    if isinstance(obj, BaseModel):
        return rec(obj.dict())

    if hasattr(obj, "to_dict"):
        return rec(obj.to_dict())

    return obj


def _truncate_strings(obj: Any, limit: int) -> Any:
    if isinstance(obj, dict):
        return {
            key: _truncate_strings(value, limit) for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [_truncate_strings(element, limit) for element in obj]

    if isinstance(obj, str) and len(obj) > limit:
        skip = len(obj) - limit
        return (
            obj[: limit // 2] + f"...({skip:_} skipped)..." + obj[-limit // 2 :]
        )

    return obj


def _truncate_lists(obj: Any, limit: int) -> Any:
    if isinstance(obj, dict):
        return {
            key: _truncate_lists(value, limit) for key, value in obj.items()
        }

    if isinstance(obj, list):
        if len(obj) > limit:
            skip = len(obj) - limit
            obj = (
                obj[: limit // 2]
                + [f"...({skip:_} skipped)..."]
                + obj[-limit // 2 :]
            )
        return [_truncate_lists(element, limit) for element in obj]

    return obj
