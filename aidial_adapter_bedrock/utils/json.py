import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel


def remove_nones(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def json_dumps_short(
    obj: Any, *, string_limit: int = 100, list_len_limit: int = 10, **kwargs
) -> str:
    def default(obj) -> str:
        return _truncate_strings(str(obj), string_limit)

    return json.dumps(
        _truncate_lists(
            _truncate_strings(to_dict(obj, **kwargs), string_limit),
            list_len_limit,
        ),
        default=default,
    )


def json_dumps(obj: Any, **kwargs) -> str:
    return json.dumps(to_dict(obj, **kwargs))


def to_dict(obj: Any, **kwargs) -> Any:
    def rec(val):
        return to_dict(val, **kwargs)

    def dict_field(key: str, val: Any) -> Any:
        if key in kwargs.get("excluded_keys", []):
            return "<excluded>"
        return val

    if isinstance(obj, bytes):
        return f"<bytes>({len(obj):_} B)"

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, dict):
        return {key: rec(dict_field(key, value)) for key, value in obj.items()}

    if isinstance(obj, list):
        return [rec(element) for element in obj]

    if isinstance(obj, tuple):
        return tuple(rec(element) for element in obj)

    if isinstance(obj, BaseModel):
        return rec(obj.dict())

    if hasattr(obj, "to_dict"):
        return rec(obj.to_dict())

    if is_dataclass(type(obj)):
        return rec(asdict(obj))

    return obj


def _truncate_strings(obj: Any, limit: int) -> Any:
    def rec(val):
        return _truncate_strings(val, limit)

    if isinstance(obj, dict):
        return {key: rec(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [rec(element) for element in obj]

    if isinstance(obj, tuple):
        return tuple(rec(element) for element in obj)

    if isinstance(obj, str) and len(obj) > limit:
        skip = len(obj) - limit
        return (
            obj[: limit // 2] + f"...({skip:_} skipped)..." + obj[-limit // 2 :]
        )

    return obj


def _truncate_lists(obj: Any, limit: int) -> Any:
    def rec(val):
        return _truncate_lists(val, limit)

    if isinstance(obj, dict):
        return {key: rec(value) for key, value in obj.items()}

    if isinstance(obj, list):
        if len(obj) > limit:
            skip = len(obj) - limit
            obj = (
                obj[: limit // 2]
                + [f"...({skip:_} skipped)..."]
                + obj[-limit // 2 :]
            )
        return [rec(element) for element in obj]

    if isinstance(obj, tuple):
        return tuple(rec(element) for element in obj)

    return obj
