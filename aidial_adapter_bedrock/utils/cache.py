import json
from asyncio import Lock
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, ParamSpec, Tuple, TypeVar

from pydantic import BaseModel

from aidial_adapter_bedrock.utils.datetime import ensure_utc, now_utc
from aidial_adapter_bedrock.utils.log_config import app_logger as log

_P = ParamSpec("_P")
_T = TypeVar("_T")


def ttl_cache(
    func: Callable[_P, Coroutine[Any, Any, Tuple[datetime | None, _T]]],
) -> Callable[_P, Coroutine[Any, Any, _T]]:
    _cache: dict[str, Tuple[datetime | None, _T]] = {}
    _locks: dict[str, Lock] = defaultdict(Lock)

    async def _wrapper(*args, **kwargs):
        key = _make_key(args, kwargs)

        async with _locks[key]:
            expiry, value = _cache.get(key, (None, None))

            if value is not None:
                if expiry is None or ensure_utc(expiry) > now_utc() + timedelta(
                    minutes=1
                ):
                    return value
                else:
                    log.debug("cache entry has expired")

            expiration, value = await func(*args, **kwargs)
            _cache[key] = (expiration, value)
            return value

    return _wrapper


def _make_key(args: tuple, kwargs: dict) -> str:
    dump_args = {"sort_keys": True, "separators": (",", ":")}

    def default(obj):
        if isinstance(obj, BaseModel):
            return json.dumps(obj.model_dump(), **dump_args)
        raise TypeError(f"Cannot serialize object of type {type(obj)!r}")

    return json.dumps([args, kwargs], **dump_args, default=default)
