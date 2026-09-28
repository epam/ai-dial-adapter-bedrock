import json
from asyncio import Lock
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta
from typing import Any, Generic, ParamSpec, Protocol, TypeVar

from pydantic import BaseModel

from aidial_adapter_bedrock.utils.datetime import ensure_utc, now_utc
from aidial_adapter_bedrock.utils.log_config import app_logger as log

_P = ParamSpec("_P")
_T_co = TypeVar("_T_co", covariant=True)
_T = TypeVar("_T")


class _SyncCachedFunction(Protocol, Generic[_P, _T_co]):
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T_co: ...
    async def clear(self) -> None: ...


def cache(
    close: Callable[[_T], Coroutine[Any, Any, None]] | None = None,
) -> Callable[[Callable[_P, _T]], _SyncCachedFunction[_P, _T]]:
    def wrapper(
        func: Callable[_P, _T],
    ) -> _SyncCachedFunction[_P, _T]:
        _cache: dict[str, _T] = {}

        class _Wrapper:
            def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
                key = _make_key(args, kwargs)

                if key not in _cache:
                    _cache[key] = func(*args, **kwargs)

                return _cache[key]

            async def clear(self) -> None:
                nonlocal _cache
                entries = _cache
                _cache = {}

                func_name = f"{func.__module__}.{func.__qualname__}"
                log.debug(f"Clearing cache {func_name}")

                if close is not None:
                    for key, value in entries.items():
                        log.debug(f"Closing cached value {func_name}({key})")

                        try:
                            await close(value)
                        except Exception as e:
                            log.error(f"Error on closing cached value: {e}")

        return _Wrapper()

    return wrapper


class _AsyncCachedFunction(Protocol, Generic[_P, _T_co]):
    async def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T_co: ...
    def clear(self) -> None: ...


def ttl_cache(
    maxsize: int | None = None,
    close: Callable[[_T], Coroutine[Any, Any, None]] | None = None,
) -> Callable[
    [Callable[_P, Coroutine[Any, Any, tuple[datetime | None, _T]]]],
    _AsyncCachedFunction[_P, _T],
]:
    """
    Caches the awaited value until the expiration it is returned with.

    `maxsize` bounds the cache, evicting the least recently used entry;
    leaving it unset makes the cache grow with the number of distinct keys.
    `close` releases the resources of a value that the cache drops, either
    because it was evicted or because its expiration passed.
    """

    def wrapper(
        func: Callable[_P, Coroutine[Any, Any, tuple[datetime | None, _T]]],
    ) -> _AsyncCachedFunction[_P, _T]:
        _cache: OrderedDict[str, tuple[datetime | None, _T]] = OrderedDict()
        _locks: dict[str, Lock] = defaultdict(Lock)

        func_name = f"{func.__module__}.{func.__qualname__}"

        async def _discard(key: str, value: _T) -> None:
            lock = _locks.get(key)
            if lock is not None and not lock.locked():
                del _locks[key]

            if close is None:
                return

            try:
                await close(value)
            except Exception as e:
                log.error(f"Error on closing cached value: {e}")

        class _Wrapper:
            async def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
                key = _make_key(args, kwargs)

                async with _locks[key]:
                    expiry, value = _cache.get(key, (None, None))

                    if value is not None:
                        if expiry is None or ensure_utc(
                            expiry
                        ) > now_utc() + timedelta(minutes=1):
                            _cache.move_to_end(key)
                            return value
                        else:
                            log.debug("cache entry has expired")

                    expiration, new_value = await func(*args, **kwargs)
                    _cache[key] = (expiration, new_value)
                    _cache.move_to_end(key)

                    # The expired value is dropped only now, because it stayed
                    # usable while its replacement was being created.
                    if value is not None:
                        await _discard(key, value)

                    while maxsize is not None and len(_cache) > maxsize:
                        evicted_key, (_, evicted) = _cache.popitem(last=False)
                        log.debug(
                            f"Evicting the least recently used entry of "
                            f"{func_name}, {len(_cache)} entries left"
                        )
                        await _discard(evicted_key, evicted)

                    return new_value

            def clear(self) -> None:
                _cache.clear()
                _locks.clear()

        return _Wrapper()

    return wrapper


def _make_key(args: tuple, kwargs: dict) -> str:
    dump_args = {"sort_keys": True, "separators": (",", ":")}

    def default(obj):
        if isinstance(obj, BaseModel):
            return json.dumps(obj.model_dump(), **dump_args)
        raise TypeError(f"Cannot serialize object of type {type(obj)!r}")

    return json.dumps([args, kwargs], **dump_args, default=default)
