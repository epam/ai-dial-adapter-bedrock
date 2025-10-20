import asyncio
import sys
from datetime import datetime as original_datetime
from datetime import timedelta, timezone

import pytest
from pydantic import BaseModel

from aidial_adapter_bedrock.utils.cache import _make_key, ttl_cache


class datetime(original_datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore
        return original_datetime(2025, 1, 1, 0, 0, 0, tzinfo=tz)


@pytest.fixture(
    params=[None, timezone.utc],
    ids=["offset-naive", "offset-aware-utc"],
)
def time_zone(request):
    return request.param


@pytest.fixture
def get_now(time_zone):
    def _f():
        return datetime.now(time_zone)

    return _f


@pytest.fixture(autouse=True)
def fixed_datetime(monkeypatch):
    monkeypatch.setattr(
        sys.modules["aidial_adapter_bedrock.utils.datetime"],
        "datetime",
        datetime,
    )


async def test_basic_caching(get_now):
    calls = 0

    async def func(x: int):
        nonlocal calls
        calls += 1
        return (get_now() + timedelta(minutes=5), x * 2)

    cached = ttl_cache(func)

    result1 = await cached(2)
    result2 = await cached(2)

    assert result1 == 4
    assert result2 == 4
    assert calls == 1


async def test_no_expiry():
    calls = 0

    async def func(x: int):
        nonlocal calls
        calls += 1
        return (None, x + 1)

    cached = ttl_cache(func)

    r1 = await cached(1)
    r2 = await cached(1)
    r3 = await cached(1)

    assert r1 == 2
    assert r2 == 2
    assert r3 == 2
    assert calls == 1


async def test_expiry_refresh(get_now):
    calls = 0

    async def func(x: int):
        nonlocal calls
        calls += 1
        if calls == 1:
            return (get_now() + timedelta(seconds=30), "first")
        return (get_now() + timedelta(minutes=5), "second")

    cached = ttl_cache(func)

    v1 = await cached(3)
    v2 = await cached(3)
    v3 = await cached(3)

    assert v1 == "first"
    assert v2 == "second"
    assert v3 == "second"
    assert calls == 2


async def test_different_args_and_kwargs(get_now):
    calls = []

    async def func(a: int, b: int = 0):
        calls.append((a, b))
        return (get_now() + timedelta(minutes=5), a + b)

    cached = ttl_cache(func)

    r1 = await cached(1, b=2)
    r2 = await cached(1, b=3)
    r3 = await cached(1, b=2)

    assert r1 == 3
    assert r2 == 4
    assert r3 == 3
    assert len(calls) == 2


async def test_model_arg_serialization(get_now):
    calls = 0

    class M(BaseModel):
        x: int

    async def func(m: M):
        nonlocal calls
        calls += 1
        return (get_now() + timedelta(minutes=5), m.x)

    cached = ttl_cache(func)

    m1 = M(x=5)
    m2 = M(x=5)
    m3 = M(x=6)

    r1 = await cached(m1)
    r2 = await cached(m2)
    r3 = await cached(m3)

    assert r1 == 5
    assert r2 == 5
    assert r3 == 6
    assert calls == 2


async def test_different_keys():
    calls = []

    async def func(x):
        calls.append(x)
        return (None, x)

    cached = ttl_cache(func)
    res1 = await cached(1)
    res2 = await cached(2)

    assert res1 == 1
    assert res2 == 2
    assert set(calls) == {1, 2}


async def test_args_kwargs_separate_keys():
    calls = []

    async def func(x):
        calls.append(x)
        return (None, x)

    cached = ttl_cache(func)
    res1 = await cached(1)
    res2 = await cached(x=1)

    assert res1 == 1
    assert res2 == 1
    assert len(calls) == 2


async def test_concurrent_requests(get_now):
    calls = 0

    async def func(x):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return (get_now() + timedelta(minutes=5), x)

    n = 100
    cached = ttl_cache(func)
    results = await asyncio.gather(*[cached(7) for _ in range(n)])
    assert results == [7] * n
    assert calls == 1


def test_make_key_order_independence():
    key1 = _make_key((1,), {"a": 2, "b": 3})
    key2 = _make_key((1,), {"b": 3, "a": 2})
    assert key1 == key2
