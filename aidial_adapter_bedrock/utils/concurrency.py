import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, cast

from aidial_adapter_bedrock.utils.constants import THREAD_POOL_SIZE

_T = TypeVar("_T")

# A single shared pool for all blocking operations including all boto calls.
#
# A thread is held per blocking call, not per request, but `next()` on a
# Botocore event stream blocks until the next chunk arrives -- most of a
# stream's wall time -- so demand stays close to the number of streams in
# flight. Beyond the pool size requests still all progress, each chunk waiting
# its turn for a thread, so this caps throughput rather than concurrency.
# Hence a size far above the `min(32, cpu_count + 4)` default.
_THREAD_POOL = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)


async def run_in_threadpool(func: Callable[[], _T]) -> _T:
    return await asyncio.get_running_loop().run_in_executor(_THREAD_POOL, func)


async def to_async_iterator(iter: Iterator[_T]) -> AsyncIterator[_T]:
    def _next() -> tuple[bool, _T | None]:
        try:
            return False, next(iter)
        except StopIteration:
            return True, None

    while True:
        is_end, item = await run_in_threadpool(lambda: _next())
        if is_end:
            break
        else:
            yield cast(_T, item)
