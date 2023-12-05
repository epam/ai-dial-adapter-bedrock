import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import (
    AsyncIterator,
    Callable,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

T = TypeVar("T")


async def make_async(func: Callable[[], T]) -> T:
    with ThreadPoolExecutor(max_workers=1) as executor:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func)


async def to_async_iterator(iter: Iterator[T]) -> AsyncIterator[T]:
    def _next() -> Tuple[bool, Optional[T]]:
        try:
            return False, next(iter)
        except StopIteration:
            return True, None

    while True:
        is_end, item = await make_async(lambda: _next())
        if is_end:
            break
        else:
            yield cast(T, item)
