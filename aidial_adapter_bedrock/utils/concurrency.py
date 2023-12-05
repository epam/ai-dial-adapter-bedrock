import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Callable, Iterator, Optional, Tuple, TypeVar

T = TypeVar("T")
A = TypeVar("A")


async def make_async(func: Callable[[A], T], arg: A) -> T:
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, arg)


async def to_async_iterator(iter: Iterator[T]) -> AsyncIterator[T]:
    def _next() -> Tuple[bool, Optional[T]]:
        try:
            return False, next(iter)
        except StopIteration:
            return True, None

    while True:
        is_end, item = await make_async(lambda _: _next(), ())
        if is_end:
            break
        else:
            assert item is not None
            yield item
