from typing import AsyncIterator, TypeVar

_T = TypeVar("_T")


async def aiter_to_list(iterator: AsyncIterator[_T]) -> list[_T]:
    return [item async for item in iterator]
