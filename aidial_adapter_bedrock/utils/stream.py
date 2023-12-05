from typing import AsyncIterator, List

import tests.utils.string as string


async def lstrip(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    start = True
    async for chunk in stream:
        if start:
            chunk = chunk.lstrip()
            if chunk != "":
                start = False
                yield chunk
        else:
            yield chunk


async def remove_prefix(
    stream: AsyncIterator[str], prefix: str
) -> AsyncIterator[str]:
    acc = ""
    start = True

    async for chunk in stream:
        if start:
            acc += chunk
            if len(acc) >= len(prefix):
                yield string.remove_prefix(prefix, acc)
                start = False
        else:
            yield chunk

    if start:
        yield acc


async def stop_at(
    stream: AsyncIterator[str], stop_sequences: List[str]
) -> AsyncIterator[str]:
    if len(stop_sequences) == 0:
        async for item in stream:
            yield item
        return

    buffer_len = max(map(len, stop_sequences)) - 1

    hold = ""
    async for chunk in stream:
        hold += chunk

        min_index = len(hold)
        for stop_sequence in stop_sequences:
            if stop_sequence in hold:
                min_index = min(min_index, hold.index(stop_sequence))

        if min_index < len(hold):
            commit = hold[:min_index]
            if commit:
                yield commit
            return

        commit, hold = hold[:-buffer_len], hold[-buffer_len:]
        if commit:
            yield commit

    if hold:
        yield hold


async def ensure_not_empty(
    gen: AsyncIterator[str], default: str
) -> AsyncIterator[str]:
    all_chunks_are_empty = True
    async for chunk in gen:
        all_chunks_are_empty = all_chunks_are_empty and chunk == ""
        yield chunk

    if all_chunks_are_empty:
        yield default
        yield default
