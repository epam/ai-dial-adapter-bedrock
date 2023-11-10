from typing import Generator, List

import tests.utils.string as string


def lstrip(gen: Generator[str, None, None]) -> Generator[str, None, None]:
    start = True
    for chunk in gen:
        if start:
            chunk = chunk.lstrip()
            if chunk != "":
                start = False
                yield chunk
        else:
            yield chunk


def remove_prefix(
    gen: Generator[str, None, None], prefix: str
) -> Generator[str, None, None]:
    acc = ""
    start = True

    for chunk in gen:
        if start:
            acc += chunk
            if len(acc) >= len(prefix):
                yield string.remove_prefix(prefix, acc)
                start = False
        else:
            yield chunk

    if start:
        yield acc


def stop_at(
    gen: Generator[str, None, None], stop_sequences: List[str]
) -> Generator[str, None, None]:
    if len(stop_sequences) == 0:
        yield from gen
        return

    buffer_len = max(map(len, stop_sequences)) - 1

    hold = ""
    for chunk in gen:
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


def ensure_not_empty(
    gen: Generator[str, None, None], default: str
) -> Generator[str, None, None]:
    all_chunks_are_empty = True
    for chunk in gen:
        all_chunks_are_empty = all_chunks_are_empty and chunk == ""
        yield chunk

    if all_chunks_are_empty:
        yield default
