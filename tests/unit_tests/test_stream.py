from typing import AsyncIterator, List, Tuple

import pytest

import tests.utils.string as string
from aidial_adapter_bedrock.utils.stream import (
    ensure_not_empty,
    lstrip,
    remove_prefix,
    stop_at,
)


async def list_to_stream(xs: List[str]) -> AsyncIterator[str]:
    for x in xs:
        yield x


async def stream_to_string(stream: AsyncIterator[str]) -> str:
    ret = ""
    async for chunk in stream:
        ret += chunk
    return ret


lstrip_test_cases: List[Tuple[List[str]]] = [
    ([],),
    (["a"],),
    ([" a"],),
    ([" a", " b"],),
    (["", " a"],),
    ([" ", "", " a"],),
    ([" \n", " ", " a"],),
    ([" a\n\tb\n\t"],),
    ([" \n   \t \n    a\n\tb\n\t"],),
    (["", " \n   \t \n    a\n\tb\n\t"],),
    ([" \n", " ", "   \t \n  ", "  a  \n\tb\n\t"],),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    lstrip_test_cases,
    ids=lambda arg: f"{arg[0]}",
)
async def test_lstrip(test):
    (xs,) = test
    stream = lstrip(list_to_stream(xs))
    actual: str = await stream_to_string(stream)
    expected: str = "".join(xs).lstrip()
    assert actual == expected


remove_prefix_test_cases: List[Tuple[str, List[str]]] = [
    ("", []),
    ("a", []),
    ("a", ["b"]),
    ("a", ["", "", "a", "b"]),
    ("a", ["b", "a"]),
    ("a", ["a", "a"]),
    ("a", ["aa"]),
    ("a", ["aaaaa"]),
    ("abc", ["!abc!"]),
    ("abc", ["abcabc"]),
    ("a", ["a", "b"]),
    ("prefix", ["prefix:xyz"]),
    ("prefix", ["prefix:prefix:xyz"]),
    ("a", ["Aa"]),
    ("abc", ["a"]),
    ("abc", ["a", "bc"]),
    ("abc", ["a", "bcd"]),
    ("abc", ["a", "bc", "d"]),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    remove_prefix_test_cases,
    ids=lambda arg: f"{arg[0]}-{arg[1]}",
)
async def test_remove_prefix(test):
    (prefix, xs) = test
    steam = remove_prefix(list_to_stream(xs), prefix)
    actual: str = await stream_to_string(steam)
    expected: str = string.remove_prefix(prefix, "".join(xs))
    assert actual == expected


stop_at_test_cases: List[Tuple[str | List[str], List[str]]] = [
    ("", []),
    ("", ["a", "b"]),
    ("a", ["b"]),
    ("a", ["ba"]),
    ("a", ["b", "a"]),
    ("a", ["b", "a", "c"]),
    ("a", ["bac"]),
    ("a", ["baca"]),
    ("abc", ["zabcy"]),
    ("ab", ["d", "a", "b", "c"]),
    ("hello", ["? hel", "lo world", "!"]),
    ("hello world", ["? hel", "lo", " wor", "ld", "!"]),
    ("hello worlD", ["? hel", "lo", " wor", "ld", "!"]),
    ("z", ["ab", "cd"]),
    ("z", ["", "", "a", " \t  ", "  z ", "tt"]),
    (["hello", "world"], ["Hel", "lo", " ", "world", "!"]),
    (["ab", "ba"], ["abba"]),
    (["ba", "ab"], ["abba"]),
    (["a", "b", "c"], ["abc"]),
    (["c", "b", "a"], ["abc"]),
    ([], ["abc", "d", "ef"]),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    stop_at_test_cases,
    ids=lambda arg: f"{arg[0]}-{arg[1]}",
)
async def test_stop_at(test):
    (stop, xs) = test
    stop_sequences: List[str] = [stop] if isinstance(stop, str) else stop
    stream = stop_at(list_to_stream(xs), stop_sequences)
    actual: str = await stream_to_string(stream)
    expected: str = string.stop_at(stop_sequences, "".join(xs))
    assert actual == expected


ensure_not_empty_test_cases: List[Tuple[str | List[str], List[str]]] = [
    ("", []),
    (" ", ["", "", "a"]),
    (" ", ["", "", "\t", ""]),
    (" ", ["abc", "de"]),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    ensure_not_empty_test_cases,
    ids=lambda arg: f"{arg[0]}-{arg[1]}",
)
async def test_ensure_not_empty(test):
    (default, xs) = test
    stream = ensure_not_empty(list_to_stream(xs), default)
    actual: str = await stream_to_string(stream)
    expected: str = string.ensure_not_empty(default, "".join(xs))
    assert actual == expected
