from typing import Generator, List, Tuple

import pytest

import tests.utils.string as string
from aidial_adapter_bedrock.utils.stream import (
    ensure_not_empty,
    lstrip,
    remove_prefix,
    stop_at,
)


def list_to_gen(xs: List[str]) -> Generator[str, None, None]:
    for x in xs:
        yield x


def gen_to_string(gen: Generator[str, None, None]) -> str:
    return "".join(x for x in gen)


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


@pytest.mark.parametrize(
    "test",
    lstrip_test_cases,
    ids=lambda arg: f"{arg[0]}",
)
def test_lstrip(test):
    (xs,) = test
    gen = lstrip(list_to_gen(xs))
    actual = gen_to_string(gen)
    expected = "".join(xs).lstrip()
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


@pytest.mark.parametrize(
    "test",
    remove_prefix_test_cases,
    ids=lambda arg: f"{arg[0]}-{arg[1]}",
)
def test_remove_prefix(test):
    (prefix, xs) = test
    gen = remove_prefix(list_to_gen(xs), prefix)
    actual = gen_to_string(gen)
    expected = string.remove_prefix(prefix, "".join(xs))
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


@pytest.mark.parametrize(
    "test",
    stop_at_test_cases,
    ids=lambda arg: f"{arg[0]}-{arg[1]}",
)
def test_stop_at(test):
    (stop, xs) = test
    stop_sequences: List[str] = [stop] if isinstance(stop, str) else stop
    gen = stop_at(list_to_gen(xs), stop_sequences)
    actual = gen_to_string(gen)
    expected = string.stop_at(stop_sequences, "".join(xs))
    assert actual == expected


ensure_not_empty_test_cases: List[Tuple[str | List[str], List[str]]] = [
    ("", []),
    (" ", ["", "", "a"]),
    (" ", ["", "", "\t", ""]),
    (" ", ["abc", "de"]),
]


@pytest.mark.parametrize(
    "test",
    ensure_not_empty_test_cases,
    ids=lambda arg: f"{arg[0]}-{arg[1]}",
)
def test_ensure_not_empty(test):
    (default, xs) = test
    gen = ensure_not_empty(list_to_gen(xs), default)
    actual = gen_to_string(gen)
    expected = string.ensure_not_empty(default, "".join(xs))
    assert actual == expected
