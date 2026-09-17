import re

import pytest

from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)

CONFORMING: re.Pattern[str] = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{2,63}$")

_LONG_MCP_NAME: str = "mcp__" + "a" * 60 + "__do_the_thing"


@pytest.mark.parametrize(
    "name",
    ["get_weather", "abc", "a" * 64, "_leading", "with-hyphen", "Mixed_9"],
)
def test_conforming_names_are_never_touched(name: str) -> None:
    assert ToolNameAliases().to_upstream(name) == name


@pytest.mark.parametrize(
    "name",
    [
        _LONG_MCP_NAME,
        "a" * 65,
        "ab",
        "9leading_digit",
        "has spaces",
        "dots.and:colons",
        "unicodé",
        "-leading-hyphen",
    ],
)
def test_non_conforming_names_become_conforming_aliases(name: str) -> None:
    alias: str = ToolNameAliases().to_upstream(name)
    assert alias != name
    assert CONFORMING.match(alias), alias
    assert ToolNameAliases().to_client(alias) == name


def test_aliasing_is_deterministic_across_registries() -> None:
    assert ToolNameAliases().to_upstream(
        _LONG_MCP_NAME
    ) == ToolNameAliases().to_upstream(_LONG_MCP_NAME)


def test_names_differing_only_past_the_truncation_point_still_differ() -> None:
    aliases: ToolNameAliases = ToolNameAliases()
    shared: str = "m" * 80
    assert aliases.to_upstream(f"{shared}_one") != aliases.to_upstream(
        f"{shared}_two"
    )


def test_an_unregistered_name_is_returned_as_sent() -> None:
    assert ToolNameAliases().to_client("never_registered") == "never_registered"


def test_aliases_are_restored_across_registry_instances() -> None:
    original: str = "long_tool_" + "x" * 70
    alias: str = ToolNameAliases().to_upstream(original)
    assert ToolNameAliases().to_client(alias) == original


def test_registry_evicts_least_recently_used_alias() -> None:
    aliases: ToolNameAliases = ToolNameAliases()
    names: list[str] = [
        "eviction_test_" + "x" * 70 + str(i) for i in range(4097)
    ]
    registered: list[str] = [aliases.to_upstream(name) for name in names[:4096]]
    assert aliases.to_client(registered[0]) == names[0]
    aliases.to_upstream(names[-1])
    assert aliases.to_client(registered[0]) == names[0]
    assert aliases.to_client(registered[1]) == registered[1]
