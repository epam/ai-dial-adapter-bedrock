import re

import pytest

from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)

CONFORMING = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{2,63}$")

_LONG_MCP_NAME = "mcp__" + "a" * 60 + "__do_the_thing"


@pytest.mark.parametrize(
    "name",
    ["get_weather", "abc", "a" * 64, "_leading", "with-hyphen", "Mixed_9"],
)
def test_conforming_names_are_never_touched(name):
    assert ToolNameAliases().to_upstream(name) == name


@pytest.mark.parametrize(
    "name",
    [
        _LONG_MCP_NAME,
        "a" * 65,
        "ab",  # shorter than the 3-character minimum
        "9leading_digit",
        "has spaces",
        "dots.and:colons",
        "unicodé",
        "-leading-hyphen",
    ],
)
def test_non_conforming_names_become_conforming_aliases(name):
    alias = ToolNameAliases().to_upstream(name)
    assert alias != name
    assert CONFORMING.match(alias), alias
    assert len(alias) <= 64


def test_an_empty_name_has_nothing_to_alias():
    assert ToolNameAliases().to_upstream("") == ""


def test_aliases_round_trip():
    aliases = ToolNameAliases()
    alias = aliases.to_upstream(_LONG_MCP_NAME)
    assert aliases.to_client(alias) == _LONG_MCP_NAME


def test_aliasing_is_deterministic_across_registries():
    # The alias derives from the name alone, so the three sites that carry a
    # name agree without sharing state.
    assert ToolNameAliases().to_upstream(
        _LONG_MCP_NAME
    ) == ToolNameAliases().to_upstream(_LONG_MCP_NAME)


def test_distinct_names_get_distinct_aliases():
    aliases = ToolNameAliases()
    first = aliases.to_upstream("mcp__server__" + "x" * 60)
    second = aliases.to_upstream("mcp__server__" + "y" * 60)
    assert first != second


def test_names_differing_only_past_the_truncation_point_still_differ():
    # The head is truncated to 55 characters, so only the digest separates
    # these two.
    aliases = ToolNameAliases()
    shared = "m" * 60
    assert aliases.to_upstream(f"{shared}_one") != aliases.to_upstream(
        f"{shared}_two"
    )


def test_an_unregistered_name_is_returned_as_sent():
    assert ToolNameAliases().to_client("never_registered") == "never_registered"
