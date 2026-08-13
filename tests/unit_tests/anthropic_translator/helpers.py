"""Builders shared by the Anthropic translator tests.

Each one carries the defaults that keep a test focused on the single thing it
exercises, so a test states only what it actually depends on.
"""

import json

from aidial_adapter_bedrock.anthropic_translator.capabilities import (
    FEATURES_HEADER,
    DeploymentProfile,
)

ALL_EFFORTS = ["low", "medium", "high"]


def make_profile(
    *,
    temperature_supported: bool = True,
    cache_supported: bool = True,
    max_completion_tokens_supported: bool = False,
    reasoning_efforts: list[str] | None = None,
) -> DeploymentProfile:
    """A deployment that supports everything the tests don't care about, so
    only the capability under test has to be stated."""
    return DeploymentProfile(
        temperature_supported=temperature_supported,
        cache_supported=cache_supported,
        max_completion_tokens_supported=max_completion_tokens_supported,
        reasoning_efforts=(
            ALL_EFFORTS if reasoning_efforts is None else reasoning_efforts
        ),
    )


def features_header(**features) -> dict[str, str]:
    """The `x-dial-deployment-features` header Core stamps on a routed request.

    `features` is fully serialised in the real contract, so the default spells
    every member the translator reads — and a few it deliberately ignores.
    """
    return {
        FEATURES_HEADER: json.dumps(
            {
                "temperature": True,
                "cache": False,
                "reasoning_efforts": [],
                "max_completion_tokens_supported": False,
                "tools": True,
                **features,
            }
        )
    }
