"""Builders shared by the Anthropic translator tests.

Each one carries the defaults that keep a test focused on the single thing it
exercises, so a test states only what it actually depends on.
"""

from typing import Any

from aidial_adapter_bedrock.anthropic_translator.capabilities import (
    DeploymentProfile,
    ReasoningKnob,
    TopLevelEffort,
)

ALL_EFFORTS = ["low", "medium", "high"]


def make_profile(
    *,
    temperature_supported: bool = True,
    cache_supported: bool = True,
    max_completion_tokens_supported: bool = False,
    max_output_tokens: int | None = None,
    reasoning: ReasoningKnob | None = None,
) -> DeploymentProfile:
    """A deployment that supports everything the tests don't care about, so
    only the capability under test has to be stated."""
    return DeploymentProfile(
        temperature_supported=temperature_supported,
        cache_supported=cache_supported,
        max_completion_tokens_supported=max_completion_tokens_supported,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning or TopLevelEffort(levels=ALL_EFFORTS),
    )


def catalog(
    deployment: str = "gpt-5.5",
    *,
    features: dict[str, Any] | None = None,
    limits: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A `GET /openai/models` body listing one deployment.

    `features` is always present and fully serialised in the real contract, so
    the default spells every member out.
    """
    entry: dict[str, Any] = {
        "id": deployment,
        "object": "model",
        "features": {
            "temperature": True,
            "cache": False,
            "reasoning_efforts": [],
            "max_tokens_supported": True,
            "max_completion_tokens_supported": False,
            "tools": True,
            **(features or {}),
        },
    }
    if limits is not None:
        entry["limits"] = limits
    if defaults is not None:
        entry["defaults"] = defaults
    return {"object": "list", "data": [entry]}
