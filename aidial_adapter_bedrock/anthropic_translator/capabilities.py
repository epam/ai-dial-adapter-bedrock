"""
Deployment capability discovery via DIAL Core's model catalog.

DIAL Core fronts several model adapters whose accepted request shapes genuinely
differ, and the difference is not inferable from the request body: a Gemini
deployment configured with a thinking budget rejects any request that also
carries `reasoning_effort`, because the Vertex adapter maps that onto Gemini's
`thinking_level`. A translator that emits one fixed shape therefore works
against some deployments and hard-fails against others, so the shape is chosen
per deployment from `GET {DIAL_URL}/openai/models`.

A lookup here must never fail — or indefinitely delay — the user's message. Any
failure yields `UNRESOLVED_PROFILE`, which means "unknown", not "unsupported":
every consumer asserts nothing and omits the capability-gated field, because
sending nothing degrades a request while sending the wrong knob fails it.
"""

import asyncio
import hashlib
import time
from collections import OrderedDict
from typing import Any

from pydantic import BaseModel

from aidial_adapter_bedrock.anthropic_translator.core_client import (
    get_http_client,
)
from aidial_adapter_bedrock.anthropic_translator.settings import (
    get_model_catalog_size,
    get_model_catalog_timeout,
    get_model_catalog_ttl,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log
from aidial_adapter_bedrock.utils.pydantic import ExtraAllowModel


class Features(ExtraAllowModel):
    temperature: bool | None = None
    cache: bool | None = None
    reasoning_efforts: list[str] | None = None
    max_completion_tokens_supported: bool | None = None


class Limits(ExtraAllowModel):
    max_completion_tokens: int | None = None


class ModelEntry(ExtraAllowModel):
    id: str
    features: Features | None = None
    limits: Limits | None = None
    # `defaults` is an opaque, operator-supplied map copied verbatim from
    # deployment configuration with no schema validation, so it is walked
    # defensively rather than typed: any level of it can be anything.
    defaults: dict[str, Any] | None = None


class ModelCatalog(ExtraAllowModel):
    data: list[ModelEntry]


class NestedEffort(BaseModel):
    """The deployment nests reasoning under `configuration.reasoning`: it takes
    an effort level merged into the defaults it already declares."""

    defaults: dict[str, Any]


class NestedBudget(BaseModel):
    """The deployment nests thinking under `configuration.thinking` (Gemini):
    it takes a raw token budget, never an effort."""

    defaults: dict[str, Any]


class TopLevelEffort(BaseModel):
    """No nested configuration: the standard top-level `reasoning_effort`,
    restricted to the levels the deployment advertises. An empty list is a real
    answer meaning the deployment supports no reasoning — as is an unresolved
    profile, where asserting nothing is the only safe move."""

    levels: list[str]


ReasoningKnob = NestedEffort | NestedBudget | TopLevelEffort


class DeploymentProfile(BaseModel):
    """The request shape one deployment accepts."""

    temperature_supported: bool
    cache_supported: bool
    max_completion_tokens_supported: bool
    max_output_tokens: int | None
    reasoning: ReasoningKnob


def _nested_dict(node: Any, *path: str) -> dict[str, Any]:
    for key in path:
        if not isinstance(node, dict):
            return {}
        node = node.get(key)
    return node if isinstance(node, dict) else {}


def _reasoning_knob(
    defaults: dict[str, Any], features: Features
) -> ReasoningKnob:
    configuration = _nested_dict(defaults, "custom_fields", "configuration")
    # Which *key* exists decides the shape; the value is coerced separately
    # because the operator may have configured anything under it.
    if "reasoning" in configuration:
        return NestedEffort(defaults=_nested_dict(configuration, "reasoning"))
    if "thinking" in configuration:
        return NestedBudget(defaults=_nested_dict(configuration, "thinking"))
    return TopLevelEffort(levels=features.reasoning_efforts or [])


def _to_profile(entry: ModelEntry | None) -> DeploymentProfile:
    """Resolve one catalog entry, or the absence of one, into a profile.

    Absent capabilities are resolved by what the wrong answer costs, which is
    why the gates don't point the same way: dropping `temperature` on a guess
    silently changes generation quality, so only an explicit `false` suppresses
    it, while every other gate stays shut until the catalog opens it, because
    emitting a field the adapter doesn't understand fails the request outright.
    """
    features = (entry.features if entry else None) or Features()
    limits = (entry.limits if entry else None) or Limits()
    defaults = (entry.defaults if entry else None) or {}

    # The spec's five-row token-budget table collapses to this: every row
    # names `limits.max_completion_tokens` when present and `defaults`
    # otherwise. The context window and input cap are not derived — clamping
    # the prompt would need a tokenizer this translator does not have.
    default_max_tokens = defaults.get("max_tokens")
    return DeploymentProfile(
        temperature_supported=features.temperature is not False,
        cache_supported=features.cache is True,
        max_completion_tokens_supported=features.max_completion_tokens_supported
        is True,
        max_output_tokens=limits.max_completion_tokens
        or (
            default_max_tokens if isinstance(default_max_tokens, int) else None
        ),
        reasoning=_reasoning_knob(defaults, features),
    )


UNRESOLVED_PROFILE = _to_profile(None)

_Catalog = dict[str, DeploymentProfile]

# One lock, not one per key: a burst of concurrent cold requests must produce a
# single upstream call. Serialising every credential behind it is what makes
# the bounded fetch timeout load-bearing.
_lock = asyncio.Lock()
_cache: OrderedDict[str, tuple[float, _Catalog]] = OrderedDict()


def clear_cache() -> None:
    _cache.clear()


def _cache_key(base_url: str, credential: str | None) -> str:
    # Core filters the listing by the caller's roles, so a catalog cached under
    # a narrow credential must never answer for a broader one — that would
    # silently disable reasoning routing and output clamping for every caller
    # sharing the entry. The credential is hashed, never stored.
    digest = hashlib.sha256((credential or "").encode()).hexdigest()
    return f"{base_url}\n{digest}"


def _read_cache(key: str, ttl: int) -> _Catalog | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    stored_at, catalog = entry
    if time.monotonic() - stored_at >= ttl:
        del _cache[key]
        return None
    _cache.move_to_end(key)
    return catalog


def _write_cache(key: str, catalog: _Catalog) -> None:
    _cache[key] = (time.monotonic(), catalog)
    _cache.move_to_end(key)
    while len(_cache) > get_model_catalog_size():
        _cache.popitem(last=False)


async def _fetch_catalog(
    base_url: str, credential: tuple[str, str] | None
) -> _Catalog | None:
    """`None` on any failure, so the caller can tell "no catalog" from "an empty
    one" and decline to cache it."""
    try:
        response = await get_http_client().get(
            f"{base_url}/openai/models",
            headers=dict([credential] if credential else []),
            timeout=get_model_catalog_timeout(),
        )
        response.raise_for_status()
        catalog = ModelCatalog.model_validate(response.json())
    except Exception:
        log.warning(
            "Could not resolve deployment capabilities from the DIAL Core "
            "model catalog; continuing without them",
            exc_info=True,
        )
        return None
    return {entry.id: _to_profile(entry) for entry in catalog.data}


async def get_deployment_profile(
    base_url: str, credential: tuple[str, str] | None, deployment: str
) -> DeploymentProfile:
    """The profile for `deployment`, as seen by the caller's own credential."""
    ttl: int = get_model_catalog_ttl()
    key: str = _cache_key(base_url, credential[1] if credential else None)

    async with _lock:
        catalog: _Catalog | None = _read_cache(key, ttl) if ttl > 0 else None
        if catalog is None:
            catalog = await _fetch_catalog(base_url, credential)
            # Failures are never cached: the next request retries.
            if catalog is not None and ttl > 0:
                _write_cache(key, catalog)

    return (catalog or {}).get(deployment, UNRESOLVED_PROFILE)
