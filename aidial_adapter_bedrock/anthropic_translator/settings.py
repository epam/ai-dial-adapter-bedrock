"""
Environment-backed settings for the Anthropic Messages translators.

Every value is read at call time rather than at import, so tests and runtime
reconfiguration see the current environment.
"""

import os

from aidial_adapter_bedrock.utils.env import get_env_int


def get_api_version() -> str:
    """The `api-version` query parameter some Chat Completions deployments
    (Azure-backed ones in particular) require."""
    return os.getenv("TRANSLATOR_API_VERSION") or "2025-01-01-preview"


def get_model_catalog_ttl() -> int:
    """Seconds a fetched model catalog stays usable. `0` disables caching."""
    return get_env_int("TRANSLATOR_MODEL_CATALOG_TTL", 600)


def get_model_catalog_size() -> int:
    """How many per-credential catalogs to keep, so a service fronting many
    credentials can't grow without limit."""
    return get_env_int("TRANSLATOR_MODEL_CATALOG_SIZE", 64)


def get_model_catalog_timeout() -> int:
    """Bounds every phase of the catalog fetch. The fetch is serialised, so an
    unbounded one lets a single unresponsive Core stall every in-flight
    request — worse than the failure the unresolved profile exists to survive.
    """
    return get_env_int("TRANSLATOR_MODEL_CATALOG_TIMEOUT", 5)


def get_stop_unsupported_deployments() -> tuple[str, ...]:
    """Deployment-id prefixes whose upstream rejects the `stop` parameter
    outright, so the translator must strip it and reproduce stop-sequence
    semantics itself. Empty disables the stripping."""
    raw: str = os.getenv("TRANSLATOR_STOP_UNSUPPORTED_DEPLOYMENTS", "gpt-5.")
    return tuple(
        prefix.strip().lower() for prefix in raw.split(",") if prefix.strip()
    )
