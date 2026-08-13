"""
Environment-backed settings for the Anthropic Messages translators.

Every value is read at call time rather than at import, so tests and runtime
reconfiguration see the current environment.
"""

import os


def get_api_version() -> str:
    """The `api-version` query parameter some Chat Completions deployments
    (Azure-backed ones in particular) require."""
    return os.getenv("TRANSLATOR_API_VERSION") or "2025-01-01-preview"


def get_stop_unsupported_deployments() -> tuple[str, ...]:
    """Deployment-id prefixes whose upstream rejects the `stop` parameter
    outright, so the translator must strip it and reproduce stop-sequence
    semantics itself. Empty disables the stripping."""
    raw: str = os.getenv("TRANSLATOR_STOP_UNSUPPORTED_DEPLOYMENTS", "gpt-5.")
    return tuple(
        prefix.strip().lower() for prefix in raw.split(",") if prefix.strip()
    )
