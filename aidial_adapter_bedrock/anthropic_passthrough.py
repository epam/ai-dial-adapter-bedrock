from typing import assert_never

from aidial_adapter_anthropic.passthrough import mount_anthropic_api
from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
)
from fastapi import FastAPI, Request

from aidial_adapter_bedrock.bedrock import (
    AnthropicClient,
    create_anthropic_client,
)
from aidial_adapter_bedrock.upstream_config import (
    UpstreamConfig,
    parse_upstream_config,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log
from aidial_adapter_bedrock.utils.session_tags import (
    is_enabled as session_tags_enabled,
)
from aidial_adapter_bedrock.utils.session_tags import resolve_session_tags

_API_KEY_HEADER_NAME = "Api-Key"


async def _get_model(request: Request) -> str | None:
    """
    The model of an Anthropic Messages API request, which carries it in the
    body rather than in the path.
    """

    try:
        body = await request.json()
        model = body["model"]
    except Exception as exc:
        log.warning(
            f"Skipping the model AWS STS session tag; failed to read the model "
            f"of the request: {type(exc).__name__}: {exc}"
        )
        return None

    return model if isinstance(model, str) else None


async def _resolve_session_tags(
    request: Request, upstream_config: UpstreamConfig
):
    if not session_tags_enabled(upstream_config):
        return None

    return await resolve_session_tags(
        request.headers.get(_API_KEY_HEADER_NAME),
        upstream_config,
        await _get_model(request),
    )


async def _get_anthropic_client(request: Request) -> AnthropicClient:
    upstream_config = await parse_upstream_config(request)
    session_tags = await _resolve_session_tags(request, upstream_config)
    return await create_anthropic_client(upstream_config, session_tags)


def _strip_unsupported_features(
    client: AnthropicClient, features: list[str]
) -> list[str]:
    _unsupported_flags_by_bedrock = {
        "oauth-2025-04-20",
        "redact-thinking-2026-02-12",
        "thinking-token-count-2026-05-13",
        "prompt-caching-scope-2026-01-05",
        "claude-code-20250219",
        "advisor-tool-2026-03-01",
    }
    match client:
        case AsyncAnthropicBedrock() | AsyncAnthropicBedrockMantle():
            return [
                f for f in features if f not in _unsupported_flags_by_bedrock
            ]
        case AsyncAnthropic():
            return features
        case _:
            assert_never(client)


def mount_anthropic_passthrough(app: FastAPI, path: str):
    mount_anthropic_api(
        app,
        _get_anthropic_client,
        path=path,
        on_anthropic_beta_header=_strip_unsupported_features,
    )
