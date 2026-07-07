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
from aidial_adapter_bedrock.upstream_config import parse_upstream_config


async def _get_anthropic_client(request: Request) -> AnthropicClient:
    upstream_config = await parse_upstream_config(request)
    return await create_anthropic_client(upstream_config)


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
        path="/anthropic",
        on_anthropic_beta_header=_strip_unsupported_features,
    )
