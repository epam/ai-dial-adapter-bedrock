from fastapi import Request

from aidial_adapter_bedrock.bedrock import (
    AnthropicClient,
    create_anthropic_client,
)
from aidial_adapter_bedrock.upstream_config import (
    SessionTag,
    UpstreamConfig,
    parse_upstream_config,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log
from aidial_adapter_bedrock.utils.session_tags import (
    is_enabled as session_tags_enabled,
)
from aidial_adapter_bedrock.utils.session_tags import (
    resolve_session_tags,
)

_API_KEY_HEADER_NAME = "Api-Key"


async def _get_model(request: Request) -> str | None:
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
) -> list[SessionTag] | None:
    if not session_tags_enabled(upstream_config):
        return None

    return await resolve_session_tags(
        request.headers.get(_API_KEY_HEADER_NAME),
        upstream_config,
        await _get_model(request),
    )


async def get_anthropic_client(request: Request) -> AnthropicClient:
    upstream_config = await parse_upstream_config(request)
    session_tags = await _resolve_session_tags(request, upstream_config)
    return await create_anthropic_client(upstream_config, session_tags)
