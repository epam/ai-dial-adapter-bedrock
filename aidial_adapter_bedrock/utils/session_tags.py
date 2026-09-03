import json
from typing import Any

from aidial_client import UserInfo

from aidial_adapter_bedrock.dial_api.client import create_dial_client
from aidial_adapter_bedrock.upstream_config import (
    AWSAssumeRoleCredentials,
    CloudUpstreamConfig,
    SessionTag,
    UpstreamConfig,
)
from aidial_adapter_bedrock.utils.env import get_env_bool, get_env_list
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

AWS_SESSION_TAGS_ENABLED = get_env_bool("AWS_SESSION_TAGS_ENABLED")

# The allow-list of DIAL UserInfo paths to pass as tags.
AWS_SESSION_TAGS_USER_INFO_FIELDS = get_env_list(
    "AWS_SESSION_TAGS_USER_INFO_FIELDS"
)

# Tag keys are namespaced by their source, so that the sources stay
# independent and their keys cannot collide.
_MODEL_ID_TAG_KEY = "Bedrock.modelId"
_USER_INFO_TAG_PREFIX = "UserInfo."

# AWS STS session tag constraints:
# https://docs.aws.amazon.com/IAM/latest/UserGuide/id_session-tags.html#id_session-tags_operations
_MAX_ENTRIES = 50
_MAX_KEY_LEN = 128
_MAX_VALUE_LEN = 256


def is_enabled(upstream_config: UpstreamConfig) -> bool:
    return (
        AWS_SESSION_TAGS_ENABLED
        and isinstance(upstream_config, CloudUpstreamConfig)
        and isinstance(upstream_config.credentials, AWSAssumeRoleCredentials)
    )


def _get_element_at_path(node: Any, path: str) -> Any:
    for segment in path.split("."):
        if isinstance(node, dict):
            node = node[segment]
        elif isinstance(node, list):
            node = node[int(segment)]
        else:
            raise TypeError(f"cannot index into {type(node).__name__}")
    return node


def resolve_paths(
    data: dict[str, Any], paths: list[str] | None = None
) -> dict[str, str]:
    if not paths:
        return {}

    result: dict[str, str] = {}
    for path in paths:
        if not path:
            continue
        try:
            element = _get_element_at_path(data, path)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            log.warning(
                f"Skipping unresolved AWS STS session tags path "
                f"{path!r}: {type(exc).__name__}: {exc}"
            )
            continue
        result[path] = (
            element if isinstance(element, str) else json.dumps(element)
        )
    return result


def _format_paths(paths: list[str]) -> str:
    return ", ".join(paths)


def _to_session_tags(flat: dict[str, str]) -> list[SessionTag]:
    safe: dict[str, str] = {}
    changed_keys: list[str] = []
    changed_values: list[str] = []
    empty_keys: list[str] = []
    collisions: list[str] = []

    items = list(flat.items())
    for index, (key, value) in enumerate(items):
        if len(safe) >= _MAX_ENTRIES:
            omitted = [path for path, _ in items[index:]]
            log.warning(
                f"AWS STS session tags entry cap reached; "
                f"omitted {len(omitted)} configured path(s): "
                f"{_format_paths(omitted)}"
            )
            break

        safe_key = key[:_MAX_KEY_LEN]
        safe_value = value[:_MAX_VALUE_LEN]

        if safe_key != key:
            changed_keys.append(key)
        if safe_value != value:
            changed_values.append(key)
        if not safe_key:
            empty_keys.append(key)
            continue
        if safe_key in safe:
            collisions.append(key)
            continue

        safe[safe_key] = safe_value

    if changed_keys:
        log.warning(
            f"Sanitized AWS STS session tags key(s): "
            f"{_format_paths(changed_keys)}"
        )
    if changed_values:
        log.warning(
            f"Sanitized AWS STS session tags value(s) for path(s): "
            f"{_format_paths(changed_values)}"
        )
    if empty_keys:
        log.warning(
            f"Dropped AWS STS session tags path(s) with empty sanitized "
            f"key(s): {_format_paths(empty_keys)}"
        )
    if collisions:
        log.warning(
            f"Dropped AWS STS session tags path(s) whose sanitized key "
            f"collides with an earlier entry: {_format_paths(collisions)}"
        )

    return [{"Key": key, "Value": value} for key, value in safe.items()]


def build_tags(
    model_id: str | None, user_info: UserInfo | None
) -> list[SessionTag]:
    # Tags extracted from other sources
    tags = {} if model_id is None else {_MODEL_ID_TAG_KEY: model_id}

    # Tags extracted from UserInfo
    if user_info is not None:
        data = user_info.model_dump(mode="json")
        resolved = resolve_paths(data, AWS_SESSION_TAGS_USER_INFO_FIELDS)
        tags.update(
            (f"{_USER_INFO_TAG_PREFIX}{path}", value)
            for path, value in resolved.items()
        )

    ret = _to_session_tags(tags)
    log.debug(f"Built AWS STS session tags: {ret}")
    return ret


async def _fetch_user_info(api_key: str | None) -> UserInfo | None:
    if not AWS_SESSION_TAGS_USER_INFO_FIELDS:
        return None

    if api_key is None:
        log.warning(
            "Skipping UserInfo AWS STS session tags; "
            "the request carries no DIAL API key"
        )
        return None

    dial_client = create_dial_client(api_key)
    if dial_client is None:
        log.warning(
            "Skipping UserInfo AWS STS session tags; "
            "DIAL_URL env variable is not set"
        )
        return None

    try:
        return await dial_client.user.info()
    except Exception as exc:
        log.warning(
            f"Skipping UserInfo AWS STS session tags; "
            f"failed to fetch DIAL user info: {type(exc).__name__}: {exc}"
        )
        return None


async def resolve_session_tags(
    api_key: str | None,
    upstream_config: UpstreamConfig,
    model_id: str | None,
) -> list[SessionTag] | None:
    if not is_enabled(upstream_config):
        return None

    return build_tags(model_id, await _fetch_user_info(api_key)) or None
