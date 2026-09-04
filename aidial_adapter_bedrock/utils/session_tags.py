import json
import unicodedata
from collections.abc import Container
from typing import Any, NamedTuple

from aidial_client import UserInfo

from aidial_adapter_bedrock.dial_api.client import create_dial_client
from aidial_adapter_bedrock.upstream_config import (
    AWSAssumeRoleCredentials,
    CloudUpstreamConfig,
    SessionTag,
    UpstreamConfig,
)
from aidial_adapter_bedrock.utils.env import get_env_list
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

# The tags to pass, each naming its source in the prefix, e.g.
# "UserInfo.roles.0,UserInfo.project,Bedrock.modelId".
# Setting it enables the feature; the entries double as the AWS tag keys, so
# the prefixes also keep the sources from colliding.
AWS_SESSION_TAGS = get_env_list("AWS_SESSION_TAGS")

# The tag sources, as named by the prefix of a configured tag.
_BEDROCK_SOURCE = "Bedrock"
_USER_INFO_SOURCE = "UserInfo"

# The only field the Bedrock source provides.
_MODEL_ID_FIELD = "modelId"

# AWS STS session tag constraints:
# https://docs.aws.amazon.com/IAM/latest/UserGuide/id_session-tags.html#id_session-tags_operations
_MAX_ENTRIES = 50
_MAX_KEY_LEN = 128
_MAX_VALUE_LEN = 256

# AWS also rejects a key or a value that doesn't match the character pattern
# [\p{L}\p{Z}\p{N}_.:/=+\-@], which the docs above don't mention. Note that
# `,` isn't allowed, so a JSON-serialized value never passes as-is.
# `re` doesn't support \p{...}, hence the Unicode categories:
# L = letters, Z = separators (spaces), N = numbers.
_ALLOWED_TAG_CATEGORIES = frozenset("LZN")
_ALLOWED_TAG_CHARS = frozenset("_.:/=+-@")
_TAG_CHAR_PLACEHOLDER = "_"


class _Tag(NamedTuple):
    key: str
    """The configured tag, passed to AWS as the tag key."""

    source: str
    field: str


def is_enabled(upstream_config: UpstreamConfig) -> bool:
    return (
        # A blank variable holds no tag, so it leaves the feature disabled.
        any(AWS_SESSION_TAGS or [])
        and isinstance(upstream_config, CloudUpstreamConfig)
        and isinstance(upstream_config.credentials, AWSAssumeRoleCredentials)
    )


def parse_tags(tags: list[str] | None) -> list[_Tag]:
    ret: list[_Tag] = []
    for tag in tags or []:
        source, _, field = tag.partition(".")

        if not field:
            if tag:
                log.warning(
                    f"Skipping AWS STS session tag {tag!r}: it names no "
                    f"source; expected {_BEDROCK_SOURCE}.<field> or "
                    f"{_USER_INFO_SOURCE}.<path>"
                )
            continue

        if source == _USER_INFO_SOURCE:
            ret.append(_Tag(tag, source, field))
        elif source == _BEDROCK_SOURCE:
            if field == _MODEL_ID_FIELD:
                ret.append(_Tag(tag, source, field))
            else:
                log.warning(
                    f"Skipping AWS STS session tag {tag!r}: the "
                    f"{_BEDROCK_SOURCE} source only provides "
                    f"{_MODEL_ID_FIELD!r}"
                )
        else:
            log.warning(
                f"Skipping AWS STS session tag {tag!r}: unknown source "
                f"{source!r}; expected {_BEDROCK_SOURCE} or "
                f"{_USER_INFO_SOURCE}"
            )

    return ret


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


def _sanitize_chars(value: str) -> str:
    """
    Replaces the characters AWS rejects, one for one so that the length of
    the value is preserved.
    """

    return "".join(
        char
        if char in _ALLOWED_TAG_CHARS
        or unicodedata.category(char)[0] in _ALLOWED_TAG_CATEGORIES
        else _TAG_CHAR_PLACEHOLDER
        for char in value
    )


def _dedupe_key(key: str, taken: Container[str]) -> str:
    """
    Postfixes a key that sanitization or truncation made collide with an
    earlier one, keeping it within the length limit.
    """

    for index in range(1, _MAX_ENTRIES + 1):
        postfix = f"_{index}"
        candidate = f"{key[: _MAX_KEY_LEN - len(postfix)]}{postfix}"
        if candidate not in taken:
            return candidate

    return key


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

        safe_key = _sanitize_chars(key)[:_MAX_KEY_LEN]
        safe_value = _sanitize_chars(value)[:_MAX_VALUE_LEN]

        if safe_key != key:
            changed_keys.append(key)
        if safe_value != value:
            changed_values.append(key)
        if not safe_key:
            empty_keys.append(key)
            continue
        if safe_key in safe:
            collisions.append(key)
            safe_key = _dedupe_key(safe_key, safe)

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
            f"Postfixed AWS STS session tags path(s) whose sanitized key "
            f"collides with an earlier entry: {_format_paths(collisions)}"
        )

    return [{"Key": key, "Value": value} for key, value in safe.items()]


def build_tags(
    model_id: str | None, user_info: UserInfo | None
) -> list[SessionTag]:
    parsed = parse_tags(AWS_SESSION_TAGS)

    user_info_resolved_paths = (
        resolve_paths(
            user_info.model_dump(mode="json"),
            [tag.field for tag in parsed if tag.source == _USER_INFO_SOURCE],
        )
        if user_info is not None
        else {}
    )

    tags: dict[str, str] = {}
    for tag in parsed:
        if tag.source == _BEDROCK_SOURCE:
            if model_id is not None:
                tags[tag.key] = model_id
        elif tag.field in user_info_resolved_paths:
            tags[tag.key] = user_info_resolved_paths[tag.field]

    ret = _to_session_tags(tags)
    log.debug(f"Built AWS STS session tags: {ret}")
    return ret


def _wants_user_info() -> bool:
    return any(
        tag.startswith(f"{_USER_INFO_SOURCE}.")
        for tag in AWS_SESSION_TAGS or []
    )


async def _fetch_user_info(api_key: str | None) -> UserInfo | None:
    if not _wants_user_info():
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
