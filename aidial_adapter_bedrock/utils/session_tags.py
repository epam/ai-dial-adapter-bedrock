import json
import unicodedata
from collections.abc import Container
from dataclasses import dataclass
from typing import Any, ClassVar, Self

from aidial_client import UserInfo

from aidial_adapter_bedrock.dial_api.client import create_dial_client
from aidial_adapter_bedrock.upstream_config import (
    AWSAssumeRoleCredentials,
    CloudUpstreamConfig,
    SessionTag,
    UpstreamConfig,
)
from aidial_adapter_bedrock.utils.env import get_str_dict
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

# The tags to pass, as a JSON object mapping the AWS tag key to the field to
# take it from, e.g.
# {"application": "Bedrock.modelId", "project": "UserInfo.project"}.
# Setting it enables the feature.
AWS_SESSION_TAGS = get_str_dict("AWS_SESSION_TAGS")

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


def is_enabled(upstream_config: UpstreamConfig) -> bool:
    return (
        bool(AWS_SESSION_TAGS)
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


def _format_tags(tags: list[str]) -> str:
    return ", ".join(tags)


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


"""A tag to pass: the field it comes from, the alias AWS keys it by, and the
resolved value."""
_Entry = tuple[str, str, str]


def _to_session_tags(entries: list[_Entry]) -> list[SessionTag]:
    # The alias is what AWS receives as the tag key, so the AWS constraints
    # are fitted to the alias rather than to the field it's configured for.
    ret: list[SessionTag] = []
    taken: set[str] = set()
    changed_keys: list[str] = []
    changed_values: list[str] = []
    empty_keys: list[str] = []
    collisions: list[str] = []

    for index, (path, alias, value) in enumerate(entries):
        if len(ret) >= _MAX_ENTRIES:
            omitted = [alias for _, alias, _ in entries[index:]]
            log.warning(
                f"AWS STS session tags entry cap reached; "
                f"omitted {len(omitted)} configured tag(s): "
                f"{_format_tags(omitted)}"
            )
            break

        safe_alias = _sanitize_chars(alias)[:_MAX_KEY_LEN]
        safe_value = _sanitize_chars(value)[:_MAX_VALUE_LEN]

        if safe_alias != alias:
            changed_keys.append(alias)
        if safe_value != value:
            changed_values.append(alias)
        if not safe_alias:
            empty_keys.append(path)
            continue
        if safe_alias in taken:
            collisions.append(alias)
            safe_alias = _dedupe_key(safe_alias, taken)

        taken.add(safe_alias)
        ret.append({"Key": path, "KeyAlias": safe_alias, "Value": safe_value})

    if changed_keys:
        log.warning(
            f"Sanitized AWS STS session tags key(s): "
            f"{_format_tags(changed_keys)}"
        )
    if changed_values:
        log.warning(
            f"Sanitized AWS STS session tags value(s): "
            f"{_format_tags(changed_values)}"
        )
    if empty_keys:
        log.warning(
            f"Dropped AWS STS session tags with an empty key, configured for "
            f"field(s): {_format_tags(empty_keys)}"
        )
    if collisions:
        log.warning(
            f"Postfixed AWS STS session tags whose sanitized key "
            f"collides with an earlier entry: {_format_tags(collisions)}"
        )

    return ret


@dataclass
class Tags:
    bedrock_model_id: list[str]
    user_info_paths: list[tuple[str, str]]

    _BEDROCK_MODEL_ID: ClassVar[str] = "Bedrock.modelId"
    _USER_INFO_PREFIX: ClassVar[str] = "UserInfo."

    @classmethod
    def parse(cls, tags: dict[str, str]) -> Self:
        bedrock_model_id: list[str] = []
        user_info_paths: list[tuple[str, str]] = []

        for tag_alias, tag_path in tags.items():
            if tag_path == cls._BEDROCK_MODEL_ID:
                bedrock_model_id.append(tag_alias)
            elif tag_path.startswith(cls._USER_INFO_PREFIX):
                user_info_paths.append(
                    (tag_alias, tag_path.removeprefix(cls._USER_INFO_PREFIX))
                )
            else:
                log.warning(
                    f"Skipping AWS STS session tag {tag_alias!r}: unknown "
                    f"field {tag_path!r}; expected {cls._BEDROCK_MODEL_ID} "
                    f"or {cls._USER_INFO_PREFIX}<path>"
                )

        return cls(
            bedrock_model_id=bedrock_model_id,
            user_info_paths=user_info_paths,
        )

    @property
    def wants_user_info(self) -> bool:
        return bool(self.user_info_paths)

    def to_session_tags(
        self, bedrock_model_id: str | None, user_info: UserInfo | None
    ) -> list[SessionTag]:
        entries: list[_Entry] = []

        if bedrock_model_id is not None:
            entries.extend(
                (self._BEDROCK_MODEL_ID, alias, bedrock_model_id)
                for alias in self.bedrock_model_id
            )

        if user_info is not None:
            resolved = resolve_paths(
                user_info.model_dump(mode="json"),
                [path for _, path in self.user_info_paths],
            )
            entries.extend(
                (f"{self._USER_INFO_PREFIX}{path}", alias, resolved[path])
                for alias, path in self.user_info_paths
                if path in resolved
            )

        ret = _to_session_tags(entries)
        log.debug(f"Built AWS STS session tags: {ret}")
        return ret


async def _fetch_user_info(api_key: str | None) -> UserInfo | None:
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

    tags = Tags.parse(AWS_SESSION_TAGS)
    user_info = (
        await _fetch_user_info(api_key) if tags.wants_user_info else None
    )

    return tags.to_session_tags(model_id, user_info) or None
