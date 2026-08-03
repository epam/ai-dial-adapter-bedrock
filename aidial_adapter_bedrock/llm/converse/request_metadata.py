import json
import os
import re
from typing import Any

from aidial_client import UserInfo

from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

CONVERSE_API_REQUEST_METADATA_FIELDS = os.getenv(
    "CONVERSE_API_REQUEST_METADATA_FIELDS"
)

_MAX_ENTRIES = 16
_MAX_KEY_LEN = 256
_MAX_VALUE_LEN = 256
_ALLOWED = re.compile(r"[^a-zA-Z0-9\s:_@$#=/+,.\-]")


def is_enabled() -> bool:
    return bool(
        CONVERSE_API_REQUEST_METADATA_FIELDS
        and CONVERSE_API_REQUEST_METADATA_FIELDS.strip()
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
    data: dict[str, Any], config: str | None = None
) -> dict[str, str]:
    if not config:
        return {}

    result: dict[str, str] = {}
    for path in (p.strip() for p in config.split(",")):
        if not path:
            continue
        try:
            element = _get_element_at_path(data, path)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            log.warning(
                f"Skipping unresolved Converse requestMetadata path "
                f"{path!r}: {type(exc).__name__}: {exc}"
            )
            continue
        result[path] = json.dumps(element)
    return result


def _format_paths(paths: list[str]) -> str:
    return ", ".join(paths)


def _to_bedrock_metadata(flat: dict[str, str]) -> dict[str, str]:
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
                f"Converse requestMetadata entry cap reached; "
                f"omitted {len(omitted)} configured path(s): "
                f"{_format_paths(omitted)}"
            )
            break

        safe_key = _ALLOWED.sub("", key)[:_MAX_KEY_LEN]
        safe_value = _ALLOWED.sub("", value)[:_MAX_VALUE_LEN]

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
            f"Sanitized Converse requestMetadata key(s): "
            f"{_format_paths(changed_keys)}"
        )
    if changed_values:
        log.warning(
            f"Sanitized Converse requestMetadata value(s) for path(s): "
            f"{_format_paths(changed_values)}"
        )
    if empty_keys:
        log.warning(
            f"Dropped Converse requestMetadata path(s) with empty sanitized "
            f"key(s): {_format_paths(empty_keys)}"
        )
    if collisions:
        log.warning(
            f"Dropped Converse requestMetadata path(s) whose sanitized key "
            f"collides with an earlier entry: {_format_paths(collisions)}"
        )

    return safe


def from_user_info(user_info: UserInfo) -> dict[str, str]:
    data = user_info.model_dump(mode="json")
    return _to_bedrock_metadata(
        resolve_paths(data, CONVERSE_API_REQUEST_METADATA_FIELDS)
    )
