import time

from aidial_sdk.chat_completion import CacheBreakpoint
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Tool as DialTool

_DIAL_CACHE_BREAKPOINT_PATH = "X-DIAL-CACHE-BREAKPOINT-PATH"
_DIAL_CACHE_EXPIRE_AT = "X-DIAL-CACHE-EXPIRE-AT"

# 5min is a default TTL for Converse API cache breakpoints
# https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
_DEFAULT_TTL_SEC = 5 * 60


def _parse_ttl(ttl: str) -> int | None:
    try:
        for unit, secs in {"h": 3600, "m": 60, "s": 1}.items():
            if ttl[-1] == unit:
                return secs * int(ttl[:-1])
    except Exception:
        return None


def _ttl_from_breakpoint(breakpoint: CacheBreakpoint) -> int:
    s = (breakpoint.model_extra or {}).get("ttl")
    if s and isinstance(s, str) and (ttl := _parse_ttl(s)):
        return ttl
    return _DEFAULT_TTL_SEC


def get_response_headers_for_caching(
    messages: list[DialMessage], tools: list[DialTool]
) -> dict | None:
    ttl = 0
    message_path = None
    tool_path = None

    for i, message in enumerate(messages):
        if (
            (cf := message.custom_fields)
            and (breakpoint := cf.cache_breakpoint)
            and (msg_ttl := _ttl_from_breakpoint(breakpoint))
        ):
            ttl = max(ttl, msg_ttl)
            message_path = f"prefix.body.messages[{i}]"

    for i, tool in enumerate(tools):
        if (
            (cf := tool.custom_fields)
            and (breakpoint := cf.cache_breakpoint)
            and (tool_ttl := _ttl_from_breakpoint(breakpoint))
        ):
            ttl = max(ttl, tool_ttl)
            tool_path = f"prefix.body.tools[{i}]"

    path = message_path or tool_path

    if path is None:
        return None

    return {
        _DIAL_CACHE_BREAKPOINT_PATH: path,
        _DIAL_CACHE_EXPIRE_AT: str(int(time.time()) + ttl),
    }
