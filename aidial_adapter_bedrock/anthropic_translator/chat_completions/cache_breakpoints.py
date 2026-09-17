import re
from datetime import UTC, datetime, timedelta

from aidial_sdk.chat_completion.request import CacheBreakpoint
from pydantic import JsonValue

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    CacheControl,
)
from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)

_TTL: re.Pattern[str] = re.compile(r"^([1-9]\d*)([mh])$")

_UNITS: dict[str, timedelta] = {
    "m": timedelta(minutes=1),
    "h": timedelta(hours=1),
}


def cache_breakpoint(
    controls: list[CacheControl], tlog: TranslationLog
) -> CacheBreakpoint | None:
    if not controls:
        return None
    expires: str | None = None
    for control in controls:
        try:
            if (ttl := _ttl(control, tlog)) is not None:
                candidate: str = _expire_at(ttl)
                expires = max(expires, candidate) if expires else candidate
        except (OverflowError, ValueError):
            tlog.warning(
                "Ignoring unreadable cache_control.ttl: %s", control.ttl
            )
    return CacheBreakpoint(expire_at=expires)


def _ttl(control: CacheControl, tlog: TranslationLog) -> timedelta | None:
    raw: JsonValue = control.ttl
    if raw is None:
        return None
    if isinstance(raw, str) and (match := _TTL.match(raw)):
        return int(match.group(1)) * _UNITS[match.group(2)]

    tlog.warning("Ignoring unreadable cache_control.ttl: %s", raw)
    return None


def _expire_at(ttl: timedelta) -> str:
    expires: datetime = datetime.now(UTC) + ttl
    return expires.replace(microsecond=0).isoformat().replace("+00:00", "Z")
