import re
from datetime import UTC, datetime, timedelta

from aidial_sdk.chat_completion.request import CacheBreakpoint
from anthropic.types.beta import BetaCacheControlEphemeralParam as CacheControl
from pydantic import JsonValue

from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

_TTL: re.Pattern[str] = re.compile(r"^([1-9]\d*)([mh])$")

_UNITS: dict[str, timedelta] = {
    "m": timedelta(minutes=1),
    "h": timedelta(hours=1),
}


def cache_breakpoint(controls: list[CacheControl]) -> CacheBreakpoint | None:
    if not controls:
        return None
    expires: str | None = None
    for control in controls:
        try:
            if (ttl := _ttl(control)) is not None:
                candidate: str = _expire_at(ttl)
                expires = max(expires, candidate) if expires else candidate
        except (OverflowError, ValueError):
            log.warning(
                "Ignoring unreadable cache_control.ttl: %s", control.get("ttl")
            )
    return CacheBreakpoint(expire_at=expires)


def _ttl(control: CacheControl) -> timedelta | None:
    raw: JsonValue = control.get("ttl")
    if raw is None:
        return None
    if isinstance(raw, str) and (match := _TTL.match(raw)):
        return int(match.group(1)) * _UNITS[match.group(2)]

    log.warning("Ignoring unreadable cache_control.ttl: %s", raw)
    return None


def _expire_at(ttl: timedelta) -> str:
    expires: datetime = datetime.now(UTC) + ttl
    return expires.replace(microsecond=0).isoformat().replace("+00:00", "Z")
