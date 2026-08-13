"""
Anthropic `cache_control` → DIAL `custom_fields.cache_breakpoint`.

Anthropic marks cache breakpoints **per content block**; DIAL's Chat
Completions dialect marks them **per message or tool object**, and computes
prefix hashes only for prefixes that end at a breakpoint. Getting this wrong is
a cost bug rather than a correctness one, and therefore easy to miss: without
markers the upstream re-reads the whole prompt every turn.

The two dialects also state a lifetime differently. Anthropic gives a duration
on the block (`{"type": "ephemeral", "ttl": "1h"}`); DIAL takes the instant it
lands on (`{"expire_at": "2026-08-11T16:05:00Z"}`).
"""

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from aidial_sdk.chat_completion.request import CacheBreakpoint

from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)

CacheControl = dict[str, Any]

_TTL = re.compile(r"^(\d+)([smhd])$")

_UNITS = {
    "s": timedelta(seconds=1),
    "m": timedelta(minutes=1),
    "h": timedelta(hours=1),
    "d": timedelta(days=1),
}


def cache_breakpoint(
    controls: list[CacheControl], tlog: TranslationLog
) -> CacheBreakpoint | None:
    """The marker for the `cache_control` blocks that collapsed onto one
    outbound message or tool, or `None` when none of them asked to cache.

    When several blocks collapse onto one object the longest ttl wins: taking
    the maximum can only keep content cached longer than one source asked for,
    while taking the minimum would silently shorten another's. `cache_control.
    type` means nothing to DIAL and is never forwarded, and a breakpoint with
    no `expire_at` leaves Core and the provider on their own default.
    """
    if not controls:
        return None
    ttls = [
        ttl for control in controls if (ttl := _ttl(control, tlog)) is not None
    ]
    return CacheBreakpoint(expire_at=_expire_at(max(ttls)) if ttls else None)


def _ttl(control: CacheControl, tlog: TranslationLog) -> timedelta | None:
    raw: Any = control.get("ttl")
    if raw is None:
        return None
    if isinstance(raw, str) and (match := _TTL.match(raw)):
        return int(match.group(1)) * _UNITS[match.group(2)]
    # Never at the cost of the marker itself: an unreadable lifetime falls back
    # to the upstream's default, while dropping the marker would stop caching.
    tlog.warning("Ignoring unreadable cache_control.ttl: %s", raw)
    return None


def _expire_at(ttl: timedelta) -> str:
    expires: datetime = datetime.now(UTC) + ttl
    return expires.replace(microsecond=0).isoformat().replace("+00:00", "Z")
