from datetime import datetime, timezone

_default_tz = timezone.utc


def now_utc() -> datetime:
    return datetime.now(_default_tz)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_default_tz)
    else:
        return dt.astimezone(_default_tz)
