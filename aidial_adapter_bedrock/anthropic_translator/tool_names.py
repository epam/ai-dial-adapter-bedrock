import hashlib
import re
from collections import OrderedDict

_CONFORMING: re.Pattern[str] = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{2,63}$")
_ALLOWED_CHAR: re.Pattern[str] = re.compile(r"[^A-Za-z0-9_-]")
_ALLOWED_LEAD: re.Pattern[str] = re.compile(r"^[A-Za-z_]")

_MAX_LENGTH: int = 64
_DIGEST_LENGTH: int = 8
_HEAD_LENGTH: int = _MAX_LENGTH - _DIGEST_LENGTH - 1


def _to_alias(name: str) -> str:
    digest: str = hashlib.sha256(name.encode()).hexdigest()[:_DIGEST_LENGTH]
    head: str = _ALLOWED_CHAR.sub("_", name)
    if not _ALLOWED_LEAD.match(head):
        head = f"t_{head}"
    return f"{head[:_HEAD_LENGTH]}_{digest}"


_ORIGINALS: OrderedDict[str, str] = OrderedDict()
_MAX_ALIASES: int = 4096


class ToolNameAliases:
    def __init__(self) -> None:
        self._originals: OrderedDict[str, str] = _ORIGINALS

    def to_upstream(self, name: str) -> str:
        if not name or _CONFORMING.match(name):
            return name
        alias: str = _to_alias(name)
        self._originals[alias] = name
        self._originals.move_to_end(alias)
        if len(self._originals) > _MAX_ALIASES:
            self._originals.popitem(last=False)
        return alias

    def to_client(self, name: str) -> str:
        if name in self._originals:
            self._originals.move_to_end(name)
        return self._originals.get(name, name)
