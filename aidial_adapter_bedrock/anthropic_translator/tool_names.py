"""
Tool-name aliasing.

Claude Code names MCP tools `mcp__<server>__<tool>`, which routinely exceeds
the 64 characters the fronted model families accept. Several upstreams reject
*the whole request* over one non-conforming name rather than ignoring that
tool, so a single long MCP name can brick a conversation. Non-conforming names
therefore travel to Core under a deterministic alias and are restored on the
response path, so the client only ever sees the name it sent.
"""

import hashlib
import re

# The strictest pattern across the fronted families: 3-64 characters, starting
# with a letter or underscore, containing only letters, digits, `_` and `-`.
_CONFORMING = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{2,63}$")
_ALLOWED_CHAR = re.compile(r"[^A-Za-z0-9_-]")
_ALLOWED_LEAD = re.compile(r"^[A-Za-z_]")

_MAX_LENGTH = 64
_DIGEST_LENGTH = 8
_HEAD_LENGTH = _MAX_LENGTH - _DIGEST_LENGTH - 1


def _to_alias(name: str) -> str:
    digest: str = hashlib.sha256(name.encode()).hexdigest()[:_DIGEST_LENGTH]
    head: str = _ALLOWED_CHAR.sub("_", name)
    if not _ALLOWED_LEAD.match(head):
        head = f"t_{head}"
    return f"{head[:_HEAD_LENGTH]}_{digest}"


class ToolNameAliases:
    """Registry of the aliases one translation needed, and their originals.

    Scoped to a single request: the aliases a request registers are exactly the
    ones its own response can name, so nothing can be evicted while a call is
    in flight and no name leaks between callers.
    """

    def __init__(self) -> None:
        self._originals: dict[str, str] = {}

    def to_upstream(self, name: str) -> str:
        """Register and return the name to send to Core. Conforming names are
        never touched, and an empty one has nothing to alias."""
        if not name or _CONFORMING.match(name):
            return name
        alias: str = _to_alias(name)
        self._originals[alias] = name
        return alias

    def to_client(self, name: str) -> str:
        """Restore the name the client sent. An unregistered name is returned
        as-is: the upstream may name a tool this request never aliased."""
        return self._originals.get(name, name)
