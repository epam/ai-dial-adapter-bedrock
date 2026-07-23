"""
Buffered logging for a single translation pass.

Translating one Anthropic request touches many helpers, each of which may drop
an unsupported field or block. Logged individually, one request emits a burst
of lines that reads like a stream of unrelated events. Instead, the public
entry point creates one `TranslationLog`, passes it down to every helper,
which record notes on it, and `flush()`es it once when the pass finishes — one
atomic translation → one log line per severity.

Flush in a `finally` so a raised error still emits whatever was collected:

    tlog = TranslationLog("Anthropic→Chat Completions request")
    try:
        ...  # thread `tlog` through the helpers
        return result
    finally:
        tlog.flush()
"""

import logging

from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class TranslationLog:
    """Collects `%`-style log notes for one translation and flushes them as at
    most one line per level, tagged with the operation name."""

    def __init__(self, operation: str) -> None:
        self._operation = operation
        self._entries: list[tuple[int, str]] = []

    def debug(self, message: str, *args: object) -> None:
        self._record(logging.DEBUG, message, args)

    def info(self, message: str, *args: object) -> None:
        self._record(logging.INFO, message, args)

    def warning(self, message: str, *args: object) -> None:
        self._record(logging.WARNING, message, args)

    def _record(self, level: int, message: str, args: tuple) -> None:
        self._entries.append((level, message % args if args else message))

    def flush(self) -> None:
        """Emit one aggregated line per level (highest first) for any level the
        logger is enabled for, then reset so a second call is a no-op."""
        for level in (logging.WARNING, logging.INFO, logging.DEBUG):
            messages = [msg for lvl, msg in self._entries if lvl == level]
            if messages and log.isEnabledFor(level):
                log.log(
                    level,
                    "%s (%d): %s",
                    self._operation,
                    len(messages),
                    "; ".join(messages),
                )
        self._entries.clear()
