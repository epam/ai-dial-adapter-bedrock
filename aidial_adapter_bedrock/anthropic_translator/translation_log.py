import logging

from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class TranslationLog:
    def __init__(self, operation: str) -> None:
        self._operation: str = operation
        self._entries: list[tuple[int, str]] = []

    def debug(self, message: str, *args: object) -> None:
        self._record(logging.DEBUG, message, args)

    def info(self, message: str, *args: object) -> None:
        self._record(logging.INFO, message, args)

    def warning(self, message: str, *args: object) -> None:
        self._record(logging.WARNING, message, args)

    def _record(
        self, level: int, message: str, args: tuple[object, ...]
    ) -> None:
        self._entries.append((level, message % args if args else message))

    def flush(self) -> None:
        for level in (logging.WARNING, logging.INFO, logging.DEBUG):
            messages: list[str] = [
                msg for lvl, msg in self._entries if lvl == level
            ]
            if messages and log.isEnabledFor(level):
                log.log(
                    level,
                    "%s (%d): %s",
                    self._operation,
                    len(messages),
                    "; ".join(messages),
                )
        self._entries.clear()
