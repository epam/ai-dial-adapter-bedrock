from __future__ import annotations

from typing import Callable

from aidial_sdk.chat_completion import Stage

_StageFactory = Callable[[str], Stage]


class LazyStage:
    title: str
    stage_factory: _StageFactory

    _stage: Stage | None = None

    def __init__(self, stage_factory: _StageFactory, title: str):
        self.stage_factory = stage_factory
        self.title = title

    def __enter__(self) -> LazyStage:
        return self

    async def __aenter__(self) -> LazyStage:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def append_content(self, text: str) -> None:
        if self._stage is None:
            self._stage = self.stage_factory(self.title)
            self._stage.open()
        self._stage.append_content(text)

    def close(self) -> None:
        if self._stage is not None:
            self._stage.close()
