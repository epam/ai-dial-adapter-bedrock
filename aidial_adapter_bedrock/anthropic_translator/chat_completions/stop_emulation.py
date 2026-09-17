from pydantic import BaseModel

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    MessagesRequest,
)


class StopMatch(BaseModel):
    text: str
    sequence: str | None = None


def strips_stop_parameter(deployment: str) -> bool:
    return deployment.lower().startswith("gpt-5.")


def emulated_stop_sequences(req: MessagesRequest, deployment: str) -> list[str]:
    if not strips_stop_parameter(deployment):
        return []
    return req.stop_sequences or []


def apply_stop_sequences(text: str, sequences: list[str]) -> StopMatch:
    best: tuple[int, int, str] | None = None
    for sequence in sequences:
        if not sequence or (start := text.find(sequence)) < 0:
            continue

        candidate: tuple[int, int, str] = (
            start + len(sequence),
            start,
            sequence,
        )
        if best is None or candidate < best:
            best = candidate

    if best is None:
        return StopMatch(text=text)
    _, start, sequence = best
    return StopMatch(text=text[:start], sequence=sequence)


class StopSequenceMatcher:
    def __init__(self, sequences: list[str]) -> None:
        self._sequences: list[str] = [
            sequence for sequence in sequences if sequence
        ]
        self._withhold: int = (
            max((len(s) for s in self._sequences), default=1) - 1
        )
        self._pending: str = ""
        self.matched: str | None = None

    def push(self, text: str) -> str:
        if self.matched is not None:
            return ""

        self._pending += text
        match: StopMatch = apply_stop_sequences(self._pending, self._sequences)
        if match.sequence is not None:
            self.matched = match.sequence
            self._pending = ""
            return match.text

        safe: int = len(self._pending) - self._withhold
        if safe <= 0:
            return ""
        emitted: str
        emitted, self._pending = self._pending[:safe], self._pending[safe:]
        return emitted

    def flush(self) -> str:
        pending: str
        pending, self._pending = self._pending, ""
        return "" if self.matched is not None else pending
