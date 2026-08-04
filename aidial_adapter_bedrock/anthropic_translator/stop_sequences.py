"""
Stop-sequence emulation.

Some deployments reject the Chat Completions `stop` parameter outright and fail
the whole request, so for those the translator omits it and reproduces
Anthropic's native semantics itself: generation halts at the first *completed*
occurrence of any sequence, the returned text excludes the sequence, and
nothing after it exists.

Both the batch and the incremental matcher rank candidates by earliest
completion. That ranking is what keeps the two in step: an incremental matcher
can only observe a sequence once its final character arrives, so with
`["abXY", "b"]` against `"abXY"` it necessarily stops at the `b`. Ranking by
start offset instead would make the batch path pick `abXY`, truncating the same
completion differently in the two modes.
"""

from aidial_adapter_bedrock.anthropic_translator.settings import (
    get_stop_unsupported_deployments,
)


def strips_stop_parameter(deployment: str) -> bool:
    return deployment.lower().startswith(get_stop_unsupported_deployments())


def apply_stop_sequences(
    text: str, sequences: list[str]
) -> tuple[str, str | None]:
    """Truncate `text` before the sequence that completes earliest, returning
    it with the sequence that matched (`None` when none did)."""
    best: tuple[int, int, str] | None = None
    for sequence in sequences:
        if not sequence or (start := text.find(sequence)) < 0:
            continue
        # Ties on the end offset go to the earlier start; both are compared at
        # once because the tuple orders by end first.
        candidate = (start + len(sequence), start, sequence)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        return text, None
    _, start, sequence = best
    return text[:start], sequence


class StopSequenceMatcher:
    """Incremental counterpart of `apply_stop_sequences`.

    Text already sent to the client cannot be recalled, so the matcher withholds
    the shortest suffix that could still grow into a sequence and releases it
    once it turns out to be ordinary text. A matcher built with no sequences
    withholds nothing and passes every fragment straight through.
    """

    def __init__(self, sequences: list[str]) -> None:
        self._sequences = [sequence for sequence in sequences if sequence]
        self._withhold = max((len(s) for s in self._sequences), default=1) - 1
        self._pending = ""
        self.matched: str | None = None

    def push(self, text: str) -> str:
        """Consume a fragment and return the part that is safe to emit."""
        if self.matched is not None:
            return ""

        self._pending += text
        emitted, self.matched = apply_stop_sequences(
            self._pending, self._sequences
        )
        if self.matched is not None:
            self._pending = ""
            return emitted

        # A single-character sequence withholds nothing, so this must emit
        # eagerly rather than buffer forever.
        safe: int = len(self._pending) - self._withhold
        if safe <= 0:
            return ""
        emitted, self._pending = self._pending[:safe], self._pending[safe:]
        return emitted

    def flush(self) -> str:
        """Release the withheld tail — it turned out to be ordinary text."""
        pending, self._pending = self._pending, ""
        return "" if self.matched is not None else pending
