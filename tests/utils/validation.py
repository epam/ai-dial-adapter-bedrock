from collections.abc import Sequence
from enum import Enum
from typing import TypeVar

_T = TypeVar("_T", bound=Enum)


def check_enum_completeness(seq: Sequence[_T]) -> None:
    if not seq:
        raise ValueError("List cannot be empty.")

    enum_cls = type(seq[0])

    if untested := (set(enum_cls) - set(seq)):
        raise ValueError(
            f"Missing enum cases: {', '.join([m.name for m in untested])}."
        )
