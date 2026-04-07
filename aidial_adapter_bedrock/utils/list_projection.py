import builtins
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Generic, Self, TypeVar

_T = TypeVar("_T")


@dataclass
class ListProjection(Generic[_T]):
    """
    The class represents a transformation of the original list which may
    include merge, removal and addition of the original list elements.

    Each derivative element is mapped onto a subset of original elements.
    The subsets must be disjoint.
    """

    lst: list[tuple[_T, set[int]]] = field(default_factory=list)

    @property
    def raw_list(self) -> builtins.list[_T]:
        return [msg for msg, _ in self.lst]

    def to_original_indices(self, indices: Iterable[int]) -> set[int]:
        return {
            orig_index for index in indices for orig_index in self.lst[index][1]
        }

    def append(self, elem: _T, idx: int) -> Self:
        self.lst.append((elem, {idx}))
        return self
