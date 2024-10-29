from dataclasses import dataclass, field
from typing import Generic, Iterable, List, Self, Set, Tuple, TypeVar

_T = TypeVar("_T")


@dataclass
class ListProjection(Generic[_T]):
    """
    The class presents a transformation of the original list which may
    include merge, removal and addition of the original list elements.

    Each derivative element is mapped onto a subset of original elements.
    The subsets must be disjoint.
    """

    list: List[Tuple[_T, Set[int]]] = field(default_factory=list)

    @property
    def raw_list(self) -> List[_T]:
        return [msg for msg, _ in self.list]

    def to_original_indices(self, idx: int | Iterable[int]) -> Set[int]:
        return {
            orig_index
            for index in _to_set(idx)
            for orig_index in self.list[index][1]
        }

    def append(self, elem: _T, idx: int | Iterable[int]) -> Self:
        self.list.append((elem, _to_set(idx)))
        return self


def _to_set(idx: int | Iterable[int]) -> Set[int]:
    return {idx} if isinstance(idx, int) else set(idx)
