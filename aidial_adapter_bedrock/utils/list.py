from typing import Container, List, TypeVar

T = TypeVar("T")


def select_by_indices(lst: List[T], indices: Container[int]) -> List[T]:
    return [elem for idx, elem in enumerate(lst) if idx in indices]


def omit_by_indices(lst: List[T], indices: Container[int]) -> List[T]:
    return [elem for idx, elem in enumerate(lst) if idx not in indices]
