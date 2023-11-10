from typing import List, Set, TypeVar

T = TypeVar("T")


def exclude_indices(input_list: List[T], indices: Set[int]) -> List[T]:
    return [
        item for index, item in enumerate(input_list) if index not in indices
    ]
