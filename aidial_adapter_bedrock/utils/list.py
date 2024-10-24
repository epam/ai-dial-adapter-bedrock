from typing import Any, Callable, Container, List, TypeVar

_T = TypeVar("_T")
_V = TypeVar("_V")


def select_by_indices(lst: List[_T], indices: Container[int]) -> List[_T]:
    return [elem for idx, elem in enumerate(lst) if idx in indices]


def omit_by_indices(lst: List[_T], indices: Container[int]) -> List[_T]:
    return [elem for idx, elem in enumerate(lst) if idx not in indices]


def group_by(
    lst: List[_T],
    key: Callable[[_T], Any],
    init: Callable[[_T], _V],
    merge: Callable[[_V, _T], _V],
) -> List[_V]:

    def _gen():
        if not lst:
            return

        prev_val = init(lst[0])
        prev_key = key(lst[0])

        for elem in lst[1:]:
            if prev_key == key(elem):
                prev_val = merge(prev_val, elem)
            else:
                yield prev_val
                prev_val = init(elem)
                prev_key = key(elem)

        yield prev_val

    return list(_gen())
