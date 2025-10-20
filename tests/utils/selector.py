from __future__ import annotations

from typing import Generic, Protocol, TypeVar

_T = TypeVar("_T", contravariant=True)


class Selector(Protocol, Generic[_T]):
    def __call__(self, x: _T, /) -> bool: ...


class Pred(Generic[_T]):
    p: Selector[_T]

    def __init__(self, p: Selector[_T]) -> None:
        self.p = p

    def __call__(self, x: _T, /) -> bool:
        return self.p(x)

    def __invert__(self) -> Pred:
        def _p(x: _T) -> bool:
            return not self(x)

        return Pred(_p)

    def __and__(self, other):
        def _p(x):
            return self(x) and other(x)

        return Pred(_p)

    def __or__(self, other):
        def _p(x):
            return self(x) or other(x)

        return Pred(_p)


pred = Pred
