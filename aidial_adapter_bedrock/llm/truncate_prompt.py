from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Set, TypeVar

from pydantic import BaseModel

from aidial_adapter_bedrock.utils.list import select_by_indices


class TruncatePromptError(ABC, BaseModel):
    @abstractmethod
    def print(self) -> str:
        pass


class InconsistentLimitsError(TruncatePromptError):
    user_limit: int
    model_limit: int

    def print(self) -> str:
        return (
            f"Maximum prompt tokens ({self.user_limit}) "
            f"exceeds the model maximum prompt tokens ({self.model_limit})."
        )


class ModelLimitOverflow(TruncatePromptError):
    model_limit: int
    token_count: int

    def print(self) -> str:
        return (
            f"Token count of all messages ({self.token_count}) "
            f"exceeds the model maximum prompt tokens ({self.model_limit})."
        )


class UserLimitOverflow(TruncatePromptError):
    user_limit: int
    token_count: int

    def print(self) -> str:
        return (
            f"Token count of the last message and all system messages ({self.token_count}) "
            f"exceeds the maximum prompt tokens ({self.user_limit})."
        )


def partition_indexer(chunks: List[int]) -> Callable[[int], List[int]]:
    """Returns a function that maps an index to indices of its partition.
    >>> [partition_indexer([2, 3])(i) for i in range(5)]
    [[0, 1], [0, 1], [2, 3, 4], [2, 3, 4], [2, 3, 4]]
    """
    mapping: dict[int, List[int]] = {}
    offset = 0
    for size in chunks:
        chunk = list(range(offset, offset + size))
        for idx in range(size):
            mapping[offset + idx] = chunk
        offset += size

    return mapping.__getitem__


T = TypeVar("T")
Messages = List[T]


def truncate_prompt(
    messages: Messages,
    tokenize_messages: Callable[[Messages], int],
    keep_message: Callable[[Messages, int], bool],
    partition_messages: Callable[[Messages], List[int]],
    model_limit: Optional[int],
    user_limit: Optional[int],
) -> Set[int] | TruncatePromptError:
    if (
        user_limit is not None
        and model_limit is not None
        and user_limit > model_limit
    ):
        return InconsistentLimitsError(
            user_limit=user_limit, model_limit=model_limit
        )

    if user_limit is None:
        if model_limit is None:
            return set()

        token_count = tokenize_messages(messages)
        if token_count <= model_limit:
            return set()

        return ModelLimitOverflow(
            model_limit=model_limit, token_count=token_count
        )

    partition_sizes = partition_messages(messages)
    if sum(partition_sizes) != len(messages):
        raise ValueError(
            "Partition sizes must add up to the number of messages."
        )

    def _tokenize_selected(indices: Set[int]) -> int:
        return tokenize_messages(select_by_indices(messages, indices))

    get_partition_indices = partition_indexer(partition_sizes)

    n = len(messages)
    kept_indices: Set[int] = {
        j
        for i in range(n)
        for j in get_partition_indices(i)
        if keep_message(messages, i)
    }

    token_count = _tokenize_selected(kept_indices)
    if token_count > user_limit:
        return UserLimitOverflow(user_limit=user_limit, token_count=token_count)

    for idx in reversed(range(n)):
        if idx in kept_indices:
            continue

        chunk_indices = get_partition_indices(idx)
        new_token_count = _tokenize_selected({*kept_indices, *chunk_indices})
        if new_token_count > user_limit:
            break

        kept_indices.update(chunk_indices)

    all_indices = set(range(n))
    return all_indices - kept_indices
