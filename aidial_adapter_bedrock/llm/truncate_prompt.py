from abc import ABC, abstractmethod
from typing import Awaitable, Callable, List, Optional, Set, Tuple, TypeVar

from aidial_sdk.exceptions import ContextLengthExceededError
from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import (
    InvalidRequestError,
    TruncatePromptSystemAndLastUserError,
)
from pydantic import BaseModel

from aidial_adapter_bedrock.utils.list import omit_by_indices, select_by_indices


class TruncatePromptError(ABC, BaseModel):
    @abstractmethod
    def to_dial_exception(self) -> DialException:
        pass

    def print(self) -> str:
        return self.to_dial_exception().message


class InconsistentLimitsError(TruncatePromptError):
    user_limit: int
    model_limit: int

    def to_dial_exception(self) -> DialException:
        return InvalidRequestError(
            f"The request maximum prompt tokens is {self.user_limit}. "
            f"However, the model's maximum context length is {self.model_limit} tokens."
        )


class ModelLimitOverflow(TruncatePromptError):
    model_limit: int
    token_count: int

    def to_dial_exception(self) -> DialException:
        return ContextLengthExceededError(self.model_limit, self.token_count)


class UserLimitOverflow(TruncatePromptError):
    user_limit: int
    token_count: int

    def to_dial_exception(self) -> DialException:
        return TruncatePromptSystemAndLastUserError(
            self.user_limit, self.token_count
        )


def _partition_indexer(chunks: List[int]) -> Callable[[int], List[int]]:
    """
    Returns a function that maps an index to indices of its partition.
    """
    mapping: dict[int, List[int]] = {}
    offset = 0
    for size in chunks:
        chunk = list(range(offset, offset + size))
        for idx in range(size):
            mapping[offset + idx] = chunk
        offset += size

    return mapping.__getitem__


_T = TypeVar("_T")
DiscardedMessages = List[int]


async def truncate_prompt(
    messages: List[_T],
    tokenizer: Callable[[List[_T]], Awaitable[int]],
    keep_message: Callable[[List[_T], int], bool],
    partitioner: Callable[[List[_T]], List[int]],
    model_limit: Optional[int],
    user_limit: Optional[int],
) -> Tuple[DiscardedMessages, List[_T]]:
    """
    Returns a list of indices of discarded messages and a list of preserved messages
    """

    result = await compute_discarded_messages(
        messages,
        tokenizer,
        keep_message,
        partitioner,
        model_limit,
        user_limit,
    )

    if isinstance(result, TruncatePromptError):
        raise result.to_dial_exception()

    return (list(result), omit_by_indices(messages, result))


async def compute_discarded_messages(
    messages: List[_T],
    tokenizer: Callable[[List[_T]], Awaitable[int]],
    keep_message: Callable[[List[_T], int], bool],
    partitioner: Callable[[List[_T]], List[int]],
    model_limit: Optional[int],
    user_limit: Optional[int],
) -> DiscardedMessages | TruncatePromptError:
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
            return []

        token_count = await tokenizer(messages)
        if token_count <= model_limit:
            return []

        return ModelLimitOverflow(
            model_limit=model_limit, token_count=token_count
        )

    partition_sizes = partitioner(messages)
    if sum(partition_sizes) != len(messages):
        raise ValueError(
            "Partition sizes must add up to the number of messages."
        )

    async def _tokenize_selected(indices: Set[int]) -> int:
        return await tokenizer(select_by_indices(messages, indices))

    get_partition_indices = _partition_indexer(partition_sizes)

    n = len(messages)
    kept_indices: Set[int] = {
        j
        for i in range(n)
        for j in get_partition_indices(i)
        if keep_message(messages, i)
    }

    token_count = await _tokenize_selected(kept_indices)
    if token_count > user_limit:
        return UserLimitOverflow(user_limit=user_limit, token_count=token_count)

    for idx in reversed(range(n)):
        if idx in kept_indices:
            continue

        chunk_indices = get_partition_indices(idx)
        new_token_count = await _tokenize_selected(
            {*kept_indices, *chunk_indices}
        )
        if new_token_count > user_limit:
            break

        kept_indices.update(chunk_indices)

    all_indices = set(range(n))
    return sorted(list(all_indices - kept_indices))
