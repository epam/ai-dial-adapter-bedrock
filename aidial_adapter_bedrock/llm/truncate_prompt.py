from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TypeVar

from aidial_sdk.exceptions import (
    ContextLengthExceededError,
    InvalidRequestError,
    TruncatePromptSystemAndLastUserError,
)
from aidial_sdk.exceptions import HTTPException as DialException
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


def _partition_indexer(chunks: list[int]) -> Callable[[int], list[int]]:
    """
    Returns a function that maps an index to indices of its partition.
    """
    mapping: dict[int, list[int]] = {}
    offset = 0
    for size in chunks:
        chunk = list(range(offset, offset + size))
        for idx in range(size):
            mapping[offset + idx] = chunk
        offset += size

    return mapping.__getitem__


_T = TypeVar("_T")
DiscardedMessages = list[int]


async def truncate_prompt(
    messages: list[_T],
    tokenizer: Callable[[list[_T]], Awaitable[int]],
    keep_message: Callable[[list[_T], int], bool],
    partitioner: Callable[[list[_T]], list[int]],
    model_limit: int | None,
    user_limit: int | None,
) -> tuple[DiscardedMessages, list[_T]]:
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
    messages: list[_T],
    tokenizer: Callable[[list[_T]], Awaitable[int]],
    keep_message: Callable[[list[_T], int], bool],
    partitioner: Callable[[list[_T]], list[int]],
    model_limit: int | None,
    user_limit: int | None,
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

    async def _tokenize_selected(indices: set[int]) -> int:
        return await tokenizer(select_by_indices(messages, indices))

    get_partition_indices = _partition_indexer(partition_sizes)

    n = len(messages)
    kept_indices: set[int] = {
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
    return sorted(all_indices - kept_indices)
