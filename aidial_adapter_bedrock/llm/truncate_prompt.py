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
            f"Token count of all messages ({self.token_count}) exceeds"
            f" the model maximum prompt tokens ({self.model_limit})."
        )


class UserLimitOverflow(TruncatePromptError):
    user_limit: int
    token_count: int

    def print(self) -> str:
        return (
            "Token count of the last message and all system messages "
            f"({self.token_count}) exceeds the maximum prompt tokens ({self.user_limit})."
        )


T = TypeVar("T")
Messages = List[T]


def truncate_prompt(
    messages: Messages,
    tokenize: Callable[[Messages], int],
    keep_message: Callable[[Messages, int], bool],
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

    def _tokenize_selected(indices: Set[int]) -> int:
        return tokenize(select_by_indices(messages, indices))

    n = len(messages)
    all_indices = set(range(0, n))

    if user_limit is None:
        if model_limit is None:
            return set()

        token_count = _tokenize_selected(all_indices)
        if token_count <= model_limit:
            return set()

        return ModelLimitOverflow(
            model_limit=model_limit, token_count=token_count
        )

    token_count: int = 0
    kept_indices: Set[int] = {
        idx for idx in range(0, n) if keep_message(messages, idx)
    }

    token_count = _tokenize_selected(kept_indices)
    if token_count > user_limit:
        return UserLimitOverflow(user_limit=user_limit, token_count=token_count)

    for idx in reversed(range(0, n)):
        if idx in kept_indices:
            continue

        new_token_count = _tokenize_selected({*kept_indices, idx})
        if new_token_count > user_limit:
            break

        kept_indices.add(idx)
        token_count = new_token_count

    return all_indices - kept_indices
