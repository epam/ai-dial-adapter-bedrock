from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    List,
    TypeVar,
    assert_never,
    cast,
)

from aidial_sdk.chat_completion.request import Attachment
from aidial_sdk.embeddings.request import EmbeddingsRequest

from aidial_adapter_bedrock.llm.errors import ValidationError

T = TypeVar("T")

Coro = Coroutine[T, Any, Any]
Tokens = List[int]


async def reject_tokens(tokens: Tokens):
    raise ValidationError(
        "Tokens in the input are not supported, provide text instead. "
        "When Langchain AzureOpenAIEmbeddings class is used, set 'check_embedding_ctx_length=False' to disable tokenization."
    )


EMPTY_INPUT_LIST_ERROR = ValidationError(
    "Empty list in an element of custom_input list"
)

ATTACHMENT_ERROR = ValidationError("Attachments are not supported")


async def reject_attachment(attachment: Attachment):
    raise ATTACHMENT_ERROR


async def collect_embedding_inputs(
    request: EmbeddingsRequest,
    *,
    on_text: Callable[[str], Coro[T]],
    on_tokens: Callable[[Tokens], Coro[T]] = reject_tokens,
    on_attachment: Callable[[Attachment], Coro[T]] = reject_attachment,
    on_mixed: Callable[[List[str | Attachment]], Coro[T]],
) -> AsyncIterator[T]:

    if isinstance(request.input, str):
        yield await on_text(request.input)
    elif isinstance(request.input, list):

        is_list_of_tokens = False
        for input in request.input:
            if isinstance(input, str):
                yield await on_text(input)
            elif isinstance(input, list):
                yield await on_tokens(input)
            else:
                is_list_of_tokens = True
                break

        if is_list_of_tokens:
            yield await on_tokens(cast(Tokens, request.input))

    else:
        assert_never(request.input)

    if request.custom_input is None:
        return

    for input in request.custom_input:
        if isinstance(input, str):
            yield await on_text(input)
        elif isinstance(input, Attachment):
            yield await on_attachment(input)
        elif isinstance(input, list):
            yield await on_mixed(input)
        else:
            assert_never(input)


def collect_embedding_inputs_without_attachments(
    request: EmbeddingsRequest,
    *,
    on_texts: Callable[[List[str]], Coro[T]],
    on_tokens: Callable[[Tokens], Coro[T]] = reject_tokens,
) -> AsyncIterator[T]:

    async def on_text(text: str) -> Coro[T]:
        return await on_texts([text])

    async def on_mixed(inputs: List[str | Attachment]) -> Coro[T]:
        if inputs == []:
            raise EMPTY_INPUT_LIST_ERROR

        texts: List[str] = []
        for input in inputs:
            if isinstance(input, str):
                texts.append(input)
            else:
                raise ATTACHMENT_ERROR

        return await on_texts(texts)

    return collect_embedding_inputs(
        request,
        on_text=on_text,
        on_tokens=on_tokens,
        on_attachment=reject_attachment,
        on_mixed=on_mixed,
    )
