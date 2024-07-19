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


async def reject_attachment(attachment: Attachment):
    raise ValidationError("Attachments are not supported")


async def collect_embedding_inputs(
    request: EmbeddingsRequest,
    *,
    on_text: Callable[[str], Coro[T]],
    on_tokens: Callable[[Tokens], Coro[T]] = reject_tokens,
    on_attachment: Callable[[Attachment], Coro[T]] = reject_attachment,
    on_mixed: Callable[[List[str | Attachment]], AsyncIterator[T]],
) -> AsyncIterator[T]:

    async def on_text_or_attachment(input: str | Attachment) -> T:
        if isinstance(input, str):
            return await on_text(input)
        else:
            return await on_attachment(input)

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

        if is_list_of_tokens:
            yield await on_tokens(cast(Tokens, request.input))

    else:
        assert_never(request.input)

    if request.custom_input is None:
        return

    for input in request.custom_input:
        if isinstance(input, (str, Attachment)):
            yield await on_text_or_attachment(input)
        elif isinstance(input, list):
            async for t in on_mixed(input):
                yield t
        else:
            assert_never(input)


def collect_embedding_inputs_no_attachments(
    request: EmbeddingsRequest,
    *,
    on_text: Callable[[str], Coro[T]],
    on_tokens: Callable[[Tokens], Coro[T]] = reject_tokens,
) -> AsyncIterator[T]:

    async def on_mixed(inputs: List[str | Attachment]) -> AsyncIterator[T]:
        if len(inputs) == 0:
            pass
        elif len(inputs) == 1:
            if isinstance(inputs[0], str):
                yield await on_text(inputs[0])
            else:
                yield await reject_attachment(inputs[0])
        else:
            raise ValidationError(
                "No more than one element is allowed in an element of custom_input list"
            )

    return collect_embedding_inputs(
        request,
        on_text=on_text,
        on_tokens=on_tokens,
        on_attachment=reject_attachment,
        on_mixed=on_mixed,
    )
