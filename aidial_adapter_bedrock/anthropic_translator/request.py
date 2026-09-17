from collections.abc import Iterator
from typing import cast

from anthropic.types.beta.message_create_params import MessageCreateParams
from pydantic import TypeAdapter, ValidationError

from aidial_adapter_bedrock.anthropic_translator.errors import (
    AnthropicErrorType,
    AnthropicHTTPError,
    format_validation_error,
)

_REQUEST: TypeAdapter[MessageCreateParams] = TypeAdapter(MessageCreateParams)


def _validate_iterables(
    value: object, original: object, path: tuple[str, ...] = ()
) -> None:
    # SDK parameters use Iterable, which Pydantic validates lazily. Consume all
    # nested iterators before sending anything upstream.
    if isinstance(value, dict) and isinstance(original, dict):
        for key, item in value.items():
            _validate_iterables(item, original[key], (*path, key))
    elif isinstance(value, list | tuple | Iterator):
        if not isinstance(original, list):
            raise AnthropicHTTPError(
                AnthropicErrorType.INVALID_REQUEST,
                f"{'.'.join(path)}: Input should be a valid list",
            )
        try:
            for index, item in enumerate(value):
                _validate_iterables(item, original[index], (*path, str(index)))
        except ValidationError as error:
            raise AnthropicHTTPError(
                AnthropicErrorType.INVALID_REQUEST,
                f"{'.'.join(path)}.{format_validation_error(error)}",
            ) from error


def validate_request(body: dict[str, object]) -> MessageCreateParams:
    for field in ("model", "max_tokens"):
        if body.get(field) is None:
            raise AnthropicHTTPError(
                AnthropicErrorType.INVALID_REQUEST, f"'{field}' is required"
            )
    try:
        _validate_iterables(_REQUEST.validate_python(body, strict=True), body)
    except ValidationError as error:
        raise AnthropicHTTPError(
            AnthropicErrorType.INVALID_REQUEST, format_validation_error(error)
        ) from error
    # Keep the original JSON: validating a TypedDict may discard unknown fields,
    # including JSON Schema keywords in a tool's input_schema.
    return cast(MessageCreateParams, body)
