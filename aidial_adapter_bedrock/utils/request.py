from typing import List, TypeGuard

from aidial_sdk.chat_completion import (
    MessageContentPart,
    MessageContentTextPart,
)
from pydantic import StrictStr

from aidial_adapter_bedrock.llm.errors import ValidationError


def get_message_content_text_content(
    content: None | str | List[MessageContentPart], strict: bool = True
) -> str:

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    texts: List[str] = []
    for part in content:
        if isinstance(part, MessageContentTextPart):
            texts.append(part.text)
        elif strict:
            raise ValidationError(
                "Can't extract text from a multi-modal content part"
            )

    return "\n".join(texts)


def is_text_content_parts(
    parts: List[MessageContentPart],
) -> TypeGuard[List[MessageContentTextPart]]:
    return all(isinstance(part, MessageContentTextPart) for part in parts)


def is_content_parts(
    content: None | StrictStr | List[MessageContentPart],
) -> TypeGuard[List[MessageContentPart]]:
    return not is_plain_text_content(content)


def is_plain_text_content(
    content: None | StrictStr | List[MessageContentPart],
) -> TypeGuard[None | StrictStr]:
    return content is None or isinstance(content, str)
