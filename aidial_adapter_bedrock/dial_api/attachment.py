import mimetypes
from typing import Optional

from aidial_sdk.chat_completion import Attachment

from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    download_file_as_base64,
)
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.utils.resource import Resource


async def download_url(
    file_storage: Optional[FileStorage], name: str, url: str
) -> Resource:
    type = _guess_url_content_type(url)
    if not type:
        raise _no_content_type_exception(name)
    data = await _download_url_as_base64(file_storage, url)
    return Resource(type=type, data=data)


async def download_attachment(
    file_storage: Optional[FileStorage], name: str, attachment: Attachment
) -> Resource:
    type = _guess_attachment_content_type(attachment)

    if type is None:
        raise _no_content_type_exception(name)

    if attachment.data:
        data = attachment.data
    elif attachment.url:
        data = await _download_url_as_base64(file_storage, attachment.url)
    else:
        raise ValidationError(
            "Invalid attachment: either data or URL is required"
        )

    return Resource(type=type, data=data)


def _guess_url_content_type(url: str) -> Optional[str]:
    return (
        Resource.parse_data_url_content_type(url)
        or mimetypes.guess_type(url)[0]
    )


def _guess_attachment_content_type(attachment: Attachment) -> Optional[str]:
    if attachment.type is None or "octet-stream" in attachment.type:
        # It's an arbitrary binary file or type is missing.
        # Trying to guess the type from the URL.
        if (url := attachment.url) and (
            url_type := _guess_url_content_type(url)
        ):
            return url_type

    return attachment.type


async def _download_url_as_base64(
    file_storage: Optional[FileStorage], url: str
) -> str:
    if (recourse := Resource.from_data_url(url)) is not None:
        return recourse.data

    if file_storage:
        return await file_storage.download_file_as_base64(url)
    else:
        return await download_file_as_base64(url)


def _no_content_type_exception(name: str) -> ValidationError:
    return ValidationError(f"Can't derive content type of the {name}")
