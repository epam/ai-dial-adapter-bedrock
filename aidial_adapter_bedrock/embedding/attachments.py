import mimetypes
from typing import List, Optional, Tuple

from aidial_sdk.chat_completion import Attachment

from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    download_file_as_base64,
)
from aidial_adapter_bedrock.llm.errors import UserError, ValidationError


async def _download_base64_data(
    url: str, file_storage: Optional[FileStorage]
) -> str:
    if not file_storage:
        return await download_file_as_base64(url)
    return await file_storage.download_file_as_base64(url)


def _validate_content_type(
    content_type: str, supported_content_types: List[str]
):
    if content_type not in supported_content_types:
        raise UserError(
            f"Unsupported attachment type: {content_type}. "
            f"Supported attachment types: {', '.join(supported_content_types)}.",
        )


async def download_base64_data(
    attachment: Attachment,
    file_storage: Optional[FileStorage],
    supported_content_types: List[str],
) -> Tuple[str, str]:
    if attachment.data:
        if not attachment.type:
            raise ValidationError(
                "Attachment type is required for provided data"
            )
        _validate_content_type(attachment.type, supported_content_types)
        return attachment.type, attachment.data

    if attachment.url:
        content_type = (
            attachment.type or mimetypes.guess_type(attachment.url)[0]
        )
        if not content_type:
            raise ValidationError(
                f"Cannot guess content type of attachment {attachment.url}"
            )
        _validate_content_type(content_type, supported_content_types)

        data = await _download_base64_data(attachment.url, file_storage)
        return content_type, data

    raise ValidationError("Attachment data or URL is required")
