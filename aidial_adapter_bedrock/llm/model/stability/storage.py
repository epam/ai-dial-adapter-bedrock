from aidial_adapter_anthropic.dial.consumer import Attachment
from aidial_adapter_anthropic.dial.storage import FileStorage


async def save_to_storage(
    storage: FileStorage, attachment: Attachment
) -> Attachment:
    if (
        attachment.type is not None
        and attachment.type.startswith("image/")
        and attachment.data is not None
    ):
        response = await storage.upload_file_as_base64(
            "images", attachment.data, attachment.type
        )
        return Attachment(
            title=attachment.title,
            type=attachment.type,
            url=response["url"],
        )

    return attachment
