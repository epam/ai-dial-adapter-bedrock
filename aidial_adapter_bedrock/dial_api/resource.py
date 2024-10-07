import mimetypes
from abc import ABC, abstractmethod

from aidial_sdk.chat_completion import Attachment
from pydantic import BaseModel, Field, root_validator

from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    download_file_as_base64,
)
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.utils.resource import Resource


class DialResource(ABC, BaseModel):
    entity_name: str = Field(default=None)

    @abstractmethod
    async def download(self, storage: FileStorage | None) -> Resource: ...

    @abstractmethod
    async def guess_content_type(self) -> str | None: ...

    @abstractmethod
    async def get_resource_name(self, storage: FileStorage | None) -> str: ...

    async def get_content_type(self) -> str:
        type = await self.guess_content_type()
        if not type:
            raise self.no_content_type_exception()
        return type

    def no_content_type_exception(self) -> ValidationError:
        return ValidationError(
            f"Can't derive content type of the {self.entity_name}"
        )


class URLResource(DialResource):
    url: str

    @root_validator
    def validator(cls, values):
        values["entity_name"] = values.get("entity_name") or "URL"
        return values

    async def download(self, storage: FileStorage | None) -> Resource:
        type = await self.get_content_type()
        data = await _download_url_as_base64(storage, self.url)
        return Resource.from_base64(type=type, data_base64=data)

    async def guess_content_type(self) -> str | None:
        return (
            Resource.parse_data_url_content_type(self.url)
            or mimetypes.guess_type(self.url)[0]
        )

    async def get_resource_name(self, storage: FileStorage | None) -> str:
        if type := Resource.parse_data_url_content_type(self.url):
            return f"data URL ({type})"

        if storage is not None:
            return await storage.get_human_readable_name(self.url)

        return self.url


class AttachmentResource(DialResource):
    attachment: Attachment

    @root_validator
    def validator(cls, values):
        values["entity_name"] = values.get("entity_name") or "attachment"
        return values

    async def download(self, storage: FileStorage | None) -> Resource:
        type = await self.get_content_type()

        if self.attachment.data:
            data = self.attachment.data
        elif self.attachment.url:
            data = await _download_url_as_base64(storage, self.attachment.url)
        else:
            raise ValidationError(f"Invalid {self.entity_name}")

        return Resource.from_base64(type=type, data_base64=data)

    async def guess_content_type(self) -> str | None:
        if (
            self.attachment.type is None
            or "octet-stream" in self.attachment.type
        ):
            # It's an arbitrary binary file or type is missing.
            # Trying to guess the type from the URL.
            if url := self.attachment.url:
                resource = URLResource(url=url, entity_name=self.entity_name)
                type = await resource.guess_content_type()
                if type:
                    return type

        return self.attachment.type

    async def get_resource_name(self, storage: FileStorage | None) -> str:
        if title := self.attachment.title:
            return title

        if self.attachment.data:
            return f"data {self.entity_name}"

        if url := self.attachment.url:
            resource = URLResource(url=url, entity_name=self.entity_name)
            return await resource.get_resource_name(storage)

        raise ValidationError(f"Invalid {self.entity_name}")


async def _download_url_as_base64(
    file_storage: FileStorage | None, url: str
) -> str:
    if (resource := Resource.from_data_url(url)) is not None:
        return resource.data_base64

    if file_storage:
        return await file_storage.download_file_as_base64(url)
    else:
        return await download_file_as_base64(url)
