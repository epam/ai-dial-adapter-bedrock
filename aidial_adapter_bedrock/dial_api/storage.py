import base64
import hashlib
import io
from typing import TypedDict

import aiohttp

from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class FileMetadata(TypedDict):
    name: str
    type: str
    path: str
    contentLength: int
    contentType: str


class FileStorage:
    base_url: str
    api_key: str

    def __init__(self, dial_url: str, base_dir: str, api_key: str):
        self.base_url = f"{dial_url}/v1/files/{base_dir}"
        self.api_key = api_key

    def auth_headers(self) -> dict[str, str]:
        return {"api-key": self.api_key}

    @staticmethod
    def to_form_data(
        filename: str, content_type: str, content: bytes
    ) -> aiohttp.FormData:
        data = aiohttp.FormData()
        data.add_field(
            "file",
            io.BytesIO(content),
            filename=filename,
            content_type=content_type,
        )
        return data

    async def list(self) -> list[FileMetadata]:
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}?purpose=metadata&path=relative"
            async with session.get(
                url, headers=self.auth_headers()
            ) as response:
                response.raise_for_status()
                ret = await response.json()
                log.debug(f"Listed files at '{url}': {ret}")
                return ret

    async def delete(self, filename: str):
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/{filename}"
            async with session.delete(
                url, headers=self.auth_headers()
            ) as response:
                response.raise_for_status()
                ret = await response.text()
                log.debug(f"Removed files at '{url}': {ret}")
                return ret

    async def upload(
        self, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        async with aiohttp.ClientSession() as session:
            data = FileStorage.to_form_data(filename, content_type, content)
            async with session.post(
                self.base_url,
                data=data,
                headers=self.auth_headers(),
            ) as response:
                response.raise_for_status()
                ret = await response.json()
                log.debug(
                    f"Uploaded to '{self.base_url}' file '{filename}': {ret}"
                )
                return ret


def hash_digest(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()


class ImageStorage:
    storage: FileStorage

    def __init__(self, dial_url: str, base_dir: str, api_key: str):
        self.storage = FileStorage(dial_url, base_dir, api_key)

    async def upload_base64_png_image(self, data: str) -> FileMetadata:
        filename = hash_digest(data) + ".png"
        content: bytes = base64.b64decode(data)
        return await self.storage.upload(filename, "image/png", content)
