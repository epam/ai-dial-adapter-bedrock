import base64
import hashlib
import io
from typing import TypedDict

import aiohttp

from aidial_adapter_bedrock.dial_api.auth import Auth
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class FileMetadata(TypedDict):
    name: str
    type: str
    path: str
    contentLength: int
    contentType: str


class FileStorage:
    base_url: str
    auth: Auth

    def __init__(self, dial_url: str, base_dir: str, auth: Auth):
        self.base_url = f"{dial_url}/v1/files/{base_dir}"
        self.auth = auth

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

    async def upload(
        self, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        async with aiohttp.ClientSession() as session:
            data = FileStorage.to_form_data(filename, content_type, content)
            async with session.post(
                self.base_url,
                data=data,
                headers=self.auth.headers,
            ) as response:
                response.raise_for_status()
                meta = await response.json()
                log.debug(
                    f"Uploaded file: path={self.base_url}, file={filename}, metadata={meta}"
                )
                return meta


def _hash_digest(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()


async def upload_base64_file(
    storage: FileStorage, data: str, content_type: str
) -> FileMetadata:
    filename = _hash_digest(data)
    content: bytes = base64.b64decode(data)
    return await storage.upload(filename, content_type, content)
