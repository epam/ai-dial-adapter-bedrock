import base64
import hashlib
import io
from typing import Mapping, Optional, TypedDict

import aiohttp

from aidial_adapter_bedrock.dial_api.auth import Auth
from aidial_adapter_bedrock.utils.env import get_env, get_env_bool
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

    def __init__(self, dial_url: str, base_dir: str, bucket: str, auth: Auth):
        self.base_url = f"{dial_url}/v1/files/{bucket}/{base_dir}"
        self.auth = auth

    @classmethod
    async def create(cls, dial_url: str, base_dir: str, auth: Auth):
        bucket = await FileStorage._get_bucket(dial_url, auth)
        return cls(dial_url, base_dir, bucket, auth)

    @staticmethod
    async def _get_bucket(dial_url: str, auth: Auth) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{dial_url}/v1/bucket", headers=auth.headers
            ) as response:
                response.raise_for_status()
                body = await response.json()
                return body["bucket"]

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
            async with session.put(
                f"{self.base_url}/{filename}",
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


async def upload_file_as_base64(
    storage: FileStorage, data: str, content_type: str
) -> FileMetadata:
    filename = _hash_digest(data)
    content: bytes = base64.b64decode(data)
    return await storage.upload(filename, content_type, content)


DIAL_USE_FILE_STORAGE = get_env_bool("DIAL_USE_FILE_STORAGE", False)

DIAL_URL: Optional[str] = None
if DIAL_USE_FILE_STORAGE:
    DIAL_URL = get_env(
        "DIAL_URL", "DIAL_URL must be set to use the DIAL file storage"
    )


async def create_file_storage(
    base_dir: str, headers: Mapping[str, str]
) -> Optional[FileStorage]:
    if not DIAL_USE_FILE_STORAGE or DIAL_URL is None:
        return None

    auth = Auth.from_headers("authorization", headers)
    if auth is None:
        log.warning(
            "The request doesn't have required headers to use the DIAL file storage. "
            "Fallback to base64 encoding of images."
        )
        return None

    return await FileStorage.create(
        dial_url=DIAL_URL,
        auth=auth,
        base_dir=base_dir,
    )
