import base64
import hashlib
import io
import re
from typing import List, Mapping, Optional, TypedDict

import aiohttp

from aidial_adapter_bedrock.dial_api.auth import Auth
from aidial_adapter_bedrock.utils.env import get_env, get_env_bool
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class FileMetadata(TypedDict):
    name: str
    parentPath: str
    bucket: str
    url: str
    type: str

    contentLength: int
    contentType: str


class FileStorage:
    dial_url: str
    bucket_url: str
    upload_url: str
    auth: Auth

    def __init__(self, dial_url: str, upload_dir: str, bucket: str, auth: Auth):
        self.dial_url = dial_url
        self.bucket_url = f"{self.dial_url}/v1/files/{bucket}"
        self.upload_url = f"{self.bucket_url}/{upload_dir}"
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
    def _to_form_data(
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
        self,
        directories: List[str],
        filename: str,
        content_type: str,
        content: bytes,
    ) -> FileMetadata:
        async with aiohttp.ClientSession() as session:
            data = FileStorage._to_form_data(filename, content_type, content)
            async with session.put(
                url="/".join([self.upload_url, *directories, filename]),
                data=data,
                headers=self.auth.headers,
            ) as response:
                response.raise_for_status()
                meta = await response.json()
                log.debug(
                    f"Uploaded file: path={self.upload_url}, file={filename}, metadata={meta}"
                )
                return meta

    async def upload_file_as_base64(
        self, data: str, content_type: str
    ) -> FileMetadata:
        path_segments = _compute_path_as_hash(data)
        directories, filename = path_segments[:-1], path_segments[-1]

        ext = _get_extension(content_type)
        filename = f"{filename}.{ext}" if ext is not None else filename

        content: bytes = base64.b64decode(data)
        return await self.upload(directories, filename, content_type, content)


def _compute_path_as_hash(file_content: str) -> List[str]:
    hash = hashlib.sha256(file_content.encode()).hexdigest()
    return _chunk_string([3, 3], hash)


def _chunk_string(chunk_sizes: List[int], string: str) -> List[str]:
    chunks = []
    start = 0
    for chunk_size in chunk_sizes:
        chunks.append(string[start : start + chunk_size])
        start += chunk_size
    chunks.append(string[start:])
    return [chunk for chunk in chunks if chunk != ""]


def _get_extension(content_type: str) -> Optional[str]:
    pattern = r"^image\/(jpeg|jpg|png|bmp)$"
    match = re.match(pattern, content_type)
    return match.group(1) if match is not None else None


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
