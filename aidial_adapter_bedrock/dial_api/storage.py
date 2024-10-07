import base64
import hashlib
import io
import mimetypes
import os
from typing import Mapping, Optional, TypedDict
from urllib.parse import unquote, urljoin

import aiohttp
from pydantic import BaseModel

from aidial_adapter_bedrock.utils.log_config import app_logger as log


class FileMetadata(TypedDict):
    name: str
    parentPath: str
    bucket: str
    url: str


class Bucket(TypedDict):
    bucket: str
    appdata: str


class FileStorage(BaseModel):
    dial_url: str
    api_key: str
    bucket: Optional[Bucket] = None

    @property
    def auth_headers(self) -> Mapping[str, str]:
        return {"api-key": self.api_key}

    async def _get_bucket(self, session: aiohttp.ClientSession) -> Bucket:
        if self.bucket is None:
            async with session.get(
                f"{self.dial_url}/v1/bucket",
                headers=self.auth_headers,
            ) as response:
                response.raise_for_status()
                self.bucket = await response.json()
                log.debug(f"bucket: {self.bucket}")

        return self.bucket

    async def _get_user_bucket(self, session: aiohttp.ClientSession) -> str:
        bucket = await self._get_bucket(session)
        appdata = bucket.get("appdata")
        if appdata is None:
            raise ValueError(
                "Can't retrieve user bucket because appdata isn't available"
            )
        return appdata.split("/", 1)[0]

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
        self, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        async with aiohttp.ClientSession() as session:
            bucket = await self._get_bucket(session)

            appdata = bucket["appdata"]
            ext = mimetypes.guess_extension(content_type) or ""
            url = f"{self.dial_url}/v1/files/{appdata}/{filename}{ext}"

            data = FileStorage._to_form_data(filename, content_type, content)

            async with session.put(
                url=url,
                data=data,
                headers=self.auth_headers,
            ) as response:
                response.raise_for_status()
                meta = await response.json()
                log.debug(f"Uploaded file: url={url}, metadata={meta}")
                return meta

    async def upload_file_as_base64(
        self, upload_dir: str, data: str, content_type: str
    ) -> FileMetadata:
        filename = f"{upload_dir}/{_compute_hash_digest(data)}"
        content: bytes = base64.b64decode(data)
        return await self.upload(filename, content_type, content)

    def attachment_link_to_url(self, link: str) -> str:
        base_url = f"{self.dial_url}/v1/"
        return urljoin(base_url, link)

    async def download_file_as_base64(self, dial_path: str) -> str:
        url = urljoin(f"{self.dial_url}/v1/", dial_path)
        headers: Mapping[str, str] = {}
        if url.lower().startswith(self.dial_url.lower()):
            headers = self.auth_headers

        return await download_file_as_base64(url, headers)

    def _url_to_attachment_link(self, url: str) -> str:
        return url.removeprefix(f"{self.dial_url}/v1/")

    async def get_human_readable_name(self, link: str) -> str:
        url = self.attachment_link_to_url(link)
        link = self._url_to_attachment_link(url)

        link = link.removeprefix("files/")

        if link.startswith("public/"):
            bucket = "public"
        else:
            async with aiohttp.ClientSession() as session:
                bucket = await self._get_user_bucket(session)

        link = link.removeprefix(f"{bucket}/")
        decoded_link = unquote(link)
        return link if link == decoded_link else repr(decoded_link)


async def _download_file(
    url: str, headers: Optional[Mapping[str, str]]
) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            return await response.read()


async def download_file_as_base64(
    url: str, headers: Optional[Mapping[str, str]] = None
) -> str:
    data = await _download_file(url, headers)
    return base64.b64encode(data).decode("ascii")


def _compute_hash_digest(file_content: str) -> str:
    return hashlib.sha256(file_content.encode()).hexdigest()


DIAL_URL = os.getenv("DIAL_URL")


def create_file_storage(api_key: str) -> Optional[FileStorage]:
    if DIAL_URL is None:
        return None

    return FileStorage(dial_url=DIAL_URL, api_key=api_key)
