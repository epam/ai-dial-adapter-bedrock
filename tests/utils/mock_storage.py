import os
import shutil
import tempfile
from typing import Dict

from aidial_adapter_bedrock.dial_api.storage import FileMetadata, FileStorage


class MockFileStorage(FileStorage):
    temp_dir: str
    files: Dict[str, bytes]

    @classmethod
    def create(cls) -> "MockFileStorage":
        storage = cls(
            dial_url="http://mock",
            api_key="mock",
            temp_dir=tempfile.mkdtemp(),
            files={},
        )
        return storage

    async def upload(
        self, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        full_path = os.path.join(self.temp_dir, filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "wb") as f:
            f.write(content)

        self.files[filename] = content

        return FileMetadata(
            name=filename,
            parentPath=os.path.dirname(filename),
            bucket="mock-bucket",
            url=f"files/mock-bucket/{filename}",
        )

    async def download_file(self, link: str) -> bytes:
        filename = link.removeprefix("files/mock-bucket/")
        return self.files[filename]

    async def get_human_readable_name(self, link: str) -> str:
        return link.removeprefix("files/mock-bucket/")

    def cleanup(self):
        shutil.rmtree(self.temp_dir)
