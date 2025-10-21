import inspect
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Protocol,
    Sequence,
    Set,
    TypeVar,
    assert_never,
    runtime_checkable,
)

from aidial_sdk.chat_completion import (
    MessageContentImagePart,
    MessageContentRefusalPart,
    MessageContentTextPart,
)
from pydantic import BaseModel

from aidial_adapter_bedrock.dial_api.resource import (
    AttachmentResource,
    DialResource,
    UnsupportedContentType,
    URLResource,
)
from aidial_adapter_bedrock.dial_api.storage import FileStorage
from aidial_adapter_bedrock.llm.errors import UserError, ValidationError
from aidial_adapter_bedrock.llm.message import BaseMessage, SystemMessage
from aidial_adapter_bedrock.utils.concurrency import aiter_to_list
from aidial_adapter_bedrock.utils.resource import Resource

_T = TypeVar("_T", covariant=True)
_Config = TypeVar("_Config", bound=BaseModel, contravariant=True)


@runtime_checkable
class Handler(Protocol, Generic[_T]):
    def __call__(self, resource: Resource) -> _T: ...


@runtime_checkable
class HandlerWithConfig(Protocol, Generic[_T, _Config]):
    def __call__(self, resource: Resource, config: _Config | None) -> _T: ...


class AttachmentProcessor(BaseModel, Generic[_T, _Config]):
    class Config:
        arbitrary_types_allowed = True

    supported_types: Dict[str, Set[str]]
    """MIME type to file extensions mapping"""

    handler: Handler[_T] | HandlerWithConfig[_T, _Config]

    def handle(self, resource: Resource, config: _Config | None) -> _T:
        sig = inspect.signature(self.handler)
        params = list(sig.parameters.values())

        with_config = (
            len(params) >= 2
            and params[1].kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ) or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)

        if with_config:
            return self.handler(resource, config)  # type: ignore
        return self.handler(resource)  # type: ignore


class AttachmentProcessors(BaseModel, Generic[_T, _Config]):
    config: _Config | None = None
    attachment_processors: Sequence[AttachmentProcessor[_T, _Config]]
    text_handler: Callable[[str], _T]
    file_storage: FileStorage | None

    @property
    def supported_types(self) -> Dict[str, Set[str]]:
        ret: Dict[str, Set[str]] = {}
        for processor in self.attachment_processors:
            for mime_type, file_exts in processor.supported_types.items():
                ret.setdefault(mime_type, set()).update(file_exts)
        return ret

    @property
    def supported_mime_types(self) -> List[str]:
        return list(self.supported_types)

    @property
    def supported_image_types(self) -> List[str]:
        return [t for t in self.supported_mime_types if t.startswith("image/")]

    async def process_attachments(self, message: BaseMessage) -> List[_T]:
        return await aiter_to_list(self.process_attachments_iter(message))

    async def process_attachments_iter(
        self, message: BaseMessage
    ) -> AsyncIterator[_T]:
        if not isinstance(message, SystemMessage):
            for attachment in message.attachments:
                yield await self._handle_dial_resource(
                    AttachmentResource(
                        attachment=attachment,
                        entity_name="attachment",
                        supported_types=self.supported_mime_types,
                    ),
                )

        content = message.content

        match content:
            case str():
                yield self.text_handler(content)
            case list():
                for part in content:
                    match part:
                        case MessageContentTextPart(text=text):
                            yield self.text_handler(text)
                        case MessageContentImagePart(image_url=image_url):
                            yield await self._handle_dial_resource(
                                URLResource(
                                    url=image_url.url,
                                    entity_name="image url",
                                    supported_types=self.supported_image_types,
                                ),
                            )
                        case MessageContentRefusalPart():
                            raise ValidationError(
                                "Refuse content parts aren't supported"
                            )
                        case _:
                            assert_never(part)
            case _:
                assert_never(content)

    async def _download_resource(self, dial_resource: DialResource) -> Resource:
        try:
            return await dial_resource.download(self.file_storage)
        except UnsupportedContentType as e:
            raise UserError(
                f"Unsupported media type: {e.type}",
                get_usage_message(self.get_file_exts(e.supported_types)),
            )

    async def _handle_resource(self, resource: Resource) -> _T:
        for processor in self.attachment_processors:
            if resource.type in processor.supported_types:
                return processor.handle(resource, self.config)

        raise UserError(
            f"Unsupported media type: {resource.type}",
            get_usage_message(self.get_file_exts(self.supported_mime_types)),
        )

    async def _handle_dial_resource(self, dial_resource: DialResource) -> _T:
        resource = await self._download_resource(dial_resource)
        return await self._handle_resource(resource)

    def get_file_exts(self, mime_types: List[str]) -> List[str]:
        return [
            file_ext
            for mime_type, file_exts in self.supported_types.items()
            if mime_type in mime_types
            for file_ext in file_exts
        ]


def get_usage_message(supported_exts: List[str]) -> str:
    document_hint = ""
    if "pdf" in supported_exts:
        document_hint = '- "Summarize the document" for a PDF document'

    return f"""
The application answers queries about attached files.
Attach file(s) and ask questions about them in the same message.

Supported attachment types: {', '.join(supported_exts)}.

Examples of queries:
- "Describe this picture" for an image
- "What are in these images? Is there any difference between them?" for multiple images
{document_hint}
""".strip()
