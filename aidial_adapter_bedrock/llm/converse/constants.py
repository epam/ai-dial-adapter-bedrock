from aidial_sdk.chat_completion import FinishReason as DialFinishReason

from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDocumentType,
    ConverseImageType,
    ConverseStopReason,
)

CONVERSE_TO_DIAL_FINISH_REASON = {
    ConverseStopReason.END_TURN: DialFinishReason.STOP,
    ConverseStopReason.TOOL_USE: DialFinishReason.TOOL_CALLS,
    ConverseStopReason.MAX_TOKENS: DialFinishReason.LENGTH,
    ConverseStopReason.STOP_SEQUENCE: DialFinishReason.STOP,
    ConverseStopReason.GUARDRAIL_INTERVENED: DialFinishReason.CONTENT_FILTER,
    ConverseStopReason.CONTENT_FILTERED: DialFinishReason.CONTENT_FILTER,
}


IMAGE_MIME_TO_CONVERSE_TYPE = {
    "image/png": ConverseImageType.PNG,
    "image/jpeg": ConverseImageType.JPEG,
    "image/gif": ConverseImageType.GIF,
    "image/webp": ConverseImageType.WEBP,
}
CONVERSE_IMAGE_TYPE_TO_MIME = {
    v: k for k, v in IMAGE_MIME_TO_CONVERSE_TYPE.items()
}

DOCUMENT_MIME_TO_CONVERSE_TYPE = {
    "application/pdf": ConverseDocumentType.PDF,
    "application/csv": ConverseDocumentType.CSV,
    "application/msword": ConverseDocumentType.DOC,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ConverseDocumentType.DOCX,
    "application/vnd.ms-excel": ConverseDocumentType.XLS,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ConverseDocumentType.XLSX,
    "text/plain": ConverseDocumentType.TXT,
    "text/markdown": ConverseDocumentType.MD,
}
CONVERSE_DOCUMENT_TYPE_TO_MIME = {
    v: k for k, v in DOCUMENT_MIME_TO_CONVERSE_TYPE.items()
}
