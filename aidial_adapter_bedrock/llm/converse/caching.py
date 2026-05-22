from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, assert_never

import pydantic
from aidial_adapter_anthropic.adapter import ValidationError
from aidial_sdk.chat_completion import CacheBreakpoint, CacheBreakpointPath
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Tool as DialTool

from aidial_adapter_bedrock.llm.converse.types import (
    ConverseCachePoint,
    ConverseCachePointPart,
)
from aidial_adapter_bedrock.utils.pydantic import ExtraForbidModel


class ConverseCacheBreakpoint(ExtraForbidModel):
    ttl: Literal["5m", "1h"] | None = None

    @classmethod
    def parse(cls, brk: CacheBreakpoint) -> ConverseCacheBreakpoint:
        try:
            return cls.model_validate(brk.model_extra or {})
        except pydantic.ValidationError as e:
            raise ValidationError(str(e)) from None

    def to_converse_cache_point_part(self) -> ConverseCachePointPart:
        cachePoint = ConverseCachePoint(type="default")
        if ttl := self.ttl:
            cachePoint["ttl"] = ttl
        return ConverseCachePointPart(cachePoint=cachePoint)

    def get_ttl(self) -> int:
        match self.ttl:
            case "1h":
                return 3600
            case "5m" | None:
                # 5 minutes is a default TTL for Converse API cache breakpoints
                # https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
                return 5 * 60
            case _:
                assert_never(self.ttl)


@dataclass
class CacheInfo:
    breakpoint_path: CacheBreakpointPath
    expire_at: str


def get_cache_info(
    messages: list[DialMessage], tools: list[DialTool]
) -> CacheInfo | None:
    ttl = 0
    path = None

    # The hashing order is tools followed by messages.
    # So a breakpoint in messages takes precedence over the one in tools.
    for i, tool in enumerate(tools):
        if (cf := tool.custom_fields) and (brk := cf.cache_breakpoint):
            ttl = max(ttl, ConverseCacheBreakpoint.parse(brk).get_ttl())
            path = CacheBreakpointPath.tools(i)

    for i, message in enumerate(messages):
        if (cf := message.custom_fields) and (brk := cf.cache_breakpoint):
            ttl = max(ttl, ConverseCacheBreakpoint.parse(brk).get_ttl())
            path = CacheBreakpointPath.messages(i)

    if path is None:
        return None

    return CacheInfo(path, str(int(time.time()) + ttl))
