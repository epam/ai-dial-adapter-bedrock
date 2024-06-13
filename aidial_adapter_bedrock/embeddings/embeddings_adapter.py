from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from pydantic import BaseModel

from aidial_adapter_bedrock.dial_api.request import EmbeddingsType
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage


class EmbeddingsAdapter(ABC, BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    async def embeddings(
        self,
        request_body: dict,
        embedding_instruction: Optional[str],
        embedding_type: EmbeddingsType,
    ) -> Tuple[List[List[float]], TokenUsage]:
        pass
