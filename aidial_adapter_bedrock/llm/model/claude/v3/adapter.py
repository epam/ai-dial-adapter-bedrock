from typing import Self

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.deployments import Claude3Deployment
from aidial_adapter_bedrock.dial_api.storage import create_file_storage
from aidial_adapter_bedrock.llm.converse.adapter import ConverseAdapter
from aidial_adapter_bedrock.llm.model.claude.v3.tokenizer import (
    create_tokenizer,
    tokenize_text,
)


class Adapter(ConverseAdapter):
    @classmethod
    def create(
        cls,
        deployment: Claude3Deployment,
        api_key: str,
        bedrock: Bedrock,
    ) -> Self:
        return cls(
            deployment=deployment,
            bedrock=bedrock,
            storage=create_file_storage(api_key),
            tokenize_text=tokenize_text,
            tokenizer_factory=create_tokenizer,
        )
