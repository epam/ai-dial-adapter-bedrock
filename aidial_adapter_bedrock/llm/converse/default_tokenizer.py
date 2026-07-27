from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDeployment,
    ConverseMessages,
    ConverseRequestWrapper,
    ConverseTokensRequest,
)
from aidial_adapter_bedrock.utils.json import remove_nones


def upstream_converse_tokenizer_factory(
    deployment: ConverseDeployment,
    params: ConverseRequestWrapper,
    bedrock: Bedrock,
):
    async def tokenizer(messages: ConverseMessages) -> int:
        body = ConverseTokensRequest(
            messages=[msg[0] for msg in messages],
            **remove_nones(
                {
                    "system": params.system,
                    "toolConfig": params.toolConfig,
                }
            ),
        )

        return await bedrock.acount_tokens_converse(deployment, body)

    return tokenizer
