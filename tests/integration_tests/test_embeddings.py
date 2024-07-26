from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, List

import pytest
from openai import AsyncAzureOpenAI
from openai.types import CreateEmbeddingResponse

from aidial_adapter_bedrock.deployments import EmbeddingsDeployment
from aidial_adapter_bedrock.llm.consumer import Attachment
from tests.conftest import DEFAULT_API_VERSION, TEST_SERVER_URL
from tests.utils.openai import sanitize_test_name


@dataclass
class ModelSpec:
    deployment: EmbeddingsDeployment
    default_dimensions: int
    supports_dimensions: bool


specs: List[ModelSpec] = [
    ModelSpec(
        deployment=EmbeddingsDeployment.AMAZON_TITAN_EMBED_TEXT_V1,
        default_dimensions=1536,
        supports_dimensions=False,
    ),
    ModelSpec(
        deployment=EmbeddingsDeployment.AMAZON_TITAN_EMBED_TEXT_V2,
        default_dimensions=1024,
        supports_dimensions=True,
    ),
    ModelSpec(
        deployment=EmbeddingsDeployment.AMAZON_TITAN_EMBED_IMAGE_V1,
        default_dimensions=1024,
        supports_dimensions=True,
    ),
]


@dataclass
class TestCase:
    __test__ = False

    deployment: EmbeddingsDeployment
    input: str | List[str]
    extra_body: dict

    expected: Callable[[CreateEmbeddingResponse], None] | Exception

    def get_id(self):
        return sanitize_test_name(
            f"{self.deployment.value} {self.extra_body} {self.input}"
        )


def check_embeddings_response(
    input: str | List[str],
    custom_input: list[Any] | None,
    dimensions: int,
) -> Callable[[CreateEmbeddingResponse], None]:
    def ret(resp: CreateEmbeddingResponse):
        n_inputs = 1 if isinstance(input, str) else len(input)
        n_inputs += len(custom_input) if custom_input else 0

        assert len(resp.data) == n_inputs
        assert len(resp.data[0].embedding) == dimensions

    return ret


def get_test_case(
    spec: ModelSpec,
    input: str | List[str],
    custom_input: list[Any] | None,
    encoding_format: str | None,
    embedding_type: str | None,
    embedding_instr: str | None,
    dimensions: int | None,
) -> TestCase:

    custom_fields = {}

    if embedding_instr:
        custom_fields["instruction"] = embedding_instr

    if embedding_type:
        custom_fields["type"] = embedding_type

    expected: Callable[[CreateEmbeddingResponse], None] | Exception = (
        check_embeddings_response(
            input, custom_input, dimensions or spec.default_dimensions
        )
    )

    if dimensions and not spec.supports_dimensions:
        expected = Exception("Dimensions parameter is not supported")
    elif embedding_instr:
        expected = Exception("Instruction prompt is not supported")
    elif embedding_type:
        expected = Exception(
            "The embedding model does not support embedding types"
        )

    return TestCase(
        deployment=spec.deployment,
        input=input,
        extra_body=(
            {
                "custom_input": custom_input,
                "custom_fields": custom_fields,
                "encoding_format": encoding_format,
                "dimensions": dimensions,
            }
        ),
        expected=expected,
    )


image_attachment = Attachment(
    type="image/png",
    url="https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_92x30dp.png",
).dict()


def get_image_test_cases(
    input: str | List[str],
    custom_input: list[Any] | None,
    exception: Exception | None,
) -> TestCase:
    expected = exception or check_embeddings_response(input, custom_input, 1024)

    return TestCase(
        deployment=EmbeddingsDeployment.AMAZON_TITAN_EMBED_IMAGE_V1,
        input=input,
        extra_body=({"custom_input": custom_input}),
        expected=expected,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    [
        get_image_test_cases(input, custom_input, exception)
        for input, custom_input, exception in [
            ("dog", ["cat", image_attachment], None),
            ([], [["image title", image_attachment]], None),
            ([], [[image_attachment, "image title"]], None),
            (
                [],
                [["text1", "text2"]],
                Exception(
                    "The first element of a custom_input list element must be a string "
                    "and the second element must be an image attachment or vice versa"
                ),
            ),
            (
                [],
                [["image title 2", image_attachment, "image title"]],
                Exception(
                    "No more than two elements are allowed in an element of custom_input list"
                ),
            ),
        ]
    ]
    + [
        get_test_case(spec, input, custom_input, format, ty, instr, dims)
        for spec, input, custom_input, format, ty, instr, dims in product(
            specs,
            ["dog", ["fish", "cat"]],
            [None, ["ball"]],
            ["base64", "float"],
            [None, "embedding_type"],
            [None, "instruction"],
            [None, 256],
        )
    ],
    ids=lambda test: test.get_id(),
)
async def test_embeddings(server, test: TestCase):
    model_id = test.deployment.value

    client = AsyncAzureOpenAI(
        azure_endpoint=TEST_SERVER_URL,
        azure_deployment=model_id,
        api_version=DEFAULT_API_VERSION,
        api_key="dummy_key",
    )

    async def run() -> CreateEmbeddingResponse:
        return await client.embeddings.create(
            model=model_id, input=test.input, extra_body=test.extra_body
        )

    if isinstance(test.expected, Exception):
        with pytest.raises(type(test.expected), match=str(test.expected)):
            await run()
    else:
        embeddings = await run()
        test.expected(embeddings)
