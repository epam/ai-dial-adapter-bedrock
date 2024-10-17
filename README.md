# Overview

The project implements [AI DIAL API](https://epam-rail.com/dial_api) for language models from [AWS Bedrock](https://aws.amazon.com/bedrock/).

## Supported models

### Chat completion models

The following models support `POST SERVER_URL/openai/deployments/DEPLOYMENT_NAME/chat/completions` endpoint along with an optional support of `POST /tokenize` and `POST /truncate_prompt` endpoints:

Note that a model supports `/truncate_prompt` endpoint if and only if it supports `max_prompt_tokens` request parameter.

|Vendor|Model|Deployment name|Modality|`/tokenize`|`/truncate_prompt`, `max_prompt_tokens`|tools/functions|
|---|---|---|---|---|---|---|
|Anthropic|Claude 3.5 Sonnet|anthropic.claude-3-5-sonnet-20240620-v1:0|text-to-text, image-to-text|🟡|🟡|✅|
|Anthropic|Claude 3 Sonnet|anthropic.claude-3-sonnet-20240229-v1:0|text-to-text, image-to-text|🟡|🟡|✅|
|Anthropic|Claude 3 Haiku|anthropic.claude-3-haiku-20240307-v1:0|text-to-text, image-to-text|🟡|🟡|✅|
|Anthropic|Claude 3 Opus|anthropic.claude-3-opus-20240229-v1:0|text-to-text, image-to-text|🟡|🟡|✅|
|Anthropic|Claude 2.1|anthropic.claude-v2:1|text-to-text|✅|✅|✅|
|Anthropic|Claude 2|anthropic.claude-v2|text-to-text|✅|✅|❌|
|Anthropic|Claude Instant 1.2|anthropic.claude-instant-v1|text-to-text|🟡|🟡|❌|
|Meta|Llama 3.1 405B Instruct|meta.llama3-1-405b-instruct-v1:0|text-to-text|🟡|🟡|❌|
|Meta|Llama 3.1 70B Instruct|meta.llama3-1-70b-instruct-v1:0|text-to-text|🟡|🟡|❌|
|Meta|Llama 3.1 8B Instruct|meta.llama3-1-8b-instruct-v1:0|text-to-text|🟡|🟡|❌|
|Meta|Llama 3 Chat 70B Instruct|meta.llama3-70b-instruct-v1:0|text-to-text|🟡|🟡|❌|
|Meta|Llama 3 Chat 8B Instruct|meta.llama3-8b-instruct-v1:0|text-to-text|🟡|🟡|❌|
|Meta|Llama 2 Chat 70B|meta.llama2-70b-chat-v1|text-to-text|🟡|🟡|❌|
|Meta|Llama 2 Chat 13B|meta.llama2-13b-chat-v1|text-to-text|🟡|🟡|❌|
|Stability AI|SDXL 1.0|stability.stable-diffusion-xl-v1|text-to-image|❌|🟡|❌|
|Amazon|Titan Text G1 - Express|amazon.titan-tg1-large|text-to-text|🟡|🟡|❌|
|AI21 Labs|Jurassic-2 Ultra|ai21.j2-jumbo-instruct|text-to-text|🟡|🟡|❌|
|AI21 Labs|Jurassic-2 Ultra v1|ai21.j2-ultra-v1|text-to-text|🟡|🟡|❌|
|AI21 Labs|Jurassic-2 Mid|ai21.j2-grande-instruct|text-to-text|🟡|🟡|❌|
|AI21 Labs|Jurassic-2 Mid v1|ai21.j2-mid-v1|text-to-text|🟡|🟡|❌|
|Cohere|Command|cohere.command-text-v14|text-to-text|🟡|🟡|❌|
|Cohere|Command Light|cohere.command-light-text-v14|text-to-text|🟡|🟡|❌|

✅, 🟡, and ❌ denote degrees of support of the given feature:

||`/tokenize`, `/truncate_prompt`, `max_prompt_token`|tools/functions|
|---|---|---|
|✅|Fully supported via an official tokenization algorithm|Fully supported via native tools API or official prompts to enable tools|
|🟡|Partially supported, because tokenization algorithm wasn't made public by the model vendor.<br>An approximate tokenization algorithm is used instead.<br>It conservatively counts **every byte in UTF-8 encoding of a string as a single token**.|Partially supported, because the model doesn't support tools natively.<br>Prompt engineering is used instead to emulate tools, which may not be very reliable.|
|❌|Not supported|Not supported|

### Embedding models

The following models support `SERVER_URL/openai/deployments/DEPLOYMENT_NAME/embeddings` endpoint:

|Model|Deployment name|Modality|
|---|---|---|
|Titan Multimodal Embeddings Generation 1 (G1)|amazon.titan-embed-image-v1|image/text-to-embedding|
|Amazon Titan Text Embeddings V2|amazon.titan-embed-text-v2:0|text-to-embedding|
|Titan Embeddings G1 – Text v1.2|amazon.titan-embed-text-v1|text-to-embedding|
|Cohere Embed English|cohere.embed-english-v3|text-to-embedding|
|Cohere Multilingual|cohere.embed-multilingual-v3|text-to-embedding|

## Developer environment

This project uses [Python>=3.11](https://www.python.org/downloads/) and [Poetry>=1.6.1](https://python-poetry.org/) as a dependency manager.

Check out Poetry's [documentation on how to install it](https://python-poetry.org/docs/#installation) on your system before proceeding.

To install requirements:

```sh
poetry install
```

This will install all requirements for running the package, linting, formatting and tests.

### IDE configuration

The recommended IDE is [VSCode](https://code.visualstudio.com/).
Open the project in VSCode and install the recommended extensions.

The VSCode is configured to use PEP-8 compatible formatter [Black](https://black.readthedocs.io/en/stable/index.html).

Alternatively you can use [PyCharm](https://www.jetbrains.com/pycharm/).

Set-up the Black formatter for PyCharm [manually](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea) or
install PyCharm>=2023.2 with [built-in Black support](https://blog.jetbrains.com/pycharm/2023/07/2023-2/#black).

## Run

Run the development server:

```sh
make serve
```

Open `localhost:5001/docs` to make sure the server is up and running.

## Environment Variables

Copy `.env.example` to `.env` and customize it for your environment:

|Variable|Default|Description|
|---|---|---|
|AWS_ACCESS_KEY_ID|NA|AWS credentials with access to Bedrock service|
|AWS_SECRET_ACCESS_KEY|NA|AWS credentials with access to Bedrock service|
|AWS_DEFAULT_REGION||AWS region e.g. `us-east-1`|
|AWS_ASSUME_ROLE_ARN|| AWS assume role arn e.g. `arn:aws:iam::123456789012:role/RoleName`|
|LOG_LEVEL|INFO|Log level. Use DEBUG for dev purposes and INFO in prod|
|AIDIAL_LOG_LEVEL|WARNING|AI DIAL SDK log level|
|DIAL_URL||URL of the core DIAL server. If defined, images generated by Stability are uploaded to the DIAL file storage and attachments are returned with URLs pointing to the images. Otherwise, the images are returned as base64 encoded strings.|
|WEB_CONCURRENCY|1|Number of workers for the server|
|TEST_SERVER_URL|http://0.0.0.0:5001|Server URL used in the integration tests|

## Load balancing

If you use DIAL Core load balancing mechanism, you can provide `extraData` upstream setting with different aws account credentials/regions to use different model deployments:

```json
{
  "upstreams": [
    {
      "extraData": {
        "region": "eu-west-1",
        "aws_access_key_id": "key_id_1",
        "aws_secret_access_key": "access_key_1"
      }
    },
    {
      "extraData": {
        "region": "eu-west-1",
        "aws_access_key_id": "key_id_2",
        "aws_secret_access_key": "access_key_2"
      }
    },
    {
      "extraData": {
        "region": "eu-west-1",
        "aws_assume_role_arn": "arn:aws:iam::123456789012:role/BedrockAccessAdapterRoleName"
      }
    }
  ]
}
```

Supported `extraData` fields:
- `region`
- `aws_access_key_id`
- `aws_secret_access_key`
- `aws_assume_role_arn`

### Docker

Run the server in Docker:

```sh
make docker_serve
```

## Lint

Run the linting before committing:

```sh
make lint
```

To auto-fix formatting issues run:

```sh
make format
```

## Test

Run unit tests locally:

```sh
make test
```

Run unit tests in Docker:

```sh
make docker_test
```

Run integration tests locally:

```sh
make integration_tests
```

## Clean

To remove the virtual environment and build artifacts:

```sh
make clean
```