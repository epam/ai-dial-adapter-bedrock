# Overview

The project implements [AI DIAL API](https://epam-rail.com/dial_api) for language models from [AWS Bedrock](https://aws.amazon.com/bedrock/).

## Supported models

The following models support `POST SERVER_URL/openai/deployments/DEPLOYMENT_NAME/chat/completions` endpoint along with optional support of `/tokenize` and `/truncate_prompt` endpoints:

|Vendor|Model|Deployment name|Modality|`/tokenize`|`/truncate_prompt`|tools/functions support|precise tokenization|
|---|---|---|---|---|---|---|---|
|Anthropic|Claude 2 Sonnet|anthropic.claude-3-sonnet-20240229-v1:0|text-to-text, image-to-text|❌|❌|✅|❌|
|Anthropic|Claude 2 Haiku|anthropic.claude-3-haiku-20240307-v1:0|text-to-text, image-to-text|❌|❌|✅|❌|
|Anthropic|Claude 2 Opus|anthropic.claude-3-opus-20240229-v1:0|text-to-text, image-to-text|❌|❌|✅|❌|
|Anthropic|Claude 2.1|anthropic.claude-v2:1|text-to-text|✅|✅|✅|✅|
|Anthropic|Claude 2|anthropic.claude-v2|text-to-text|✅|✅|❌|✅|
|Anthropic|Claude Instant 1.2|anthropic.claude-instant-v1|text-to-text|✅|✅|❌|❌|
|Meta|Llama 3 Chat 70B Instruct|meta.llama3-70b-instruct-v1:0|text-to-text|✅|✅|❌|❌|
|Meta|Llama 3 Chat 8B Instruct|meta.llama3-8b-instruct-v1:0|text-to-text|✅|✅|❌|❌|
|Meta|Llama 2 Chat 70B|meta.llama2-70b-chat-v1|text-to-text|✅|✅|❌|❌|
|Meta|Llama 2 Chat 13B|meta.llama2-13b-chat-v1|text-to-text|✅|✅|❌|❌|
|Stability AI|SDXL 1.0|stability.stable-diffusion-xl-v1|text-to-image|❌|✅|❌|❌|
|Amazon|Titan Text G1 - Express|amazon.titan-tg1-large|text-to-text|✅|✅|❌|❌|
|AI21 Labs|Jurassic-2 Ultra|ai21.j2-jumbo-instruct|text-to-text|✅|✅|❌|❌|
|AI21 Labs|Jurassic-2 Mid|ai21.j2-grande-instruct|text-to-text|✅|✅|❌|❌|
|Cohere|Command|cohere.command-text-v14|text-to-text|✅|✅|❌|❌|
|Cohere|Command Light|cohere.command-light-text-v14|text-to-text|✅|✅|❌|❌|

The models that support `/truncate_prompt` do also support `max_prompt_tokens` request parameter.

Certain model do not support precise tokenization, because the tokenization algorithm is not known. Instead an approximate tokenization algorithm is used. It conservatively counts every byte in UTF-8 encoding of a string as a single token.

The following models support `SERVER_URL/openai/deployments/DEPLOYMENT_NAME/embeddings` endpoint:

|Model|Deployment name|Modality|
|---|---|---|
|Titan Embeddings G1 – Text v1.2|amazon.titan-embed-text-v1|text-to-embedding|
|Amazon Titan Text Embeddings V2|amazon.titan-embed-text-v2:0|text-to-embedding|

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
|LOG_LEVEL|INFO|Log level. Use DEBUG for dev purposes and INFO in prod|
|AIDIAL_LOG_LEVEL|WARNING|AI DIAL SDK log level|
|DIAL_URL||URL of the core DIAL server. If defined, images generated by Stability are uploaded to the DIAL file storage and attachments are returned with URLs pointing to the images. Otherwise, the images are returned as base64 encoded strings.|
|WEB_CONCURRENCY|1|Number of workers for the server|
|TEST_SERVER_URL|http://0.0.0.0:5001|Server URL used in the integration tests|

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
