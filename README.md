## Overview

The project implements [AI DIAL API](https://epam-rail.com/dial_api) for language models from [AWS Bedrock](https://aws.amazon.com/bedrock/).

## Supported models

The following models support `POST SERVER_URL/openai/deployments/MODEL_NAME/chat/completions` endpoint along with optional support of `/tokenize` and `/truncate_prompt` endpoints:

|Model|Modality|`/tokenize`|`/truncate_prompt`|tools/functions support|precise tokenization|
|---|---|---|---|---|---|
|amazon.titan-tg1-large|text-to-text|✅|✅|❌|❌|
|ai21.j2-grande-instruct|text-to-text|✅|✅|❌|❌|
|ai21.j2-jumbo-instruct|text-to-text|✅|✅|❌|❌|
|anthropic.claude-instant-v1|text-to-text|✅|✅|❌|❌|
|anthropic.claude-v1|text-to-text|✅|✅|❌|✅|
|anthropic.claude-v2|text-to-text|✅|✅|❌|✅|
|anthropic.claude-v2:1|text-to-text|✅|✅|✅|✅|
|anthropic.claude-3-sonnet-20240229-v1:0|text-to-text, image-tot-text|❌|❌|❌|❌|
|stability.stable-diffusion-xl|text-to-image|❌|✅|❌|❌|
|meta.llama2-13b-chat-v1|text-to-text|✅|✅|❌|❌|
|meta.llama2-70b-chat-v1|text-to-text|✅|✅|❌|❌|
|cohere.command-text-v14|text-to-text|✅|✅|❌|❌|
|cohere.command-light-text-v14|text-to-text|✅|✅|❌|❌|

The models that support `/truncate_prompt` do also support `max_prompt_tokens` request parameter.

Certain model do not support precise tokenization, because the tokenization algorithm is not known. Instead an approximate tokenization algorithm is used. It conservatively counts every byte in UTF-8 encoding of a string as a single token.

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
|AWS_DEFAULT_REGION||AWS region e.g. "us-east-1"|
|LOG_LEVEL|INFO|Log level. Use DEBUG for dev purposes and INFO in prod|
|AIDIAL_LOG_LEVEL|WARNING|AI DIAL SDK log level|
|DIAL_USE_FILE_STORAGE|False|Save model artifacts to DIAL File storage (particularly, Stability images are uploaded to the files storage and their base64 encodings are replaced with links to the storage). The creds for the file storage must be passed in `api-key` header of the incoming request. The file storage won't be used if the header isn't set.|
|DIAL_URL||URL of the core DIAL server (required when DIAL_USE_FILE_STORAGE=True)|
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
