# Overview

The project implements [AI DIAL API](https://epam-rail.com/dial_api) for language models from [AWS Bedrock](https://aws.amazon.com/bedrock/).

## Supported models

### Chat completion models

The following models support `POST SERVER_URL/openai/deployments/DEPLOYMENT_NAME/chat/completions` endpoint along with an optional support of `POST /tokenize` and `POST /truncate_prompt` endpoints:

Note that a model supports `/truncate_prompt` endpoint if and only if it supports `max_prompt_tokens` request parameter.

|Vendor|Model|Deployment name|Modality|`/tokenize`|`/truncate_prompt`, `max_prompt_tokens`|tools/functions|`/configuration`|Implementation|
|---|---|---|---|---|---|---|---|---|
|Anthropic|Claude 3.7 Sonnet|us.anthropic.claude-3-7-sonnet-20250219-v1:0|(text/image)-to-text|🟡|🟡|✅|✅|Anthropic SDK|
|Anthropic|Claude 3.5 Sonnet|anthropic.claude-3-5-sonnet-20240620-v1:0|(text/image)-to-text|🟡|🟡|✅|✅|Anthropic SDK|
|Anthropic|Claude 3.5 Sonnet 2.0|anthropic.claude-3-5-sonnet-20241022-v2:0|(text/image)-to-text|🟡|🟡|✅|✅|Anthropic SDK|
|Anthropic|Claude 3 Sonnet|anthropic.claude-3-sonnet-20240229-v1:0|(text/image)-to-text|🟡|🟡|✅|✅|Anthropic SDK|
|Anthropic|Claude 3 Haiku|anthropic.claude-3-haiku-20240307-v1:0|(text/image)-to-text|🟡|🟡|✅|✅|Anthropic SDK|
|Anthropic|Claude 3.5 Haiku|anthropic.claude-3-5-haiku-20241022-v1:0|text-to-text|🟡|🟡|✅|✅|Anthropic SDK|
|Anthropic|Claude 3 Opus|anthropic.claude-3-opus-20240229-v1:0|(text/image)-to-text|🟡|🟡|✅|✅|Anthropic SDK|
|DeepSeek|DeepSeek R1|deepseek.r1-v1:0|text-to-text|🟡|🟡|❌|✅|Converse API|
|Anthropic|Claude 2.1|anthropic.claude-v2:1|text-to-text|✅|✅|✅|❌|Bedrock API|
|Anthropic|Claude 2|anthropic.claude-v2|text-to-text|✅|✅|❌|❌|Bedrock API|
|Anthropic|Claude Instant 1.2|anthropic.claude-instant-v1|text-to-text|🟡|🟡|❌|❌|Bedrock API|
|Meta|Llama 3.3 70B Instruct|meta.llama3-3-70b-instruct-v1:0|text-to-text|🟡|🟡|✅|✅|Converse API|
|Meta|Llama 3.2 90B Instruct|us.meta.llama3-2-90b-instruct-v1:0|(text/image)-to-text|🟡|🟡|✅|✅|Converse API|
|Meta|Llama 3.2 11B Instruct|us.meta.llama3-2-11b-instruct-v1:0|(text/image)-to-text|🟡|🟡|❌|✅|Converse API|
|Meta|Llama 3.2 3B Instruct|us.meta.llama3-2-3b-instruct-v1:0|text-to-text|🟡|🟡|❌|✅|Converse API|
|Meta|Llama 3.2 1B Instruct|us.meta.llama3-2-1b-instruct-v1:0|text-to-text|🟡|🟡|❌|✅|Converse API|
|Meta|Llama 3.1 405B Instruct|meta.llama3-1-405b-instruct-v1:0|text-to-text|🟡|🟡|✅|✅|Converse API|
|Meta|Llama 3.1 70B Instruct|meta.llama3-1-70b-instruct-v1:0|text-to-text|🟡|🟡|✅|✅|Converse API|
|Meta|Llama 3.1 8B Instruct|meta.llama3-1-8b-instruct-v1:0|text-to-text|🟡|🟡|❌|✅|Converse API|
|Meta|Llama 3 Chat 70B Instruct|meta.llama3-70b-instruct-v1:0|text-to-text|🟡|🟡|❌|✅|Converse API|
|Meta|Llama 3 Chat 8B Instruct|meta.llama3-8b-instruct-v1:0|text-to-text|🟡|🟡|❌|✅|Converse API|
|Stability AI|SDXL 1.0|stability.stable-diffusion-xl-v1|text-to-image|❌|🟡|❌|❌|Bedrock API|
|Stability AI|SD3 Large 1.0|stability.sd3-large-v1:0|(text/image)-to-image|❌|🟡|❌|✅|Bedrock API|
|Stability AI|Stable Diffusion 3.5 Large|stability.sd3-5-large-v1:0|(text/image)-to-image|❌|🟡|❌|✅|Bedrock API|
|Stability AI|Stable Image Ultra 1.0|stability.stable-image-ultra-v1:0|text-to-image|❌|🟡|❌|✅|Bedrock API|
|Stability AI|Stable Image Core 1.0|stability.stable-image-core-v1:0|text-to-image|❌|🟡|❌|✅|Bedrock API|
|Amazon|Titan Text G1 - Express|amazon.titan-tg1-large|text-to-text|🟡|🟡|❌|❌|Bedrock API|
|Amazon|Nova Pro|amazon.nova-pro-v1:0|(text/image/document)-to-text|🟡|🟡|✅|✅|Converse API|
|Amazon|Nova Lite|amazon.nova-lite-v1:0|(text/image/document)-to-text|🟡|🟡|✅|✅|Converse API|
|Amazon|Nova Micro|amazon.nova-micro-v1:0|text-to-text|🟡|🟡|❌|✅|Converse API|
|AI21 Labs|Jamba 1.5 Large|ai21.jamba-1-5-large-v1:0|text-to-text|🟡|🟡|✅|✅|Converse API|
|AI21 Labs|Jamba 1.5 Mini|ai21.jamba-1-5-mini-v1:0|text-to-text|🟡|🟡|✅|✅|Converse API|
|Cohere|Command R|cohere.command-r-v1:0|(text/document)-to-text|🟡|🟡|✅|✅|Converse API|
|Cohere|Command R+|cohere.command-r-plus-v1:0|(text/document)-to-text|🟡|🟡|✅|✅|Converse API|
|Cohere|Command|cohere.command-text-v14|text-to-text|🟡|🟡|❌|❌|Bedrock API|
|Cohere|Command Light|cohere.command-light-text-v14|text-to-text|🟡|🟡|❌|❌|Bedrock API|

✅, 🟡, and ❌ denote degrees of support of the given feature:

||`/tokenize`, `/truncate_prompt`, `max_prompt_token`|tools/functions|`/configuration`|
|---|---|---|---|
|✅|Fully supported via an official tokenization algorithm|Fully supported via native tools API or official prompts to enable tools|Configurable via the `/configuration` endpoint|
|🟡|Partially supported, because tokenization algorithm wasn't made public by the model vendor.<br>An approximate tokenization algorithm is used instead.<br>It conservatively counts **every byte in UTF-8 encoding of a string as a single token**.|Partially supported, because the model doesn't support tools natively.<br>Prompt engineering is used instead to emulate tools, which may not be very reliable.|Not applicable|
|❌|Not supported|Not supported|Not configurable|

#### Implementation basis

The model adapters differ in what SDKs/APIs they are based on:

1. [Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html) - the single API unifying different chat completion models
2. [Bedrock API](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model.html) - the original Bedrock API for calling chat completion models
3. [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) - the SDK for Anthropic Claude models that provides finer control over the model than the Converse API.

#### Configurable models

Certain models support configuration via the `/configuration` endpoint.
GET request to this endpoint returns the schema of the model configuration in [JSON Schema](https://json-schema.org/) format.
Such models expect that `custom_fields.configuration` field of the `chat/completions` request will contain a JSON value that conforms to the schema.
The `custom_fields.configuration` field is optional iff each field in the schema is optional too.

##### Converse API models

All models based on Bedrock Converse API accept a configuration parameter that enables the [optimized latency mode](https://docs.aws.amazon.com/bedrock/latest/userguide/latency-optimized-inference.html):

|Configuration|Comment|
|---|---|
|`{"performanceConfig": {"latency":"standard"}}`|Default latency|
|`{"performanceConfig": {"latency":"optimized"}}`|Optimized latency|

> [!NOTE]
> Not all Bedrock models actually support the optimized latency mode. Check the [official documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/latency-optimized-inference.html) before use.

> [!WARNING]
> Currently **Claude 3** adapter is based on the [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) that **doesn't** support optimized latency model.

##### Claude 3.7 Sonnet

The model accepts optional configuration that enables [thinking feature](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking):

|Configuration|Comment|
|---|---|
|`{"thinking": {"type": "enabled", "budget_tokens": 1024}}`|Thinking enabled with the given limit on reasoning tokens|
|`{"thinking": {"type": "disabled"}}`|Thinking disabled|

##### Claude models

The Claude models accept an optional list of beta feature flags.
The whole list of flags could be found in the [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/anthropic_beta_param.py).

|Beta flag|Comment|Scope|
|---|---|---|
|`{"betas": ["token-efficient-tools-2025-02-19"]}`|[Token-efficient tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use)|Claude 3.7 Sonnet|
|`{"betas": ["output-128k-2025-02-19"]}`|[Extended output length](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-output-capabilities-beta)|Claude 3.7 Sonnet|

Not every model supports all flags. Refer to the official documentation before utilizing any flags.

##### Stability AI models

The models accept optional configuration with the following fields:

* `aspect_ratio: str` - one of "16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"
* `negative_prompt: str` - a prompt to be used for negative examples

#### Prompt caching

Certain chat completion models support prompt caching via cache breakpoint inserted in tool definitions or request messages.

The adapter supports cache breakpoint for the models based on Converse API and Claude 3 models.

<details><summary>System cache breakpoint</summary>

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Long system prompt",
      "custom_fields": {
        "cache_breakpoint": {}
      }
    },
    {
      "role": "user",
      "content": "user query"
    }
  ]
}
```

</details>

<details><summary>Message cache breakpoint</summary>

```json
{
  "messages": [
    {
      "role": "system",
      "content": "System prompt"
    },
    {
      "role": "user",
      "content": "user query"
      "custom_fields": {
        "cache_breakpoint": {}
      }
    }
  ]
}
```

</details>

<details><summary>Tools cache breakpoint</summary>

```json
{
  "tools": [
    {
      "type": "function",
      "name": "get_weather",
      "description": "Get current temperature for a given location.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City and country e.g. Bogotá, Colombia"
          }
        },
        "required": [
          "location"
        ]
      },
      "custom_fields": {
        "cache_breakpoint": {}
      }
    }
  ],
  "messages": [
    {
      "role": "system",
      "content": "System prompt"
    },
    {
      "role": "user",
      "content": "user query"
    }
  ]
}
```

</details>

> [!NOTE]
> Not every model supports prompt caching. Refer to the [official documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html#prompt-caching-models) before utilizing any cache breakpoints.

#### Cross-region inference

The adapter supports cross-region inference for US, EU and APAC regions for the listed models.

E.g. `Claude 3.5 Sonnet 2.0` model can be accessed via the following deployment names:

1. `anthropic.claude-3-5-sonnet-20241022-v2:0`
2. `us.anthropic.claude-3-5-sonnet-20241022-v2:0`
3. `eu.anthropic.claude-3-5-sonnet-20241022-v2:0`
4. `apac.anthropic.claude-3-5-sonnet-20241022-v2:0`

[Check](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html#inference-profiles-support-system) that your AWS Bedrock account supports cross-region inference for a particular model before using it.

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
|AWS_ASSUME_ROLE_ARN|| AWS assume role ARN e.g. `arn:aws:iam::123456789012:role/RoleName`|
|LOG_LEVEL|INFO|Log level. Use DEBUG for dev purposes and INFO in prod|
|AIDIAL_LOG_LEVEL|WARNING|AI DIAL SDK log level|
|DIAL_URL||URL of the core DIAL server. If defined, images generated by Stability are uploaded to the DIAL file storage and attachments are returned with URLs pointing to the images. Otherwise, the images are returned as base64 encoded strings.|
|WEB_CONCURRENCY|1|Number of workers for the server|
|COMPATIBILITY_MAPPING|{}|A JSON dictionary that maps Bedrock deployments that **aren't supported** by the Adapter to the Bedrock deployments that **are supported** by the Adapter _(see the [Supported models](#supported-models)_ section). Find more details in the [compatibility mode](#compatibility-mode) section.|
|CLAUDE_DEFAULT_MAX_TOKENS|1536|The default value of `max_tokens` chat completion parameter if it is not provided in the request.<br>**:warning: Using the variable is discouraged**.<br>Consider configuring the default in the DIAL Core Config instead as demonstrated in the [example below](#default-max_tokens-for-claude-models).|

### Resource limits

The following environment variables reveal adapter's implementation details and therefore are more susceptible to changes in future than the variables discussed so far.

:warning: Don't use the variables unless you are absolutely sure you know what you are doing.

|Variable|Applicable to models implemented via|Default|Description|
|---|---|---|---|
|ANTHROPIC_MAX_CONNECTIONS|[Anthropic SDK](#implementation-basis)|1000|The maximum number of concurrent requests. Corresponds to `max_connections` [parameter](https://www.python-httpx.org/advanced/resource-limits/) of the HTTPX client.|
|ANTHROPIC_MAX_KEEPALIVE_CONNECTIONS|[Anthropic SDK](#implementation-basis)|100|The maximum number of idle connections kept in a connection pool. Corresponds to the `max_keepalive_connections` [parameter](https://www.python-httpx.org/advanced/resource-limits/) of the HTTPX client.|
|BOTOCORE_CLIENT_MAX_POOL_CONNECTIONS|[Bedrock API & Conserve API](#implementation-basis)|1000|The maximum number of connections kept in a connection pool.|

## Default `max_tokens` for Claude models

Unlike OpenAI models, Claude models require the `max_tokens` parameter in the chat completion request.

We recommend configuring `max_tokens` default value on a per-model basis in the DIAL Core Config, for example:

```json
{
    "models": {
        "dial-claude-deployment-id": {
            "type": "chat",
            "description": "...",
            "endpoint": "...",
            "defaults": {
                "max_tokens": 2048
            }
        }
    }
}
```

If the default is missing in the DIAL Core Config, it will be taken from the `CLAUDE_DEFAULT_MAX_TOKENS` environment variable.
However, we strongly recommend not to rely on this variable and instead configure the defaults in the DIAL Core Config.
Such a **per-model** configuration is operationally cleaner since all the information relevant to tokens _(like pricing and token limits)_ is kept in the same place.

The default value set in the DIAL Core Config takes precedence over the one configured in the adapter.

Make sure the default doesn't exceed Claude's [max output tokens](https://docs.anthropic.com/en/docs/about-claude/models/all-models#model-comparison-table), otherwise, you will receive an error like this one: `The maximum tokens you requested exceeds the model limit of 131072`.

## Compatibility mode

The Adapter supports a predefined list of AWS Bedrock deployments. The [Supported models](#supported-models) section lists the models. These models could be accessed via `/openai/deployments/{deployment_name}/(chat_completions|embeddings)` endpoints. The Adapter won't recognize any other deployment name and will result in 404 error.

Now, suppose AWS Bedrock released a new version of a model, e.g. `anthropic.claude-3-5-sonnet-20250210-v3:0` which is a better version of an older `anthropic.claude-3-5-sonnet-20241022-v2:0` model.

Immediately after the release, the former model is **unsupported** by the Adapter, but the latter is **supported**.
Therefore, the request to `openai/deployments/anthropic.claude-3-5-sonnet-20250210-v3:0/chat/completions` will result in 404 error.

It will take some time for the Adapter to catch up with AWS Bedrock - support the v3 model and publish the release with the fix.

What to do in the meantime? Presumably, the v3 model is backward compatible with v2, so we may try to run v3 in **the compatibility mode** - that is to convince the Adapter to process v3 request as if it's v2 request with the only difference that the final upstream request to AWS Bedrock will be to v3 and not v2.

The `COMPATIBILITY_MAPPING` env variable enables exactly this scenario.

When it's defined like this:

```ini
COMPATIBILITY_MAPPING={"anthropic.claude-3-5-sonnet-20250210-v3:0": "anthropic.claude-3-5-sonnet-20241022-v2:0"}
```

the Adapter will be able to handle requests to `anthropic.claude-3-5-sonnet-20250210-v3:0` deployment.
The requests will be processed by the same pipeline as `anthropic.claude-3-5-sonnet-20241022-v2:0`, but the call to AWS Bedrock will be done to `anthropic.claude-3-5-sonnet-20250210-v3:0` deployment name.

Naturally, this will only work if the APIs of v2 and v3 deployments are compatible:

1. The requests utilizing the modalities supported by both v2 and v3 will work just fine.
2. However, the requests with modalities that are supported by v3 *(e.g. audio)* and aren't supported by v2, won't be processed correctly. You will have to wait until the Adapter supports the v3 deployment natively.

When a version of the Adapter supporting the v3 model is released, you may migrate to it and safely remove the entry from the `COMPATIBILITY_MAPPING` dictionary.

Note that a mapping such as this one would be ineffectual:

```ini
COMPATIBILITY_MAPPING={"anthropic.claude-3-5-sonnet-20250210-v3:0": "stability.stable-image-ultra-v1:0"}
```

since the APIs and capabilities of these two models are drastically different.

## Load balancing

If you use DIAL Core load balancing mechanism, you can provide `extraData` upstream setting with different AWS account credentials/regions to use different model deployments:

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
    },
    {
      "key": "anthropic-api-key"
    }
  ]
}
```

The fields in the extra data override the corresponding environment variables:

|`extraData` field|Env variable|
|---|---|
|`region`|`AWS_DEFAULT_REGION`|
|`aws_access_key_id`|`AWS_ACCESS_KEY_ID`|
|`aws_secret_access_key`|`AWS_SECRET_ACCESS_KEY`|
|`aws_assume_role_arn`|`AWS_ASSUME_ROLE_ARN`|

## Authentication

### AWS Bedrock

Authentication with AWS Bedrock is configured either:

1. globally via `AWS_*` environment vars, or
2. on a [per upstream basis](#load-balancing) via `upstreams.extraData` fields in DIAL Core Config.

### Anthropic API

Claude>=3 deployments could be accessed via API key. The API keys should be configured per-upstream in the DIAL Core config:

```json
{
  "models": {
    "claude-3-5-sonnet-20241022": {
      "endpoint": "...",
      "upstreams": [
        {
          "key": "anthropic-api-key"
        }
      ]
    }
  }
}
```

Keep in mind that the same Anthropic models have [different identifiers](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) in Anthropic API and AWS Bedrock.

E.g. `anthropic.claude-3-5-sonnet-20241022-v2:0` in AWS Bedrock corresponds to `claude-3-5-sonnet-20241022` in Anthropic API.

The adapter uses deployment identifiers from **AWS Bedrock**.
Therefore, in order to use Anthropic API model you need to map its identifier to a corresponding identifier in AWS Bedrock using the [compatibility mapping](#compatibility-mode):

```ini
COMPATIBILITY_MAPPING={"claude-3-5-sonnet-20241022":"anthropic.claude-3-5-sonnet-20241022-v2:0"}
```

Otherwise, the adapter will return 404 on requests to `claude-3-5-sonnet-20241022`.

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
