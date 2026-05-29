# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI adapter that bridges the [AI DIAL API](https://dialx.ai/dial_api) (OpenAI-compatible protocol) to AWS Bedrock. It translates DIAL/OpenAI chat completion and embedding requests into AWS Bedrock, Converse API, or Anthropic SDK calls, and translates responses back.

## Commands

```sh
# Setup
make install             # poetry install (all dep groups)

# Dev server
make serve               # uvicorn on port 5001 with --reload and .env

# Test
make test                # pytest tests/unit_tests/ (via nox)
make integration_tests   # pytest tests/integration_tests/

# Lint / Format
make lint                # ruff check + ruff format --check + pyright
make format              # ruff check --fix + ruff format
```

After making source code changes, always run `make lint && make test` to verify nothing is broken.

Find the additional specialized guides:

- [General Code Style](.claude/engineering/general-code-style.md) for coding standards (typing, module placement, visibility, protocol conformance).
- [Tests Code Style](.claude/engineering/tests-code-style.md) for coding standards for tests specifically. Note that the general code style principles apply as well.
- [Direct Output Style](.claude/output-styles/direct.md) for guidance on how to communicate when describing code changes, issues, or blockers.

Tools: **Poetry** for deps, **nox** for task runner, **ruff** (line-length 80, py311), **pyright** (basic mode), **pytest-asyncio** + **respx** for tests.

## Architecture

### Request Flow

```
POST /openai/deployments/{deployment_id}/chat/completions
  → BedrockChatCompletion (chat_completion.py)
      → resolve_upstream_deployment_id_from_request()   # compat mapping
      → parse_upstream_config()                          # AWS creds or API key
      → get_bedrock_adapter()                            # factory dispatch
          ├── anthropic_claude.create_adapter()          # Anthropic SDK (Claude 3/4 default)
          ├── ConverseAdapterFactory.create()            # AWS Converse API (most models)
          └── StabilityV2Adapter.create()                # Direct Bedrock invoke_model
```

### Key Modules

| Path | Role |
|------|------|
| `aidial_adapter_bedrock/app.py` | FastAPI app factory, route registration |
| `aidial_adapter_bedrock/deployments.py` | `ChatCompletionDeployment` and `EmbeddingsDeployment` enums with all supported model IDs |
| `aidial_adapter_bedrock/chat_completion.py` | `BedrockChatCompletion` — main handler (chat, tokenize, truncate_prompt, configuration) |
| `aidial_adapter_bedrock/bedrock.py` | Boto3 client wrapper with TTL-cached client creation |
| `aidial_adapter_bedrock/upstream_config.py` | `UpstreamConfig` union — parses AWS creds or API key from request headers |
| `aidial_adapter_bedrock/llm/model/adapter.py` | `get_bedrock_adapter()` — central factory dispatching by deployment enum |
| `aidial_adapter_bedrock/llm/converse/` | Converse API adapter (Meta Llama, Amazon Nova, Cohere, AI21, DeepSeek) |
| `aidial_adapter_bedrock/llm/model/claude/` | Claude adapter via Anthropic SDK |
| `aidial_adapter_bedrock/llm/model/stability/` | Stability AI image generation |
| `aidial_adapter_bedrock/llm/decorator/` | Composable decorators: caching, message preprocessing, replication |
| `aidial_adapter_bedrock/embedding/` | Embedding adapters (Amazon Titan text/image, Cohere) |
| `aidial_adapter_bedrock/utils/adapter_deployments.py` | `resolve_upstream_deployment_id` — compatibility mapping logic |
| `aidial_adapter_bedrock/anthropic_api.py` | Minimal FastAPI sub-app for Anthropic API passthrough at `/anthropic` |

### Key Abstractions

**`UpstreamConfig`** = `ApiKeyUpstreamConfig | CloudUpstreamConfig`: parsed from `x-upstream-extra-data` / `x-upstream-key` headers. Determines whether requests route to Anthropic API directly (API key) or to AWS Bedrock (IAM / assume-role).

**Three backend implementations**:
- **Anthropic SDK** (`AsyncAnthropicBedrock` or `AsyncAnthropic`) — default for Claude 3/4 models
- **Converse API** (boto3) — for Llama, Nova, Cohere, AI21, DeepSeek; also for Claude when guardrails/latency optimization are configured
- **Direct `invoke_model`** — Stability AI only

**`ChatCompletionDecorator`** (`llm/decorator/base.py`): decorators compose in order — `preprocess_messages_decorator` → `replicator_decorator` → `caching_decorator`.

**Compatibility mapping**: `COMPATIBILITY_MAPPING` env var and `compatible_model_id` in `extraData` header allow routing unsupported model IDs through a supported model's pipeline. The `AdapterDeployment[T]` type holds both `upstream_deployment_id` (what client sent) and `reference_deployment_id` (canonical enum for dispatch).

### Adding a New Model

1. Add an entry to `ChatCompletionDeployment` enum in `deployments.py`
2. Wire it in `get_bedrock_adapter()` (`llm/model/adapter.py`) to the appropriate factory
3. If it needs a new Converse-based adapter, extend `ConverseAdapterFactory`; otherwise implement `ChatCompletionAdapter`
