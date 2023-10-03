# Bedrock models

## Amazon Titan

### Tokenization

Returns number of used tokens in response:

```json
{
  "inputTextTokenCount": 10,
  "results": [
    {
      "content": "foo",
      "tokenCount": 20
    }
  ]
}
```

```python
return TokenUsage(
    prompt_tokens=resp.inputTextTokenCount,
    completion_tokens=resp.results[0].tokenCount,
)
```

Tokenizer is unknown.

### Token limits

Tokens limits are unknown.

Experimentally found tokens limits (see `./test/find_token_limits.py`):

- amazon.titan-tg1-large: 4096

## Anthropic Claude

### Tokenization

`anthropic` package provides methods to [calculate tokens](https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/tokens.py).

```python
from anthropic.tokenizer import count_tokens

return TokenUsage(
    prompt_tokens=count_tokens(query.prompt),
    completion_tokens=count_tokens(resp.completion),
)
```

The tokenizer could be migrated to the frontend.

### Streaming

The streaming is supported by [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/streaming.py), but it's not supported by the Bedrock API.

### Token limits

As per [documentation](https://docs.anthropic.com/claude/docs/introduction-to-prompt-design#prompt-length) the limits aren't strictly specified:

```
The maximum prompt length that Claude can see is its context window. Claude's context window is currently ~75,000 words / ~100,000 tokens / ~340,000 Unicode characters.

Right now when this context window is exceeded in the API Claude is likely to return an incoherent response. We apologize for this “sharp edge”.
```

However [pricing page](https://www.anthropic.com/pricing) explicitly says that context windows is 100k tokens.

So:

- max input tokens: 100k
- max outputs tokens: unknown

However, it doesn't match with experimentally found tokens limits (see `./test/find_token_limits.py`):

- anthropic.claude-v1: 12288
- anthropic.claude-instant-v1: 12288

REST: [completion call](https://docs.anthropic.com/claude/reference/complete_post)

## AI21 models

### Rate limits

See [rate limits](https://docs.ai21.com/docs/rate-limits).

### Token limits

REST (and Bedrock presumably): max context window = [8191 tokens](https://docs.ai21.com/reference/j2-complete-ref)

AWS: see [token limits](https://docs.ai21.com/docs/choosing-the-right-instance-type-for-amazon-sagemaker-models#foundation-models) for various model instances

```
The context window acts as a threshold for the amount of tokens in the prompt and the completion, namely: prompt + completion <= context window.
```

Experimentally found tokens limits confirm the documentation (see `./test/find_token_limits.py`):

- ai21.j2-grande-instruct: 8191
- ai21.j2-jumbo-instruct: 8191

### Tokenization

Response contains tokens explicitly as arrays. One needs only compute the len of the arrays:

```python
return TokenUsage(
    prompt_tokens=len(resp.prompt.tokens),
    completion_tokens=len(resp.completions[0].data.tokens),
)
```

Tokenizer is [unknown](https://docs.ai21.com/docs/tokenizer-tokenization):

> AI21 Studio uses a large token dictionary (250K)

Token counting is only possible using AI21 API Key.

**Tokenization via Bedrock API is currently unsupported** (see `./local_client_ai21.py`).

SDK: `ai21` package calls `tokenize` endpoint to do tokenization (see `ai21/modules/tokenization.py`).

REST: There is API for [tokenization](https://docs.ai21.com/reference/tokenize-ref)

## Stable diffusion

There is no meaningful size limit on the prompt size.
