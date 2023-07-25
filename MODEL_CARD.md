# Bedrock models

## Amazon Titan

Returns number of used tokens in response:

```json
{ inputTextTokenCount: 10,
, results: [{
    content: "XYZ",
    tokenCount: 20
  }]
}
```

```
return TokenUsage(
    prompt_tokens=resp.inputTextTokenCount,
    completion_tokens=resp.results[0].tokenCount,
)
```

Tokenizer is unknown.

## Anthropic Claude

`anthropic` package provides methods to [calculate tokens](https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/tokens.py).

Could be migrated to the frontend.

## AI21 models

Response contains tokens explicitly as arrays. One needs only compute the len of the arrays:

```
return TokenUsage(
    prompt_tokens=len(resp.prompt.tokens),
    completion_tokens=len(resp.completions[0].data.tokens),
)
```

Tokenizer is unknown.

### Tokenization via AI21 API Key

SDK: `ai21` package calls `tokenize` endpoint to do tokenization (see `ai21/modules/tokenization.py`).

REST: There is API for [tokenization](https://docs.ai21.com/reference/tokenize-ref)

**Tokenization via Bedrock API is currently unsupported** (see `./local_client_ai21.py`).
