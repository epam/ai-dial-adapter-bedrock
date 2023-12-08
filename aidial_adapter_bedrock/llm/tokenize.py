def default_tokenize(string: str) -> int:
    """
    The number of bytes is a proxy for the number of tokens for
    models which do not provide any means to count tokens.

    Any token number estimator should satisfy the following requirements:
    1. Overestimation of number of tokens is allowed.
    It's ok to truncate the chat history more than necessary.
    2. Underestimation of number of tokens is prohibited.
    It's wrong to leave the chat history as is when the truncation was actually required.
    """
    return len(string.encode("utf-8"))
