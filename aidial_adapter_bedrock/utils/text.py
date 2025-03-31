def truncate_string(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "..."


def capitalize(s: str) -> str:
    if not s:
        return s
    return s[0].upper() + s[1:]
