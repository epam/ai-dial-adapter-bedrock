from typing import List


def remove_prefix(prefix: str, string: str) -> str:
    if string.startswith(prefix):
        return string[len(prefix) :]
    return string


def stop_at(stop_sequences: List[str], string: str) -> str:
    min_index = len(string)
    for stop_sequence in stop_sequences:
        if stop_sequence in string:
            min_index = min(min_index, string.index(stop_sequence))
    return string[:min_index]


def ensure_not_empty(default: str, string: str) -> str:
    return default if string == "" else string
