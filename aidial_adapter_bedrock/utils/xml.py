from collections import defaultdict
from typing import Any, List, overload

from defusedxml import ElementTree

Arg = None | str | List[str | None]


@overload
def _arg_to_str(arg: List[str | None]) -> str: ...


@overload
def _arg_to_str(arg: None) -> None: ...


@overload
def _arg_to_str(arg: str) -> str: ...


def _arg_to_str(arg: Arg) -> str | None:
    if isinstance(arg, list):
        return "\n".join([x for x in arg if x is not None])
    return arg


@overload
def tag(name: str, arg: List[str | None]) -> str: ...


@overload
def tag(name: str, arg: None) -> None: ...


@overload
def tag(name: str, arg: str) -> str: ...


def tag(name: str, arg: Arg) -> str | None:
    content = _arg_to_str(arg)
    return f"<{name}>{content}</{name}>" if content is not None else None


@overload
def tag_nl(name: str, arg: List[str | None]) -> str: ...


@overload
def tag_nl(name: str, arg: None) -> None: ...


@overload
def tag_nl(name: str, arg: str) -> str: ...


def tag_nl(name: str, arg: Arg) -> str | None:
    content = _arg_to_str(arg)
    if content is None:
        return None
    content = "\n" if content == "" else f"\n{content}\n"
    return f"<{name}>{content}</{name}>"


def _xml_to_dict(t) -> dict[str, Any]:
    d = {t.tag: {}}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(_xml_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(("@" + k, v) for k, v in t.attrib.items())
    if t.text and t.text.strip():
        if children or t.attrib:
            d[t.tag]["#text"] = t.text
        else:
            d[t.tag] = t.text
    return d


def parse_xml(string: str) -> Any:
    return _xml_to_dict(ElementTree.fromstring(string))
