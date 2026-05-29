# Code Style

## General Principles

Follow the Google Python Style Guide where practical, with project-specific adjustments described below.

Prioritize, in order:

1. Correctness
2. Readability
3. Simplicity
4. Maintainability

Prefer explicit, boring, easy-to-follow code over clever abstractions.

---

## Typing

Strict typing is required throughout the codebase.

* `Any` and `Unknown` are disallowed unless accompanied by a clear justification.
* Collections must always be fully typed:

  * `list[Foo]`, not `list`
  * `dict[str, Bar]`, not `dict`
  * `tuple[int, str]`, not `tuple`
* All public functions and methods must declare return types.
* Avoid `cast()` unless it documents a genuine typing limitation rather than masking a design issue.
* Prefer structured types over untyped dictionaries:

  * `TypedDict`
  * `Protocol`
  * `dataclass`
  * `BaseModel`

Follow SDK/library types where possible.

---

## Formatting

Avoid trailing commas in multiline argument lists when they are not required. This allows Ruff to collapse expressions into a single line if it's possible.

BAD:

```python
async def parse_upstream_config(
    request: fastapi.Request,
) -> UpstreamConfig:
    ...
```

GOOD:

```python
async def parse_upstream_config(
    request: fastapi.Request
) -> UpstreamConfig:
    ...
```

## Naming

Use concise, descriptive, and context-independent names.

Names should be understandable without requiring readers to inspect call sites.

Avoid:

* unnecessary abbreviations
* redundant prefixes/suffixes
* overly verbose names

Before adding a new constants/class/function on the top-level of the module, consider whether it will be used only with that module or should be private. If the latter, prefix it with an underscore.

---

## Testing

Prefer mocking HTTP boundaries rather than internal implementation details.

Use dependency injection so components depend on interfaces/protocols instead of concrete implementations. Prefer this over:

* monkey patching
* replacing entire modules/classes

Tests should validate externally observable behavior, not implementation details.

Avoid using `unittest.mock` to mock HTTP responses. Use `httpx` instead.

Avoid using `monkeypatch` pytest fixture as much as possible in favor of `httpx` and mocking via dependency injection. `monkeypatch` is a powerful tool but can lead to brittle tests if overused.

---

## Design Principles

### Readability

Readability is the primary optimization target.

Prefer simple control flow and explicit behavior over abstraction-heavy designs.

### Composability

Prefer small, reusable components with clear semantics over monolithic classes or functions.
A component’s behavior should ideally be understandable from its name and type signature alone.

### DRY

Avoid duplicated logic, but do not sacrifice readability for aggressive deduplication.

This is especially important in tests. Large amounts of repeated setup quickly make tests bloated and obscure the behavior being validated.

Extract shared setup into helper functions and fixtures, while keeping the test logic itself explicit, focused, and easy to read.

### Law of Demeter

Modules and functions should interact only with their direct dependencies.

BAD:

```python
def foo(bar: Bar):
    foo = bar.x.foo
    # use foo, bar isn't used anymore
```

GOOD:

```python
def foo(x: Foo):
    # use x directly
```
