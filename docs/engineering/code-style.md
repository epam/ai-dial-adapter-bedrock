# Code Style

## General guidelines

By default try to follow Google [Python Style Guide](https://google.github.io/styleguide/pyguide.html), with some adjustments for our specific use case and preferences. The main principles are:

## Python Typing

Strict typing is required throughout the codebase.

- `Any` or `Unknown` is a blocker unless an explicit comment explains why it cannot be avoided.
- Untyped or partially-typed collections are blockers: use `list[Foo]` not `list`, `dict[str, Bar]` not `dict`, `tuple[int, str]` not `tuple`.
- All public functions and methods must have return type annotations.
- `cast()` calls that paper over a real typing gap instead of fixing it are a code smell.
- Follow the types provided by SDKs and libraries where possible. Where not possible - use `TypedDict`, `Protocol`, `dataclass` or `BaseModel` to create your own types instead of `dict` or `Any`.

## Naming Conventions

- Strive for succinct and descriptive names. Avoid unnecessary verbosity.
- Namings should be context-independent where possible. Meaning that a distinct function should be possible to understand from its name and code alone, without needing to consult its call sites.

## Testing

- Try a much as possible to mock only HTTP requests.
- To facilitate the above principle, use Dependency Injection to that class depend on interfaces/protocols that could be mocked (instead of monkey patching or using fixtures to replace entire modules or classes).

## Misc

- Keep the code DRY
- Keep the code simple and easy to read - that the MAIN metric to optimize the code against
- Follow principle of least knowledge ("Law of Demeter") - modules should only interact with their direct dependencies, not with the internals of other modules. That translates to method too:

BAD:

```python
def foo(bar: Bar):
  foo = x.foo
  # do something with foo, bar is unused after this
```

GOOD:

```python
def foo(x: Foo):
  # do something with foo
```
