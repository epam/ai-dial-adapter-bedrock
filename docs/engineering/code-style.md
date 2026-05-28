# Code Style

## Python Typing

Strict typing is required throughout the codebase.

- `Any` or `Unknown` is a blocker unless an explicit comment explains why it cannot be avoided.
- Untyped or partially-typed collections are blockers: use `list[Foo]` not `list`, `dict[str, Bar]` not `dict`, `tuple[int, str]` not `tuple`.
- All public functions and methods must have return type annotations.
- `cast()` calls that paper over a real typing gap instead of fixing it are a code smell.
