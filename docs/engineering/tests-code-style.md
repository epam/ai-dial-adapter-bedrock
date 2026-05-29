# Code Style for tests

Prefer mocking HTTP boundaries rather than internal implementation details.

Use dependency injection so components depend on interfaces/protocols instead of concrete implementations. Prefer this over:

* monkey patching
* replacing entire modules/classes

Tests should validate externally observable behavior, not implementation details.

Avoid using `unittest.mock` to mock HTTP responses. Use `httpx` instead.

Avoid using `monkeypatch` pytest fixture as much as possible in favor of `httpx` and mocking via dependency injection. `monkeypatch` is a powerful tool but can lead to brittle tests if overused.
