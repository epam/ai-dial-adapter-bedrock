# Code Style for tests

Prefer mocking HTTP boundaries rather than internal implementation details.

Use dependency injection so components depend on interfaces/protocols instead of concrete implementations. Prefer this over:

* monkey patching
* replacing entire modules/classes

Tests should validate externally observable behavior, not implementation details.

Avoid using `unittest.mock` to mock HTTP responses. Use `respx` instead.

Avoid using `monkeypatch` pytest fixture as much as possible in favor of `respx` and mocking via dependency injection. `monkeypatch` is a powerful tool but can lead to brittle tests if overused.

## DRY

Avoid duplicated logic in tests that repeats itself in each test especially in the arrange and assert parts. Bloated tests are hard to read and understand the contract that is actually being tested.

To remove the bloat, introduce helper functions and fixtures to abstract away common setup, but keep the test logic itself straightforward and focused on the specific behavior being validated.

## Examples

Refer to `./tests/unit_tests/test_claude_api.py` as an example of how to write tests that follow these guidelines.
