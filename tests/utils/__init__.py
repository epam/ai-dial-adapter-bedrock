import pytest

pytest.register_assert_rewrite("tests.utils.openai")
pytest.register_assert_rewrite("tests.utils.json")
pytest.register_assert_rewrite("tests.utils.exception")
