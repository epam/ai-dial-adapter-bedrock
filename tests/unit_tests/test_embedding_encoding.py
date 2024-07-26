import base64
from typing import List

import numpy as np
import pytest

from aidial_adapter_bedrock.embedding.encoding import (
    base64_to_vector,
    vector_to_base64,
)

vectors = [
    [],
    [0.0],
    [1.0, -1.0, 3.5, 2.2],
    [
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
    [1.123456789] * 1000,
]


@pytest.mark.parametrize("vector", vectors)
def test_to_base64_and_back(vector: List[float]):
    actual_vector = base64_to_vector(vector_to_base64(vector))

    assert np.allclose(
        vector, actual_vector, equal_nan=True
    ), f"Expected: {vector}, Actual: {actual_vector}"


@pytest.mark.parametrize("vector", vectors)
def test_to_vector_and_back(vector: List[float]):
    str = base64.b64encode(np.array(vector, dtype="float32").tobytes()).decode(
        "utf-8"
    )

    actual_str = vector_to_base64(base64_to_vector(str))

    assert str == actual_str, f"Expected: {str}, Actual: {actual_str}"
