import base64
from typing import List

import numpy as np


def vector_to_base64(vector: List[float]) -> str:
    array = np.array(vector, dtype="float32")
    byte_data = array.tobytes()
    base64_encoded = base64.b64encode(byte_data).decode("utf-8")
    return base64_encoded


def base64_to_vector(data: str) -> List[float]:
    return np.frombuffer(base64.b64decode(data), dtype="float32").tolist()
