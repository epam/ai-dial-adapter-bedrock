"""Guards the protobuf backend the OTLP span exporter runs on.

protobuf publishes its `upb` C extension as manylinux wheels only. The images
are built on `python:3.11-alpine` (musl), where no platform wheel matches and
pip silently installs `protobuf-*-py3-none-any.whl` — the pure-Python
implementation, which serializes roughly 5x slower.

Nothing warns when that happens: the only visible symptom is the OTLP
exporter thread burning CPU, which for a streaming adapter competes with the
event loop relaying chunks. `poetry.toml` pins `no-binary = ["protobuf"]` to
force a source build; this test fails if that is removed, if the sdist stops
building the extension, or if a base image change reintroduces the fallback.
"""

from google.protobuf.internal import api_implementation


def test_protobuf_uses_the_c_extension():
    assert api_implementation.Type() == "upb", (
        "protobuf fell back to its pure-Python implementation. On musl this "
        "happens when the wheel is used instead of a source build — check "
        'that poetry.toml still sets `no-binary = ["protobuf"]`.'
    )


def test_upb_extension_is_importable():
    """The backing module, imported directly, so the failure names itself."""
    from google._upb import _message  # noqa: F401
