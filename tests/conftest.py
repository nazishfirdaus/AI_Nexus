"""Shared pytest configuration.

Real-model tests (marked `real_models` / the opt-in retrieval test) are excluded by
default so CI and normal `pytest` runs never download the ~1.3 GB BGE model. Enable
them locally with ANEXUS_RUN_REAL_TESTS=1.
"""
import os

import pytest

_RUN_REAL = os.getenv("ANEXUS_RUN_REAL_TESTS", "0") == "1"
_REAL_SKIP_REASON = "real-model test; set ANEXUS_RUN_REAL_TESTS=1 to run it"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_models: loads the real BGE embedding model (opt-in, ANEXUS_RUN_REAL_TESTS=1)",
    )


def pytest_collection_modifyitems(config, items):
    if _RUN_REAL:
        return
    skip = pytest.mark.skip(reason=_REAL_SKIP_REASON)
    for item in items:
        if "real_models" in item.keywords or item.name.startswith("test_real_"):
            item.add_marker(skip)