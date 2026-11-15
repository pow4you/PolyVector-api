"""Shared fixtures.  Tests needing a compiled backend or PyTorch skip themselves
when those are unavailable, so the plain unit tests always run."""

from __future__ import annotations

import importlib.util

import pytest

from polyvec.build import TOOLS
from polyvec.errors import BackendNotAvailableError
from polyvec.paths import EXAMPLE_IMAGE, EXAMPLE_KEYPOINTS
from polyvec.runners import get_runner


@pytest.fixture(scope="session")
def example_image():
    return EXAMPLE_IMAGE


@pytest.fixture(scope="session")
def example_keypoints():
    return EXAMPLE_KEYPOINTS


@pytest.fixture(scope="session")
def backend_runner():
    """A runner that can execute the backends, or skip."""
    try:
        runner = get_runner("auto", TOOLS[0])
    except BackendNotAvailableError as exc:
        pytest.skip(f"no compiled backend available: {exc}")
    return runner


@pytest.fixture(scope="session")
def needs_torch():
    for module in ("torch", "skimage", "matplotlib"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"optional dependency '{module}' is not installed")
