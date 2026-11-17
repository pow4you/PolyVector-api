"""The in-process backends (``polyvec._native``) that binary wheels ship.

These run wherever the extension modules are installed (a wheel, or a source
build that found the C++ dependencies) and skip otherwise.  They check that

* ``runner="auto"`` picks the extension module,
* the in-process and the subprocess paths give the same strokes,
* repeated calls are independent (upstream keeps its state in globals),
* the programs' console output is captured instead of printed,
* the helper process recovers from a timeout.

The comparison against the command-line programs (the file-based subprocess
path through ``LocalRunner`` / ``DockerRunner``) is opt-in, because it needs
those programs: set ``POLYVEC_TEST_CLI_RUNNER=local`` (or ``docker``).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import polyvec
from polyvec import _native
from polyvec.errors import VectorizationError
from polyvec.paths import EXAMPLE_IMAGE, EXAMPLE_KEYPOINTS
from polyvec.runners import InProcessRunner, SubprocessRunner, get_runner

#: Fewer flow steps than upstream's 1500 keep the tests quick; every comparison
#: below uses the same value on both sides.
FLOW_ITERATIONS = 300

needs_field = pytest.mark.skipif(not _native.available("polyvec-field"),
                                 reason="polyvec._native._field is not installed")
needs_flow = pytest.mark.skipif(not _native.available("polyvec-flow"),
                                reason="polyvec._native._flow is not installed")
pytestmark = pytest.mark.native


def field(runner, **options):
    return polyvec.PolyVectorField(runner=runner, **options)


def flow(runner, **options):
    return polyvec.PolyVectorFlow(runner=runner, flow_iterations=FLOW_ITERATIONS, **options)


def assert_same_strokes(actual, expected, atol=0.0):
    assert len(actual) == len(expected), f"{len(actual)} strokes vs {len(expected)}"
    for index, (a, b) in enumerate(zip(actual.strokes, expected.strokes)):
        assert a.shape == b.shape, f"stroke {index}: {a.shape} vs {b.shape}"
        np.testing.assert_allclose(a, b, rtol=0, atol=atol, err_msg=f"stroke {index}")


@pytest.fixture(scope="module")
def field_in_process():
    return field("inprocess").vectorize(EXAMPLE_IMAGE)


@pytest.fixture(scope="module")
def flow_in_process():
    return flow("inprocess").vectorize(EXAMPLE_IMAGE, keypoints=EXAMPLE_KEYPOINTS)


# ----------------------------------------------------------------- selection
@needs_field
def test_auto_prefers_the_extension_module(monkeypatch):
    monkeypatch.delenv("POLYVEC_RUNNER", raising=False)
    assert type(get_runner("auto", "polyvec-field")) is InProcessRunner
    assert type(get_runner("subprocess", "polyvec-field")) is SubprocessRunner


# ---------------------------------------------------------------- field (2019)
@needs_field
def test_field_in_process(field_in_process):
    drawing = field_in_process
    assert drawing.metadata["runner"] == "inprocess"
    assert len(drawing) > 10
    assert all(stroke.ndim == 2 and stroke.shape[1] == 2 for stroke in drawing)


@needs_field
def test_field_in_process_matches_subprocess(field_in_process):
    in_helper = field("subprocess").vectorize(EXAMPLE_IMAGE)
    assert in_helper.metadata["runner"] == "subprocess"
    assert_same_strokes(in_helper, field_in_process)


@needs_field
def test_field_repeated_calls_are_independent(field_in_process):
    vectorizer = field("inprocess")
    vectorizer.vectorize(np.full((40, 60), 255, np.uint8))      # different size and content
    assert_same_strokes(vectorizer.vectorize(EXAMPLE_IMAGE), field_in_process)


@needs_field
def test_field_blank_image_has_no_strokes():
    assert len(field("inprocess").vectorize(np.full((32, 32), 255, np.uint8))) == 0


@needs_field
def test_console_output_is_captured(capfd):
    vectorizer = field("inprocess")
    vectorizer.vectorize(EXAMPLE_IMAGE)
    printed = capfd.readouterr()
    assert printed.out == "" and printed.err == ""
    assert "COMPONENT" in vectorizer.last_run.output     # upstream's std::cout ...
    assert "/" in vectorizer.last_run.output             # ... and its printf progress


@needs_field
def test_subprocess_timeout_kills_and_recovers(field_in_process):
    with pytest.raises(VectorizationError, match="did not finish"):
        field("subprocess", timeout=0.01).vectorize(EXAMPLE_IMAGE)
    assert_same_strokes(field("subprocess").vectorize(EXAMPLE_IMAGE), field_in_process)


@needs_field
def test_bad_input_raises_in_both_modes():
    tiny = np.zeros((1, 1), np.uint8)
    for runner in ("inprocess", "subprocess"):
        with pytest.raises(VectorizationError, match="at least 2x2"):
            field(runner).vectorize(tiny)


# ----------------------------------------------------------------- flow (2021)
@needs_flow
def test_flow_in_process(flow_in_process):
    drawing = flow_in_process
    assert drawing.metadata["runner"] == "inprocess"
    assert len(drawing) > 10
    assert len(drawing.metadata["keypoints"]) == len(polyvec.Keypoints.load(EXAMPLE_KEYPOINTS))


@needs_flow
def test_flow_console_output_is_captured(capfd):
    vectorizer = polyvec.PolyVectorFlow(runner="inprocess", flow_iterations=20)
    vectorizer.vectorize(EXAMPLE_IMAGE, keypoints=EXAMPLE_KEYPOINTS)
    printed = capfd.readouterr()
    assert printed.out == "" and printed.err == ""
    assert "Starting FLOW" in vectorizer.last_run.output     # upstream
    assert "HiGHS" in vectorizer.last_run.output             # the solver, via the shim


@needs_flow
def test_flow_in_process_matches_subprocess(flow_in_process):
    in_helper = flow("subprocess").vectorize(EXAMPLE_IMAGE, keypoints=EXAMPLE_KEYPOINTS)
    assert_same_strokes(in_helper, flow_in_process)


@needs_flow
@needs_field
def test_flow_and_field_interleave(flow_in_process, field_in_process):
    """The two modules are separate libraries; neither disturbs the other."""
    assert_same_strokes(field("inprocess").vectorize(EXAMPLE_IMAGE), field_in_process)
    assert_same_strokes(flow("inprocess").vectorize(EXAMPLE_IMAGE, keypoints=EXAMPLE_KEYPOINTS),
                        flow_in_process)


# ----------------------------------------------- against the command-line programs
CLI_RUNNER = os.environ.get("POLYVEC_TEST_CLI_RUNNER")
needs_cli = pytest.mark.skipif(
    not CLI_RUNNER, reason="set POLYVEC_TEST_CLI_RUNNER=local|docker to compare with the programs")

#: The programs write coordinates to SVG with 6 significant digits.
SVG_PRECISION = 2e-3
#: Flow curves are iterated hundreds of times, so builds with different OpenCV,
#: Eigen or compiler versions drift apart slightly (measured: 0.02 px between
#: Ubuntu 22.04's packages and the wheel's vcpkg build); the curves chosen are
#: the same.
FLOW_PRECISION = 5e-2


@needs_cli
@needs_field
def test_field_matches_command_line_program(field_in_process):
    from_program = field(CLI_RUNNER).vectorize(EXAMPLE_IMAGE)
    assert from_program.metadata["runner"] == CLI_RUNNER
    assert_same_strokes(field_in_process, from_program, atol=SVG_PRECISION)


@needs_cli
@needs_flow
def test_flow_matches_command_line_program(flow_in_process):
    from_program = flow(CLI_RUNNER).vectorize(EXAMPLE_IMAGE, keypoints=EXAMPLE_KEYPOINTS)
    assert from_program.metadata["runner"] == CLI_RUNNER
    assert_same_strokes(flow_in_process, from_program, atol=FLOW_PRECISION)
