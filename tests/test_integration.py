"""End-to-end tests against the real compiled programs and the real network.

They run when a backend is available (local binaries, or Docker - in which case
the first run builds the image) and skip otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import polyvec
from polyvec import Keypoints, PolyVectorField, PolyVectorFlow, PyramidalKeypointDetector


def _ink_coverage(drawing, gray, radius=3.0):
    """Fraction of stroke vertices lying within ``radius`` px of an ink pixel."""
    ink_rows, ink_cols = np.nonzero(gray < 165)
    ink = np.c_[ink_cols + 0.5, ink_rows + 0.5]
    points = np.concatenate(drawing.strokes)
    nearest = np.sqrt(((points[:, None, :] - ink[None, :, :]) ** 2).sum(-1)).min(1)
    return float((nearest <= radius).mean())


@pytest.mark.backend
def test_field_on_example(example_image, backend_runner):
    drawing = PolyVectorField(runner=backend_runner).vectorize(example_image)

    assert drawing.method == "field"
    assert (drawing.width, drawing.height) == Image.open(example_image).size
    assert len(drawing) > 10
    assert _ink_coverage(drawing, polyvec.load_grayscale(example_image)) > 0.95


@pytest.mark.backend
def test_field_file_and_array_inputs_agree(example_image, backend_runner):
    vectorizer = PolyVectorField(runner=backend_runner)
    from_file = vectorizer.vectorize(example_image)
    from_array = vectorizer.vectorize(np.asarray(Image.open(example_image).convert("RGB")))

    assert len(from_file) == len(from_array)
    for a, b in zip(from_file, from_array):
        np.testing.assert_allclose(a, b)


@pytest.mark.backend
def test_flow_with_upstream_keypoints(example_image, example_keypoints, backend_runner):
    flow = PolyVectorFlow(runner=backend_runner)
    drawing = flow.vectorize(example_image, keypoints=example_keypoints)

    assert drawing.method == "flow"
    assert len(drawing) > 10
    assert len(drawing.metadata["keypoints"]) == len(Keypoints.load(example_keypoints))
    assert _ink_coverage(drawing, polyvec.load_grayscale(example_image)) > 0.95


@pytest.mark.torch
def test_keypoint_detector(example_image, needs_torch):
    detector = PyramidalKeypointDetector(device="cpu")
    keypoints = detector.detect(example_image)

    assert 20 <= len(keypoints) <= 60
    width, height = Image.open(example_image).size
    assert keypoints.points[:, 0].max() < width and keypoints.points[:, 1].max() < height
    assert set(detector.last_heatmaps) == {"detection", "endpoint", "sharp_corner", "junction"}
    assert detector.last_heatmaps["detection"].shape == (height, width)


@pytest.mark.backend
@pytest.mark.torch
def test_flow_end_to_end(example_image, backend_runner, needs_torch):
    """The one-call path: detector and C++ program chained inside vectorize()."""
    drawing = polyvec.vectorize(example_image, method="flow", runner=backend_runner,
                                flow_iterations=300)

    assert len(drawing) > 10
    assert len(drawing.metadata["keypoints"]) > 0
    assert _ink_coverage(drawing, polyvec.load_grayscale(example_image)) > 0.9
