"""Keypoint-Driven Line Drawing Vectorization via PolyVector Flow (Puhachov et al. 2021)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from ..drawing import VectorDrawing
from ..images import ImageLike, load_grayscale
from ..keypoints import KeypointDetector, Keypoints, PyramidalKeypointDetector
from ..native import NativeJob, NativeVectorizer

#: Forms in which keypoints may be supplied to :meth:`PolyVectorFlow.vectorize`.
KeypointsLike = Union[Keypoints, str, "os.PathLike[str]", np.ndarray]


class PolyVectorFlow(NativeVectorizer):
    """Keypoint-driven vectorization with PolyVector flow.

    Wraps the authors' implementation
    (https://github.com/ivanpuhachov/line-drawing-vectorization-polyvector-flow,
    bundled in ``polyvec/vendor/pvflow``), which works in two stages:

    1. a neural network marks the drawing's *keypoints* - stroke endpoints,
       junctions and sharp corners (:class:`~polyvec.keypoints.PyramidalKeypointDetector`,
       upstream's model with its released weights);
    2. a C++ program connects the keypoints with candidate curves, snaps the
       curves to the drawing by flowing them along a PolyVector frame field, and
       selects the subset that best explains the ink.

    Both stages happen inside :meth:`vectorize`; the detected keypoints are
    returned in ``drawing.metadata["keypoints"]``.

    Example::

        from polyvec import PolyVectorFlow

        flow = PolyVectorFlow()
        drawing = flow.vectorize("sketch.png")              # detects keypoints itself
        drawing = flow.vectorize("sketch.png", keypoints="sketch.pts")   # or reuse some

    Args:
        keypoint_detector: Detector used when no keypoints are passed to
            :meth:`vectorize`.  Default: a :class:`PyramidalKeypointDetector`
            (requires the ``flow`` extra, i.e. PyTorch).
        flow_iterations: Number of PolyVector-flow steps applied to each
            candidate curve.  1500 is upstream's default; fewer is faster and
            less precisely snapped.
        **options: See :class:`~polyvec.native.NativeVectorizer`
            (``runner``, ``timeout``, ``ink_threshold``, ``workdir``, ``invert``).
    """

    name = "flow"
    aliases = ("polyvector-flow", "puhachov2021")
    title = "PolyVector Flow"
    reference = ("I. Puhachov, W. Neveu, E. Chien and M. Bessmeltsev. Keypoint-Driven Line "
                 "Drawing Vectorization via PolyVector Flow. ACM Transactions on Graphics "
                 "40(6) (SIGGRAPH Asia), 2021.")
    tool = "polyvec-flow"

    #: File names used inside the scratch directory.
    KEYPOINTS_NAME = "input.pts"
    OUTPUT_NAME = "output.svg"

    def __init__(self, *, keypoint_detector: Optional[KeypointDetector] = None,
                 flow_iterations: int = 1500, **options: Any) -> None:
        super().__init__(**options)
        if flow_iterations < 1:
            raise ValueError(f"flow_iterations must be positive, got {flow_iterations}")
        self.keypoint_detector = keypoint_detector or PyramidalKeypointDetector()
        self.flow_iterations = int(flow_iterations)

    def vectorize(self, image: ImageLike,
                  keypoints: Optional[KeypointsLike] = None) -> VectorDrawing:
        """Vectorize one drawing.

        Args:
            image: Path to an image file, a NumPy array or a PIL image.
            keypoints: Keypoints to use instead of running the detector: a
                :class:`~polyvec.keypoints.Keypoints`, the path of an upstream
                ``.pts`` file, or an ``(N, 3)`` array of ``x, y, label`` rows
                (labels: 1 endpoint, 3 junction, 5 sharp corner).

        Returns:
            The strokes; ``metadata["keypoints"]`` holds the keypoints used.
        """
        return super().vectorize(image, keypoints=keypoints)

    def detect_keypoints(self, image: ImageLike) -> Keypoints:
        """Run only stage 1 - e.g. to inspect or edit the keypoints before
        passing them back to :meth:`vectorize`.

        The image goes through the same preprocessing as in :meth:`vectorize`,
        so the result is exactly what stage 2 would have received.
        """
        return self.keypoint_detector.detect(self.preprocess(load_grayscale(image)))

    def _stage(self, workdir: Path, image_name: str, gray: np.ndarray,
               keypoints: Optional[KeypointsLike] = None, **inputs: Any) -> NativeJob:
        if inputs:
            raise TypeError(f"{type(self).__name__}.vectorize() got unexpected "
                            f"argument(s): {', '.join(sorted(inputs))}")
        if keypoints is None:
            points = self.keypoint_detector.detect(gray)
        else:
            points = Keypoints.coerce(keypoints)
        points.save(workdir / self.KEYPOINTS_NAME)

        # Upstream usage: `vectorize IMAGE POINTS OUTPUT.svg ITERATIONS`; giving
        # an output path is also what keeps its debug window from opening.
        return NativeJob(
            args=[image_name, self.KEYPOINTS_NAME, self.OUTPUT_NAME, str(self.flow_iterations)],
            output=self.OUTPUT_NAME,
            metadata={"keypoints": points},
        )
