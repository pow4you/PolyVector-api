"""polyvec - line-drawing vectorization with PolyVector fields, behind one small API.

Two published methods, each wrapped (not re-implemented) around its authors' code:

* :class:`PolyVectorField` - Bessmeltsev & Solomon, *Vectorization of Line
  Drawings via PolyVector Fields* (2019)
* :class:`PolyVectorFlow` - Puhachov et al., *Keypoint-Driven Line Drawing
  Vectorization via PolyVector Flow* (2021)

Quick start::

    import polyvec

    drawing = polyvec.vectorize("sketch.png", method="flow")   # or a NumPy array
    for stroke in drawing:          # (N, 2) arrays of x, y pixel coordinates
        ...
    drawing.save("sketch.svg")

Class hierarchy::

    Vectorizer                 image in -> VectorDrawing out
    ├── NativeVectorizer       ... via a compiled upstream program
    │   ├── PolyVectorField
    │   └── PolyVectorFlow     + KeypointDetector (PyramidalKeypointDetector)
    └── TextVectorizer         words traced separately around one of the above
                               + WordDetector (docTR, PaddleOCR, Tesseract)

    Runner                     where the compiled program executes
    ├── LocalRunner            a binary built on this machine
    └── DockerRunner           a container (image built automatically)
"""

from __future__ import annotations

from typing import Any

from .base import Vectorizer, available_methods, get_vectorizer, get_vectorizer_class
from .drawing import VectorDrawing
from .errors import (
    BackendBuildError,
    BackendNotAvailableError,
    InvalidImageError,
    MissingDependencyError,
    PolyvecError,
    VectorizationError,
)
from .images import ImageLike, load_grayscale
from .keypoints import KeypointDetector, Keypoints, KeypointType, PyramidalKeypointDetector
from .methods import PolyVectorField, PolyVectorFlow
from .native import NativeJob, NativeVectorizer
from .runners import DockerRunner, LocalRunner, Runner, RunResult, get_runner
from .text import TextVectorizer, WordDetector, Words, WritingDirection

__version__ = "0.1.0"

__all__ = [
    "vectorize",
    # vectorizers
    "Vectorizer", "NativeVectorizer", "NativeJob", "PolyVectorField", "PolyVectorFlow",
    "TextVectorizer",
    "available_methods", "get_vectorizer", "get_vectorizer_class",
    # results and inputs
    "VectorDrawing", "ImageLike", "load_grayscale",
    # keypoints
    "KeypointDetector", "PyramidalKeypointDetector", "Keypoints", "KeypointType",
    # text
    "WordDetector", "Words", "WritingDirection",
    # runners
    "Runner", "RunResult", "LocalRunner", "DockerRunner", "get_runner",
    # errors
    "PolyvecError", "InvalidImageError", "MissingDependencyError",
    "BackendNotAvailableError", "BackendBuildError", "VectorizationError",
]


def vectorize(image: ImageLike, method: str = "flow", **options: Any) -> VectorDrawing:
    """Vectorize a line drawing in one call.

    Args:
        image: Path to an image file, a NumPy array or a PIL image.
        method: ``"flow"`` (PolyVector Flow, 2021), ``"field"`` (PolyVector
            Fields, 2019) or ``"text"`` (words traced separately around one of
            them, :class:`TextVectorizer`); see :func:`available_methods`.
        **options: Constructor options of the chosen vectorizer, e.g.
            ``runner="docker"``, ``ink_threshold=190`` or ``flow_iterations=500``.

    Returns:
        The strokes as a :class:`VectorDrawing`.

    For repeated use, create the vectorizer once (``PolyVectorFlow(...)``) and
    call its ``vectorize`` method; that keeps the keypoint network loaded.
    """
    return get_vectorizer(method, **options).vectorize(image)
