"""Keypoints - the extra input PolyVector Flow needs - and the detector interface."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Union

import numpy as np

from ..images import ImageLike


class KeypointType(IntEnum):
    """Kinds of keypoint; the values are the codes of the upstream ``.pts`` format."""

    ENDPOINT = 1       #: a stroke ends here
    JUNCTION = 3       #: strokes meet or cross (T-, Y-, X-junction)
    SHARP_CORNER = 5   #: a single stroke turns sharply


@dataclass(frozen=True)
class Keypoints:
    """A set of labelled keypoints.

    Attributes:
        points: ``(N, 2)`` float array of ``x`` (column), ``y`` (row) pixel
            coordinates - the same axes as :class:`~polyvec.drawing.VectorDrawing`.
        labels: ``(N,)`` int array of :class:`KeypointType` codes.
    """

    points: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)
        labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if len(points) != len(labels):
            raise ValueError(f"{len(points)} points but {len(labels)} labels")
        unknown = set(labels.tolist()) - {int(kind) for kind in KeypointType}
        if unknown:
            raise ValueError(f"unknown keypoint label(s) {sorted(unknown)}; expected 1, 3 or 5")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "labels", labels)

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        counts = ", ".join(f"{len(self.of_type(kind))} {kind.name.lower()}" for kind in KeypointType)
        return f"Keypoints({len(self)}: {counts})"

    def of_type(self, kind: KeypointType) -> np.ndarray:
        """Coordinates ``(M, 2)`` of the keypoints of one kind."""
        return self.points[self.labels == int(kind)]

    # ------------------------------------------------------------ .pts format
    def save(self, path: Union[str, "os.PathLike[str]"]) -> Path:
        """Write the upstream ``.pts`` format.

        First line: the number of points.  Then one ``row column label`` line
        per point - note the *row first* order used by the upstream tools.
        """
        path = Path(path)
        lines = [f"{len(self)}"]
        lines += [f"{y:g} {x:g} {label}" for (x, y), label in zip(self.points, self.labels)]
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return path

    @classmethod
    def load(cls, path: Union[str, "os.PathLike[str]"]) -> "Keypoints":
        """Read an upstream ``.pts`` file (see :meth:`save` for the layout)."""
        tokens = Path(path).read_text(encoding="ascii").split()
        if not tokens:
            raise ValueError(f"{path}: empty keypoint file")
        count = int(float(tokens[0]))
        values = np.array(tokens[1:1 + 3 * count], dtype=np.float64)
        if values.size != 3 * count:
            raise ValueError(f"{path}: expected {count} 'row column label' triples")
        rows = values.reshape(count, 3)
        return cls(points=rows[:, [1, 0]], labels=rows[:, 2].astype(np.int64))

    @classmethod
    def coerce(cls, value: Union["Keypoints", str, "os.PathLike[str]", np.ndarray]) -> "Keypoints":
        """Accept keypoints in any supported form.

        Args:
            value: A :class:`Keypoints`, the path of a ``.pts`` file, or an
                ``(N, 3)`` array of ``x, y, label`` rows.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, os.PathLike)):
            return cls.load(value)
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"keypoint array must be shaped (N, 3) as x, y, label; got {array.shape}")
        return cls(points=array[:, :2], labels=array[:, 2].astype(np.int64))


class KeypointDetector(ABC):
    """Finds endpoints, junctions and sharp corners in a line drawing.

    Implement :meth:`detect` to plug a different detector into
    :class:`~polyvec.methods.flow.PolyVectorFlow`.
    """

    @abstractmethod
    def detect(self, image: ImageLike) -> Keypoints:
        """Detect the keypoints of one drawing.

        Args:
            image: Anything :func:`polyvec.images.load_grayscale` accepts.
        """

    def __call__(self, image: ImageLike) -> Keypoints:
        """Alias for :meth:`detect`."""
        return self.detect(image)
