"""Writing direction: which way strokes, words and lines run.

A direction has a *primary* axis (the way characters follow each other in a
line) and a perpendicular *secondary* axis (the way lines follow each other).
``"ltr-ttb"`` - left to right, lines top to bottom - is the default and covers
Latin, Cyrillic, Greek ...; ``"rtl-ttb"`` is Arabic or Hebrew, ``"ttb-rtl"``
traditional vertical CJK, and so on: every combination of a horizontal and a
vertical axis, in either order, is allowed.

The ordering rules (:func:`orient_stroke`, :func:`order_strokes`,
:func:`order_boxes`) make a tracing read like writing: the first stroke starts
at the corner where writing starts (top-left for ``ltr-ttb``), strokes follow
each other along the primary axis, and every polyline's points run the way
the pen would have moved - along the primary axis, or along the secondary one
for strokes that run across it (the stem of a ``l`` is drawn top to bottom).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Sequence, Tuple, Union

import numpy as np

#: Axis name -> unit vector in image coordinates (x right, y down).
_AXES: Dict[str, Tuple[float, float]] = {
    "ltr": (1.0, 0.0), "rtl": (-1.0, 0.0), "ttb": (0.0, 1.0), "btt": (0.0, -1.0),
}
_ALIASES = {"l2r": "ltr", "r2l": "rtl", "t2b": "ttb", "b2t": "btt"}


@dataclass(frozen=True)
class WritingDirection:
    """A primary (within a line) and secondary (line to line) writing direction.

    Args:
        primary: ``"ltr"``, ``"rtl"``, ``"ttb"`` or ``"btt"`` (``l2r`` etc. also accepted).
        secondary: One of the two axes perpendicular to ``primary``.

    Use :meth:`parse` for the ``"ltr-ttb"`` string form.
    """

    primary: str = "ltr"
    secondary: str = "ttb"

    DEFAULT: ClassVar[str] = "ltr-ttb"

    def __post_init__(self) -> None:
        primary = _ALIASES.get(self.primary.lower(), self.primary.lower())
        secondary = _ALIASES.get(self.secondary.lower(), self.secondary.lower())
        for axis in (primary, secondary):
            if axis not in _AXES:
                raise ValueError(f"unknown writing axis {axis!r}; use one of "
                                 f"{', '.join(_AXES)} (or l2r, r2l, t2b, b2t)")
        if abs(float(np.dot(_AXES[primary], _AXES[secondary]))) > 0:
            raise ValueError(f"the secondary direction must be perpendicular to the primary one; "
                             f"got {primary}-{secondary}")
        object.__setattr__(self, "primary", primary)
        object.__setattr__(self, "secondary", secondary)

    @classmethod
    def parse(cls, value: Union[str, "WritingDirection", Sequence[str]]) -> "WritingDirection":
        """``"ltr-ttb"``, ``"rtl_ttb"``, ``("ttb", "rtl")`` or a :class:`WritingDirection`."""
        if isinstance(value, WritingDirection):
            return value
        if isinstance(value, str):
            parts = value.replace("_", "-").replace("/", "-").replace(" ", "-").split("-")
            parts = [part for part in parts if part]
        else:
            parts = list(value)
        if len(parts) == 1:          # primary only: lines run top to bottom, columns right to left
            primary = _ALIASES.get(parts[0].lower(), parts[0].lower())
            parts.append("rtl" if primary in ("ttb", "btt") else "ttb")
        if len(parts) != 2:
            raise ValueError(f"writing direction must look like 'ltr-ttb'; got {value!r}")
        return cls(parts[0], parts[1])

    def __str__(self) -> str:
        return f"{self.primary}-{self.secondary}"

    @property
    def axes(self) -> np.ndarray:
        """``(2, 2)`` array: the primary and the secondary unit vector (x, y)."""
        return np.array([_AXES[self.primary], _AXES[self.secondary]], dtype=np.float64)

    def coordinates(self, points: np.ndarray) -> np.ndarray:
        """``(N, 2)`` image points -> ``(N, 2)`` (along primary, along secondary)."""
        return np.asarray(points, dtype=np.float64).reshape(-1, 2) @ self.axes.T


# --------------------------------------------------------------------------- strokes
def orient_stroke(stroke: np.ndarray, direction: WritingDirection, *,
                  across: float = 2.0, closed_tolerance: float = 1.5) -> np.ndarray:
    """Return ``stroke`` with its points running the way the pen would move.

    * An open stroke runs forward along the primary axis; if it mostly runs
      across it (its secondary extent is more than ``across`` times its primary
      extent, e.g. a vertical bar in horizontal writing) it runs forward along
      the secondary axis instead.
    * A closed stroke (ends within ``closed_tolerance`` pixels) is rotated to
      start at its first point in writing order (topmost, then leftmost, for
      ``ltr-ttb``) and keeps its turning sense.
    """
    stroke = np.asarray(stroke, dtype=np.float64)
    if len(stroke) < 2:
        return stroke
    local = direction.coordinates(stroke)
    if np.linalg.norm(stroke[-1] - stroke[0]) <= closed_tolerance and len(stroke) > 3:
        ring = stroke[:-1]
        start = int(np.lexsort((local[:-1, 0], local[:-1, 1]))[0])
        return np.vstack([ring[start:], ring[:start], ring[start:start + 1]])
    delta = local[-1] - local[0]
    extent = local.max(axis=0) - local.min(axis=0)
    axis = 1 if extent[1] > across * max(extent[0], 1e-9) else 0
    if delta[axis] < 0 or (delta[axis] == 0 and delta[1 - axis] < 0):
        return stroke[::-1].copy()
    return stroke


def order_strokes(strokes: Sequence[np.ndarray], direction: WritingDirection,
                  **orient_options: float) -> List[np.ndarray]:
    """Orient every stroke (:func:`orient_stroke`) and sort them into writing order.

    Strokes follow each other by where they begin along the primary axis (the
    leading edge of their extent), ties broken along the secondary axis - so
    for ``ltr-ttb`` the first stroke is the leftmost, topmost one and the
    rest follow to the right.
    """
    oriented = [orient_stroke(stroke, direction, **orient_options) for stroke in strokes]

    def key(stroke: np.ndarray) -> Tuple[float, float]:
        local = direction.coordinates(stroke)
        return float(local[:, 0].min()), float(local[0, 1])

    return sorted(oriented, key=key)


# --------------------------------------------------------------------------- word boxes
def order_boxes(boxes: np.ndarray, direction: WritingDirection, *,
                line_overlap: float = 0.5) -> np.ndarray:
    """Indices that put word boxes (``x0, y0, x1, y1`` rows) into reading order.

    Boxes are grouped into lines - a box joins a line when their extents along
    the secondary axis overlap by at least ``line_overlap`` of the smaller one -
    lines are ordered along the secondary axis and the words of each line
    along the primary axis.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    corners = np.stack([boxes[:, [0, 1]], boxes[:, [2, 3]]], axis=1)          # (N, 2, 2)
    local = direction.coordinates(corners.reshape(-1, 2)).reshape(-1, 2, 2)
    lo, hi = local.min(axis=1), local.max(axis=1)                             # (N, 2) each

    lines: List[List[int]] = []
    spans: List[List[float]] = []
    for index in np.argsort((lo[:, 1] + hi[:, 1]) / 2.0, kind="stable"):
        a, b = lo[index, 1], hi[index, 1]
        for line, span in zip(lines, spans):
            overlap = min(b, span[1]) - max(a, span[0])
            if overlap >= line_overlap * min(b - a, span[1] - span[0]):
                line.append(int(index))
                span[0], span[1] = min(a, span[0]), max(b, span[1])
                break
        else:
            lines.append([int(index)])
            spans.append([a, b])

    order = sorted(range(len(lines)), key=lambda i: (spans[i][0] + spans[i][1]) / 2.0)
    return np.array([index for i in order
                     for index in sorted(lines[i], key=lambda j: (lo[j, 0], lo[j, 1]))], dtype=int)
