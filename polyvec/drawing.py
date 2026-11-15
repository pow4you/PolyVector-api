"""The result type shared by every vectorizer: a set of polylines."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from xml.sax.saxutils import quoteattr

import numpy as np

_SVG_TAG = re.compile(r"<svg\b[^>]*>", re.IGNORECASE | re.DOTALL)
_POLYLINE_TAG = re.compile(r"<polyline\b[^>]*>", re.IGNORECASE | re.DOTALL)
_ATTRIBUTE = r"""\b{name}\s*=\s*(["'])(.*?)\1"""
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class VectorDrawing:
    """A vectorized line drawing.

    Attributes:
        strokes: One ``(N, 2)`` float array per curve, ``N >= 2``.  Columns are
            ``x`` (to the right) and ``y`` (downwards) in pixels of the input
            image, with the origin at the image's top-left corner, so the
            centre of the pixel in row ``i``, column ``j`` is ``(j + 0.5, i + 0.5)``.
        width: Width of the source image in pixels.
        height: Height of the source image in pixels.
        method: Registry name of the vectorizer that produced the drawing.
        metadata: Method-specific extras (run time, native log, keypoints, ...).
    """

    strokes: List[np.ndarray]
    width: int
    height: int
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ container API
    def __len__(self) -> int:
        return len(self.strokes)

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.strokes)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.strokes[index]

    def __repr__(self) -> str:
        return (
            f"VectorDrawing({len(self.strokes)} strokes, {self.num_points} points, "
            f"{self.width}x{self.height}, method={self.method!r})"
        )

    # --------------------------------------------------------------- properties
    @property
    def num_points(self) -> int:
        """Total number of vertices over all strokes."""
        return sum(len(stroke) for stroke in self.strokes)

    def lengths(self) -> np.ndarray:
        """Arc length of every stroke, in pixels."""
        return np.array(
            [float(np.linalg.norm(np.diff(stroke, axis=0), axis=1).sum()) for stroke in self.strokes]
        )

    # ---------------------------------------------------------------------- SVG
    def to_svg(self, stroke: str = "black", stroke_width: float = 1.0,
               background: Optional[str] = None) -> str:
        """Serialise as a standalone SVG document (one ``<polyline>`` per stroke).

        Args:
            stroke: Any SVG colour.
            stroke_width: Line width in pixels.
            background: Optional fill colour for a full-size backdrop rectangle.
        """
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
        ]
        if background:
            lines.append(f'  <rect width="100%" height="100%" fill={quoteattr(background)}/>')
        lines.append(
            f'  <g fill="none" stroke={quoteattr(stroke)} stroke-width="{stroke_width:g}" '
            'stroke-linecap="round" stroke-linejoin="round">'
        )
        for points in self.strokes:
            coords = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
            lines.append(f'    <polyline points="{coords}"/>')
        lines += ["  </g>", "</svg>", ""]
        return "\n".join(lines)

    def save(self, path: Union[str, "os.PathLike[str]"], **svg_options: Any) -> Path:
        """Write the drawing as an SVG file; ``svg_options`` go to :meth:`to_svg`."""
        path = Path(path)
        path.write_text(self.to_svg(**svg_options), encoding="utf-8")
        return path

    @classmethod
    def from_svg(cls, source: Union[str, "os.PathLike[str]"], *,
                 width: Optional[int] = None, height: Optional[int] = None,
                 method: str = "") -> "VectorDrawing":
        """Read the polylines of an SVG file or SVG text.

        Understands what the upstream programs (and :meth:`to_svg`) write: plain
        ``<polyline points="x,y x,y ...">`` elements in pixel coordinates.  It is
        not a general SVG reader - paths, transforms and so on are ignored.

        Upstream output may contain ``nan`` coordinates; those points are
        dropped, and strokes left with fewer than two points are discarded.

        Args:
            source: Path to an ``.svg`` file, or the SVG markup itself.
            width, height: Override the size declared on the ``<svg>`` element.
            method: Stored on the result.
        """
        text = str(source)
        if "<" not in text:
            text = Path(source).read_text(encoding="utf-8", errors="replace")

        if width is None or height is None:
            declared = _declared_size(text)
            width = declared[0] if width is None else width
            height = declared[1] if height is None else height

        strokes: List[np.ndarray] = []
        for tag in _POLYLINE_TAG.finditer(text):
            points = _parse_points(_attribute(tag.group(0), "points") or "")
            if len(points) >= 2:
                strokes.append(points)
        return cls(strokes=strokes, width=int(width), height=int(height), method=method)


# --------------------------------------------------------------------------- helpers
def _attribute(tag: str, name: str) -> Optional[str]:
    match = re.search(_ATTRIBUTE.format(name=name), tag, re.IGNORECASE | re.DOTALL)
    return match.group(2) if match else None


def _declared_size(svg_text: str) -> "tuple[int, int]":
    """Width and height from the ``<svg>`` element (``305px`` -> 305); 0 if absent."""
    tag = _SVG_TAG.search(svg_text)
    size = []
    for name in ("width", "height"):
        value = _attribute(tag.group(0), name) if tag else None
        number = _NUMBER.match(value.strip()) if value else None
        size.append(int(round(float(number.group(0)))) if number else 0)
    return size[0], size[1]


def _parse_points(points_attribute: str) -> np.ndarray:
    """``"1,2 3.5,4 nan,nan"`` -> ``[[1, 2], [3.5, 4]]`` (non-finite points dropped)."""
    values: List[float] = []
    for token in points_attribute.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            values.append(float("nan"))
    if len(values) % 2:
        values.pop()
    points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    return points[np.isfinite(points).all(axis=1)]
