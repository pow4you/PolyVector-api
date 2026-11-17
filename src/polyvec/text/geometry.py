"""Stroke geometry for the text pipeline: cutting at word boxes, rasterizing, measuring."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

#: ``(x0, y0, x1, y1)``
Rect = Sequence[float]


def inside_any(points: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """``(N,)`` bool: which points lie inside (or on the edge of) at least one rect."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    rects = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
    if len(rects) == 0:
        return np.zeros(len(points), dtype=bool)
    x, y = points[:, :1], points[:, 1:]
    return ((x >= rects[:, 0]) & (x <= rects[:, 2]) & (y >= rects[:, 1]) & (y <= rects[:, 3])).any(axis=1)


def _segment_crossings(a: np.ndarray, b: np.ndarray, rects: np.ndarray) -> List[float]:
    """Parameters ``t`` in (0, 1) where segment ``a + t (b - a)`` enters or leaves a rect
    (Liang-Barsky clipping against each rect)."""
    d = b - a
    ts: List[float] = []
    for x0, y0, x1, y1 in rects:
        lo, hi = -np.inf, np.inf
        for p, q0, q1 in ((d[0], x0 - a[0], x1 - a[0]), (d[1], y0 - a[1], y1 - a[1])):
            if p == 0:
                if q0 > 0 or q1 < 0:              # parallel and outside this slab
                    lo, hi = 1.0, 0.0
                    break
                continue
            t0, t1 = sorted((q0 / p, q1 / p))
            lo, hi = max(lo, t0), min(hi, t1)
        if lo <= hi:
            ts.extend(t for t in (lo, hi) if 0.0 < t < 1.0)
    return sorted(set(ts))


def split_by_rects(stroke: np.ndarray,
                   rects: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Cut a polyline where it crosses the boundary of a set of rectangles.

    Returns:
        ``(inside, outside)``: the pieces inside the union of ``rects`` and the
        pieces outside it.  Crossing points are computed exactly (also for
        segments that pass through a rect without a vertex inside it) and end
        both neighbouring pieces, so the pieces meet.  Pieces shorter than two
        points are dropped.
    """
    stroke = np.asarray(stroke, dtype=np.float64).reshape(-1, 2)
    rects = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
    if len(stroke) == 0:
        return [], []
    if len(rects) == 0:
        return [], ([stroke] if len(stroke) >= 2 else [])

    inside: List[np.ndarray] = []
    outside: List[np.ndarray] = []
    state = bool(inside_any(stroke[:1], rects)[0])
    current = [stroke[0]]
    for i in range(1, len(stroke)):
        a, b = stroke[i - 1], stroke[i]
        ts = _segment_crossings(a, b, rects) + [1.0]
        previous = 0.0
        for t in ts:
            mid_state = bool(inside_any(a + (previous + t) / 2.0 * (b - a), rects)[0])
            if mid_state != state:                # the boundary was crossed at `previous`
                (inside if state else outside).append(np.array(current))
                current = [a + previous * (b - a)]
                state = mid_state
            current.append(a + t * (b - a))       # t == 1 is b itself
            previous = t
    (inside if state else outside).append(np.array(current))

    def keep(pieces: List[np.ndarray]) -> List[np.ndarray]:
        return [piece for piece in pieces
                if len(piece) >= 2 and np.linalg.norm(np.diff(piece, axis=0), axis=1).sum() > 1e-6]

    return keep(inside), keep(outside)


def clip_strokes(strokes: Sequence[np.ndarray], rects: np.ndarray,
                 keep: str = "inside") -> List[np.ndarray]:
    """The parts of ``strokes`` inside (``keep="inside"``) or outside the union of ``rects``."""
    index = 0 if keep == "inside" else 1
    return [piece for stroke in strokes for piece in split_by_rects(stroke, rects)[index]]


def stroke_length(strokes: Sequence[np.ndarray]) -> float:
    """Total arc length of a set of polylines."""
    return float(sum(np.linalg.norm(np.diff(np.asarray(s, dtype=np.float64), axis=0), axis=1).sum()
                     for s in strokes if len(s) >= 2))


def estimate_pen_width(ink: np.ndarray, strokes: Sequence[np.ndarray],
                       minimum: float = 1.0) -> float:
    """Pen width that makes ``strokes`` cover about as much as ``ink``: ink area / centreline length."""
    length = stroke_length(strokes)
    if length <= 0:
        return minimum
    return max(minimum, float(np.count_nonzero(ink)) / length)


def rasterize(strokes: Sequence[np.ndarray], width: int, height: int, pen_width: float,
              offset: Tuple[float, float] = (0.0, 0.0), scale: float = 1.0) -> np.ndarray:
    """Draw polylines as dark ink on white paper: an ``(height, width)`` ``uint8`` image.

    Points are mapped with ``(p - offset) * scale`` first, so strokes in page
    coordinates can be drawn into an upscaled crop.  Single-point strokes become dots.
    """
    canvas = Image.new("L", (int(width), int(height)), 255)
    draw = ImageDraw.Draw(canvas)
    radius = max(pen_width, 1.0) / 2.0
    line_width = max(1, int(round(pen_width)))
    for stroke in strokes:
        points = (np.asarray(stroke, dtype=np.float64).reshape(-1, 2) - offset) * scale - 0.5
        if len(points) == 0:
            continue
        xy = [tuple(point) for point in points]
        if len(xy) >= 2:
            draw.line(xy, fill=0, width=line_width, joint="curve")
        for x, y in (xy[0], xy[-1]):            # round caps
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=0)
    return np.asarray(canvas, dtype=np.uint8)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a boolean mask by ``radius`` pixels (square structuring element)."""
    if radius <= 0:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.filter(ImageFilter.MaxFilter(2 * int(radius) + 1))) > 0


def upscale(gray: np.ndarray, scale: float) -> np.ndarray:
    """Resize an ``(H, W)`` ``uint8`` image by ``scale`` with Lanczos resampling."""
    if scale == 1:
        return gray
    height, width = gray.shape
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.asarray(Image.fromarray(gray).resize(size, Image.LANCZOS), dtype=np.uint8)
