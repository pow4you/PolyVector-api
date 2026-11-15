"""Vectorization of Line Drawings via PolyVector Fields (Bessmeltsev & Solomon 2019)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..native import NativeJob, NativeVectorizer


class PolyVectorField(NativeVectorizer):
    """Frame-field-based vectorization of clean line drawings.

    Wraps the authors' C++ implementation (https://github.com/bmpix/PolyVectorization,
    bundled in ``polyvec/vendor/pv``): a smooth PolyVector frame field is fitted
    to the drawing, traced into curves, and the traces are merged into a graph
    whose topology gives the final strokes.

    The program needs nothing but the image.  Its tuning constants are compiled
    in (``vendor/pv/src/Params.h``); the one that matters in practice - the
    ink/paper threshold - is exposed here as ``ink_threshold``.

    Example::

        from polyvec import PolyVectorField

        drawing = PolyVectorField().vectorize("sketch.png")
        drawing.save("sketch.svg")

    Args:
        **options: See :class:`~polyvec.native.NativeVectorizer`
            (``runner``, ``timeout``, ``ink_threshold``, ``workdir``, ``invert``).
    """

    name = "field"
    aliases = ("polyvector-field", "polyvector", "bessmeltsev2019")
    title = "PolyVector Fields"
    reference = ("M. Bessmeltsev and J. Solomon. Vectorization of Line Drawings via "
                 "PolyVector Fields. ACM Transactions on Graphics 38(1), 2019.")
    tool = "polyvec-field"

    def _stage(self, workdir: Path, image_name: str, gray: np.ndarray, **inputs: Any) -> NativeJob:
        if inputs:
            raise TypeError(f"{type(self).__name__}.vectorize() got unexpected "
                            f"argument(s): {', '.join(sorted(inputs))}")
        # Upstream usage: `polyvector_thing FILE`, which writes FILE.svg next to it.
        return NativeJob(args=[image_name], output=f"{image_name}.svg")
