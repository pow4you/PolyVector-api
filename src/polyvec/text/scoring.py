"""How good a traced word is: does it still read as the word, and does it follow the ink?"""

from __future__ import annotations

import unicodedata
from typing import Optional

import numpy as np

from .geometry import dilate


def normalize_text(text: Optional[str], *, case_sensitive: bool = False) -> str:
    """NFKC, whitespace collapsed, optionally case-folded - before comparing readings."""
    text = unicodedata.normalize("NFKC", text or "")
    text = " ".join(text.split())
    return text if case_sensitive else text.casefold()


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insertions, deletions, substitutions each cost 1)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def text_error(read: Optional[str], reference: Optional[str], *,
               case_sensitive: bool = False) -> float:
    """Character error rate of ``read`` against ``reference``: edits / reference length.

    0 is a perfect reading; 1 or more is no resemblance.  Both empty is 0; an
    empty reference with a non-empty reading is 1.
    """
    a = normalize_text(read, case_sensitive=case_sensitive)
    b = normalize_text(reference, case_sensitive=case_sensitive)
    if not b:
        return 0.0 if not a else 1.0
    return edit_distance(a, b) / len(b)


def shape_error(ink: np.ndarray, traced: np.ndarray, tolerance: int = 1) -> float:
    """How badly a rasterized tracing covers the original ink: ``1 - F1``.

    Precision is the share of traced pixels that lie on (dilated) original
    ink, recall the share of original ink that the (dilated) tracing covers;
    ``tolerance`` is the dilation radius in pixels.  0 is a perfect match, 1
    no overlap at all.

    Args:
        ink, traced: Boolean masks of the same shape.
    """
    ink = np.asarray(ink, dtype=bool)
    traced = np.asarray(traced, dtype=bool)
    if not ink.any() and not traced.any():
        return 0.0
    if not ink.any() or not traced.any():
        return 1.0
    precision = np.count_nonzero(traced & dilate(ink, tolerance)) / np.count_nonzero(traced)
    recall = np.count_nonzero(ink & dilate(traced, tolerance)) / np.count_nonzero(ink)
    if precision + recall == 0:
        return 1.0
    return float(1.0 - 2.0 * precision * recall / (precision + recall))
