"""Turning "an image" - a file, a NumPy array or a PIL image - into the one form
the vectorizers work on: an 8-bit grayscale array with dark ink on light paper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from .errors import InvalidImageError

#: Anything :func:`load_grayscale` accepts.
ImageLike = Union[str, "os.PathLike[str]", np.ndarray, Image.Image]


def load_grayscale(image: ImageLike) -> np.ndarray:
    """Convert any supported input to an ``(H, W)`` ``uint8`` array.

    Accepted inputs:

    * a path to an image file (anything Pillow can open),
    * a :class:`PIL.Image.Image`,
    * a NumPy array shaped ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` (RGB) or
      ``(H, W, 4)`` (RGBA).  ``uint8`` is used as is; ``bool`` maps False/True
      to 0/255; other integers are scaled from their full range; floats are
      taken as 0..1 (or 0..255 if any value exceeds 1).

    Colour is reduced with the ITU-R 601 luma weights (the same ones OpenCV
    uses in the upstream programs) and transparency is flattened onto white.
    """
    if isinstance(image, (str, os.PathLike)):
        path = Path(image)
        if not path.is_file():
            raise InvalidImageError(f"image file not found: {path}")
        try:
            with Image.open(path) as opened:
                return _from_pil(opened)
        except OSError as exc:
            raise InvalidImageError(f"cannot read image file {path}: {exc}") from exc
    if isinstance(image, Image.Image):
        return _from_pil(image)
    if isinstance(image, np.ndarray):
        return _from_pil(_array_to_pil(image))
    raise InvalidImageError(
        f"unsupported image type {type(image).__name__}; "
        "expected a file path, a NumPy array or a PIL image"
    )


def save_png(gray: np.ndarray, path: Union[str, "os.PathLike[str]"]) -> Path:
    """Write an ``(H, W)`` ``uint8`` array as a grayscale PNG and return the path."""
    path = Path(path)
    Image.fromarray(np.ascontiguousarray(gray, dtype=np.uint8)).save(path, format="PNG")
    return path


def remap_threshold(gray: np.ndarray, threshold: int, target: int) -> np.ndarray:
    """Stretch gray levels so that ``threshold`` lands on ``target``.

    The upstream programs separate ink from paper at a compiled-in gray level
    (``target``).  Rather than recompiling, the image is remapped piecewise
    linearly - 0 stays 0, ``threshold`` becomes ``target``, 255 stays 255 - which
    is the contrast adjustment the upstream README recommends doing by hand.
    """
    if not 0 < threshold < 255:
        raise ValueError(f"ink threshold must be between 1 and 254, got {threshold}")
    levels = np.arange(256, dtype=np.float64)
    lookup = np.interp(levels, [0.0, float(threshold), 255.0], [0.0, float(target), 255.0])
    return np.rint(lookup).astype(np.uint8)[gray]


# --------------------------------------------------------------------------- helpers
def _from_pil(image: Image.Image) -> np.ndarray:
    """Flatten transparency onto white and reduce to 8-bit gray."""
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        paper = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(paper, rgba).convert("RGB")
    elif image.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        image = _array_to_pil(np.asarray(image))
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.ndim != 2 or gray.size == 0:
        raise InvalidImageError(f"image has no pixels (shape {gray.shape})")
    return gray


def _array_to_pil(array: np.ndarray) -> Image.Image:
    """Validate an array's shape and bring its values to 8 bits."""
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    # gray, RGB or RGBA - Pillow picks the matching mode from the shape
    if not (array.ndim == 2 or (array.ndim == 3 and array.shape[2] in (3, 4))):
        raise InvalidImageError(
            f"expected an array shaped (H, W), (H, W, 1), (H, W, 3) or (H, W, 4); got {array.shape}"
        )
    if array.size == 0:
        raise InvalidImageError(f"image has no pixels (shape {array.shape})")
    return Image.fromarray(np.ascontiguousarray(_to_uint8(array)))


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scaled = (array.astype(np.float64) - info.min) / float(info.max - info.min)
        return np.rint(scaled * 255.0).astype(np.uint8)
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise InvalidImageError("image contains NaN or infinite values")
        scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
        return np.rint(np.clip(array.astype(np.float64) * scale, 0.0, 255.0)).astype(np.uint8)
    raise InvalidImageError(f"unsupported array dtype {array.dtype}")
