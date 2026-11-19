"""Word boxes - where on a page the text is - and the detector interface.

Borrowed from ``inksight``.  The text pipeline (:class:`polyvec.TextVectorizer`)
is ``find the words -> trace each on an upscaled crop -> put the strokes back``.
A :class:`WordDetector` does the first step, and - through :meth:`recognize` -
also reads rasterized results back for verification.  Subclasses that set a
``name`` register themselves and can be chosen by name
(``TextVectorizer(word_detector="paddleocr")``, ``--detector``).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from ...images import ImageLike, load_rgb


@dataclass(frozen=True)
class Words:
    """A set of word boxes, in reading order.

    Attributes:
        boxes: ``(N, 4)`` float array of ``x0, y0, x1, y1`` pixel coordinates
            (left, top, right, bottom) - the same axes as
            :class:`~polyvec.drawing.VectorDrawing`.
        texts: ``N`` strings the detector read (None where it reads nothing);
            the reference for re-reading the traced result.
        scores: ``(N,)`` detector confidences, NaN where unknown.
    """

    boxes: np.ndarray
    texts: Tuple[Optional[str], ...] = ()
    scores: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        count = len(boxes)
        texts = tuple(self.texts) if len(self.texts) else (None,) * count
        scores = (np.full(count, np.nan) if self.scores is None
                  else np.asarray(self.scores, dtype=np.float64).reshape(-1))
        if len(texts) != count or len(scores) != count:
            raise ValueError(f"{count} boxes but {len(texts)} texts and {len(scores)} scores")
        if count and np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("every box needs x1 > x0 and y1 > y0")
        object.__setattr__(self, "boxes", boxes)
        object.__setattr__(self, "texts", texts)
        object.__setattr__(self, "scores", scores)

    def __len__(self) -> int:
        return len(self.boxes)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[str]]]:
        """``(box, text)`` pairs."""
        return iter(zip(self.boxes, self.texts))

    def __repr__(self) -> str:
        read = sum(text is not None for text in self.texts)
        return f"Words({len(self)} boxes, {read} with text)"

    @classmethod
    def whole(cls, width: int, height: int, text: Optional[str] = None) -> "Words":
        """One box covering a whole ``width`` x ``height`` image."""
        return cls(boxes=[[0, 0, width, height]], texts=(text,))

    # ------------------------------------------------------------ JSON format
    def to_list(self) -> List[Dict[str, Any]]:
        """``[{"box": [x0, y0, x1, y1], "text": ..., "score": ...}, ...]``."""
        return [{"box": [float(v) for v in box], "text": text,
                 "score": None if np.isnan(score) else float(score)}
                for box, text, score in zip(self.boxes, self.texts, self.scores)]

    def save(self, path: Union[str, "os.PathLike[str]"]) -> Path:
        """Write the boxes as JSON (see :meth:`to_list`)."""
        path = Path(path)
        path.write_text(json.dumps(self.to_list(), indent=1) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_list(cls, items: Sequence[Dict[str, Any]]) -> "Words":
        return cls(boxes=[item["box"] for item in items],
                   texts=tuple(item.get("text") for item in items),
                   scores=[np.nan if item.get("score") is None else item["score"] for item in items])

    @classmethod
    def load(cls, path: Union[str, "os.PathLike[str]"]) -> "Words":
        """Read a file written by :meth:`save`."""
        return cls.from_list(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def coerce(cls, value: Union["Words", str, "os.PathLike[str]", np.ndarray, Sequence[Any]]) -> "Words":
        """Accept word boxes in any supported form.

        Args:
            value: A :class:`Words`, the path of a JSON file written by
                :meth:`save`, an ``(N, 4)`` array of ``x0, y0, x1, y1`` rows, or
                a list of ``{"box": ..., "text": ...}`` dicts.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, os.PathLike)):
            return cls.load(value)
        if len(value) and isinstance(value[0], dict):
            return cls.from_list(value)
        array = np.asarray(value, dtype=np.float64)
        if array.size and (array.ndim != 2 or array.shape[1] != 4):
            raise ValueError(f"word box array must be shaped (N, 4) as x0, y0, x1, y1; got {array.shape}")
        return cls(boxes=array.reshape(-1, 4))


def expand_box(box: Sequence[float], fraction: float, width: int, height: int) -> List[int]:
    """Grow a box by ``fraction`` of its size on every side, clip it, truncate to ints.

    ``fraction=0.1`` is what the official InkSight notebook does with docTR's boxes.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    dw, dh = (x1 - x0) * fraction, (y1 - y0) * fraction
    return [int(max(0.0, x0 - dw)), int(max(0.0, y0 - dh)),
            int(min(float(width), x1 + dw)), int(min(float(height), y1 + dh))]


class WordDetector(ABC):
    """Finds the words in an image of handwriting.

    Implement :meth:`detect` to plug a different engine into
    :class:`~polyvec.text.TextVectorizer`.  Setting ``name`` registers
    the class so that it can be selected by that name.
    """

    name: ClassVar[str] = ""
    aliases: ClassVar[Tuple[str, ...]] = ()
    _registry: ClassVar[Dict[str, Type["WordDetector"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("name"):
            for key in (cls.name, *cls.__dict__.get("aliases", ())):
                WordDetector._registry[key.lower()] = cls

    @abstractmethod
    def detect(self, image: ImageLike) -> Words:
        """Detect the words of one image, in reading order.

        Args:
            image: Anything :func:`polyvec.images.load_rgb` accepts.
        """

    def recognize(self, image: ImageLike) -> str:
        """Read all text in ``image`` as one string (words joined by spaces).

        Used to re-read rasterized tracings.  The default runs :meth:`detect`;
        engines with a dedicated single-word mode may override it.
        """
        return " ".join(text for text in self.detect(image).texts if text)

    def __call__(self, image: ImageLike) -> Words:
        """Alias for :meth:`detect`."""
        return self.detect(image)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class WholeImageDetector(WordDetector):
    """Treats the entire image as one word (reads nothing)."""

    name = "whole"

    def detect(self, image: ImageLike) -> Words:
        height, width = load_rgb(image).shape[:2]
        return Words.whole(width, height)


# --------------------------------------------------------------------------- registry
def available_detectors() -> List[str]:
    """Primary names of all registered word detectors."""
    return sorted({cls.name for cls in WordDetector._registry.values()})


def get_detector(spec: Union[str, WordDetector], **options: Any) -> WordDetector:
    """A detector instance from a registered name (or an instance, returned as is)."""
    if isinstance(spec, WordDetector):
        return spec
    try:
        cls = WordDetector._registry[spec.lower()]
    except KeyError:
        known = ", ".join(sorted(WordDetector._registry))
        raise ValueError(f"unknown word detector {spec!r}; known: {known}") from None
    return cls(**options)
