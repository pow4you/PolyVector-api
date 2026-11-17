"""Word boxes from PaddleOCR 3 (PP-OCR text detection + recognition) (borrowed from ``inksight``)."""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, Optional

import numpy as np

from ...errors import MissingDependencyError
from ...images import ImageLike, load_rgb
from ._process import EngineProcess
from .base import WordDetector, Words, expand_box

#: The parts of a PaddleOCR result this detector uses.
_RESULT_KEYS = ("rec_texts", "rec_scores", "rec_polys", "text_word", "text_word_region")


def _build(options: Dict[str, Any]) -> Any:
    from paddleocr import PaddleOCR
    return PaddleOCR(**options)


def _run(pipeline: Any, bgr: np.ndarray) -> Dict[str, Any]:
    """The used parts of a PaddleOCR result, as plain (picklable) Python data."""
    result = pipeline.predict(bgr)[0]
    out = {}
    for key in _RESULT_KEYS:
        value = result.get(key)
        out[key] = [np.asarray(v).tolist() if isinstance(v, np.ndarray) else v
                    for v in (value if value is not None else [])]
    return out


class PaddleOCRWordDetector(WordDetector):
    """A newer alternative to docTR: PaddleOCR's PP-OCR pipeline (``paddleocr>=3``).

    PP-OCR detects text *lines*; with ``return_word_box`` its recogniser also
    places each recognised word inside its line, which gives word boxes.  Set
    ``level="line"`` to trace whole lines as one unit instead.

    PaddleOCR runs in a persistent worker process by default (see
    :mod:`polyvec.text.words._process`: on Linux, Paddle crashes when certain
    native libraries were loaded first), started on the first detection and stopped with
    :meth:`close` or at exit.  It downloads its models on first use.  Document
    unwarping and orientation models are off: handwriting photos rarely need
    them and they slow everything down.

    Args:
        box_expansion: Fraction to grow each box by on every side.
        level: ``"word"`` or ``"line"``.
        lang: PaddleOCR language code (``"en"``, ``"de"``, ``"ch"`` ...).
        ocr_version: PP-OCR generation, e.g. ``"PP-OCRv5"``; None for PaddleOCR's default.
        enable_mkldnn: Use oneDNN on the CPU.  Off by default: Paddle 3.x's
            oneDNN kernels fail on these models (Windows and Linux alike).
        device: ``"cpu"``, ``"gpu"``, ``"gpu:0"`` ...; None for PaddleOCR's choice.
        isolated: Run PaddleOCR in a worker process (default) or in this one.
        **options: Further ``paddleocr.PaddleOCR`` arguments.
    """

    name = "paddleocr"
    aliases = ("paddle",)

    def __init__(self, box_expansion: float = 0.1, level: str = "word", lang: Optional[str] = "en",
                 ocr_version: Optional[str] = None, enable_mkldnn: bool = False,
                 device: Optional[str] = None, isolated: bool = True, **options: Any) -> None:
        if level not in ("word", "line"):
            raise ValueError("level must be 'word' or 'line'")
        self.box_expansion = box_expansion
        self.level = level
        self.isolated = isolated
        self.options = dict(
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False, return_word_box=(level == "word"),
            enable_mkldnn=enable_mkldnn,
        )
        for key, value in (("lang", lang), ("ocr_version", ocr_version), ("device", device)):
            if value is not None:
                self.options[key] = value
        self.options.update(options)
        self._engine: Optional[Any] = None

    def _predict(self, bgr: np.ndarray) -> Dict[str, Any]:
        if self._engine is None:
            if importlib.util.find_spec("paddleocr") is None or importlib.util.find_spec("paddle") is None:
                raise MissingDependencyError(
                    "the PaddleOCR word detector needs paddleocr and paddlepaddle. Install the "
                    "'paddle' extra:  uv sync --extra paddle   (or  pip install -e '.[paddle]')")
            self._engine = (EngineProcess(_build, _run, self.options, "PaddleOCR") if self.isolated
                            else _build(self.options))
        if self.isolated:
            return self._engine(bgr)
        return _run(self._engine, bgr)

    def close(self) -> None:
        """Stop the worker process, if any (it restarts on the next detection)."""
        if isinstance(self._engine, EngineProcess):
            self._engine.close()

    def detect(self, image: ImageLike) -> Words:
        rgb = load_rgb(image)
        height, width = rgb.shape[:2]
        # PaddleOCR expects OpenCV's channel order.
        result = self._predict(np.ascontiguousarray(rgb[:, :, ::-1]))

        boxes, texts, scores = [], [], []

        def add(polygon: Any, text: str, score: float) -> None:
            points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
            box = expand_box([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(),
                              points[:, 1].max()], self.box_expansion, width, height)
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
                texts.append(text)
                scores.append(score)

        if self.level == "word":
            for words, regions, score in zip(result["text_word"], result["text_word_region"],
                                             result["rec_scores"]):
                for word, region in zip(words, regions):
                    if word.strip():
                        add(region, word.strip(), float(score))
        else:
            for polygon, text, score in zip(result["rec_polys"], result["rec_texts"],
                                            result["rec_scores"]):
                if text.strip():
                    add(polygon, text.strip(), float(score))
        return Words(boxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                     texts=tuple(texts), scores=scores)

    def __repr__(self) -> str:
        return f"PaddleOCRWordDetector(level={self.level!r}, box_expansion={self.box_expansion})"
