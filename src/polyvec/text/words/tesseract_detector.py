"""Word boxes from Tesseract OCR (borrowed from ``inksight``)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...errors import PolyvecError, MissingDependencyError
from ...images import ImageLike, load_rgb
from .base import WordDetector, Words, expand_box


class TesseractWordDetector(WordDetector):
    """Tesseract's word boxes (``pytesseract`` plus the ``tesseract`` program).

    Args:
        box_expansion: Fraction to grow each box by on every side.
        min_confidence: Drop words below this confidence (0-100); None keeps all.
        lang: Tesseract language(s), e.g. ``"eng+deu"``.
        config: Extra Tesseract command-line options.
    """

    name = "tesseract"

    def __init__(self, box_expansion: float = 0.0, min_confidence: Optional[float] = None,
                 lang: Optional[str] = None, config: str = "") -> None:
        self.box_expansion = box_expansion
        self.min_confidence = min_confidence
        self.lang = lang
        self.config = config

    @staticmethod
    def _pytesseract():
        try:
            import pytesseract
        except ImportError as exc:
            raise MissingDependencyError(
                "the Tesseract word detector needs pytesseract and the tesseract program. "
                "Install the 'tesseract' extra and tesseract itself (apt install tesseract-ocr; "
                "on Windows the UB Mannheim installer)") from exc
        return pytesseract

    def recognize(self, image: ImageLike) -> str:
        """Read ``image`` as a single text line (``--psm 7``), for re-reading word crops."""
        pytesseract = self._pytesseract()
        try:
            text = pytesseract.image_to_string(load_rgb(image), lang=self.lang,
                                               config=f"--psm 7 {self.config}".strip())
        except pytesseract.TesseractNotFoundError as exc:
            raise PolyvecError("the tesseract program was not found on PATH") from exc
        return " ".join(text.split())

    def detect(self, image: ImageLike) -> Words:
        pytesseract = self._pytesseract()
        rgb = load_rgb(image)
        height, width = rgb.shape[:2]
        try:
            data = pytesseract.image_to_data(rgb, lang=self.lang, config=self.config,
                                             output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractNotFoundError as exc:
            raise PolyvecError("the tesseract program was not found on PATH") from exc

        boxes, texts, scores = [], [], []
        for i, text in enumerate(data["text"]):
            confidence = float(data["conf"][i])
            if not text.strip() or (self.min_confidence is not None and confidence < self.min_confidence):
                continue
            x0, y0 = data["left"][i], data["top"][i]
            box = expand_box([x0, y0, x0 + data["width"][i], y0 + data["height"][i]],
                             self.box_expansion, width, height)
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
                texts.append(text)
                scores.append(confidence / 100.0)
        return Words(boxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                     texts=tuple(texts), scores=scores)
