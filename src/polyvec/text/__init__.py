"""Text-aware vectorization.

* :mod:`~polyvec.text.words` - word detectors borrowed from ``inksight``
  (Tesseract, docTR, PaddleOCR)
* :mod:`~polyvec.text.direction` - writing directions and stroke ordering
* :mod:`~polyvec.text.scoring` - re-readability (character error rate) and
  shape error
* :class:`TextVectorizer` - the ``"text"`` method that ties them together
"""

from .direction import WritingDirection, order_boxes, order_strokes, orient_stroke
from .scoring import shape_error, text_error
from .vectorizer import Candidate, TextVectorizer, WordTrace
from .words import (DoctrWordDetector, PaddleOCRWordDetector, TesseractWordDetector,
                    WholeImageDetector, WordDetector, Words, available_detectors, get_detector)

__all__ = ["TextVectorizer", "Candidate", "WordTrace", "WritingDirection", "orient_stroke",
           "order_strokes", "order_boxes", "text_error", "shape_error", "Words", "WordDetector",
           "WholeImageDetector", "DoctrWordDetector", "PaddleOCRWordDetector",
           "TesseractWordDetector", "available_detectors", "get_detector"]
