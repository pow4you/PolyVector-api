"""Word detection (borrowed from ``inksight``): Tesseract, docTR and PaddleOCR.

The OCR engines are optional extras (``doctr``, ``paddle``, ``tesseract``);
importing this package does not need any of them.
"""

from .base import (WholeImageDetector, WordDetector, Words, available_detectors, expand_box,
                   get_detector)
from .doctr_detector import DoctrWordDetector
from .paddle_detector import PaddleOCRWordDetector
from .tesseract_detector import TesseractWordDetector

__all__ = ["Words", "WordDetector", "WholeImageDetector", "DoctrWordDetector",
           "PaddleOCRWordDetector", "TesseractWordDetector", "available_detectors",
           "expand_box", "get_detector"]
