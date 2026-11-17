"""Word boxes from docTR - the engine the official InkSight notebook uses."""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...errors import MissingDependencyError
from ...images import ImageLike, load_rgb
from ._process import EngineProcess
from .base import WordDetector, Words, expand_box

#: ``(text, ((x0, y0), (x1, y1)) relative to the image, confidence)``
_Word = Tuple[str, Tuple[Tuple[float, float], Tuple[float, float]], float]


def _build(options: Dict[str, Any]) -> Any:
    from doctr.models import ocr_predictor
    return ocr_predictor(pretrained=True, **options)


def _run(predictor: Any, rgb: np.ndarray) -> List[_Word]:
    """The words of a docTR result, as plain (picklable) Python data."""
    result = predictor([rgb])
    return [(word.value, tuple(tuple(map(float, corner)) for corner in word.geometry),
             float(getattr(word, "confidence", np.nan)))
            for page in result.pages for block in page.blocks
            for line in block.lines for word in line.words]


class DoctrWordDetector(WordDetector):
    """docTR's pretrained OCR predictor, the word finder of the official notebook.

    Uses docTR >= 1.0, which runs on PyTorch.  Each word box is grown by
    ``box_expansion`` of its size on every side, as the InkSight notebook does.

    The predictor runs in a persistent worker process by default (see
    :mod:`polyvec.text.words._process`: OCR engines and other native libraries
    do not always share a process), started on the first detection and stopped with
    :meth:`close` or at exit.  docTR downloads its weights on first use.

    Args:
        box_expansion: Fraction to grow each box by (notebook: 0.1).
        predictor: A ready docTR ``ocr_predictor`` to use in this process
            (implies ``isolated=False``).
        isolated: Run docTR in a worker process (default) or in this one.
        **options: Further ``doctr.models.ocr_predictor`` arguments
            (``det_arch``, ``reco_arch``, ...).
    """

    name = "doctr"

    def __init__(self, box_expansion: float = 0.1, predictor: Optional[Any] = None,
                 isolated: bool = True, **options: Any) -> None:
        self.box_expansion = box_expansion
        self.options = options
        self.isolated = isolated and predictor is None
        self._engine: Optional[Any] = predictor

    def _predict(self, rgb: np.ndarray) -> List[_Word]:
        if self._engine is None:
            if importlib.util.find_spec("doctr") is None:
                raise MissingDependencyError(
                    "the docTR word detector needs python-doctr. Install the 'doctr' extra:  "
                    "uv sync --extra doctr   (or  pip install -e '.[doctr]')")
            self._engine = (EngineProcess(_build, _run, self.options, "docTR") if self.isolated
                            else _build(self.options))
        if self.isolated:
            return self._engine(rgb)
        return _run(self._engine, rgb)

    def close(self) -> None:
        """Stop the worker process, if any (it restarts on the next detection)."""
        if isinstance(self._engine, EngineProcess):
            self._engine.close()

    def detect(self, image: ImageLike) -> Words:
        rgb = load_rgb(image)
        height, width = rgb.shape[:2]
        boxes, texts, scores = [], [], []
        for value, ((rx0, ry0), (rx1, ry1)), confidence in self._predict(rgb):
            if not value.strip():
                continue
            box = expand_box([int(rx0 * width), int(ry0 * height), int(rx1 * width), int(ry1 * height)],
                             self.box_expansion, width, height)
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
                texts.append(value)
                scores.append(confidence)
        return Words(boxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                     texts=tuple(texts), scores=scores)

    def __repr__(self) -> str:
        return f"DoctrWordDetector(box_expansion={self.box_expansion})"
