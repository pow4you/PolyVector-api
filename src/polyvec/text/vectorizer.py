"""Text-aware vectorization: words are traced on their own, upscaled, in writing order."""

from __future__ import annotations

import logging
import math
import time
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..base import Vectorizer, get_vectorizer
from ..drawing import VectorDrawing
from ..errors import PolyvecError
from ..images import ImageLike, load_grayscale
from ..native import NativeVectorizer
from ..runners import InProcessRunner, SubprocessRunner
from .direction import WritingDirection, order_boxes, order_strokes
from .geometry import (clip_strokes, estimate_pen_width, rasterize, upscale)
from .scoring import shape_error, text_error
from .words import WordDetector, Words, expand_box, get_detector

#: Forms in which word boxes may be supplied to :meth:`TextVectorizer.vectorize`.
WordsLike = Union[Words, str, "os.PathLike[str]", np.ndarray, Sequence[Any]]

log = logging.getLogger(__name__)

#: Gray level below which a pixel counts as ink when no ``ink_threshold`` is set
#: (the compiled programs' own cut, see ``NativeVectorizer.NATIVE_INK_THRESHOLD``).
DEFAULT_INK_THRESHOLD = 165


@dataclass
class Candidate:
    """One attempt at tracing one word."""

    strokes: List[np.ndarray]
    variant: Dict[str, Any]
    read: Optional[str] = None
    text_error: Optional[float] = None
    shape_error: Optional[float] = None
    passed: bool = True
    score: float = 0.0
    error: Optional[str] = None


@dataclass
class WordTrace:
    """The outcome for one word: the chosen candidate and how it was found."""

    box: np.ndarray
    region: List[int]
    text: Optional[str]
    chosen: Candidate
    candidates: List[Candidate] = field(default_factory=list)
    fallback: bool = False


class TextVectorizer(Vectorizer):
    """Vectorizes images that contain writing, giving the words special care.

    Line-drawing vectorizers do well on drawings and poorly on small
    handwriting or type: a stroke is a few pixels wide, letters touch, and the
    output comes in no particular order.  This method wraps any other
    vectorizer (``base``) and handles text in six steps:

    1. **Find the words** with an OCR word detector borrowed from ``inksight``
       - docTR, PaddleOCR or Tesseract (:mod:`polyvec.text.words`) - and put
       them into reading order for ``direction``.
    2. **Trace each word on an upscaled crop** (``scale``, Lanczos), with some
       ``padding`` of context around the box, and map the strokes back.
    3. **Order the strokes** like writing (:mod:`polyvec.text.direction`):
       starting where writing starts (top-left for the default ``"ltr-ttb"``),
       following the primary direction, every polyline running the way the
       pen moves.  All combinations of primary and secondary direction work.
    4. **Re-read** (optional, off by default: ``verify=True``): the traced word
       is rasterized again and read by the recognizer; the character error
       rate against the detector's reading must be within ``max_text_error``.
       It does not have to pass - a word that never passes keeps its best
       attempt and is marked ``passed: False``.
    5. **Iterate or pick the best of N** (``attempts``): further attempts vary
       scale and ink threshold (``variants``).  ``selection="first"`` stops at
       the first attempt within the limits; ``selection="best"`` runs all of
       them and keeps the one with the lowest score - re-readability
       (``text_weight`` x text error, when verifying) plus shape following
       (``shape_weight`` x shape error: 1 - F1 between the rasterized tracing
       and the original ink).
    6. **Patch the text back in**: the base vectorizer's tracing of the whole
       image (``trace_page``) has its strokes cut away inside the word regions
       and the word strokes take their place.  The text that was read goes
       into ``metadata["text"]``, per-word details into ``metadata["words"]``.

    Example::

        from polyvec import TextVectorizer

        text = TextVectorizer("flow", word_detector="paddleocr", verify=True,
                              attempts=4, selection="best")
        drawing = text.vectorize("note.png")
        drawing.metadata["text"]

    Args:
        base: The vectorizer that traces the page and each word: a registered
            name (``"flow"``, ``"field"``) or a :class:`Vectorizer`.
        word_detector: A :class:`~polyvec.text.words.WordDetector` or its name
            (``"doctr"``, ``"paddleocr"``, ``"tesseract"``, ``"whole"``).
        recognizer: Reads rasterized tracings back when ``verify`` is on;
            default: the word detector.
        direction: Writing direction, ``"<primary>-<secondary>"`` with axes
            ``ltr``, ``rtl``, ``ttb``, ``btt`` (default ``"ltr-ttb"``), or a
            :class:`~polyvec.text.direction.WritingDirection`.
        scale: Upscaling factor for word crops.  2 is the measured sweet spot
            for ``flow``: at 4 the strokes get so thick that its curve selection
            times out or crashes on about half the words.
        max_side: Upper limit for the longer side of an upscaled crop, in
            pixels; ``scale`` is reduced for large words.
        padding: Context around each word box in the crop, as a fraction of
            the box size on each side.
        margin: The region a word's strokes are kept in (and the page's
            strokes removed from) is the box grown by this fraction.
        trace_page: Also trace the whole image with ``base`` and keep its
            strokes outside the words.  Off: only text strokes are returned.
        verify: Re-read each traced word (step 4).
        max_text_error: Largest character error rate that passes (0-1).
        max_shape_error: Largest shape error that passes (0-1); None: no limit.
        case_sensitive: Compare readings case-sensitively.
        attempts: Attempts per word (step 5).
        selection: ``"first"`` (stop at the first passing attempt) or ``"best"``.
        variants: Parameter overrides, one dict per attempt, with keys
            ``scale``, ``padding`` and ``ink_threshold``; default: a built-in
            schedule around ``scale`` that tries smaller scales first.  After
            an attempt that fails (timeout, crash), the next one never uses a
            larger scale than 0.75 x the failed one.
        text_weight, shape_weight: Weights of the two errors in the score.
        word_timeout: Seconds a single word's tracing may take (the base's
            ``timeout`` while tracing crops); a word that runs over falls back
            to the page tracing.  None: the base's own timeout.  A base
            running in-process (``runner="inprocess"``, which ``"auto"``
            picks when the wheel's extension is installed) cannot be timed
            out, so while a timeout is set its word crops are traced in the
            helper process instead (:class:`~polyvec.runners.SubprocessRunner`),
            which also survives a crash; the page tracing stays in-process.
        invert: See :class:`~polyvec.base.Vectorizer`.
        **base_options: Constructor options for ``base`` when it is given by
            name (``runner``, ``ink_threshold``, ``flow_iterations`` ...).
    """

    name = "text"
    aliases = ("words", "handwriting")
    title = "Text-aware tracing"
    reference = ("Word detection from inksight (docTR / PaddleOCR / Tesseract) around a "
                 "PolyVector method.")

    SELECTIONS = ("first", "best")

    def __init__(self, base: Union[str, Vectorizer] = "flow", *,
                 word_detector: Union[str, WordDetector] = "doctr",
                 recognizer: Union[str, WordDetector, None] = None,
                 direction: Union[str, WritingDirection] = WritingDirection.DEFAULT,
                 scale: float = 2.0, max_side: int = 1600, padding: float = 0.25,
                 margin: float = 0.05, trace_page: bool = True,
                 verify: bool = False, max_text_error: float = 0.25,
                 max_shape_error: Optional[float] = None, case_sensitive: bool = False,
                 attempts: int = 1, selection: str = "first",
                 variants: Optional[Sequence[Mapping[str, Any]]] = None,
                 text_weight: float = 1.0, shape_weight: float = 1.0,
                 word_timeout: Optional[float] = 120.0,
                 invert: bool = False, **base_options: Any) -> None:
        super().__init__(invert=invert)
        if isinstance(base, Vectorizer):
            if base_options:
                raise TypeError("base_options are only used when base is given by name")
            self.base = base
        else:
            if str(base).lower() in (self.name, *self.aliases):
                raise ValueError("the text method cannot be its own base")
            self.base = get_vectorizer(base, **base_options)
        if scale <= 0 or max_side < 8:
            raise ValueError("scale must be positive and max_side at least 8")
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {attempts}")
        if selection not in self.SELECTIONS:
            raise ValueError(f"selection must be one of {self.SELECTIONS}, got {selection!r}")
        self.word_detector = get_detector(word_detector)
        self.recognizer = self.word_detector if recognizer is None else get_detector(recognizer)
        self.direction = WritingDirection.parse(direction)
        self.scale = float(scale)
        self.max_side = int(max_side)
        self.padding = float(padding)
        self.margin = float(margin)
        self.trace_page = trace_page
        self.verify = verify
        self.max_text_error = float(max_text_error)
        self.max_shape_error = None if max_shape_error is None else float(max_shape_error)
        self.case_sensitive = case_sensitive
        self.attempts = int(attempts)
        self.selection = selection
        self.variants = [dict(v) for v in variants] if variants is not None else None
        self.text_weight = float(text_weight)
        self.shape_weight = float(shape_weight)
        self.word_timeout = word_timeout

    # ------------------------------------------------------------- public API
    def vectorize(self, image: ImageLike, words: Optional[WordsLike] = None) -> VectorDrawing:
        """Vectorize one image.

        Args:
            image: Path to an image file, a NumPy array or a PIL image.
            words: Word boxes to use instead of running the detector: a
                :class:`~polyvec.text.words.Words`, the path of a JSON file
                written by ``Words.save``, an ``(N, 4)`` array of
                ``x0, y0, x1, y1`` rows, or a list of ``{"box", "text"}`` dicts.

        Returns:
            The strokes; ``metadata["words"]`` holds per-word details,
            ``metadata["text"]`` the text read, ``metadata["text_strokes"]``
            the ``(start, stop)`` range of the word strokes.
        """
        return super().vectorize(image, words=words)

    def detect_words(self, image: ImageLike) -> Words:
        """Run only step 1 (in reading order) - e.g. to check or edit the boxes
        before passing them back to :meth:`vectorize`."""
        return self._ordered(self.word_detector.detect(self.preprocess(load_grayscale(image))))

    @property
    def ink_threshold(self) -> int:
        """The gray level separating ink from paper for scoring (the base's, if it has one)."""
        value = getattr(self.base, "ink_threshold", None)
        return int(value) if value is not None else DEFAULT_INK_THRESHOLD

    def variant(self, attempt: int) -> Dict[str, Any]:
        """Parameters of attempt number ``attempt`` (0-based)."""
        if self.variants is not None:
            if not self.variants:
                return {"scale": self.scale, "padding": self.padding, "ink_threshold": None}
            chosen = dict(self.variants[attempt % len(self.variants)])
        else:
            # smaller first: larger crops are what make the backends time out or crash
            schedule = [{}, {"scale": 0.75}, {"ink": 25}, {"ink": -25}, {"scale": 0.5},
                        {"scale": 0.75, "ink": 25}, {"scale": 1.5}, {"scale": 2.0}]
            step = schedule[attempt % len(schedule)]
            chosen = {}
            if "scale" in step:
                chosen["scale"] = self.scale * step["scale"]
            if "ink" in step:
                chosen["ink_threshold"] = int(np.clip(self.ink_threshold + step["ink"], 1, 254))
        chosen.setdefault("scale", self.scale)
        chosen.setdefault("padding", self.padding)
        chosen.setdefault("ink_threshold", None)
        unknown = set(chosen) - {"scale", "padding", "ink_threshold"}
        if unknown:
            raise ValueError(f"unknown variant key(s): {', '.join(sorted(unknown))}")
        return chosen

    # --------------------------------------------------------------- pipeline
    def _vectorize(self, gray: np.ndarray, words: Optional[WordsLike] = None,
                   **inputs: Any) -> VectorDrawing:
        if inputs:
            raise TypeError(f"{type(self).__name__}.vectorize() got unexpected "
                            f"argument(s): {', '.join(sorted(inputs))}")
        height, width = gray.shape
        detector = "given" if words is not None else self.word_detector.name
        boxes = self._ordered(self.word_detector.detect(gray) if words is None
                              else Words.coerce(words))

        page_strokes: List[np.ndarray] = []
        page_meta: Dict[str, Any] = {}
        if self.trace_page or len(boxes) == 0:
            page = self.base.vectorize(gray)
            page_strokes, page_meta = list(page.strokes), dict(page.metadata)

        traces = []
        for number, (box, text) in enumerate(boxes, 1):
            started = time.perf_counter()
            trace = self._trace_word(gray, box, text, page_strokes)
            traces.append(trace)
            log.info("word %d/%d %r: %d strokes, shape error %s%s (%.1f s)", number, len(boxes),
                     text, len(trace.chosen.strokes),
                     "-" if trace.chosen.shape_error is None else f"{trace.chosen.shape_error:.3f}",
                     " [fallback]" if trace.fallback else "", time.perf_counter() - started)
        regions = np.array([trace.region for trace in traces], dtype=np.float64).reshape(-1, 4)

        strokes = clip_strokes(page_strokes, regions, keep="outside")
        start = len(strokes)
        word_meta = []
        for trace in traces:
            first = len(strokes)
            strokes.extend(trace.chosen.strokes)
            word_meta.append(self._describe(trace, (first, len(strokes))))

        metadata = dict(page_meta)
        metadata.update(
            base=self.base.name, detector=detector, direction=str(self.direction),
            words=word_meta, text_strokes=(start, len(strokes)),
            text=" ".join(trace.text for trace in traces if trace.text),
        )
        return VectorDrawing(strokes=strokes, width=width, height=height, method=self.name,
                             metadata=metadata)

    def _ordered(self, words: Words) -> Words:
        order = order_boxes(words.boxes, self.direction)
        return Words(boxes=words.boxes[order], texts=tuple(words.texts[i] for i in order),
                     scores=words.scores[order])

    # ---------------------------------------------------------------- one word
    def _trace_word(self, gray: np.ndarray, box: np.ndarray, text: Optional[str],
                    page_strokes: Sequence[np.ndarray]) -> WordTrace:
        height, width = gray.shape
        region = expand_box(box, self.margin, width, height)
        # without readable text to compare against, read the original crop once
        reference = text
        if self.verify and not (text or "").strip():
            reference = self.recognizer.recognize(self._score_view(gray, region))

        candidates: List[Candidate] = []
        ceiling = math.inf                # lowered by every failed attempt
        for attempt in range(self.attempts):
            variant = self.variant(attempt)
            variant["scale"] = min(variant["scale"], ceiling)
            candidate = self._attempt(gray, box, region, reference, variant)
            if candidate.error is not None:
                ceiling = 0.75 * candidate.variant["scale"]
            candidates.append(candidate)
            if self.selection == "first" and candidate.passed and candidate.error is None:
                break

        usable = [c for c in candidates if c.error is None and c.strokes]
        if usable:
            passing = [c for c in usable if c.passed]
            pool = passing if (self.selection == "first" and passing) else usable
            chosen = (pool[0] if self.selection == "first" and passing
                      else min(pool, key=lambda c: c.score))
            return WordTrace(box, region, text, chosen, candidates)

        # every attempt failed: keep what the page tracing had inside the region
        fallback = order_strokes(clip_strokes(page_strokes, np.array([region]), keep="inside"),
                                 self.direction)
        chosen = Candidate(fallback, {}, passed=False, score=math.inf,
                           error=candidates[-1].error if candidates else None)
        return WordTrace(box, region, text, chosen, candidates, fallback=True)

    def _attempt(self, gray: np.ndarray, box: np.ndarray, region: List[int],
                 reference: Optional[str], variant: Dict[str, Any]) -> Candidate:
        height, width = gray.shape
        x0, y0, x1, y1 = expand_box(box, variant["padding"], width, height)
        crop = gray[y0:y1, x0:x1]
        scale = min(variant["scale"], self.max_side / max(crop.shape))
        scale = max(scale, 1e-3)
        try:
            with self._override("ink_threshold", variant["ink_threshold"]), \
                    self._override("timeout", self.word_timeout), self._word_runner():
                traced = self.base.vectorize(upscale(crop, scale))
        except PolyvecError as exc:
            return Candidate([], dict(variant, scale=float(scale)), passed=False, score=math.inf,
                             error=f"{type(exc).__name__}: {exc}")
        # upscaled-crop pixels -> page pixels (the image is resized by round(size * scale))
        sx = crop.shape[1] / traced.width if traced.width else 1.0 / scale
        sy = crop.shape[0] / traced.height if traced.height else 1.0 / scale
        strokes = [np.asarray(s, dtype=np.float64) * (sx, sy) + (x0, y0) for s in traced.strokes]
        strokes = order_strokes(clip_strokes(strokes, np.array([region]), keep="inside"),
                                self.direction)
        candidate = Candidate(strokes, dict(variant, scale=float(scale)))
        self._score(gray, region, reference, candidate)
        return candidate

    def _score(self, gray: np.ndarray, region: List[int], reference: Optional[str],
               candidate: Candidate) -> None:
        """Fill in shape error, re-reading, pass/fail and score."""
        x0, y0, x1, y1 = region
        view = self._score_view(gray, region)
        scale = view.shape[1] / max(1, x1 - x0)
        ink = view < self.ink_threshold
        pen = estimate_pen_width(ink, [s * scale for s in candidate.strokes])
        raster = rasterize(candidate.strokes, view.shape[1], view.shape[0], pen,
                           offset=(x0, y0), scale=scale)
        candidate.shape_error = shape_error(ink, raster < 128,
                                            tolerance=max(1, int(round(scale))))
        candidate.score = self.shape_weight * candidate.shape_error
        passed = self.max_shape_error is None or candidate.shape_error <= self.max_shape_error

        if self.verify:
            pad = max(4, int(round(0.2 * raster.shape[0])))
            page = np.pad(raster, pad, constant_values=255)
            candidate.read = self.recognizer.recognize(page)
            if (reference or "").strip():
                candidate.text_error = text_error(candidate.read, reference,
                                                  case_sensitive=self.case_sensitive)
                candidate.score += self.text_weight * candidate.text_error
                passed = passed and candidate.text_error <= self.max_text_error
        candidate.passed = passed

    def _score_view(self, gray: np.ndarray, region: Sequence[int]) -> np.ndarray:
        """The word region, upscaled at the base scale: what scoring and re-reading look at."""
        x0, y0, x1, y1 = region
        crop = gray[y0:y1, x0:x1]
        return upscale(crop, min(self.scale, self.max_side / max(crop.shape)))

    @contextmanager
    def _override(self, attribute: str, value: Any) -> Iterator[None]:
        """Temporarily set an attribute of the base vectorizer (if it has it; None: leave it)."""
        if value is None or not hasattr(self.base, attribute):
            yield
            return
        previous = getattr(self.base, attribute)
        setattr(self.base, attribute, value)
        try:
            yield
        finally:
            setattr(self.base, attribute, previous)

    @contextmanager
    def _word_runner(self) -> Iterator[None]:
        """Trace word crops in the helper process when the base runs in-process.

        Some crops make the research code hang or crawl (seen on rendered
        text: two words of six ran past 120 s).  In-process that cannot be cut
        off and would stall the whole page, so with a ``word_timeout`` set the
        crops go through :class:`~polyvec.runners.SubprocessRunner`, which
        honours the timeout.  Other runners (local, Docker, subprocess, custom)
        are left alone.
        """
        base = self.base
        if self.word_timeout is None or not isinstance(base, NativeVectorizer):
            yield
            return
        try:
            runner = base.runner
        except PolyvecError:        # no backend at all: let vectorize() report it
            yield
            return
        if type(runner) is not InProcessRunner:
            yield
            return
        base._runner = SubprocessRunner()
        try:
            yield
        finally:
            base._runner = runner

    @staticmethod
    def _describe(trace: WordTrace, strokes: Tuple[int, int]) -> Dict[str, Any]:
        chosen = trace.chosen
        return {
            "box": [float(v) for v in trace.box], "region": list(trace.region),
            "text": trace.text, "read": chosen.read, "text_error": chosen.text_error,
            "shape_error": chosen.shape_error, "passed": chosen.passed,
            "attempt": next((i for i, c in enumerate(trace.candidates) if c is chosen), None),
            "attempts": len(trace.candidates), "variant": chosen.variant,
            "fallback": trace.fallback, "error": chosen.error, "strokes": strokes,
            "candidates": [{"variant": c.variant, "read": c.read, "text_error": c.text_error,
                            "shape_error": c.shape_error, "passed": c.passed, "score": c.score,
                            "error": c.error} for c in trace.candidates],
        }

    def __repr__(self) -> str:
        return (f"TextVectorizer(base={self.base!r}, word_detector={self.word_detector!r}, "
                f"direction='{self.direction}', verify={self.verify}, attempts={self.attempts}, "
                f"selection={self.selection!r})")
