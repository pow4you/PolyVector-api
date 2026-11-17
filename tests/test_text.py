"""Unit tests for the text pipeline, with a fake base vectorizer and a fake OCR engine."""

from __future__ import annotations

import numpy as np
import pytest

import polyvec
from polyvec import TextVectorizer, VectorDrawing, Vectorizer, VectorizationError
from polyvec.text import (WordDetector, Words, WritingDirection, order_boxes, order_strokes,
                          orient_stroke, shape_error, text_error)
from polyvec.text.geometry import rasterize, split_by_rects


# ------------------------------------------------------------------ fakes
class RowTracer(Vectorizer):
    """Traces every ink row-band as one horizontal stroke, drawn right to left
    (so the text pipeline has to flip it)."""

    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.ink_threshold = None
        self.calls = []
        self.fail = fail

    def _vectorize(self, gray, **inputs):
        self.calls.append((gray.shape, self.ink_threshold))
        if self.fail and len(self.calls) > 1:
            raise VectorizationError("boom")
        height, width = gray.shape
        rows = np.flatnonzero((gray < 128).any(axis=1))
        strokes = []
        for band in np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1) if len(rows) else []:
            cols = np.flatnonzero((gray[band] < 128).any(axis=0))
            y = band.mean() + 0.5
            strokes.append(np.array([[cols.max() + 1.0, y], [cols.min() + 0.0, y]]))
        return VectorDrawing(strokes, width, height)


class ScriptedOCR(WordDetector):
    """Reads whatever it was told to, one answer per call."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.seen = []

    def detect(self, image):
        return Words(boxes=[[1, 1, 39, 19]], texts=("hello",))

    def recognize(self, image):
        self.seen.append(image)
        return self.answers.pop(0) if self.answers else ""


def page_with_two_words():
    """60 x 40 white page: two 'words' (bars) on one line, a frame line at the bottom."""
    gray = np.full((40, 60), 255, np.uint8)
    gray[8:11, 5:20] = 0          # word 1
    gray[8:11, 30:50] = 0         # word 2
    gray[35:37, 2:58] = 0         # not text
    return gray


WORDS = Words(boxes=[[30, 5, 52, 14], [3, 5, 22, 14]], texts=("two", "one"))


# ------------------------------------------------------------------ direction
class TestWritingDirection:
    @pytest.mark.parametrize("spec, expected", [("ltr-ttb", "ltr-ttb"), ("r2l_t2b", "rtl-ttb"),
                                                ("ttb", "ttb-rtl"), (("btt", "ltr"), "btt-ltr")])
    def test_parse(self, spec, expected):
        assert str(WritingDirection.parse(spec)) == expected

    @pytest.mark.parametrize("bad", ["ltr-rtl", "ttb-btt", "up-down", "ltr-ttb-rtl"])
    def test_rejects(self, bad):
        with pytest.raises(ValueError):
            WritingDirection.parse(bad)

    def test_orient_open_strokes(self):
        ltr = WritingDirection.parse("ltr-ttb")
        backwards = np.array([[10.0, 5], [0, 6]])
        np.testing.assert_array_equal(orient_stroke(backwards, ltr), backwards[::-1])
        upward_bar = np.array([[3.0, 20], [3.5, 0]])          # mostly across: top to bottom
        np.testing.assert_array_equal(orient_stroke(upward_bar, ltr), upward_bar[::-1])
        rtl = WritingDirection.parse("rtl-ttb")
        np.testing.assert_array_equal(orient_stroke(backwards, rtl), backwards)

    def test_closed_stroke_starts_at_top_left(self):
        ring = np.array([[2.0, 2], [0, 1], [1, 0], [2, 1], [2, 2]])
        out = orient_stroke(ring, WritingDirection())
        np.testing.assert_array_equal(out[0], [1, 0])
        np.testing.assert_array_equal(out[0], out[-1])
        assert len(out) == len(ring)

    def test_order_strokes(self):
        strokes = [np.array([[9.0, 0], [5, 0]]), np.array([[0.0, 3], [2, 3]])]
        ordered = order_strokes(strokes, WritingDirection())
        np.testing.assert_array_equal(ordered[0], [[0, 3], [2, 3]])
        np.testing.assert_array_equal(ordered[1], [[5, 0], [9, 0]])
        rtl = order_strokes(strokes, WritingDirection("rtl", "ttb"))
        np.testing.assert_array_equal(rtl[0], [[9, 0], [5, 0]])

    def test_order_boxes_by_lines(self):
        boxes = np.array([[50, 0, 60, 10], [0, 30, 10, 40], [0, 1, 10, 11], [20, 32, 30, 41]])
        assert order_boxes(boxes, WritingDirection()).tolist() == [2, 0, 1, 3]
        assert order_boxes(boxes, WritingDirection("rtl", "ttb")).tolist() == [0, 2, 3, 1]
        assert order_boxes(boxes, WritingDirection("ttb", "ltr")).tolist() == [2, 1, 3, 0]


# ------------------------------------------------------------------ geometry and scores
def test_split_by_rects_meets_at_the_boundary():
    inside, outside = split_by_rects(np.array([[0.0, 5], [20, 5]]), np.array([[5, 0, 10, 10]]))
    assert len(inside) == 1 and len(outside) == 2
    np.testing.assert_allclose(inside[0], [[5, 5], [10, 5]], atol=1e-5)
    np.testing.assert_allclose(outside[0][-1], [5, 5], atol=1e-5)
    np.testing.assert_allclose(outside[1][0], [10, 5], atol=1e-5)


def test_text_error():
    assert text_error("Hello", "hello") == 0
    assert text_error("Hello", "hello", case_sensitive=True) == pytest.approx(0.2)
    assert text_error("helo", "hello") == pytest.approx(0.2)
    assert text_error("", "") == 0 and text_error("x", "") == 1


def test_shape_error_of_a_faithful_tracing_is_small():
    ink = np.zeros((20, 40), bool)
    ink[9:12, 5:35] = True
    good = rasterize([np.array([[5.5, 10.5], [34.5, 10.5]])], 40, 20, pen_width=3) < 128
    bad = rasterize([np.array([[5.5, 2.5], [34.5, 2.5]])], 40, 20, pen_width=3) < 128
    assert shape_error(ink, good) < 0.1 < 0.9 < shape_error(ink, bad)


# ------------------------------------------------------------------ the vectorizer
class TestTextVectorizer:
    def test_registered(self):
        assert "text" in polyvec.available_methods()

    def test_words_are_traced_upscaled_ordered_and_patched_in(self):
        base = RowTracer()
        text = TextVectorizer(base, word_detector="whole", scale=4, padding=0.2)
        drawing = text.vectorize(page_with_two_words(), words=WORDS)

        assert base.calls[0][0] == (40, 60)                         # the page
        assert all(shape[0] > 30 for shape, _ in base.calls[1:])    # upscaled crops
        meta = drawing.metadata
        assert [w["text"] for w in meta["words"]] == ["one", "two"]  # reading order
        assert meta["text"] == "one two"
        start, stop = meta["text_strokes"]
        assert stop - start == 2
        first, second = drawing.strokes[start:stop]
        assert first[0, 0] < first[-1, 0] < second[0, 0]             # left to right, flipped
        assert first[0, 1] == pytest.approx(9.5, abs=0.3)            # mapped back to the page
        # the bottom line survives; the page's own strokes inside the words are gone
        page = drawing.strokes[:start]
        assert any(s[:, 1].mean() > 30 for s in page)
        # (the fake joins both bars into one stroke: only the gap between the words remains)
        upper = [s for s in page if s[:, 1].mean() < 20]
        assert len(upper) == 1
        assert sorted(upper[0][:, 0]) == pytest.approx([22, 28])     # regions are whole pixels
        assert meta["words"][0]["shape_error"] < 0.5

    def test_rtl_direction(self):
        drawing = TextVectorizer(RowTracer(), word_detector="whole", direction="rtl-ttb",
                                 trace_page=False).vectorize(page_with_two_words(), words=WORDS)
        assert [w["text"] for w in drawing.metadata["words"]] == ["two", "one"]
        assert drawing.strokes[0][0, 0] > drawing.strokes[0][-1, 0]
        assert len(drawing) == 2

    def test_verify_first_passing_attempt(self):
        # "one" is traced first: "xxx" fails, "onx" (1 error in 3) passes; then "two" passes
        ocr = ScriptedOCR(["xxx", "onx", "two"])
        text = TextVectorizer(RowTracer(), word_detector="whole", recognizer=ocr, verify=True,
                              attempts=3, max_text_error=0.34, trace_page=False)
        drawing = text.vectorize(page_with_two_words(), words=WORDS)
        one, two = drawing.metadata["words"]
        assert one["attempts"] == 2 and one["attempt"] == 1 and one["passed"]
        assert one["read"] == "onx" and one["text_error"] == pytest.approx(1 / 3)
        assert two["attempts"] == 1 and two["passed"] and two["text_error"] == 0
        assert all(image.ndim == 2 and image.shape[0] > 2 * 9 for image in ocr.seen)  # upscaled

    def test_word_that_never_passes_keeps_its_best_attempt(self):
        ocr = ScriptedOCR(["xxx", "oxx"])
        drawing = TextVectorizer(RowTracer(), word_detector="whole", recognizer=ocr, verify=True,
                                 attempts=2, shape_weight=0.0, trace_page=False).vectorize(
            page_with_two_words(), words=Words(boxes=[[3, 5, 22, 14]], texts=("one",)))
        word = drawing.metadata["words"][0]
        assert not word["passed"] and word["attempt"] == 1 and len(drawing) == 1

    def test_selection_best_keeps_lowest_score(self):
        ocr = ScriptedOCR(["onx", "one", "oxx"])
        text = TextVectorizer(RowTracer(), word_detector="whole", recognizer=ocr, verify=True,
                              attempts=3, selection="best", shape_weight=0.0, trace_page=False)
        drawing = text.vectorize(page_with_two_words(),
                                 words=Words(boxes=[[3, 5, 22, 14]], texts=("one",)))
        word = drawing.metadata["words"][0]
        assert word["attempts"] == 3 and word["attempt"] == 1
        assert word["read"] == "one" and word["text_error"] == 0 and word["passed"]

    def test_verification_off_by_default_never_reads(self):
        ocr = ScriptedOCR([])
        TextVectorizer(RowTracer(), word_detector="whole", recognizer=ocr,
                       trace_page=False).vectorize(page_with_two_words(), words=WORDS)
        assert ocr.seen == []

    def test_variants_reach_the_base(self):
        base = RowTracer()
        TextVectorizer(base, word_detector="whole", attempts=2, selection="best", trace_page=False,
                       variants=[{"scale": 2}, {"scale": 3, "ink_threshold": 200}]
                       ).vectorize(page_with_two_words(), words=Words(boxes=[[3, 5, 22, 14]]))
        (shape1, thr1), (shape2, thr2) = base.calls
        assert thr1 is None and thr2 == 200 and shape2[0] > shape1[0]
        assert base.ink_threshold is None                           # restored

    def test_default_schedule_starts_at_2x_and_tries_smaller_first(self):
        text = TextVectorizer(RowTracer(), word_detector="whole")
        assert text.scale == 2
        scales = [text.variant(i)["scale"] for i in range(3)]
        assert scales[0] == 2 and scales[1] < 2

    def test_after_a_failure_the_next_attempt_is_smaller(self):
        class FailsLarge(RowTracer):
            """Times out on crops taller than 35 px (the 4x and 3x ones here)."""

            def _vectorize(self, gray, **inputs):
                if gray.shape[0] > 35:
                    self.calls.append((gray.shape, self.ink_threshold))
                    raise VectorizationError("timeout")
                return super()._vectorize(gray, **inputs)

        drawing = TextVectorizer(FailsLarge(), word_detector="whole", attempts=3,
                                 trace_page=False, variants=[{"scale": 4}]).vectorize(
            page_with_two_words(), words=Words(boxes=[[3, 5, 22, 14]]))
        word = drawing.metadata["words"][0]
        scales = [c["variant"]["scale"] for c in word["candidates"]]
        assert scales == pytest.approx([4, 3, 2.25])            # each failure: 0.75x
        assert [bool(c["error"]) for c in word["candidates"]] == [True, True, False]
        assert word["attempt"] == 2 and not word["fallback"]

    def test_failed_attempts_fall_back_to_the_page_tracing(self):
        drawing = TextVectorizer(RowTracer(fail=True), word_detector="whole").vectorize(
            page_with_two_words(), words=Words(boxes=[[3, 5, 22, 14]]))
        word = drawing.metadata["words"][0]
        assert word["fallback"] and "boom" in word["error"]
        start, stop = word["strokes"]
        assert stop - start == 1 and drawing.strokes[start][0, 0] < drawing.strokes[start][-1, 0]

    def test_detect_words_orders_detector_output(self):
        class Backwards(WordDetector):
            def detect(self, image):
                return Words(boxes=[[30, 0, 40, 10], [0, 0, 10, 10]], texts=("b", "a"))

        words = TextVectorizer(RowTracer(), word_detector=Backwards()).detect_words(
            page_with_two_words())
        assert words.texts == ("a", "b")

    @pytest.mark.parametrize("bad", [dict(attempts=0), dict(selection="worst"), dict(scale=0)])
    def test_rejects_bad_options(self, bad):
        with pytest.raises(ValueError):
            TextVectorizer(RowTracer(), word_detector="whole", **bad)
