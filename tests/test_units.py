"""Unit tests that need neither a compiled backend nor PyTorch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import polyvec
from polyvec import (InvalidImageError, Keypoints, KeypointType, NativeJob, NativeVectorizer,
                     Runner, RunResult, VectorDrawing, VectorizationError, Vectorizer)
from polyvec.images import load_grayscale, remap_threshold


# ------------------------------------------------------------------ image loading
class TestLoadGrayscale:
    def test_all_input_forms_agree(self, tmp_path):
        rgb = np.full((6, 8, 3), 255, np.uint8)
        rgb[2:4, 1:7] = (10, 20, 30)
        path = tmp_path / "drawing.png"
        Image.fromarray(rgb).save(path)

        from_array = load_grayscale(rgb)
        assert from_array.shape == (6, 8) and from_array.dtype == np.uint8
        np.testing.assert_array_equal(from_array, load_grayscale(path))
        np.testing.assert_array_equal(from_array, load_grayscale(str(path)))
        np.testing.assert_array_equal(from_array, load_grayscale(Image.fromarray(rgb)))

    def test_array_dtypes(self):
        gray = np.array([[0, 128, 255]], np.uint8)
        np.testing.assert_array_equal(load_grayscale(gray), gray)
        np.testing.assert_array_equal(load_grayscale(gray / 255.0), gray)
        np.testing.assert_array_equal(load_grayscale(gray.astype(np.float32)), gray)
        np.testing.assert_array_equal(load_grayscale(np.array([[False, True]])), [[0, 255]])
        np.testing.assert_array_equal(load_grayscale(np.array([[0, 65535]], np.uint16)), [[0, 255]])
        np.testing.assert_array_equal(load_grayscale(gray[:, :, None]), gray)

    def test_transparency_becomes_white_paper(self):
        rgba = np.zeros((2, 2, 4), np.uint8)       # black, fully transparent
        rgba[0, 0] = (0, 0, 0, 255)                # one opaque black pixel
        np.testing.assert_array_equal(load_grayscale(rgba), [[0, 255], [255, 255]])

    @pytest.mark.parametrize("bad", [np.zeros((2, 2, 2)), np.zeros(5), np.zeros((0, 4)), 42,
                                     np.array([[np.nan]])])
    def test_rejects_non_images(self, bad):
        with pytest.raises(InvalidImageError):
            load_grayscale(bad)

    def test_missing_file(self, tmp_path):
        with pytest.raises(InvalidImageError, match="not found"):
            load_grayscale(tmp_path / "nope.png")


def test_remap_threshold_moves_the_cut():
    gray = np.array([[0, 99, 100, 101, 255]], np.uint8)
    remapped = remap_threshold(gray, threshold=100, target=165)
    assert remapped[0, 0] == 0 and remapped[0, 4] == 255
    assert remapped[0, 2] == 165
    assert remapped[0, 1] < 165 < remapped[0, 3]
    assert np.all(np.diff(remapped[0].astype(int)) > 0)          # still monotone


# --------------------------------------------------------------------- drawings
UPSTREAM_STYLE_SVG = """<?xml version="1.0" standalone="no" ?>
<svg width="305px" height="342px" xmlns="http://www.w3.org/2000/svg" version="1.1" >
<g>	<image width="305" height="342" xlink:href="example.png" />
</g>
<g>	<polyline points="1.5,2.5 3,4 5.25,6 " fill="none" stroke-width="1" stroke="rgb(0,0,255)" />
	<polyline points="10,10 nan,nan 12,12 -nan,3 " stroke="rgb(0,0,255)" />
	<polyline points="7,7 " />
</g></svg>"""


class TestVectorDrawing:
    def test_reads_upstream_svg(self):
        drawing = VectorDrawing.from_svg(UPSTREAM_STYLE_SVG)
        assert (drawing.width, drawing.height) == (305, 342)
        assert len(drawing) == 2                                   # single-point stroke dropped
        np.testing.assert_allclose(drawing[0], [[1.5, 2.5], [3, 4], [5.25, 6]])
        np.testing.assert_allclose(drawing[1], [[10, 10], [12, 12]])   # nan points dropped

    def test_svg_round_trip(self, tmp_path):
        strokes = [np.array([[0.0, 0.0], [3.0, 4.0]]), np.array([[1.0, 1.0], [2.0, 5.5], [9.0, 2.0]])]
        drawing = VectorDrawing(strokes=strokes, width=10, height=8, method="demo")
        path = drawing.save(tmp_path / "out.svg", stroke="red")
        again = VectorDrawing.from_svg(path)
        assert (again.width, again.height) == (10, 8)
        for a, b in zip(again, drawing):
            np.testing.assert_allclose(a, b, atol=1e-3)

    def test_measures(self):
        drawing = VectorDrawing([np.array([[0.0, 0.0], [3.0, 4.0]])], width=5, height=5)
        assert drawing.num_points == 2
        np.testing.assert_allclose(drawing.lengths(), [5.0])


# -------------------------------------------------------------------- keypoints
class TestKeypoints:
    def test_pts_round_trip_is_row_first(self, tmp_path):
        keypoints = Keypoints(points=[[10, 20], [30.5, 40]], labels=[1, 5])
        path = keypoints.save(tmp_path / "k.pts")
        assert path.read_text().splitlines() == ["2", "20 10 1", "40 30.5 5"]
        loaded = Keypoints.load(path)
        np.testing.assert_allclose(loaded.points, keypoints.points)
        np.testing.assert_array_equal(loaded.labels, keypoints.labels)

    def test_reads_upstream_example(self, example_keypoints):
        keypoints = Keypoints.load(example_keypoints)
        assert len(keypoints) == 34
        # first line of the file is "319.0 188.0 5.0": row 319, column 188, sharp corner
        np.testing.assert_allclose(keypoints.points[0], [188.0, 319.0])
        assert keypoints.labels[0] == KeypointType.SHARP_CORNER

    def test_coerce(self, example_keypoints):
        from_array = Keypoints.coerce(np.array([[1.0, 2.0, 3]]))
        assert from_array.labels.tolist() == [KeypointType.JUNCTION]
        assert Keypoints.coerce(from_array) is from_array
        assert len(Keypoints.coerce(example_keypoints)) == 34
        np.testing.assert_allclose(from_array.of_type(KeypointType.JUNCTION), [[1.0, 2.0]])

    def test_validation(self):
        with pytest.raises(ValueError):
            Keypoints(points=[[0, 0]], labels=[2])
        with pytest.raises(ValueError):
            Keypoints.coerce(np.zeros((3, 2)))


# ------------------------------------------------------- hierarchy & extension
class FakeRunner(Runner):
    """Pretends to be the compiled program: writes a fixed SVG."""

    name = "fake"

    def __init__(self, returncode=0, write_output=True):
        self.returncode, self.write_output, self.calls = returncode, write_output, []

    def is_available(self, tool):
        return True

    def run(self, tool, args, workdir, timeout=None):
        self.calls.append((tool, list(args), sorted(p.name for p in Path(workdir).iterdir())))
        if self.write_output:
            output = next(arg for arg in args if arg.endswith(".svg"))
            (Path(workdir) / output).write_text(
                '<svg width="4px" height="3px"><polyline points="0,0 1,1 2,1 "/></svg>')
        return RunResult([tool, *args], self.returncode, "log line", 0.25)


class EchoVectorizer(NativeVectorizer):
    """A third method added from outside the package, as a user would."""

    name = "echo-test"
    aliases = ("echo-alias",)
    tool = "echo-tool"

    def _stage(self, workdir, image_name, gray, **inputs):
        (workdir / "extra.txt").write_text("staged")
        return NativeJob(args=[image_name, "result.svg"], output="result.svg",
                         metadata={"shape": gray.shape})


class TestHierarchy:
    def test_builtin_methods_are_registered(self):
        assert {"field", "flow"} <= set(polyvec.available_methods())
        assert polyvec.get_vectorizer_class("flow") is polyvec.PolyVectorFlow
        assert polyvec.get_vectorizer_class("POLYVECTOR-FIELD") is polyvec.PolyVectorField
        for cls in (polyvec.PolyVectorField, polyvec.PolyVectorFlow):
            assert issubclass(cls, NativeVectorizer) and issubclass(cls, Vectorizer)
        with pytest.raises(ValueError, match="unknown vectorization method"):
            polyvec.get_vectorizer("nope")

    def test_subclass_registers_and_runs_through_the_template(self):
        assert polyvec.get_vectorizer_class("echo-alias") is EchoVectorizer
        runner = FakeRunner()
        vectorizer = polyvec.get_vectorizer("echo-test", runner=runner)

        drawing = vectorizer.vectorize(np.full((3, 4), 255, np.uint8))

        tool, args, staged_files = runner.calls[0]
        assert tool == "echo-tool" and args == ["input.png", "result.svg"]
        assert staged_files == ["extra.txt", "input.png"]            # staged before the run
        assert isinstance(drawing, VectorDrawing) and len(drawing) == 1
        assert (drawing.width, drawing.height) == (4, 3)             # taken from the image
        assert drawing.method == "echo-test"
        assert drawing.metadata == {"elapsed": 0.25, "runner": "fake", "shape": (3, 4)}
        assert vectorizer.last_run.output == "log line"

    def test_preprocessing_reaches_the_program(self, tmp_path):
        vectorizer = EchoVectorizer(runner=FakeRunner(), invert=True, ink_threshold=100,
                                    workdir=tmp_path)
        vectorizer.vectorize(np.array([[255, 155, 0]] * 2, np.uint8))
        staged = np.asarray(Image.open(tmp_path / "input.png"))
        # inverted, then 100 remapped onto the program's fixed threshold of 165
        np.testing.assert_array_equal(staged[0], [0, 165, 255])
        assert (tmp_path / "log.txt").read_text() == "log line"

    @pytest.mark.parametrize("runner", [FakeRunner(returncode=3), FakeRunner(write_output=False)])
    def test_program_failure_raises_with_log(self, runner):
        with pytest.raises(VectorizationError) as raised:
            EchoVectorizer(runner=runner).vectorize(np.zeros((2, 2), np.uint8))
        assert raised.value.log == "log line"

    def test_flow_stages_given_keypoints_without_the_detector(self, tmp_path):
        class NoDetector(polyvec.KeypointDetector):
            def detect(self, image):
                raise AssertionError("detector must not run when keypoints are supplied")

        runner = FakeRunner()
        flow = polyvec.PolyVectorFlow(runner=runner, keypoint_detector=NoDetector(),
                                      flow_iterations=7, workdir=tmp_path)
        drawing = flow.vectorize(np.full((3, 4), 255, np.uint8), keypoints=np.array([[1, 2, 5]]))

        assert runner.calls[0][:2] == ("polyvec-flow", ["input.png", "input.pts", "output.svg", "7"])
        assert (tmp_path / "input.pts").read_text().splitlines() == ["1", "2 1 5"]
        assert len(drawing.metadata["keypoints"]) == 1

    def test_field_rejects_unknown_inputs(self):
        with pytest.raises(TypeError, match="keypoints"):
            polyvec.PolyVectorField(runner=FakeRunner()).vectorize(
                np.zeros((2, 2), np.uint8), keypoints=np.array([[0, 0, 1]]))
