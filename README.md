# polyvec

Line-drawing vectorization with PolyVector fields: **an image goes in, strokes
come out.**

`polyvec` is a small, class-based Python API around the reference
implementations of two papers. It wraps the authors' code - it does not
re-implement it - and hides everything those research codebases expect you to
do by hand (compiling, temp files, `.pts` files, shell scripts, SVG parsing).

| Method | Class | Paper | Upstream code |
|--------|-------|-------|---------------|
| `"field"` | `PolyVectorField` | Bessmeltsev & Solomon, *Vectorization of Line Drawings via PolyVector Fields*, TOG 2019 | [bmpix/PolyVectorization](https://github.com/bmpix/PolyVectorization) |
| `"flow"` | `PolyVectorFlow` | Puhachov, Neveu, Chien & Bessmeltsev, *Keypoint-Driven Line Drawing Vectorization via PolyVector Flow*, SIGGRAPH Asia 2021 | [ivanpuhachov/line-drawing-vectorization-polyvector-flow](https://github.com/ivanpuhachov/line-drawing-vectorization-polyvector-flow) |

```python
import polyvec

drawing = polyvec.vectorize("sketch.png", method="flow")   # a path, a NumPy array or a PIL image

for stroke in drawing:        # each stroke: (N, 2) float array of x, y pixel coordinates
    print(stroke.shape)

drawing.save("sketch.svg")
```

## Install

```bash
pip install polyvec               # add [flow] for PolyVector Flow's keypoint network
```

Binary wheels contain both C++ programs compiled as an extension module, so
nothing else is needed: no compiler, no Docker. Wheels are configured for
Windows (x86_64), Linux (x86_64, manylinux_2_28) and macOS (arm64, x86_64),
CPython 3.10 - 3.14; see [Platform support](#platform-support) for which of
these have actually been built and tested so far.

* The base install needs only NumPy and Pillow and is enough for `PolyVectorField`.
* The `flow` extra adds PyTorch, scikit-image and matplotlib, which the
  PolyVector Flow keypoint network (upstream's model, weights included) needs.
  A GPU is used when PyTorch can see one, but the CPU is fine: a few seconds
  per image. (On Windows the default PyTorch wheel is CPU-only; install a CUDA
  build from pytorch.org if you want the GPU.)

For development the project is set up for [uv](https://docs.astral.sh/uv/):
`uv sync --extra flow` (creates `.venv`, including pytest), or
`pip install -e ".[flow]"`.

### The compiled part

Both algorithms are C++ programs. A *runner* decides how they are executed;
`runner="auto"` (the default) takes the first one that is available:

| Runner | What runs | Needs |
|--------|-----------|-------|
| `"inprocess"` | the extension module `polyvec._native`, inside your Python process: no files, no process start - by far the fastest | a binary wheel (or a source build with the C++ dependencies, see below) |
| `"subprocess"` | the same extension module in a long-lived helper Python process | the same; a crash in the research code then only kills the helper (it is restarted on the next call), and `timeout` works |
| `"local"` | the command-line programs, compiled on this machine | `python -m polyvec build --local` |
| `"docker"` | the command-line programs in a throw-away container | Docker; the image is built from the bundled sources on first use (about five minutes, once) |

`auto` tries `inprocess`, then `local`, then `docker`; `$POLYVEC_RUNNER`
overrides it. `subprocess` is never chosen automatically - pick it when the
host process must survive whatever the research code does, or for timeouts:

```python
polyvec.vectorize("sketch.png", method="field", runner="subprocess", timeout=60)
```

```bash
python -m polyvec info      # shows the methods and which runners are ready
```

**Installing from source.** `pip install .` (or from the sdist) compiles the
extension with CMake through scikit-build-core *if* it finds OpenCV, Eigen 3,
Boost and nanobind; otherwise it warns and installs without it, and the
`local`/`docker` runners remain. The release wheels get those libraries from
[vcpkg](https://vcpkg.io), linked statically; to do the same:

```bash
ci/install-vcpkg-deps.sh x64-linux-polyvec        # Windows: ci\install-vcpkg-deps.ps1
# set the two variables it prints (CMAKE_TOOLCHAIN_FILE, SKBUILD_CMAKE_DEFINE), then
POLYVEC_REQUIRE_NATIVE=ON pip install .
```

**The command-line programs** (for the `local` runner) need CMake, a C++17
compiler and the OpenCV, Eigen 3, Boost and Qt 5 development packages - on
Ubuntu/Debian
`build-essential cmake ninja-build git qtbase5-dev libopencv-dev libboost-graph-dev libeigen3-dev`:

```bash
python -m polyvec build --local
```

## Usage

### One call

```python
import numpy as np, polyvec

image = np.asarray(...)                                  # (H, W), (H, W, 3) or (H, W, 4)
drawing = polyvec.vectorize(image, method="field")
drawing = polyvec.vectorize("sketch.png", method="flow", flow_iterations=500)
```

### Reusable objects

Create a vectorizer once when processing several images; `PolyVectorFlow`
then keeps its keypoint network loaded.

```python
from polyvec import PolyVectorField, PolyVectorFlow

field = PolyVectorField()
flow = PolyVectorFlow(ink_threshold=190)        # pick up fainter lines

for path in paths:
    flow.vectorize(path).save(path.with_suffix(".svg"))
```

### The result: `VectorDrawing`

```python
drawing.strokes        # list of (N, 2) float arrays: x right, y down, in input pixels
drawing.width, drawing.height
len(drawing), drawing.num_points, drawing.lengths()
drawing.to_svg(stroke="black", stroke_width=1.5)        # SVG text
drawing.save("out.svg")
drawing.metadata       # {"elapsed": ..., "runner": ..., "keypoints": ... (flow only)}
```

The centre of the pixel in row `i`, column `j` is the point `(j + 0.5, i + 0.5)`.

### Keypoints (PolyVector Flow)

PolyVector Flow first finds *keypoints* - endpoints, junctions and sharp
corners - with a neural network, then connects them. `vectorize()` does both,
but each stage is reachable on its own:

```python
from polyvec import PolyVectorFlow, Keypoints, KeypointType

flow = PolyVectorFlow()
keypoints = flow.detect_keypoints("sketch.png")          # stage 1 only
keypoints.points, keypoints.labels                        # (N, 2) x,y   and   (N,) of 1 / 3 / 5
keypoints.of_type(KeypointType.JUNCTION)

drawing = flow.vectorize("sketch.png", keypoints=keypoints)      # stage 2 with given keypoints
drawing = flow.vectorize("sketch.png", keypoints="sketch.pts")   # ... or an upstream .pts file
drawing = flow.vectorize("sketch.png", keypoints=np.array([[x, y, label], ...]))
```

### Text (`method="text"`)

Both methods trace drawings well and small writing poorly. `TextVectorizer`
wraps either of them and gives words special treatment:

1. **Find the words** with an OCR word detector borrowed from
   [`inksight`](../inksight-api) - docTR, PaddleOCR or Tesseract - and sort
   them into reading order.
2. **Trace each word on an upscaled crop** (`scale`, default 4x, Lanczos) with
   some context around the box, and map the strokes back.
3. **Order the strokes like writing**: the first stroke starts where writing
   starts (top-left for the default `"ltr-ttb"`), the rest follow along the
   primary direction, and every polyline's points run the way the pen moves
   (forwards along the line; top to bottom for strokes that run across it).
   Any primary/secondary pair works: `"rtl-ttb"`, `"ttb-rtl"`, `"btt-ltr"` ...
4. **Re-read (optional, off by default)**: with `verify=True` each traced word
   is rasterized again and read by the OCR engine; the character error rate
   against the detector's reading should be within `max_text_error`. It does
   not have to pass - words that never pass keep their best attempt and are
   marked `passed: False`.
5. **Iterate, or pick the best of N** (`attempts`): later attempts vary scale
   and ink threshold (or your own `variants`). `selection="first"` stops at the
   first attempt within the limits; `selection="best"` keeps the attempt with
   the lowest score: re-read error (when verifying) plus shape error (1 - F1
   between the rasterized tracing and the original ink; `max_shape_error`
   optionally limits it too).
6. **Patch the text back in**: the base method's tracing of the whole image
   loses its strokes inside the word regions and the word strokes take their
   place (`trace_page=False` returns only the words).

```python
from polyvec import TextVectorizer

text = TextVectorizer("flow", word_detector="paddleocr", direction="ltr-ttb",
                      verify=True, max_text_error=0.25, attempts=4, selection="best")
drawing = text.vectorize("note.png")
drawing.metadata["text"]                 # what was read, in reading order
drawing.metadata["text_strokes"]         # (start, stop): the word strokes come last
drawing.metadata["words"][0]             # box, region, text, read, text_error, shape_error,
                                         # passed, attempt, attempts, variant, strokes, candidates

words = text.detect_words("note.png")    # stage 1 only; edit, save, pass back:
drawing = text.vectorize("note.png", words=words)
```

Install a detector with `uv sync --extra paddle` (or `doctr`, `tesseract`).
The engines run in their own worker process, as in `inksight`.

### Options

| Option | Applies to | Meaning |
|--------|-----------|---------|
| `runner` | both | `"auto"` (extension module if installed, else local build, else Docker), `"inprocess"`, `"subprocess"`, `"local"`, `"docker"`, or a `Runner` object |
| `ink_threshold` | both | Gray level (1-254) below which a pixel is ink. The programs hard-code 165; other values are applied by adjusting image contrast first, as the upstream README advises. Raise it if faint lines vanish, lower it if neighbouring lines merge. |
| `invert` | both | Set for light-on-dark drawings |
| `timeout` | both | Abort the compiled program after this many seconds (every runner except `inprocess`: native code cannot be interrupted) |
| `workdir` | both | Keep the staged inputs, raw output and `log.txt` in this folder (debugging; the extension runners stage no files and only write `log.txt`) |
| `flow_iterations` | flow | PolyVector-flow steps per curve (upstream default 1500) |
| `keypoint_detector` | flow | Any `KeypointDetector`; default `PyramidalKeypointDetector(device=None, threshold=0.5)` |

After a call, `vectorizer.last_run.output` holds the program's complete log
(in-process too: the module captures what upstream prints rather than letting
it reach your stdout). If the program fails, a `VectorizationError` is raised
with the log in `.log`.

### Command line

```bash
python -m polyvec vectorize sketch.png -m flow -o sketch.svg
python -m polyvec vectorize sketch.png -m field --ink-threshold 190
python -m polyvec keypoints sketch.png -o sketch.pts
python -m polyvec vectorize note.png -m text --base flow -d paddleocr --verify --attempts 4 --select best
python -m polyvec words note.png -d paddleocr -o note.words.json
python -m polyvec build --docker        # or --local
```

## Design

```
Vectorizer                    abstract: image in -> VectorDrawing out
└── NativeVectorizer          abstract: ... by running a compiled upstream program
    ├── PolyVectorField       Bessmeltsev & Solomon 2019
    └── PolyVectorFlow        Puhachov et al. 2021  ── has a ──┐
                                                               ▼
KeypointDetector              abstract: image in -> Keypoints out
└── PyramidalKeypointDetector upstream network + released weights

Runner                        abstract: execute a backend program
├── InProcessRunner           extension module polyvec._native in this process (arrays in/out)
│   └── SubprocessRunner      ... in a helper process (crash isolation, timeouts)
├── LocalRunner               a binary built on this machine (files in a directory)
└── DockerRunner              a container; builds its image on demand
```

Each level adds one responsibility and leaves one method to the level below:

* **`Vectorizer`** (`src/polyvec/base.py`) fixes the public contract. `vectorize()`
  is a template method - load any image form as 8-bit gray, `preprocess()`,
  `_vectorize()`, `postprocess()` - and subclasses that set a `name` register
  themselves for `polyvec.vectorize(..., method=name)` and the command line.
* **`NativeVectorizer`** (`src/polyvec/native.py`) owns the plumbing around a
  compiled program. With a command-line runner: scratch directory, writing the
  image, invoking the program, failure detection, reading the SVG back. With
  an extension runner: handing over the image (and keypoints) as arrays and
  taking the curves back as arrays. A concrete method implements `_stage()`
  (command line and input files) and `_arguments()` (the same inputs as arrays).
* **`PolyVectorField` / `PolyVectorFlow`** (`src/polyvec/methods/`) are therefore
  tiny: the field method is one line of substance; the flow method adds the
  keypoint stage.
* **`Runner`** (`src/polyvec/runners/`) decouples *what* runs from *where*. The
  command-line programs always run inside a directory with bare file names as
  arguments, so the same call works natively, in Docker, or wherever a new
  runner sends it. The extension runners skip files: `call(tool, arrays)`.

#### The extension module

`polyvec._native` is two shared libraries, `_field` and `_flow`, built by
`src/polyvec/native/CMakeLists.txt` with nanobind and scikit-build-core. They
are separate on purpose: the two upstream trees define many functions with
the same names but different code, which must never end up in one library.

Each module is the vendored program, compiled unchanged, plus one entry file
(`src/polyvec/native/python/*_module.cpp`) in place of its `main()`:

* The entry file `#include`s upstream's `main.cpp` with `main` renamed, which
  brings in upstream's globals and helper routines, and then runs the same
  pipeline steps as `main()` - except that the image (and keypoints) come
  from NumPy arrays instead of files and the final polylines are returned
  instead of written to SVG. It applies the same `+0.5` pixel offset as
  upstream's SVG writer, so both paths share one coordinate convention.
* No Qt: the algorithm files merely include Qt headers, and
  `native/compat/headless` provides no-op stand-ins for the few names they
  mention (the debug viewer, a timer). OpenCV's `highgui` is replaced the
  same way (only one debugging routine uses it).
* Repeated calls: upstream keeps its working state in globals, and every one
  of them is reassigned at the start of a call. Calls are serialised by a lock
  and release the GIL. Upstream's `assert`s (which would abort the process)
  are compiled out, as in release builds of the programs; upstream never calls
  `exit()`.
* Output: upstream's `std::cout` and `printf` progress is captured per call
  and returned as the log; C++ exceptions become `VectorizationError`.

For PolyVector Flow the module uses the HiGHS solver through
`native/compat/gurobi`, as the Docker image does (no Gurobi in wheels).

### Repository layout

```
src/polyvec/
├── base.py, native.py        the two abstract levels
├── methods/                  PolyVectorField, PolyVectorFlow
├── keypoints/                Keypoints, KeypointDetector, PyramidalKeypointDetector
├── runners/                  InProcessRunner, SubprocessRunner, LocalRunner, DockerRunner
├── _native/                  extension modules (in wheels), subprocess helper
├── drawing.py, images.py     result type, image loading
├── build.py, __main__.py     backend builds, command line
├── native/                   CMake project, extension entry points, vcpkg manifest,
│                             Dockerfile, Qt/highgui/Gurobi stand-ins  (ours)
└── vendor/                   upstream sources and model weights       (theirs, untouched)
tests/      examples/
ci/                           vcpkg dependency install (used by cibuildwheel)
.github/workflows/wheels.yml  binary wheels for Windows, Linux and macOS
```

### Extending

**A new method backed by a program** - subclass `NativeVectorizer`:

```python
from polyvec import NativeVectorizer, NativeJob

class MyTracer(NativeVectorizer):
    name = "mytracer"            # registers it: polyvec.vectorize(img, method="mytracer")
    tool = "mytracer-cli"        # executable the runner must find

    def __init__(self, *, smoothness=1.0, **options):
        super().__init__(**options)
        self.smoothness = smoothness

    def _stage(self, workdir, image_name, gray, **inputs):
        return NativeJob(args=[image_name, "out.svg", str(self.smoothness)], output="out.svg")
```

**A pure-Python method** - subclass `Vectorizer` and implement
`_vectorize(gray) -> VectorDrawing`.

**A different keypoint detector** - subclass `KeypointDetector`, implement
`detect(image) -> Keypoints`, pass it as `PolyVectorFlow(keypoint_detector=...)`.

**A different place to run** (WSL, SSH, a cluster) - subclass `Runner`,
implement `is_available()` and `run()`, pass it as `runner=...`.

**Different pre/post-processing** - override `preprocess(gray)` or
`postprocess(drawing)` on any vectorizer.

## What is wrapped, and how faithfully

The upstream sources live unmodified in `src/polyvec/vendor/` (commits and licences
in [`src/polyvec/vendor/UPSTREAM.md`](src/polyvec/vendor/UPSTREAM.md)). What `polyvec`
adds sits beside them:

* `src/polyvec/native/CMakeLists.txt` builds both programs, as command-line
  programs and as the extension module. It replaces the upstream CMake files,
  which hard-code machine-specific paths. The extension's entry files mirror
  upstream's `main()` with the file I/O swapped for arrays (see
  [The extension module](#the-extension-module)).
* The keypoint detector imports upstream's `layers.py`, `sketchyimage.py` and
  `usemodel.py` and calls them in the order upstream's script does; it returns
  the same keypoints as that script (checked on the bundled example).
* **Solver.** PolyVector Flow is written against the commercial Gurobi
  optimizer. So that it runs without a licence, `src/polyvec/native/compat/gurobi`
  implements the few Gurobi classes the code uses on top of the open-source
  [HiGHS](https://highs.dev) solver; the upstream code compiles against it
  unchanged. On the bundled example this reproduces upstream's published
  result (same 38 strokes, largest deviation 0.04 px). The optimisation
  problems can have several equally good solutions, so on other drawings an
  occasional different-but-equivalent curve choice is possible. To use real
  Gurobi instead:

  ```bash
  python -m polyvec build --local --solver gurobi --gurobi-home /opt/gurobi1103/linux64
  ```

### The keypoint (heat-map) model

The network the PolyVector Flow paper describes in Sec. 3 and Appendix B is
real, released, and bundled here:

| | |
|-|-|
| Code | `src/polyvec/vendor/pvflow/prediction/layers.py` - class `PyramidalNet` (hourglass with skip connections, 1.9 M parameters) |
| Weights | `src/polyvec/vendor/pvflow/prediction/best_model_checkpoint.pth` (23.6 MB, the authors' released checkpoint) |
| Output | as in the paper: four full-resolution heat maps - any keypoint, endpoint, junction, sharp corner - plus two down-scaled copies of the first (0.25x, 0.5x) that only matter for training |
| Post-processing | peaks of the "any keypoint" map above a threshold (non-maximum suppression, minimum distance 1 px); each peak takes the type whose heat map is highest there |

`PyramidalKeypointDetector` runs exactly this, and keeps the four heat maps of
the last image for inspection:

```python
detector = polyvec.PyramidalKeypointDetector()
keypoints = detector.detect("sketch.png")
detector.last_heatmaps["detection"]      # (H, W) float arrays; also "endpoint",
                                         # "junction" and "sharp_corner"
```

What upstream did **not** release in that repository is the training script;
the training data is in a
[separate repository](https://github.com/ivanpuhachov/line-drawing-vectorization-polyvector-flow-dataset).
So the model can be used and inspected, but not retrained from what is here.

### Overlap between the two C++ codebases

PolyVector Flow's C++ is a fork of the 2019 code, extended with the flow.
Of the 75 files in `vendor/pv/src`, 74 reappear in `vendor/pvflow/src` under
the same name - 48 byte-identical, 26 diverged (`main.cpp` substantially) -
which accounts for roughly half of the flow tree; the other half is 37 new
files. The 2019 repository has also moved on since (2025: faster linear solve,
per-component processing), so the two differ in both directions.

The trees are deliberately kept and compiled separately: nearly all of the
"identical" `.cpp` files include headers that differ between the trees
(`stdafx.h`, `typedefs.h`, `graph_typedefs.h`), so they do not compile to the
same thing, and merging them would mean editing upstream code. On the Python
side there is no duplication - both methods share `NativeVectorizer`.

### Things to know

* Both programs separate ink from paper with a fixed threshold; most
  "missing line" problems are fixed with `ink_threshold` (or by increasing the
  image contrast yourself).
* The tuning constants in upstream's `Params.h` are compile-time. Edit the
  file in `src/polyvec/vendor/*/src/` and rebuild if you need to change them; the
  Docker image is rebuilt automatically when sources change.
* Run time grows quickly with image size and stroke count; the papers'
  examples are a few hundred pixels across. Downscale large scans first.

### Platform support

The Python package itself is platform-neutral (pure Python, `pathlib`
throughout). What differs per platform is how the C++ programs get built and
run:

| Platform | Binary wheel (`inprocess` / `subprocess`) | Docker runner | Local build (`build --local`) |
|----------|-------------------------------------------|---------------|-------------------------------|
| Linux x86_64 | **built and tested** locally: manylinux_2_28 wheels for CPython 3.10 - 3.14 built in the manylinux container exactly as CI does (vcpkg, static); the full test suite passes on the installed 3.10 and 3.14 wheels, and on Ubuntu 22.04 the in-process results match the command-line programs (field: identical up to SVG rounding; flow: same 38 strokes, at most 0.02 px apart). Not yet run through GitHub Actions. | expected to work (the image *is* Linux); not run from a Linux host | **tested** on Ubuntu 22.04 (GCC 11, Python 3.10) |
| Windows x86_64 | **configured, not built yet**: the workflow builds it with MSVC + vcpkg (`x64-windows-polyvec`); nobody has run that job | **tested** (Windows 11, Docker Desktop, Python 3.13 and 3.14) | not tested. Needs Visual Studio's C++ tools plus OpenCV, Eigen, Boost and Qt 5 (e.g. via vcpkg) and Qt's DLLs next to the executables. |
| macOS arm64 / x86_64 | **configured, not built yet** (workflow jobs on `macos-15` / `macos-15-intel`). Apple Clang has no OpenMP, so these wheels run single-threaded. | not tested | not tested. Would need Homebrew `cmake qt@5 opencv boost eigen libomp`. |

Without a wheel for your platform, `pip install` builds from the sdist: with
OpenCV/Eigen/Boost available the extension is compiled, without them (or
without any C++ compiler) you get the pure-Python package and the Docker or
local runners.

## Tests

```bash
uv run pytest               # unit tests always run; end-to-end tests need a backend
```

Tests that need a compiled backend or PyTorch skip themselves when those are
missing. `tests/test_native.py` covers the extension module (in-process vs.
helper process, repeated calls, captured output, timeouts); with
`POLYVEC_TEST_CLI_RUNNER=local` (or `docker`) it also compares the in-process
strokes with those of the command-line programs. Note that `pythonpath = ["src"]`
makes pytest import the source tree, which has no compiled modules; to test an
installed wheel, run `pytest -o pythonpath= tests` (as CI does).

## Licence and citation

The wrapper code is MIT-licensed (see `LICENSE`). The bundled upstream code
keeps its own licences (MIT for both methods, Boost Software License for the
PAAL headers); see `src/polyvec/vendor/UPSTREAM.md`. If you use a method in
research, cite its paper:

```bibtex
@article{Bessmeltsev2019,
  author  = {Bessmeltsev, Mikhail and Solomon, Justin},
  title   = {Vectorization of Line Drawings via Polyvector Fields},
  journal = {ACM Trans. Graph.}, volume = {38}, number = {1}, year = {2019},
  articleno = {9}, doi = {10.1145/3202661}
}
@article{Puhachov2021KeypointPolyvector,
  author  = {Ivan Puhachov and William Neveu and Edward Chien and Mikhail Bessmeltsev},
  title   = {Keypoint-Driven Line Drawing Vectorization via PolyVector Flow},
  journal = {ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia)},
  volume  = {40}, number = {6}, year = {2021}, doi = {10.1145/3478513.3480529}
}
```
