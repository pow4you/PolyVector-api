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
pip install -e ".[flow]"
```

* The base install needs only NumPy and Pillow and is enough for `PolyVectorField`.
* The `flow` extra adds PyTorch, scikit-image and matplotlib, which the
  PolyVector Flow keypoint network (upstream's model, weights included) needs.
  A GPU is used when PyTorch can see one, but the CPU is fine: a few seconds
  per image.

### The compiled part

Both algorithms are C++ programs. You do not have to build anything by hand:

* **Docker (default when no local build exists).** With Docker running, the
  first `vectorize()` call builds a `polyvec-backends` image from the bundled
  sources (about five minutes, once) and every call after that runs in a
  throw-away container. Nothing else to install - this is the route that works
  out of the box on Windows and macOS.
* **Local build.** On a machine with CMake, a C++17 compiler and the OpenCV,
  Eigen 3, Boost and Qt 5 development packages:

  ```bash
  python -m polyvec build --local
  ```

  On Ubuntu/Debian the packages are
  `build-essential cmake ninja-build git qtbase5-dev libopencv-dev libboost-graph-dev libeigen3-dev`.
  Local binaries are preferred automatically once they exist, and avoid the
  container start-up on every call.

```bash
python -m polyvec info      # shows the methods and which backends are ready
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

### Options

| Option | Applies to | Meaning |
|--------|-----------|---------|
| `runner` | both | `"auto"` (local build if present, else Docker), `"local"`, `"docker"`, or a `Runner` object |
| `ink_threshold` | both | Gray level (1-254) below which a pixel is ink. The programs hard-code 165; other values are applied by adjusting image contrast first, as the upstream README advises. Raise it if faint lines vanish, lower it if neighbouring lines merge. |
| `invert` | both | Set for light-on-dark drawings |
| `timeout` | both | Abort the compiled program after this many seconds |
| `workdir` | both | Keep the staged inputs, raw output and `log.txt` in this folder (debugging) |
| `flow_iterations` | flow | PolyVector-flow steps per curve (upstream default 1500) |
| `keypoint_detector` | flow | Any `KeypointDetector`; default `PyramidalKeypointDetector(device=None, threshold=0.5)` |

After a call, `vectorizer.last_run.output` holds the program's complete log. If
the program fails, a `VectorizationError` is raised with the log in `.log`.

### Command line

```bash
python -m polyvec vectorize sketch.png -m flow -o sketch.svg
python -m polyvec vectorize sketch.png -m field --ink-threshold 190
python -m polyvec keypoints sketch.png -o sketch.pts
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

Runner                        abstract: execute a backend program in a directory
├── LocalRunner               a binary built on this machine
└── DockerRunner              a container; builds its image on demand
```

Each level adds one responsibility and leaves one method to the level below:

* **`Vectorizer`** (`polyvec/base.py`) fixes the public contract. `vectorize()`
  is a template method - load any image form as 8-bit gray, `preprocess()`,
  `_vectorize()`, `postprocess()` - and subclasses that set a `name` register
  themselves for `polyvec.vectorize(..., method=name)` and the command line.
* **`NativeVectorizer`** (`polyvec/native.py`) owns the plumbing around a
  command-line program: scratch directory, writing the image, invoking the
  program through a runner, failure detection, reading the SVG back. A concrete
  method only implements `_stage()`, which says what to run.
* **`PolyVectorField` / `PolyVectorFlow`** (`polyvec/methods/`) are therefore
  tiny: the field method is one line of substance; the flow method adds the
  keypoint stage.
* **`Runner`** (`polyvec/runners/`) decouples *what* runs from *where*. Programs
  always run inside a directory with bare file names as arguments, so the same
  call works natively, in Docker, or wherever a new runner sends it.

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

The upstream sources live unmodified in `polyvec/vendor/` (commits and licences
in [`polyvec/vendor/UPSTREAM.md`](polyvec/vendor/UPSTREAM.md)). What `polyvec`
adds sits beside them:

* `polyvec/native/CMakeLists.txt` builds both programs. It replaces the
  upstream CMake files, which hard-code machine-specific paths.
* The keypoint detector imports upstream's `layers.py`, `sketchyimage.py` and
  `usemodel.py` and calls them in the order upstream's script does; it returns
  the same keypoints as that script (checked on the bundled example).
* **Solver.** PolyVector Flow is written against the commercial Gurobi
  optimizer. So that it runs without a licence, `polyvec/native/compat/gurobi`
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

Things to know:

* Both programs separate ink from paper with a fixed threshold; most
  "missing line" problems are fixed with `ink_threshold` (or by increasing the
  image contrast yourself).
* The tuning constants in upstream's `Params.h` are compile-time. Edit the
  file in `polyvec/vendor/*/src/` and rebuild if you need to change them; the
  Docker image is rebuilt automatically when sources change.
* Run time grows quickly with image size and stroke count; the papers'
  examples are a few hundred pixels across. Downscale large scans first.
* Tested on Windows 11 through the Docker runner only. The Docker image
  compiles the same CMake project on Linux/GCC, but `build --local` itself and
  a native Windows/MSVC build have not been tried yet.

## Tests

```bash
pip install -e ".[flow,dev]"
pytest                      # unit tests always run; end-to-end tests need a backend
```

## Licence and citation

The wrapper code is MIT-licensed (see `LICENSE`). The bundled upstream code
keeps its own licences (MIT for both methods, Boost Software License for the
PAAL headers); see `polyvec/vendor/UPSTREAM.md`. If you use a method in
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
