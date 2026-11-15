# Bundled upstream code

Everything in this directory is third-party code, copied **unmodified** from
the sources below. polyvec builds and calls it; it does not patch it. To
update, replace the files with those of a newer upstream commit and note the
commit here.

| Directory | Source | Commit | Licence |
|-----------|--------|--------|---------|
| `pv/` | https://github.com/bmpix/PolyVectorization | `d296439370b2b7398d9ef70098fa0c42f67e0e8d` (2025-11-14) | MIT (`pv/LICENSE`) |
| `pvflow/` | https://github.com/ivanpuhachov/line-drawing-vectorization-polyvector-flow | `705bfeb048ec3727c2276a909d300439821900b1` (2025-03-23) | MIT (`pvflow/LICENSE`) |
| `paal/` | http://paal.mimuw.edu.pl/ (`paal.zip`, downloaded 2026-11-15; headers dated 2014) | - | Boost Software License 1.0 (`paal/LICENSE_1_0.txt`) |

## What was copied

* `pv/` - `src/`, `LICENSE`, `README.md`.
  Not copied: `sample_inputs/` (third-party artwork), the upstream `CMakeLists.txt`,
  `Dockerfile` and `Makefile` (replaced by `polyvec/native`).
* `pvflow/` - `src/`, `LBFGS/`, `prediction/` (network code **and the trained
  checkpoint** `best_model_checkpoint.pth`), `inputs/example.png` + `example.pts`,
  `LICENSE`, `readme.md`.
  Not copied: `matlab/`, `labeled_drawings/`, `logs/`, shell scripts, the upstream
  CMake files.
* `paal/` - only the four headers that `pvflow/src/TopologyOptimizer.cpp` pulls
  in through `paal/greedy/steiner_tree_greedy.hpp`. The upstream README asks you
  to download PAAL separately; bundling the headers removes that step.

## Papers

* M. Bessmeltsev, J. Solomon. *Vectorization of Line Drawings via PolyVector
  Fields.* ACM Transactions on Graphics 38(1), 2019.
* I. Puhachov, W. Neveu, E. Chien, M. Bessmeltsev. *Keypoint-Driven Line Drawing
  Vectorization via PolyVector Flow.* ACM Transactions on Graphics 40(6)
  (SIGGRAPH Asia), 2021. doi:10.1145/3478513.3480529
