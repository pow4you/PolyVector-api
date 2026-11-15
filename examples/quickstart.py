"""Vectorize the bundled example drawing with both methods.

    python examples/quickstart.py [image] [output-folder]

Writes <name>_field.svg, <name>_flow.svg and <name>.pts into the output folder
(default: the current directory).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

import polyvec
from polyvec import KeypointType, PolyVectorField, PolyVectorFlow
from polyvec.paths import EXAMPLE_IMAGE


def main() -> None:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else EXAMPLE_IMAGE
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # --- PolyVector Fields (2019): needs only the image.  Here from a file path.
    field = PolyVectorField()
    drawing = field.vectorize(image_path)
    print("field:", drawing, f"in {drawing.metadata['elapsed']:.1f} s")
    drawing.save(out_dir / f"{stem}_field.svg")

    # --- PolyVector Flow (2021): keypoints + flow.  Here from a NumPy array.
    pixels = np.asarray(Image.open(image_path).convert("RGB"))
    flow = PolyVectorFlow()

    keypoints = flow.detect_keypoints(pixels)                 # stage 1 on its own
    print("keypoints:", keypoints)
    print("  junctions at:", keypoints.of_type(KeypointType.JUNCTION)[:3].tolist(), "...")
    keypoints.save(out_dir / f"{stem}.pts")

    drawing = flow.vectorize(pixels, keypoints=keypoints)     # stage 2 with those keypoints
    print("flow: ", drawing, f"in {drawing.metadata['elapsed']:.1f} s")
    drawing.save(out_dir / f"{stem}_flow.svg", stroke="crimson", stroke_width=1.5)

    # The strokes are plain arrays of x, y pixel coordinates.
    longest = max(drawing, key=len)
    print(f"longest stroke: {len(longest)} points, from {longest[0].round(1)} to {longest[-1].round(1)}")

    # The same thing in one call:
    #   polyvec.vectorize(image_path, method="flow").save("out.svg")
    print("available methods:", polyvec.available_methods())


if __name__ == "__main__":
    main()
