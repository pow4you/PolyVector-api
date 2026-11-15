"""The keypoint network released with PolyVector Flow, behind a plain ``detect()``.

Nothing here re-implements the model.  The network definition (``layers.py``),
the image normalisation and peak extraction (``sketchyimage.py``), the tiling
inference loop (``usemodel.py``) and the trained weights are the upstream files
in ``polyvec/vendor/pvflow/prediction``; this module only loads them and calls
them in the same order as upstream's ``predict_image`` - minus its plotting and
log files.
"""

from __future__ import annotations

import importlib.util
import os
import pickle
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional, Union

import numpy as np

from ..errors import MissingDependencyError, PolyvecError
from ..images import ImageLike, load_grayscale, save_png
from ..paths import KEYPOINT_CHECKPOINT, PREDICTION_DIR
from .base import KeypointDetector, Keypoints

#: Upstream modules, in dependency order.  They import each other by these
#: flat names (``from layers import ...``), as scripts run from their folder.
_UPSTREAM_MODULES = ("layers", "sketchyimage", "usemodel")

_upstream: Optional[SimpleNamespace] = None


def _load_upstream() -> SimpleNamespace:
    """Import the upstream prediction scripts without touching ``sys.path``.

    Each file is loaded under a private module name.  While loading, it is also
    visible under its flat upstream name so the files' imports of one another
    resolve; afterwards any module the caller already had under that name
    (a project-level ``layers.py``, say) is put back.
    """
    global _upstream
    if _upstream is not None:
        return _upstream

    try:
        import matplotlib  # noqa: F401  (upstream imports pyplot at module level)
        import skimage  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError(
            f"keypoint detection needs PyTorch, scikit-image and matplotlib ({exc}). "
            "Install them with:  pip install 'polyvec[flow]'"
        ) from exc

    loaded = {}
    shadowed = {name: sys.modules.get(name) for name in _UPSTREAM_MODULES}
    try:
        for name in _UPSTREAM_MODULES:
            spec = importlib.util.spec_from_file_location(
                f"polyvec._upstream_prediction_{name}", PREDICTION_DIR / f"{name}.py")
            module: ModuleType = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name, previous in shadowed.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    _upstream = SimpleNamespace(**loaded)
    return _upstream


def _numpy_scalar_globals() -> list:
    """The NumPy callables a pickled NumPy scalar refers to.

    The scalar constructor is given with the module path under which NumPy 1.x
    pickles recorded it, because NumPy 2 moved it to ``numpy._core``.
    """
    recorded_path = "numpy.core.multiarray"
    scalar = (getattr(np, "_core", None) or np.core).multiarray.scalar
    if scalar.__module__ != recorded_path:
        scalar = (scalar, f"{recorded_path}.scalar")
    return [scalar, np.dtype, type(np.dtype(np.float64))]


class _OnDevice:
    """Presents the network to upstream's inference loop on a device of our choice.

    Upstream's ``model_predict`` moves the input to ``cuda`` whenever a GPU
    exists, regardless of where the model is.  This stand-in for the model
    moves each input tile to the model's own device (and to float32) before the
    forward pass, so the caller - not upstream - decides the device.
    """

    can_classify = True

    def __init__(self, model: Any, device: Any) -> None:
        self._model = model
        self._device = device

    def eval(self) -> "_OnDevice":
        self._model.eval()
        return self

    def forward_class_heatmaps(self, tile: Any) -> Any:
        return self._model.forward_class_heatmaps(tile.to(device=self._device, dtype=self._dtype()))

    def _dtype(self) -> Any:
        return next(self._model.parameters()).dtype


class PyramidalKeypointDetector(KeypointDetector):
    """Keypoint detector of Puhachov et al. 2021 (``PyramidalNet`` + released weights).

    The network and its weights are loaded on first use, so constructing a
    detector is free and PyTorch is only required once :meth:`detect` runs.

    Args:
        checkpoint: Weights file; defaults to the checkpoint bundled with the
            package (``best_model_checkpoint.pth`` from the upstream repository).
        device: ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ...  Default: the GPU if
            PyTorch can use one, else the CPU.
        threshold: Heat-map value (0-1) above which a peak counts as a keypoint.
            0.5 is upstream's value; lower it to detect more points.

    Attributes:
        last_heatmaps: After :meth:`detect`, the network's raw outputs for the
            most recent image: ``"detection"``, ``"endpoint"``, ``"sharp_corner"``
            and ``"junction"`` heat maps as ``(H, W)`` arrays.
    """

    def __init__(self, checkpoint: Union[str, "os.PathLike[str]", None] = None,
                 device: Optional[str] = None, threshold: float = 0.5) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint is not None else KEYPOINT_CHECKPOINT
        self.device = device
        self.threshold = threshold
        self.last_heatmaps: dict = {}
        self._model: Optional[_OnDevice] = None

    # ------------------------------------------------------------------ model
    def load(self) -> "PyramidalKeypointDetector":
        """Load the network now instead of lazily on the first :meth:`detect`."""
        if self._model is not None:
            return self
        upstream = _load_upstream()
        import torch

        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"keypoint checkpoint not found: {self.checkpoint}")
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = self._read_checkpoint(torch, device)

        network = upstream.layers.PyramidalNet()
        network.load_state_dict(state["model_state_dict"])
        network.to(device).eval()
        self._model = _OnDevice(network, device)
        return self

    def _read_checkpoint(self, torch: Any, device: Any) -> dict:
        """Read the checkpoint onto ``device`` with PyTorch's restricted unpickler.

        Besides tensors, the released checkpoint stores its training curves as
        NumPy scalars, which the restricted ("weights only") loader rejects
        unless those few NumPy types are allow-listed - so that is what we do,
        rather than fall back to unrestricted unpickling.
        """
        def load(**options: Any) -> dict:
            return torch.load(self.checkpoint, map_location=device, **options)

        allow_list = getattr(torch.serialization, "safe_globals", None)
        if allow_list is None:
            # PyTorch < 2.4: no allow-list API; unrestricted loading was its default.
            return load()
        try:
            with allow_list(_numpy_scalar_globals()):
                return load(weights_only=True)
        except pickle.UnpicklingError as exc:
            raise PolyvecError(
                f"{self.checkpoint} contains pickled objects other than tensors and NumPy "
                f"scalars and was not loaded for safety ({str(exc).splitlines()[0]})"
            ) from exc

    # ----------------------------------------------------------------- detect
    def detect(self, image: ImageLike) -> Keypoints:
        self.load()
        upstream = _load_upstream()
        gray = load_grayscale(image)

        # Upstream reads its input from a file, so hand it one.
        with tempfile.TemporaryDirectory(prefix="polyvec-kp-") as scratch:
            staged = save_png(gray, Path(scratch) / "input.png")
            sketch = upstream.sketchyimage.SketchyImage.from_image(str(staged), normalize=True)

        sketch.heatmap_threshold = self.threshold
        upstream.usemodel.process_image(mymodel=self._model, sketchyimage=sketch)

        height, width = gray.shape
        endpoint, sharp, junction = (hm[:height, :width] for hm in sketch.pred_class_heatmaps)
        self.last_heatmaps = {
            "detection": sketch.pred_heatmap[:height, :width],
            "endpoint": endpoint, "sharp_corner": sharp, "junction": junction,
        }

        points = np.asarray(sketch.pred_points, dtype=np.float64).reshape(-1, 2)   # x, y
        labels = np.asarray(sketch.pred_labels, dtype=np.int64).reshape(-1)
        # Upstream pads the image to a multiple of 8; ignore anything found in the padding.
        inside = (points[:, 0] < width) & (points[:, 1] < height)
        return Keypoints(points=points[inside], labels=labels[inside])

    def __repr__(self) -> str:
        return (f"PyramidalKeypointDetector(checkpoint={self.checkpoint.name!r}, "
                f"device={self.device!r}, threshold={self.threshold})")
