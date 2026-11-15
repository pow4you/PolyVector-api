"""Where things live: bundled upstream sources, build glue, and the per-user install."""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: The installed ``polyvec`` package directory.
PACKAGE_DIR = Path(__file__).resolve().parent

#: CMake project, Dockerfile and solver shim used to compile the backends.
NATIVE_DIR = PACKAGE_DIR / "native"

#: Unmodified upstream sources (see ``vendor/UPSTREAM.md``).
VENDOR_DIR = PACKAGE_DIR / "vendor"

#: Upstream keypoint-prediction code and the released network weights.
PREDICTION_DIR = VENDOR_DIR / "pvflow" / "prediction"
KEYPOINT_CHECKPOINT = PREDICTION_DIR / "best_model_checkpoint.pth"

#: Small drawing shipped with PolyVector Flow; handy for smoke tests.
EXAMPLE_IMAGE = VENDOR_DIR / "pvflow" / "inputs" / "example.png"
EXAMPLE_KEYPOINTS = VENDOR_DIR / "pvflow" / "inputs" / "example.pts"


def user_data_dir() -> Path:
    """Per-user directory for locally compiled backends.

    ``$POLYVEC_HOME`` if set, else ``%LOCALAPPDATA%\\polyvec`` on Windows,
    ``~/Library/Application Support/polyvec`` on macOS and
    ``$XDG_DATA_HOME/polyvec`` (default ``~/.local/share/polyvec``) elsewhere.
    """
    override = os.environ.get("POLYVEC_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "polyvec"


def default_bin_dir() -> Path:
    """Where ``python -m polyvec build --local`` installs the executables."""
    return user_data_dir() / "bin"
