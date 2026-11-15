"""Compiling the native programs - as a Docker image or directly on this machine.

Both routes build the same CMake project (``polyvec/native``) from the same
bundled sources (``polyvec/vendor``).  You rarely call this module yourself:
:class:`~polyvec.runners.DockerRunner` builds its image on first use, and the
command line exposes both routes as ``python -m polyvec build``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from .errors import BackendBuildError
from .paths import NATIVE_DIR, PACKAGE_DIR, VENDOR_DIR, user_data_dir

logger = logging.getLogger("polyvec")

#: Names of the executables the CMake project installs.
TOOLS = ("polyvec-field", "polyvec-flow")

#: Repository part of the Docker image name; the tag is a hash of the sources.
DOCKER_REPOSITORY = "polyvec-backends"

_SOURCE_SUFFIXES = {".cpp", ".h", ".hpp", ".ui", ".txt", ".cmake"}


# --------------------------------------------------------------------------- sources
def native_source_files() -> List[Path]:
    """Every file that influences the compiled programs, in a stable order."""
    roots = [NATIVE_DIR, VENDOR_DIR / "pv" / "src", VENDOR_DIR / "pvflow" / "src",
             VENDOR_DIR / "pvflow" / "LBFGS", VENDOR_DIR / "paal"]
    files = [
        path for root in roots for path in root.rglob("*")
        if path.is_file() and (path.suffix in _SOURCE_SUFFIXES or path.name == "Dockerfile")
    ]
    return sorted(files)


def source_fingerprint(extra: Iterable[str] = ()) -> str:
    """Short hash of the native sources (plus ``extra`` strings, e.g. build options).

    Used as the Docker image tag so that editing the C++ code, the CMake file
    or the build options transparently triggers a rebuild.  Line endings are
    ignored, so a Windows checkout (CRLF) and a Linux one (LF) of the same
    sources agree on the tag.
    """
    digest = hashlib.sha256()
    for path in native_source_files():
        digest.update(path.relative_to(PACKAGE_DIR).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    for item in extra:
        digest.update(b"\0" + item.encode())
    return digest.hexdigest()[:12]


# --------------------------------------------------------------------------- docker
def default_image_name(build_args: Optional[Dict[str, str]] = None) -> str:
    """Image name for the current sources and build arguments."""
    options = sorted(f"{key}={value}" for key, value in (build_args or {}).items()
                     if key != "POLYVEC_BUILD_JOBS")       # speed only, same result
    return f"{DOCKER_REPOSITORY}:{source_fingerprint(options)}"


def build_docker_image(image: Optional[str] = None, *, docker: str = "docker",
                       build_args: Optional[Dict[str, str]] = None,
                       quiet: bool = True) -> str:
    """Build the image containing both backends and return its name.

    Args:
        image: Name to tag the image with; defaults to :func:`default_image_name`.
        docker: Docker-compatible CLI to invoke (``"docker"``, ``"podman"``, ...).
        build_args: ``--build-arg`` values, see ``native/Dockerfile``
            (``POLYVEC_BUILD_JOBS``).
        quiet: Capture the build log (shown only on failure) instead of streaming it.

    Raises:
        BackendBuildError: Docker is missing or the build failed.
    """
    image = image or default_image_name(build_args)
    command = [docker, "build", "--file", str(NATIVE_DIR / "Dockerfile"), "--tag", image]
    for key, value in (build_args or {}).items():
        command += ["--build-arg", f"{key}={value}"]
    command.append(str(PACKAGE_DIR))

    logger.warning("polyvec: building Docker image %s - a one-time step that takes "
                   "several minutes", image)
    _run_build_step(command, quiet=quiet, what="Docker image build")
    return image


# --------------------------------------------------------------------------- local
def build_local(prefix: Union[str, "os.PathLike[str]", None] = None, *,
                solver: str = "auto", gurobi_home: Optional[str] = None,
                jobs: Optional[int] = None, cmake_args: Sequence[str] = (),
                build_dir: Union[str, "os.PathLike[str]", None] = None,
                quiet: bool = False) -> Path:
    """Compile the backends with the local toolchain and return the ``bin`` directory.

    Needs CMake >= 3.16, a C++17 compiler and the development packages of
    OpenCV, Eigen 3, Boost and Qt 5 (see the README for per-platform hints).

    Args:
        prefix: Install prefix; executables land in ``<prefix>/bin``.  Defaults
            to the per-user location that :class:`~polyvec.runners.LocalRunner`
            searches automatically.
        solver: ``"auto"``, ``"gurobi"`` or ``"highs"`` (``POLYVEC_SOLVER``).
        gurobi_home: Gurobi installation to link against when using Gurobi.
        jobs: Parallel compile jobs (each needs ~1.5 GB of RAM).  Default: 4.
        cmake_args: Extra arguments for the configure step, e.g.
            ``["-DCMAKE_PREFIX_PATH=C:/Qt/5.15.2/msvc2019_64"]``.
        build_dir: Where to keep the build tree; a temporary directory if omitted.
        quiet: Capture the build log (shown only on failure) instead of streaming it.

    Raises:
        BackendBuildError: CMake is missing or a build step failed.
    """
    if shutil.which("cmake") is None:
        raise BackendBuildError("cmake was not found on PATH; install CMake >= 3.16 "
                                "or use the Docker runner instead")
    prefix_path = Path(prefix) if prefix is not None else user_data_dir()
    jobs = jobs or 4

    def configure_build_install(tree: Path) -> None:
        configure = ["cmake", "-S", str(NATIVE_DIR), "-B", str(tree),
                     "-DCMAKE_BUILD_TYPE=Release", f"-DPOLYVEC_SOLVER={solver}"]
        if gurobi_home:
            configure.append(f"-DGUROBI_HOME={gurobi_home}")
        if shutil.which("ninja") and not any(arg.startswith("-G") for arg in cmake_args):
            configure += ["-G", "Ninja"]
        configure += list(cmake_args)
        _run_build_step(configure, quiet=quiet, what="CMake configure")
        _run_build_step(["cmake", "--build", str(tree), "--config", "Release",
                         "--parallel", str(jobs)], quiet=quiet, what="compilation")
        _run_build_step(["cmake", "--install", str(tree), "--config", "Release",
                         "--prefix", str(prefix_path)], quiet=quiet, what="installation")

    if build_dir is not None:
        Path(build_dir).mkdir(parents=True, exist_ok=True)
        configure_build_install(Path(build_dir))
    else:
        with tempfile.TemporaryDirectory(prefix="polyvec-build-") as tree:
            configure_build_install(Path(tree))

    return prefix_path / "bin"


# --------------------------------------------------------------------------- helpers
def _run_build_step(command: Sequence[str], *, quiet: bool, what: str) -> None:
    """Run one build command; on failure raise with the tail of its output."""
    try:
        if quiet:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            output = completed.stdout or ""
        else:
            completed = subprocess.run(command)
            output = ""
    except FileNotFoundError as exc:
        raise BackendBuildError(f"{what} failed: '{command[0]}' is not installed") from exc
    if completed.returncode != 0:
        tail = "\n".join(output.splitlines()[-40:])
        raise BackendBuildError(
            f"{what} failed (exit status {completed.returncode}).\n"
            f"Command: {' '.join(map(str, command))}\n{tail}".rstrip()
        )
