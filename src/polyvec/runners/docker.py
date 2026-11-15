"""Run backend programs inside a Docker container - no local C++ toolchain needed."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..build import build_docker_image, default_image_name
from ..errors import BackendNotAvailableError, VectorizationError
from .base import Runner, RunResult

#: Seconds allowed for cheap Docker queries (``docker version``, ``image inspect``).
_QUERY_TIMEOUT = 30


class DockerRunner(Runner):
    """Executes the backends from the ``polyvec-backends`` Docker image.

    The image is compiled from the bundled sources the first time it is needed
    (several minutes, once) and reused afterwards; it is rebuilt automatically
    when those sources change.  Each run mounts the scratch directory at
    ``/work`` in a fresh, network-less container that is removed afterwards.

    Args:
        image: Image to use.  Default: a name derived from the bundled sources.
        auto_build: Build the image on first use if it does not exist yet.
        docker: Docker-compatible CLI to invoke (``"docker"``, ``"podman"``, ...).
        build_args: ``--build-arg`` values for the image build, e.g.
            ``{"POLYVEC_BUILD_JOBS": "8"}`` (see ``native/Dockerfile``).
        run_args: Extra ``docker run`` arguments, e.g. ``["--cpus", "4"]``.
        network: Container network mode.  The programs need no network access,
            so the default is ``"none"``.
    """

    name = "docker"

    def __init__(self, image: Optional[str] = None, *, auto_build: bool = True,
                 docker: str = "docker", build_args: Optional[Dict[str, str]] = None,
                 run_args: Sequence[str] = (), network: str = "none") -> None:
        self.docker = docker
        self.auto_build = auto_build
        self.build_args = dict(build_args or {})
        self.run_args = list(run_args)
        self.network = network
        self._image = image
        self._image_ready = False

    @property
    def image(self) -> str:
        """Name of the image this runner uses (computed lazily from the sources)."""
        if self._image is None:
            self._image = default_image_name(self.build_args)
        return self._image

    # ----------------------------------------------------------- availability
    def daemon_running(self) -> bool:
        """Whether the Docker CLI exists and can reach its engine."""
        if shutil.which(self.docker) is None:
            return False
        return self._query([self.docker, "version", "--format", "{{.Server.Version}}"])

    def image_exists(self) -> bool:
        return self._query([self.docker, "image", "inspect", self.image])

    def is_available(self, tool: str) -> bool:
        return self.daemon_running() and (self.auto_build or self.image_exists())

    def ensure_image(self) -> str:
        """Make sure the image exists, building it if allowed; return its name."""
        if self._image_ready:
            return self.image
        if not self.daemon_running():
            raise BackendNotAvailableError(
                f"cannot reach Docker via '{self.docker}': is Docker installed and running?"
            )
        if not self.image_exists():
            if not self.auto_build:
                raise BackendNotAvailableError(
                    f"Docker image '{self.image}' does not exist and auto_build is off; "
                    "run `python -m polyvec build --docker`"
                )
            build_docker_image(self.image, docker=self.docker, build_args=self.build_args)
        self._image_ready = True
        return self.image

    # -------------------------------------------------------------------- run
    def run(self, tool: str, args: Sequence[str], workdir: Path,
            timeout: Optional[float] = None) -> RunResult:
        image = self.ensure_image()
        container = f"polyvec-{uuid.uuid4().hex[:12]}"
        command: List[str] = [
            self.docker, "run", "--rm", "--name", container, "--network", self.network,
            "--mount", f"type=bind,source={Path(workdir).resolve()},target=/work",
            "--workdir", "/work",
        ]
        if hasattr(os, "getuid"):
            # Write results as the calling user so the scratch directory stays removable.
            command += ["--user", f"{os.getuid()}:{os.getgid()}",
                        "--env", "HOME=/tmp", "--env", "XDG_RUNTIME_DIR=/tmp"]
        command += [*self.run_args, image, tool, *map(str, args)]

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            # Killing the CLI does not stop the container; do that explicitly.
            subprocess.run([self.docker, "kill", container],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            partial = exc.output if isinstance(exc.output, str) else ""
            raise VectorizationError(f"'{tool}' did not finish within {timeout:g} s", partial) from exc
        return RunResult(command, completed.returncode, completed.stdout or "",
                         time.perf_counter() - started)

    def describe(self) -> str:
        return f"Docker image {self.image}"

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _query(command: Sequence[str]) -> bool:
        """True if a quick Docker command succeeds."""
        try:
            return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  timeout=_QUERY_TIMEOUT).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
