"""Runners decide where a compiled backend executes (this machine, Docker, ...).

Vectorizers accept ``runner=`` as a :class:`Runner` instance or one of the
strings ``"auto"``, ``"local"`` and ``"docker"``; :func:`get_runner` resolves it.
"""

from __future__ import annotations

import os
from typing import Union

from ..errors import BackendNotAvailableError
from .base import Runner, RunResult
from .docker import DockerRunner
from .local import LocalRunner

__all__ = ["Runner", "RunResult", "LocalRunner", "DockerRunner", "get_runner"]


def get_runner(spec: Union[str, Runner, None], tool: str) -> Runner:
    """Resolve a runner specification for the program ``tool``.

    Args:
        spec: A :class:`Runner` (returned unchanged), ``"local"``, ``"docker"``,
            or ``"auto"``/None.  Auto honours ``$POLYVEC_RUNNER`` if set, then
            prefers locally built binaries and falls back to Docker.
        tool: Program the runner must be able to execute.

    Raises:
        BackendNotAvailableError: Auto-selection found no way to run ``tool``.
        ValueError: ``spec`` is not a recognised name.
    """
    if isinstance(spec, Runner):
        return spec
    name = (spec or "auto").lower()
    if name == "auto":
        name = os.environ.get("POLYVEC_RUNNER", "auto").lower()
    if name == "local":
        return LocalRunner()
    if name == "docker":
        return DockerRunner()
    if name != "auto":
        raise ValueError(f"unknown runner {spec!r}; expected 'auto', 'local', 'docker' or a Runner")

    local = LocalRunner()
    if local.is_available(tool):
        return local
    docker = DockerRunner()
    if docker.is_available(tool):
        return docker
    raise BackendNotAvailableError(
        f"no way to run '{tool}': it is not built locally and Docker is not reachable.\n"
        "  - start Docker (the backend image is then built automatically on first use), or\n"
        "  - compile the backends here with `python -m polyvec build --local`."
    )
