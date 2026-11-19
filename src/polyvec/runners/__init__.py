"""Runners decide where a compiled backend executes.

Vectorizers accept ``runner=`` as a :class:`Runner` instance or one of these
strings, which :func:`get_runner` resolves:

``"inprocess"``  the extension module shipped in binary wheels, in this process
``"subprocess"`` the same module in a helper process (survives crashes, honours timeouts)
``"local"``      programs compiled on this machine (``python -m polyvec build --local``)
``"docker"``     programs in a Docker container (image built on first use)
``"auto"``       ``$POLYVEC_RUNNER`` if set, else the first available of
                 inprocess, local, docker
"""

from __future__ import annotations

import os
from typing import Union

from ..errors import BackendNotAvailableError
from .base import Runner, RunResult
from .docker import DockerRunner
from .local import LocalRunner
from .native import InProcessRunner, NativeResult, SubprocessRunner

__all__ = ["Runner", "RunResult", "InProcessRunner", "SubprocessRunner", "NativeResult",
           "LocalRunner", "DockerRunner", "get_runner", "RUNNER_NAMES"]

#: Accepted runner names, aliases included.
_ALIASES = {
    "inprocess": "inprocess", "in-process": "inprocess", "native": "inprocess",
    "subprocess": "subprocess", "isolated": "subprocess",
    "local": "local", "docker": "docker", "auto": "auto",
}
RUNNER_NAMES = ("auto", "inprocess", "subprocess", "local", "docker")


def get_runner(spec: Union[str, Runner, None], tool: str) -> Runner:
    """Resolve a runner specification for the program ``tool``.

    Args:
        spec: A :class:`Runner` (returned unchanged), one of
            :data:`RUNNER_NAMES` (see the module docstring), or None for
            ``"auto"``.  Auto honours ``$POLYVEC_RUNNER`` if set, then prefers
            the in-process extension module, then locally built programs, and
            falls back to Docker.
        tool: Program the runner must be able to execute.

    Raises:
        BackendNotAvailableError: Auto-selection found no way to run ``tool``.
        ValueError: ``spec`` is not a recognised name.
    """
    if isinstance(spec, Runner):
        return spec
    name = _ALIASES.get((spec or "auto").lower())
    if name == "auto":
        name = _ALIASES.get(os.environ.get("POLYVEC_RUNNER", "auto").lower())
        if name is None:
            raise ValueError(f"unknown runner {os.environ['POLYVEC_RUNNER']!r} in $POLYVEC_RUNNER; "
                             f"expected one of {', '.join(RUNNER_NAMES)}")
    if name is None:
        raise ValueError(f"unknown runner {spec!r}; expected one of "
                         f"{', '.join(RUNNER_NAMES)} or a Runner")
    if name != "auto":
        return {"inprocess": InProcessRunner, "subprocess": SubprocessRunner,
                "local": LocalRunner, "docker": DockerRunner}[name]()

    for candidate in (InProcessRunner(), LocalRunner(), DockerRunner()):
        if candidate.is_available(tool):
            return candidate
    raise BackendNotAvailableError(
        f"no way to run '{tool}': this installation has no compiled extension module, "
        "it is not built locally and Docker is not reachable.\n"
        "  - install a binary wheel (`pip install polyvec` on Windows, Linux or macOS), or\n"
        "  - start Docker (the backend image is then built automatically on first use), or\n"
        "  - compile the backends here with `python -m polyvec build --local`."
    )
