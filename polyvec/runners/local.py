"""Run backend programs that were compiled on this machine."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from ..errors import BackendNotAvailableError, VectorizationError
from ..paths import default_bin_dir
from .base import Runner, RunResult


class LocalRunner(Runner):
    """Executes natively built ``polyvec-field`` / ``polyvec-flow`` binaries.

    The executables are looked up, in order, in

    1. ``bin_dir`` passed to the constructor,
    2. the directory named by ``$POLYVEC_BIN_DIR``,
    3. the per-user install location written by ``python -m polyvec build --local``
       (:func:`polyvec.paths.default_bin_dir`),
    4. ``PATH``.

    Args:
        bin_dir: Extra directory to search first.
        env: Environment variables to add or override for the child process.
    """

    name = "local"

    def __init__(self, bin_dir: Union[str, "os.PathLike[str]", None] = None,
                 env: Optional[Dict[str, str]] = None) -> None:
        self.bin_dir = Path(bin_dir) if bin_dir is not None else None
        self.env = dict(env or {})

    # ----------------------------------------------------------------- lookup
    def search_dirs(self) -> List[Path]:
        """Directories searched before ``PATH``, most specific first."""
        dirs: List[Path] = []
        if self.bin_dir is not None:
            dirs.append(self.bin_dir)
        if os.environ.get("POLYVEC_BIN_DIR"):
            dirs.append(Path(os.environ["POLYVEC_BIN_DIR"]))
        dirs.append(default_bin_dir())
        return dirs

    def find(self, tool: str) -> Optional[Path]:
        """Absolute path of ``tool``, or None if it is not installed."""
        for directory in self.search_dirs():
            found = shutil.which(tool, path=str(directory))
            if found:
                return Path(found)
        found = shutil.which(tool)
        return Path(found) if found else None

    def is_available(self, tool: str) -> bool:
        return self.find(tool) is not None

    # -------------------------------------------------------------------- run
    def run(self, tool: str, args: Sequence[str], workdir: Path,
            timeout: Optional[float] = None) -> RunResult:
        executable = self.find(tool)
        if executable is None:
            searched = ", ".join(str(d) for d in self.search_dirs())
            raise BackendNotAvailableError(
                f"'{tool}' is not built on this machine (searched {searched} and PATH). "
                "Run `python -m polyvec build --local`, or use the Docker runner."
            )

        env = dict(os.environ)
        if sys.platform.startswith("linux"):
            # polyvec-flow constructs (but never shows) a Qt window; without this
            # it aborts on machines that have no display.
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
        if sys.platform == "win32":
            # Let the program find DLLs deployed next to it.
            env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
        env.update(self.env)

        command = [str(executable), *map(str, args)]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command, cwd=str(workdir), env=env, timeout=timeout,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            partial = exc.output if isinstance(exc.output, str) else ""
            raise VectorizationError(f"'{tool}' did not finish within {timeout:g} s", partial) from exc
        return RunResult(command, completed.returncode, completed.stdout or "",
                         time.perf_counter() - started)

    def describe(self) -> str:
        return f"local binaries ({', '.join(str(d) for d in self.search_dirs())}, PATH)"
