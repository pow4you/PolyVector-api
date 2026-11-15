"""Runner interface: *how* a compiled backend program gets executed."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional, Sequence


@dataclass
class RunResult:
    """Outcome of one program invocation.

    Attributes:
        command: The full command line that was executed.
        returncode: Process exit status.
        output: Everything written to stdout and stderr, interleaved.
        elapsed: Wall-clock seconds.
    """

    command: List[str]
    returncode: int
    output: str
    elapsed: float


class Runner(ABC):
    """Executes one of the native programs inside a working directory.

    A vectorizer stages its input files in a scratch directory and then asks a
    runner to execute ``tool`` there.  The contract is deliberately small:

    * the program runs with ``workdir`` as its current directory,
    * every file argument is a *bare file name inside* ``workdir``,
    * output files appear in ``workdir``.

    Because no absolute host paths are ever passed, the same call works whether
    the program runs directly on this machine (:class:`~polyvec.runners.LocalRunner`),
    in a container (:class:`~polyvec.runners.DockerRunner`), or anywhere else a
    subclass cares to send it (WSL, SSH, a job queue, ...).
    """

    #: Short identifier used in messages and by :func:`polyvec.runners.get_runner`.
    name: ClassVar[str] = ""

    @abstractmethod
    def is_available(self, tool: str) -> bool:
        """Whether this runner can execute ``tool`` right now (must not raise)."""

    @abstractmethod
    def run(self, tool: str, args: Sequence[str], workdir: Path,
            timeout: Optional[float] = None) -> RunResult:
        """Execute ``tool args...`` in ``workdir`` and wait for it to finish.

        Args:
            tool: Program name, e.g. ``"polyvec-flow"``.
            args: Command-line arguments (file names relative to ``workdir``).
            workdir: Directory holding the inputs; receives the outputs.
            timeout: Seconds after which the program is killed, or None.

        Raises:
            BackendNotAvailableError: The program cannot be started.
            VectorizationError: The program exceeded ``timeout``.
        """

    def describe(self) -> str:
        """One-line human-readable description, used by ``python -m polyvec info``."""
        return self.name or type(self).__name__

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.describe()}>"
