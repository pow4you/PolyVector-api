"""Run the backends from the compiled extension module (``polyvec._native``).

No compiler, no Docker and no files: the image goes to the upstream pipeline as
a NumPy array and the curves come back as arrays.  Two flavours:

* :class:`InProcessRunner` calls the extension directly - the fastest option,
  but a crash inside the research code (a segfault, say) takes the calling
  Python process with it, and a call cannot be interrupted.
* :class:`SubprocessRunner` makes the same call in a long-lived helper Python
  process.  It costs one array round trip per call and a process start on
  first use, but a crash only kills the helper (it is restarted on the next
  call), and ``timeout`` works.
"""

from __future__ import annotations

import atexit
import os
import pickle
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import _native
from ..errors import BackendNotAvailableError, VectorizationError
from .base import Runner, RunResult


@dataclass
class NativeResult:
    """Outcome of one extension call.

    Attributes:
        strokes: ``(N, 2)`` arrays of ``x, y`` pixel coordinates, exactly as
            the command-line program would have written them to its SVG.
        run: Command, (zero) exit status, the program's console output and
            the wall-clock time, in the same form the file-based runners report.
    """

    strokes: List[np.ndarray]
    run: RunResult


def call_extension(tool: str, arguments: Dict[str, Any]) -> Tuple[List[np.ndarray], str]:
    """Call ``polyvec._native`` for ``tool`` in this process.

    Returns:
        ``(strokes, log)``.

    Raises:
        BackendNotAvailableError: The extension module is not installed.
        VectorizationError: The upstream code raised an error.
    """
    try:
        module = _native.load(tool)
    except ImportError as exc:
        raise BackendNotAvailableError(
            f"the in-process backend for '{tool}' is not installed ({exc}).  Binary "
            "wheels include it; from a source install, use runner='docker' or 'local'."
        ) from exc
    try:
        strokes, log = module.vectorize(**arguments)
    except (RuntimeError, ValueError) as exc:
        # RuntimeError(message, log) from the upstream code; ValueError for
        # inputs the module rejects before running it.
        message, log = (tuple(exc.args) + ("", ""))[:2]
        raise VectorizationError(f"{tool} failed: {message}", str(log)) from exc
    return list(strokes), log


class InProcessRunner(Runner):
    """Executes the backends inside this Python process via ``polyvec._native``.

    Calls are serialised (the upstream code keeps its state in globals) and
    release the GIL while they run.  ``timeout`` is not supported: native code
    cannot be interrupted safely; use :class:`SubprocessRunner` for that.
    """

    name: ClassVar[str] = "inprocess"

    def is_available(self, tool: str) -> bool:
        return _native.available(tool)

    def call(self, tool: str, arguments: Dict[str, Any],
             timeout: Optional[float] = None) -> NativeResult:
        """Run ``tool`` on in-memory inputs.

        Args:
            tool: ``"polyvec-field"`` or ``"polyvec-flow"``.
            arguments: Keyword arguments of the module's ``vectorize``
                (``image`` and, for flow, ``points``, ``labels``, ``iterations``).
            timeout: Ignored here (see the class docstring).
        """
        started = time.perf_counter()
        strokes, log = call_extension(tool, arguments)
        return NativeResult(strokes, self._result(tool, log, started))

    def run(self, tool: str, args: Sequence[str], workdir: Path,
            timeout: Optional[float] = None) -> RunResult:
        raise BackendNotAvailableError(
            f"{type(self).__name__} works on in-memory arrays, not files; it is used through "
            "NativeVectorizer (PolyVectorField / PolyVectorFlow), which calls .call()."
        )

    def describe(self) -> str:
        return f"extension module in this process ({self._modules()})"

    @staticmethod
    def _modules() -> str:
        found = [_native.MODULES[tool] for tool in _native.MODULES if _native.available(tool)]
        return ", ".join(found) or "not installed"

    @staticmethod
    def _result(tool: str, log: str, started: float) -> RunResult:
        command = [f"{_native.MODULES[tool]}.vectorize"]
        return RunResult(command, 0, log, time.perf_counter() - started)


class SubprocessRunner(InProcessRunner):
    """Executes the backends via ``polyvec._native`` in a helper Python process.

    The helper (``python -m polyvec._native.worker``) is started on first use
    and shared by all :class:`SubprocessRunner` instances; if it crashes or
    exceeds ``timeout`` it is killed and the call raises
    :class:`~polyvec.errors.VectorizationError`, and a fresh helper starts on
    the next call.
    """

    name: ClassVar[str] = "subprocess"

    def call(self, tool: str, arguments: Dict[str, Any],
             timeout: Optional[float] = None) -> NativeResult:
        started = time.perf_counter()
        strokes, log = _WORKER.call(tool, arguments, timeout)
        result = self._result(tool, log, started)
        result.command = [sys.executable, "-m", "polyvec._native.worker", *result.command]
        return NativeResult(strokes, result)

    def describe(self) -> str:
        return f"extension module in a helper process ({self._modules()})"


# --------------------------------------------------------------------------- worker
#: Frames on the worker's pipes: 8-byte little-endian length, then a pickle.
_HEADER = struct.Struct("<Q")


def write_frame(stream: BinaryIO, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(data)))
    stream.write(data)
    stream.flush()


def read_frame(stream: BinaryIO) -> Any:
    """The next object on ``stream``, or None at end of stream."""
    header = stream.read(_HEADER.size)
    if len(header) < _HEADER.size:
        return None
    (length,) = _HEADER.unpack(header)
    data = stream.read(length)
    if len(data) < length:
        return None
    return pickle.loads(data)


class _Worker:
    """One helper process, used by one call at a time."""

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _start(self) -> subprocess.Popen:
        env = dict(os.environ)
        # Make the helper import this very copy of polyvec, wherever it lives.
        package_parent = str(Path(_native.__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_parent, env.get("PYTHONPATH")]))
        return subprocess.Popen(
            [sys.executable, "-m", "polyvec._native.worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
        )

    def call(self, tool: str, arguments: Dict[str, Any],
             timeout: Optional[float]) -> Tuple[List[np.ndarray], str]:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._process = self._start()
            process = self._process

            reply: List[Any] = []
            def receive() -> None:
                reply.append(read_frame(process.stdout))

            try:
                write_frame(process.stdin, (tool, arguments))
            except (BrokenPipeError, OSError):
                reply.append(None)
            else:
                reader = threading.Thread(target=receive, daemon=True)
                reader.start()
                reader.join(timeout)
                if reader.is_alive():
                    self._kill()
                    raise VectorizationError(f"'{tool}' did not finish within {timeout:g} s")

            message = reply[0] if reply else None
            if message is None:
                status = self._kill()
                raise VectorizationError(
                    f"the helper process running '{tool}' died (exit status {status}); "
                    "the upstream code probably crashed on this input."
                )
            kind, *payload = message
            if kind == "ok":
                strokes, log = payload
                return strokes, log
            if kind == "unavailable":
                raise BackendNotAvailableError(payload[0])
            error, log = payload
            raise VectorizationError(error, log)

    def _kill(self) -> Optional[int]:
        process, self._process = self._process, None
        if process is None:
            return None
        if process.poll() is None:
            process.kill()
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return None

    def close(self) -> None:
        with self._lock:
            process, self._process = self._process, None
        if process is not None and process.poll() is None:
            try:
                process.stdin.close()   # the helper exits at end of input
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()


_WORKER = _Worker()
atexit.register(_WORKER.close)
