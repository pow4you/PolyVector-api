"""Running an OCR engine in its own process (borrowed from ``inksight``).

OCR engines do not always share a process with other native libraries: on
Linux Paddle crashes when TensorFlow was loaded first, and PyTorch builds can
clash over bundled LLVMs.  PolyVector Flow's keypoint network loads PyTorch
too, so the detectors run their engine in a persistent worker process (a
fresh, spawned interpreter), started on first use and kept for later calls.
"""

from __future__ import annotations

import multiprocessing
import weakref
from typing import Any, Callable, Dict, Optional

import numpy as np

from ...errors import PolyvecError

#: ``build(options) -> engine`` and ``run(engine, image) -> picklable result``;
#: both must be module-level functions (they are sent to the worker by name).
Build = Callable[[Dict[str, Any]], Any]
Run = Callable[[Any, np.ndarray], Any]


def _serve(connection: Any, build: Build, run: Run, options: Dict[str, Any]) -> None:
    """Worker loop: build the engine once, then answer images until told to stop."""
    try:
        engine = build(options)
        connection.send(("ready", None))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    while True:
        try:
            image = connection.recv()
        except EOFError:
            return
        if image is None:
            return
        try:
            connection.send(("ok", run(engine, image)))
        except BaseException as exc:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))


def _stop(process: Any, connection: Any) -> None:
    """Ask the worker to exit, then make sure it does."""
    try:
        connection.send(None)
    except (OSError, ValueError):
        pass
    connection.close()
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()


class EngineProcess:
    """A persistent worker process holding one OCR engine.

    The worker stops when :meth:`close` is called, when this object is garbage
    collected (e.g. its detector goes away), or at interpreter exit.
    """

    def __init__(self, build: Build, run: Run, options: Dict[str, Any], name: str) -> None:
        self.build, self.run, self.options, self.name = build, run, options, name
        self._process: Optional[Any] = None
        self._connection: Optional[Any] = None
        self._finalizer: Optional[weakref.finalize] = None

    def __call__(self, image: np.ndarray) -> Any:
        if self._process is None or not self._process.is_alive():
            self._start()
        try:
            self._connection.send(image)
            status, payload = self._connection.recv()
        except (EOFError, OSError) as exc:
            self.close()
            raise PolyvecError(f"the {self.name} worker process died ({exc})") from exc
        if status != "ok":
            raise PolyvecError(f"{self.name} failed: {payload}")
        return payload

    def _start(self) -> None:
        context = multiprocessing.get_context("spawn")      # a clean interpreter
        parent, child = context.Pipe()
        self._process = context.Process(target=_serve, args=(child, self.build, self.run, self.options),
                                        name=f"polyvec-{self.name}", daemon=True)
        self._process.start()
        child.close()
        self._connection = parent
        try:
            status, payload = parent.recv()
        except EOFError:
            status, payload = "error", "the worker exited during start-up"
        if status != "ready":
            self.close()
            raise PolyvecError(f"could not start {self.name}: {payload}")
        self._finalizer = weakref.finalize(self, _stop, self._process, parent)

    def close(self) -> None:
        """Stop the worker (it is restarted on the next call)."""
        if self._finalizer is not None:
            self._finalizer()                     # runs _stop once
            self._finalizer = None
        elif self._connection is not None:        # failed during start-up
            _stop(self._process, self._connection)
        self._process = self._connection = None
