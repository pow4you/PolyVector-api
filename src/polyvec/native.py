"""Middle of the hierarchy: vectorizers implemented by a compiled upstream program."""

from __future__ import annotations

import os
import tempfile
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .base import Vectorizer
from .drawing import VectorDrawing
from .errors import VectorizationError
from .images import remap_threshold, save_png
from .runners import InProcessRunner, Runner, RunResult, get_runner


@dataclass
class NativeJob:
    """What a :class:`NativeVectorizer` subclass asks to be executed.

    Attributes:
        args: Command-line arguments for the program.  File arguments are bare
            names of files inside the scratch directory.
        output: Name of the SVG file the program will write there.
        metadata: Extra entries to attach to the resulting drawing.
    """

    args: List[str]
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class NativeVectorizer(Vectorizer):
    """A vectorizer whose algorithm lives in an upstream command-line program.

    This class owns all the plumbing the upstream tools expect from a human at
    a shell: a scratch directory, the image written as a file, the program
    invocation (through a :class:`~polyvec.runners.Runner`, so it may be a local
    binary or a container), error detection and reading the SVG back.

    A concrete method only declares which program to run and how to call it:

    * set :attr:`tool` to the executable's name,
    * implement :meth:`_stage` to write any extra input files and return the
      command line as a :class:`NativeJob`,
    * implement :meth:`_arguments` to give the same inputs as arrays to the
      compiled extension module (:class:`~polyvec.runners.InProcessRunner`
      and :class:`~polyvec.runners.SubprocessRunner` skip files altogether).

    Args:
        runner: Where the program runs - ``"auto"`` (the extension module if
            installed, else a local binary if built, else Docker),
            ``"inprocess"``, ``"subprocess"``, ``"local"``, ``"docker"`` or a
            :class:`~polyvec.runners.Runner` instance (see :mod:`polyvec.runners`).
        timeout: Seconds after which the program is aborted (None: no limit).
            Not supported by the ``"inprocess"`` runner.
        ink_threshold: Gray level (1-254) below which a pixel counts as ink.
            Upstream hard-codes this as 165 (``BACKGROUND_FOREGROUND_THRESHOLD``
            in ``Params.h``); other values are realised by adjusting the image
            contrast before the program sees it, as the upstream README
            suggests.  Raise it if faint lines go missing, lower it if separate
            lines merge.  None leaves the image untouched.
        workdir: Keep the staged inputs, the program's output and its log in
            this directory instead of a temporary one (useful for debugging).
            The extension runners stage no files and only write the log there.
        invert: See :class:`~polyvec.base.Vectorizer`.

    Attributes:
        last_run: :class:`~polyvec.runners.RunResult` of the most recent
            program invocation (command, exit status, full log), or None.
    """

    #: Name of the executable, e.g. ``"polyvec-flow"``.
    tool: ClassVar[str] = ""

    #: Gray level at which the compiled program separates ink from paper
    #: (255 - BACKGROUND_FOREGROUND_THRESHOLD in the upstream ``Params.h``).
    NATIVE_INK_THRESHOLD: ClassVar[int] = 165

    #: File name under which the input image is staged.
    IMAGE_NAME: ClassVar[str] = "input.png"

    def __init__(self, *, runner: Union[str, Runner, None] = "auto",
                 timeout: Optional[float] = None, ink_threshold: Optional[int] = None,
                 workdir: Union[str, "os.PathLike[str]", None] = None,
                 invert: bool = False) -> None:
        super().__init__(invert=invert)
        self._runner_spec = runner
        self._runner: Optional[Runner] = runner if isinstance(runner, Runner) else None
        self.timeout = timeout
        self.ink_threshold = ink_threshold
        self.workdir = Path(workdir) if workdir is not None else None
        self.last_run: Optional[RunResult] = None

    @property
    def runner(self) -> Runner:
        """The runner in use; resolved on first access so construction stays cheap."""
        if self._runner is None:
            self._runner = get_runner(self._runner_spec, self.tool)
        return self._runner

    # ------------------------------------------------------------------ hooks
    def preprocess(self, gray: np.ndarray) -> np.ndarray:
        gray = super().preprocess(gray)
        if self.ink_threshold is not None and self.ink_threshold != self.NATIVE_INK_THRESHOLD:
            gray = remap_threshold(gray, self.ink_threshold, self.NATIVE_INK_THRESHOLD)
        return gray

    @abstractmethod
    def _stage(self, workdir: Path, image_name: str, gray: np.ndarray, **inputs: Any) -> NativeJob:
        """Prepare one program invocation.

        Called after the image has been written to ``workdir / image_name``.
        Write any additional input files into ``workdir`` and describe the call.

        Args:
            workdir: Scratch directory the program will run in.
            image_name: File name of the staged input image.
            gray: The same image as an ``(H, W)`` ``uint8`` array.
            **inputs: Extra keyword arguments given to :meth:`vectorize`.
        """

    def _arguments(self, gray: np.ndarray, **inputs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare one call of the extension module (the in-memory :meth:`_stage`).

        Returns:
            The keyword arguments for the module's ``vectorize`` other than
            ``image``, and extra entries for the drawing's metadata.
        """
        raise NotImplementedError(f"{type(self).__name__} has no in-process backend")

    # --------------------------------------------------------------- pipeline
    def _vectorize(self, gray: np.ndarray, **inputs: Any) -> VectorDrawing:
        if isinstance(self.runner, InProcessRunner):
            return self._vectorize_in_memory(gray, **inputs)
        height, width = gray.shape
        with self._scratch_dir() as workdir:
            save_png(gray, workdir / self.IMAGE_NAME)
            job = self._stage(workdir, self.IMAGE_NAME, gray, **inputs)

            result = self.runner.run(self.tool, job.args, workdir, self.timeout)
            self.last_run = result
            if self.workdir is not None:
                (workdir / "log.txt").write_text(result.output, encoding="utf-8")

            output = workdir / job.output
            if result.returncode != 0 or not output.is_file():
                raise VectorizationError(self._failure_message(result, job), result.output)
            drawing = VectorDrawing.from_svg(output, width=width, height=height, method=self.name)

        drawing.metadata.update(elapsed=result.elapsed, runner=self.runner.name, **job.metadata)
        return drawing

    def _vectorize_in_memory(self, gray: np.ndarray, **inputs: Any) -> VectorDrawing:
        height, width = gray.shape
        arguments, metadata = self._arguments(gray, **inputs)
        arguments = dict(arguments, image=np.ascontiguousarray(gray, dtype=np.uint8))
        try:
            result = self.runner.call(self.tool, arguments, self.timeout)
        except VectorizationError as exc:
            self._write_log(exc.log)
            raise
        self.last_run = result.run
        self._write_log(result.run.output)

        # Same clean-up VectorDrawing.from_svg applies to the program's SVG.
        strokes = []
        for stroke in result.strokes:
            stroke = np.asarray(stroke, dtype=np.float64)
            stroke = stroke[np.isfinite(stroke).all(axis=1)]
            if len(stroke) >= 2:
                strokes.append(stroke)
        drawing = VectorDrawing(strokes=strokes, width=width, height=height, method=self.name)
        drawing.metadata.update(elapsed=result.run.elapsed, runner=self.runner.name, **metadata)
        return drawing

    def _write_log(self, log: str) -> None:
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)
            (self.workdir / "log.txt").write_text(log, encoding="utf-8")

    @contextmanager
    def _scratch_dir(self) -> Iterator[Path]:
        """The user's ``workdir`` if one was given, else a self-deleting temporary one."""
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)
            yield self.workdir
        else:
            with tempfile.TemporaryDirectory(prefix="polyvec-") as temporary:
                yield Path(temporary)

    def _failure_message(self, result: RunResult, job: NativeJob) -> str:
        tail = "\n".join(result.output.strip().splitlines()[-15:])
        reason = (f"exited with status {result.returncode}" if result.returncode != 0
                  else f"finished without writing {job.output}")
        return (f"{self.tool} {reason}.\nLast output:\n{tail}\n"
                "(the full log is in the exception's .log attribute)")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(runner={self._runner_spec!r})"
