"""Exception hierarchy.  Everything polyvec raises derives from :class:`PolyvecError`."""

from __future__ import annotations


class PolyvecError(Exception):
    """Base class of all polyvec errors."""


class InvalidImageError(PolyvecError, ValueError):
    """The input could not be interpreted as an image."""


class MissingDependencyError(PolyvecError, ImportError):
    """An optional Python dependency (e.g. PyTorch for keypoint detection) is not installed."""


class BackendNotAvailableError(PolyvecError):
    """No runner is able to execute the requested native program."""


class BackendBuildError(PolyvecError):
    """Compiling the native programs (locally or as a Docker image) failed."""


class VectorizationError(PolyvecError):
    """A native program ran but did not produce a usable result.

    Attributes:
        log: Everything the program printed, for diagnosis.
    """

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(message)
        self.log = log
