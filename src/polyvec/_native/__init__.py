"""The upstream C++ programs compiled as Python extension modules.

Binary wheels ship two modules built from the vendored sources (see
``polyvec/native/CMakeLists.txt``)::

    polyvec._native._field   PolyVector Fields  (vendor/pv)
    polyvec._native._flow    PolyVector Flow    (vendor/pvflow, HiGHS solver)

They are separate shared libraries on purpose: the two code bases define many
functions with the same names but different behaviour, which must never be
linked together.

Each exposes ``vectorize(image, ...) -> (strokes, log)``; this package maps the
programs' tool names onto them.  Use it through
:class:`~polyvec.runners.InProcessRunner` / :class:`~polyvec.runners.SubprocessRunner`
rather than directly.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict, Optional

#: Program name (as used by the runners) -> extension module.
MODULES: Dict[str, str] = {
    "polyvec-field": "polyvec._native._field",
    "polyvec-flow": "polyvec._native._flow",
}

_import_errors: Dict[str, str] = {}


def load(tool: str) -> ModuleType:
    """Import the extension module implementing ``tool``.

    Raises:
        KeyError: ``tool`` is not one of :data:`MODULES`.
        ImportError: The module is not part of this installation (e.g. a
            source install without the C++ dependencies).
    """
    return importlib.import_module(MODULES[tool])


def available(tool: str) -> bool:
    """Whether ``tool`` can be called in-process (never raises)."""
    if tool not in MODULES:
        return False
    try:
        load(tool)
    except ImportError as exc:
        _import_errors[tool] = str(exc)
        return False
    return True


def import_error(tool: str) -> Optional[str]:
    """Why :func:`available` returned False for ``tool``, if it did."""
    return _import_errors.get(tool)
