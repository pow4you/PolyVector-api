"""Helper process for :class:`polyvec.runners.SubprocessRunner`.

Started as ``python -m polyvec._native.worker``.  Reads ``(tool, arguments)``
requests from stdin and answers each on stdout, both as length-prefixed
pickles (see :mod:`polyvec.runners.native`), until stdin closes.  The real
stdout is reserved for that channel: anything else that writes to file
descriptor 1 ends up on stderr instead.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    channel = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    requests = sys.stdin.buffer

    # Imported only after the redirection, so nothing they print can corrupt the channel.
    from ..errors import BackendNotAvailableError, VectorizationError
    from ..runners.native import call_extension, read_frame, write_frame

    while True:
        request = read_frame(requests)
        if request is None:
            return 0
        tool, arguments = request
        try:
            strokes, log = call_extension(tool, arguments)
            reply = ("ok", strokes, log)
        except BackendNotAvailableError as exc:
            reply = ("unavailable", str(exc))
        except VectorizationError as exc:
            reply = ("error", str(exc), exc.log)
        except Exception as exc:  # bad arguments etc.: report, keep serving
            reply = ("error", f"{type(exc).__name__}: {exc}", "")
        write_frame(channel, reply)


if __name__ == "__main__":
    sys.exit(main())
