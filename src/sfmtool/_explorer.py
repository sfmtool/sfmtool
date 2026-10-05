# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Run SfM Explorer in this process: ``python -m sfmtool._explorer [ARGS ...]``.

``sfm explorer`` starts this module as a child process rather than calling the
viewer itself, because the viewer needs a process to itself: it creates a window
event loop, which macOS allows only on the main thread and ``winit`` allows only
once per process, and it sets process-wide state, the logger and on Windows the
DPI awareness, that should not carry over into the ``sfm`` process. The
arguments are the viewer's own command line, as its ``--help`` describes, and
the process exits with the status ``run_explorer`` returns: 0 once the window
closes, 2 for a command line the viewer cannot act on, and 1 when it cannot
start.
"""

import signal
import sys

from sfmtool._sfmtool import run_explorer


def main(argv: list[str]) -> int:
    # Python's own SIGINT handler only sets a flag for the interpreter to act
    # on, and the interpreter does not run while the viewer holds the main
    # thread, so Ctrl+C would do nothing. Restore the default, which ends the
    # process as it does for any other program. Python installs its handler only
    # over an inherited default, so an inherited "ignore" (a background job of a
    # non-interactive shell) is left as it is.
    if signal.getsignal(signal.SIGINT) is signal.default_int_handler:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    return run_explorer(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
