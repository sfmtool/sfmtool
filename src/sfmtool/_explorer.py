# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Run SfM Explorer in this process: ``python -m sfmtool._explorer [ARGS ...]``.

``sfm explorer`` starts this module as a child process rather than calling the
viewer itself. The viewer owns the process while it runs: it creates a window
event loop, which macOS allows only on the main thread and ``winit`` allows only
once per process, and it ends the process with ``exit`` on a bad option or a
port it cannot bind. The arguments are the viewer's own command line, as its
``--help`` describes.
"""

import signal
import sys

from sfmtool._sfmtool import run_explorer


def main(argv: list[str]) -> None:
    # Python's own SIGINT handler only sets a flag for the interpreter to act
    # on, and the interpreter does not run while the viewer holds the main
    # thread, so Ctrl+C would do nothing. Restore the default, which ends the
    # process as it does for any other program. Python installs its handler only
    # over an inherited default, so an inherited "ignore" (a background job of a
    # non-interactive shell) is left as it is.
    if signal.getsignal(signal.SIGINT) is signal.default_int_handler:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    run_explorer(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
