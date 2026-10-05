# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import signal
import sys

import click

from sfmtool._sfmtool import run_explorer

# What ``--mcp`` binds when given no number. Kept in step with
# ``DEFAULT_MCP_PORT`` in ``crates/sfm-explorer/src/cli.rs``; the value is
# repeated here rather than queried because Click needs it to build ``--help``,
# before the viewer is launched.
DEFAULT_MCP_PORT = 8787


@click.command()
@click.option(
    "--mcp",
    "mcp_port",
    is_flag=False,
    flag_value=str(DEFAULT_MCP_PORT),
    default=None,
    # The viewer refuses an out-of-range port as well; checking it here gives
    # the error as a Click usage error, before the viewer is started.
    type=click.IntRange(0, 65535),
    metavar="PORT",
    help=(
        "Host a Model Context Protocol endpoint on 127.0.0.1, so an agent can "
        f"drive the viewer window. Off unless asked for. Defaults to port "
        f"{DEFAULT_MCP_PORT}; 0 takes an ephemeral port, printed at startup."
    ),
)
@click.option(
    "--no-default-layout",
    is_flag=True,
    default=False,
    help=(
        "Start with the stock panel grid, ignoring any layout saved at "
        "~/.sfm-explorer-default-layout.json."
    ),
)
@click.argument("sfmr_files", nargs=-1, type=click.Path(exists=True))
def explorer(mcp_port, no_default_layout, sfmr_files):
    """Launch the SfM Explorer 3D viewer.

    Every path given is loaded as its own node in the viewer's scene graph, so
    several reconstructions can be compared side by side in one 3D space.

    The viewer comes up in whatever window placement and panel arrangement was
    saved to ~/.sfm-explorer-default-layout.json by its Panels > Save Layout...
    menu item; --no-default-layout starts from the stock grid instead.

    With --mcp the viewer also hosts an MCP endpoint an agent can drive it
    through: the scene graph, the selection, the 3D camera, the window and its
    panels, and a screenshot of the viewport. The window says so while it is
    live, in its title bar and in the Scene panel. See specs/gui/mcp-server.md.
    """
    args = [] if mcp_port is None else ["--mcp", str(mcp_port)]
    if no_default_layout:
        args.append("--no-default-layout")
    sys.exit(run_viewer([*args, *sfmr_files]))


def run_viewer(viewer_args: list[str]) -> int:
    """Run the viewer in this process with ``viewer_args``, its own command
    line, and return its exit status once its window closes.

    ``run_explorer`` takes over the main thread for the window's event loop
    until the window closes, so this is the last thing ``sfm explorer`` does.
    """
    # Python's own SIGINT handler only sets a flag for the interpreter to act
    # on, and the interpreter does not run while the viewer holds the main
    # thread, so Ctrl+C would do nothing. Restore the default, which ends the
    # process as it does for any other program. Python installs its handler only
    # over an inherited default, so an inherited "ignore" (a background job of a
    # non-interactive shell) is left as it is. Python's handler is put back once
    # the viewer returns, for a caller that goes on running.
    python_handler = signal.getsignal(signal.SIGINT)
    if python_handler is signal.default_int_handler:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    # The viewer writes to the process's stdout and stderr directly, not through
    # `sys.stdout` and `sys.stderr`, so anything still buffered on the Python
    # side is written first to keep the two in order.
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        return run_explorer(viewer_args)
    finally:
        if python_handler is signal.default_int_handler:
            signal.signal(signal.SIGINT, python_handler)
