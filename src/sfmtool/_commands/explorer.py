# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
import sys

import click

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
    type=int,
    metavar="PORT",
    help=(
        "Host a Model Context Protocol endpoint on 127.0.0.1, so an agent can "
        f"drive the viewer window. Off unless asked for. Defaults to port "
        f"{DEFAULT_MCP_PORT}; 0 takes an ephemeral port, printed at startup."
    ),
)
@click.argument("sfmr_files", nargs=-1, type=click.Path(exists=True))
def explorer(mcp_port, sfmr_files):
    """Launch the SfM Explorer 3D viewer.

    Every path given is loaded as its own node in the viewer's scene graph, so
    several reconstructions can be compared side by side in one 3D space.

    With --mcp the viewer also hosts an MCP endpoint an agent can drive it
    through: the scene graph, the selection, the 3D camera, and a screenshot of
    the viewport. The window says so while it is live, in its title bar and in
    the Scene panel. See specs/gui/mcp-server.md.
    """
    exe = shutil.which("launch-sfm-explorer")
    if exe is None:
        raise click.ClickException(
            "launch-sfm-explorer executable not found. "
            "Install sfmtool with binary support or build with: "
            "pixi run cargo build --release -p sfmtool-py"
        )

    args = [] if mcp_port is None else ["--mcp", str(mcp_port)]
    result = subprocess.run([exe, *args, *sfmr_files])
    sys.exit(result.returncode)
