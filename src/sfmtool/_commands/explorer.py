# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
import sys

import click


@click.command()
@click.argument("sfmr_files", nargs=-1, type=click.Path(exists=True))
def explorer(sfmr_files):
    """Launch the SfM Explorer 3D viewer.

    Every path given is loaded as its own node in the viewer's scene graph, so
    several reconstructions can be compared side by side in one 3D space.
    """
    exe = shutil.which("launch-sfm-explorer")
    if exe is None:
        raise click.ClickException(
            "launch-sfm-explorer executable not found. "
            "Install sfmtool with binary support or build with: "
            "pixi run cargo build --release -p sfmtool-py"
        )

    result = subprocess.run([exe, *sfmr_files])
    sys.exit(result.returncode)
