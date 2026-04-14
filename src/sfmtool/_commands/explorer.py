# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
import sys

import click


@click.command()
@click.argument("sfmr_file", required=False, type=click.Path(exists=True))
def explorer(sfmr_file):
    """Launch the SfM Explorer 3D viewer."""
    exe = shutil.which("launch-sfm-explorer")
    if exe is None:
        raise click.ClickException(
            "launch-sfm-explorer executable not found. "
            "Install sfmtool with binary support or build with: "
            "pixi run cargo build --release -p sfmtool-py"
        )

    cmd = [exe]
    if sfmr_file:
        cmd.append(sfmr_file)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)
