# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The shared check that a command's path argument names a `.sfmr` file.

Every command that reads or writes a reconstruction checks the file extension
before it does any work, so that a wrong argument fails with a usage error
rather than deep inside a reader or writer. The check and its error text live
here so that every command reports the mistake in the same words.
"""

from pathlib import Path

import click


def check_sfmr_path(path: str | Path, label: str) -> Path:
    """Return `path` as a `Path`, or raise `click.UsageError` if it is not `.sfmr`.

    The extension is compared case-insensitively. `label` names the argument
    in the error, for example "Input path" or "Reconstruction path", and the
    error reads "<label> must be a .sfmr file, got: <path>".
    """
    path = Path(path)
    if path.suffix.lower() != ".sfmr":
        raise click.UsageError(f"{label} must be a .sfmr file, got: {path}")
    return path
