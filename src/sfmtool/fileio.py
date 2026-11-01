# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Readers, writers and verifiers for the `.sfmr`, `.sift`, `.matches` and
`.camrig` formats, COLMAP binary and database interop, the image readers
`read_image_rgb` and `read_image_rgba` (the decoder the Rust code reads
photographs with; `y_x_rgb` / `y_x_rgba` `uint8` arrays, EXIF orientation
ignored), their inverses the image writers `write_image_rgb` and
`write_image_rgba` (the Rust encoder; the format comes from the extension), the
header reads `image_dimensions` and `image_has_alpha`, and
`write_web_export`.

This module is the public home of the `sfmtool._sfmtool.fileio` bindings. It
is named `fileio` rather than `io` so that `from sfmtool import *` does not
replace the standard library's `io` in the importing namespace. Each format is
read and written through these bindings directly, with no Python wrapper:
`SiftFile` and `MatchesFile` open a `.sift` and a `.matches` file, and the
writers check their own arguments.
"""

from ._sfmtool.fileio import *  # noqa: F401, F403
from ._sfmtool.fileio import __all__  # noqa: F401
