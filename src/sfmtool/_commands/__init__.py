# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI command implementations, one module per top-level command.

The package imports none of its modules: `sfmtool.cli` imports a command's
module only when that command is looked up (see `sfmtool._cli_group`), so a
module here should do no work at import time beyond defining its command, and
should import heavy libraries (numpy, OpenCV, pycolmap) only if its own command
needs them.
"""
