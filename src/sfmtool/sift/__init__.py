# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""SIFT feature file I/O and extraction (OpenCV and COLMAP backends)."""

# Re-export the native SIFT detection/extraction bindings so they live under
# the public `sfmtool.sift` name (matching their `__module__`), alongside the
# Python-side file I/O and OpenCV/COLMAP backends in this package.
from sfmtool._sfmtool.sift import *  # noqa: E402, F401, F403
