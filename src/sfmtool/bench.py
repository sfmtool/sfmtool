# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The track-editing bench: `Bench`, `EditableTrack`, the steps that edit a
track (`add_observation`, `split`, `commit`, ...), and the track-at-pixel,
nearby-track, distance-range, depth-layer and far-field sources it reads.

This module is the public home of the `sfmtool._sfmtool.bench` bindings. Its
step names say what they do to a track, so read them through the module,
as `bench.commit(...)`.
"""

from ._sfmtool.bench import *  # noqa: F401, F403
from ._sfmtool.bench import __all__  # noqa: F401
