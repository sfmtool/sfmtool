# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Descriptor matching, sweep matching (one-way, mutual and polar),
track-cluster matching and cluster covisibility: `match_image_pair`,
`background_floor_clusters`, `clusters_to_pair_matches`,
`ClusterCovisibility` and the functions beside them.

This module is the public home of the `sfmtool._sfmtool.matching` bindings.
`sfmtool.feature_match.match_image_pair` is a Python function that takes
different arguments and calls the `match_image_pair` here.
"""

from ._sfmtool.matching import *  # noqa: F401, F403
from ._sfmtool.matching import __all__  # noqa: F401
