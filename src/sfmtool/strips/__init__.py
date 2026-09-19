# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Patch-strip montages: the engines behind ``compare --strips`` and ``inspect --strips``.

One closed pipeline. ``_solve`` renders a reconstruction's points as oriented-surfel
patch strips over ``_ncc``'s window/NCC/strip primitives; ``_compare`` and ``_inspect``
choose and label the rows; ``_montage`` lays them out and writes the PNG. Outside code
enters only through the two commands' entry points, re-exported here.
"""

from ._compare import render_comparison_strips
from ._inspect import parse_point_specs, render_inspect_strips

__all__ = [
    "parse_point_specs",
    "render_comparison_strips",
    "render_inspect_strips",
]
