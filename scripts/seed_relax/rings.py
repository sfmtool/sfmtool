# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The radius bands the fill-in admits clusters in.

Provenance: the study's `v2/densify/ring_edges.py` (97-116), which derived the
grid and wrote it to a JSON file the run then read, restated here as pure
functions of the fleet constant; and `v2/densify/densifylib.assign_rings`
(198-208).  The admission order the study computed inline at `densify_run.py`
221 is a function here so the rule lives in one place.

A ring is a band of feature radius, in units of the member's OWN admission
floor, because an absolute pixel radius is not commensurable across captures
whose sensors and refine radii differ by an order of magnitude.  The bands are
octaves, which is the spacing the detector's own pyramid puts the radii on.

A ring admits its WHOLE band.  The population a ring stands for is the band
itself -- every candidate whose radius falls between its edges -- so there is
no count to choose.  A count admits a fraction of a band instead, and which
fraction differs per member, because a member's supply within one octave is a
property of its own capture: the same number is one member's whole band and a
twentieth of another's, so what it keeps is a sub-band nothing states and two
members' fill-ins stop being the same operation.  What bounds the work is the
member's own candidate supply and the number of bands.
"""

from __future__ import annotations

import math

import numpy as np


def octave_edges(ratio_p1):
    """The ring boundaries, in units of the member's own admission floor.

    Decreasing: ``edges[k]`` is ring ``k``'s upper bound and ``edges[k + 1]``
    its lower one.  The grid is anchored at the floor (ratio 1.0) and runs
    down in octaves to ``ratio_p1``, and the top ring is OPEN above the floor:
    a member's admission is a group-local top-N over its own images rather
    than a capture-wide radius bar, so a cluster coarser than the floor can be
    outside it, and it is the coarsest evidence there is."""
    p1 = float(ratio_p1)
    if not (0.0 < p1 < 1.0):
        raise ValueError(f"ratio_p1 must lie in (0, 1), got {ratio_p1!r}")
    n_oct = int(math.ceil(math.log2(1.0 / p1)))
    return [float("inf")] + [1.0 / (2.0**k) for k in range(n_oct + 1)]


def assign_rings(cand_radius, floor, edges):
    """The ring index of every candidate cluster, or ``-1`` past the last.

    Half open: ring ``k`` holds the radii in ``[floor * edges[k + 1],
    floor * edges[k])``.  The banding is the core's
    ``analysis::source_clusters`` kernel, which the fill-in reaches through
    the join; this is the same rule on a radius array the caller already
    holds."""
    from sfmtool._sfmtool.analysis import assign_bands

    return np.asarray(
        assign_bands(
            np.ascontiguousarray(np.asarray(cand_radius, float)),
            float(floor),
            np.ascontiguousarray(np.asarray(edges, float)),
        ),
        np.int64,
    )


def band_order(cand, cand_radius):
    """The order a band is admitted in: coarsest first, cluster id on a tie.

    A function of the file alone, so which clusters a ring holds does not
    depend on how the candidates were enumerated."""
    return np.lexsort((np.asarray(cand, np.int64), -np.asarray(cand_radius, float)))


def ring_cap(opts):
    """How many clusters one ring may admit, or ``None`` for its whole band.

    ``None`` is the default, and it is the absence of a count rather than a
    large one: the ring stands for the band it is.  A positive ``ring_cap``
    option puts an absolute count back, for a caller bounding the work by
    something other than the member's own supply; the band is then cut in
    :func:`band_order`, coarsest first."""
    cap = int(getattr(opts, "ring_cap", 0) or 0)
    return cap if cap > 0 else None
