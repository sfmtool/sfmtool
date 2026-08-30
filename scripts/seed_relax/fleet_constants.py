# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The two constants the relaxation reads off a fleet rather than tuning.

Each value ships with the fleet it came from, the table it was read out of and
that table's checksum, and the rule in one sentence.  `derive_constants.py`
recomputes both from a study directory, so a later fleet re-derives them
instead of re-tuning them.

Neither is exposed as an option.  A per-member ring grid would change the band
count per member and make two members' fill-ins incomparable, and a tunable
settling bar would turn a population statement about how much evidence a lens
reading needs into a knob.
"""

from __future__ import annotations

#: Finite points the filled-in state needs before the late lens release runs on
#: a perspective chart.  Below it a member's released equivalent focal is still
#: moving from ring to ring by more than the focal vote's own spread, so the
#: release would be reading a lens the evidence has not pinned down.  The
#: fisheye chart has no such bar: that family's released focal is settled from
#: the first reading.
SETTLING_FINITE_COUNT = 1778

SETTLING_FINITE_COUNT_PROVENANCE = {
    "fleet": "evo-survey-20260823 / relax-20260827 densify study set (39 members)",
    "source_table": "v2/densify/densify_stability.tsv",
    "source_table_md5": "a49172bb37a7b2f9acefd427bd277b6a",
    "member_set_table": "v2/densify/ring_pool.tsv",
    "member_set_table_md5": "0d9057e6cdce98568028a4ffee2894b7",
    "rule": (
        "Over the study members at the adopted pinhole knot count (2), split "
        "by whether the released equivalent focal was already settled at the "
        "pre-fill position (settle_pos == 0); the bar is the smallest "
        "n_finite_pre among the settled members that exceeds every unsettled "
        "member's n_finite_pre."
    ),
}

#: The pooled first percentile of candidate radius over the member's own
#: admission floor.  It sets how many octaves the ring grid runs to, and
#: nothing else: `rings.octave_edges` turns it into the boundaries.
RING_RATIO_P1 = 0.0887

RING_RATIO_P1_PROVENANCE = {
    "fleet": "evo-survey-20260823 / relax-20260827 densify study set (39 members)",
    "source_table": "v2/densify/ring_edges.json",
    "source_table_md5": "04f504fed2f4bcc4335c493d2fe3dd63",
    "member_set_table": "v2/densify/ring_pool.tsv",
    "member_set_table_md5": "0d9057e6cdce98568028a4ffee2894b7",
    "rule": (
        "The first percentile of the pooled ratio of every study member's "
        "unadmitted source-cluster radius to that member's own admission "
        "floor, over 1491855 candidate clusters; the ring grid is octaves "
        "from the floor down to it, top ring open."
    ),
}
