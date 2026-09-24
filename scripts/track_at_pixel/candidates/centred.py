# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Centred: the ensemble, choosing within the winning group the track that peaks on the pixel.

The ensemble's vote settles the depth, then takes the group's track in the
cascade's order. The group's tracks differ in one reading the vote does not
use: where the queried view's own correlation peak sits (its
``seed_shift_px``). A fit localizes every view, the queried one included,
against the consensus of the others. On a patch with a stronger feature
beside the pixel, the whole patch slides toward that feature, and the anchor
then slides it back across its plane until its centre is on the pixel again.
The other views keep the correspondences they found at the slid position,
so the triangulated point sits at the wrong depth along the pixel's ray.
The reading that shows this is the queried view's peak: it sits where the
fit wanted the patch, not on the pixel.

So among the winning group's tracks, those whose queried view peaks within
``prefer_centred_px`` of the pixel are preferred. The ensemble's order
decides among them, and among the rest when none is centred.
"""

from __future__ import annotations

from candidates import ensemble

DEFAULTS = {
    **ensemble.DEFAULTS,
    "prefer_centred_px": 0.5,
}


def build_track(ctx, image: int, pixel, options: dict | None = None):
    return ensemble.build_track(ctx, image, pixel, {**DEFAULTS, **(options or {})})
