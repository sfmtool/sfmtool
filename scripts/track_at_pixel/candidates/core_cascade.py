# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Core cascade: the cascade as ``sfmtool_core::bench::build_track_at_pixel`` runs it.

The same four members in the same order as ``candidates/cascade.py``, with the
same finish and the same default thresholds, ported to Rust and called through
``bench.build_track_at_pixel``. This module only adapts: it builds the
``TrackAtPixelSources`` once per dataset (the SIFT index, every image's
keypoints and the cluster-patches ``.matches`` clusters) and turns the
binding's result and refusal into the harness contract.

The held-out point is deleted from ``ctx.edited``, and the Rust neighbourhood
queries are built from that version's live points, so the operation cannot
see it. Nothing else it is handed holds the reconstruction's points.

The member the Python cascade calls ``baseline`` is ``constellation`` here.
"""

from __future__ import annotations

from api import TrackAtPixelError, TrackAtPixelResult

DEFAULTS = {
    # The members to try, in order; None is all four in the cascade's order.
    "members": None,
}


def _sources(ctx):
    """The dataset's ``TrackAtPixelSources``, built on first use and kept."""
    from sfmtool._sfmtool import bench as B

    ds = ctx.dataset
    sources = getattr(ds, "_core_track_at_pixel_sources", None)
    if sources is None:
        keypoints = [(xy, affine) for xy, affine in ds.keypoints]
        sources = B.TrackAtPixelSources(ctx.edited, ds.forest, keypoints, ds.matches)
        ds._core_track_at_pixel_sources = sources
    return sources


def build_track(ctx, image: int, pixel, options: dict | None = None):
    from sfmtool._sfmtool import bench as B

    opts = {**DEFAULTS, **(options or {})}
    try:
        track, report = B.build_track_at_pixel(
            ctx.edited,
            ctx.pyramids,
            _sources(ctx),
            int(image),
            (float(pixel[0]), float(pixel[1])),
            members=opts["members"],
        )
    except B.TrackAtPixelError as e:
        raise TrackAtPixelError(e.stage, e.reason, e.diagnostics) from None
    query = report.pop("query_observation")
    return TrackAtPixelResult(track=track, query_observation=query, diagnostics=report)
