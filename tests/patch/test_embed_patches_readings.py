# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""``embed_patches`` and the keypoint xforms store each observation's readings
on its own render: the self-similarity ellipse, the render's angle, tilt and
zoom, and its float32 scores against the stored bitmap."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.xform import RefineKeypointsTransform

from .conftest import load_images


def _check_ranges(readings: dict, recon: SfmrReconstruction) -> None:
    m = len(np.asarray(recon.track_point_indexes))
    axes = readings["zncc_self_similarity_ellipse_axes"]
    assert axes.shape == (m, 2) and axes.dtype == np.float32
    measured = ~np.isnan(axes[:, 0])
    assert measured.mean() > 0.9
    r = readings["options"]["max_radius"]
    assert ((axes[measured, 0] >= 0) & (axes[measured, 0] <= r)).all()
    assert (axes[measured, 1] <= axes[measured, 0]).all()
    for key in (
        "zncc_self_similarity_ellipse_major_angle",
        "zncc_self_similarity_tilt_angle",
    ):
        a = readings[key][measured]
        a = a[~np.isnan(a)]
        assert ((a >= 0) & (a < np.pi)).all(), key
    cos = readings["zncc_self_similarity_cos_view_angle"][measured]
    assert ((cos >= -1) & (cos <= 1)).all()
    assert np.median(cos) > 0.5
    zoom = readings["zncc_self_similarity_zoom"][measured]
    assert ((zoom[:, 0] > 0) & (zoom[:, 0] <= zoom[:, 1])).all()
    for key in ("plain_bitmap_zncc", "blur_matched_bitmap_zncc"):
        assert readings[key].dtype == np.float32


def test_embed_patches_stores_each_observations_readings(seoul_bull_workspace: Path):
    from sfmtool._embed_patches import embed_patches

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    out = embed_patches(recon, images, patch_size=10.0, resolution=12, rounds=1)
    readings = out.observation_readings
    assert readings is not None
    _check_ranges(readings, out)
    assert readings["options"]["anisotropic_threshold"] == 1.5

    # Each point's reference observation scores 1 against its own render.
    refs = np.asarray(out.reference_observations)
    offsets = np.concatenate(
        [[0], np.cumsum(np.asarray(out.observation_counts))]
    ).astype(np.int64)
    referenced = np.flatnonzero(refs >= 0)
    assert referenced.size > 0
    rows = offsets[referenced] + refs[referenced]
    measured = ~np.isnan(readings["zncc_self_similarity_ellipse_axes"][rows, 0])
    np.testing.assert_array_equal(readings["plain_bitmap_zncc"][rows][measured], 1.0)
    np.testing.assert_array_equal(
        readings["blur_matched_bitmap_zncc"][rows][measured], 1.0
    )
    # The other scores are the bitmap scorer's against the stored bitmaps.
    others = np.ones(len(readings["plain_bitmap_zncc"]), dtype=bool)
    others[rows] = False
    plain = readings["plain_bitmap_zncc"][others]
    assert np.isfinite(plain).mean() > 0.9
    assert np.nanmedian(plain) > 0.5

    # Reading the stored file again gives the stored rows: the scores are the
    # stored bitmaps' at the stored keypoints, nothing from before compaction.
    again = out.patches.read_observations(out, images, resolution=12)
    for key, value in readings.items():
        if key != "options":
            np.testing.assert_array_equal(
                np.asarray(again[key]), np.asarray(value), key
            )


def test_refine_keypoints_reads_every_observation_again(seoul_bull_workspace: Path):
    recon = SfmrReconstruction.load(seoul_bull_workspace).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )
    with_bitmaps = RefineKeypointsTransform(resolution=12, max_gn_steps=3).apply(recon)
    _check_ranges(with_bitmaps.observation_readings, with_bitmaps)
    assert np.isfinite(with_bitmaps.observation_readings["plain_bitmap_zncc"]).any()

    # Without bitmaps there is nothing to score against.
    without = RefineKeypointsTransform(
        resolution=12, max_gn_steps=3, bitmaps=False
    ).apply(recon)
    readings = without.observation_readings
    _check_ranges(readings, without)
    assert np.isnan(readings["plain_bitmap_zncc"]).all()
