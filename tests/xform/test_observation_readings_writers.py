# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""What each xform writer does with the stored observation readings: the
writers that render read every observation again under their own options, the
ones that do not carry the rows, and a change of bitmap resolution or sampler
drops readings taken under the old one."""

import numpy as np

from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.xform import (
    AddPatchBitmapsTransform,
    DropPatchBitmapsTransform,
    FindPointsAtInfinityTransform,
    LocalizeKeypointsTransform,
    RefineKeypointsTransform,
    RefineNormalsTransform,
)

OPTIONS = {
    "resolution": 24,
    "sampler": "per_view",
    "score_window": "gaussian_disk",
    "score_window_sigma": 0.6,
    "max_radius": 3,
    "flat_floor": 0.5,
    "noise": 2.0,
    "relative_tolerance": 0.05,
    "anisotropic_threshold": 1.5,
}


def _tagged(m: int, options=OPTIONS) -> dict:
    tag = np.arange(1, m + 1, dtype=np.float32)
    return {
        "zncc_self_similarity_ellipse_axes": np.stack([tag, tag / 2], axis=1),
        "zncc_self_similarity_ellipse_axes_is_at_least": np.zeros((m, 2), np.uint8),
        "zncc_self_similarity_ellipse_major_angle": np.full(m, 0.5, np.float32),
        "zncc_self_similarity_cos_view_angle": np.full(m, 0.75, np.float32),
        "zncc_self_similarity_tilt_angle": np.full(m, 1.0, np.float32),
        "zncc_self_similarity_zoom": np.tile(np.float32([0.25, 0.5]), (m, 1)),
        "plain_bitmap_zncc": (tag * 1e-4).astype(np.float32),
        "blur_matched_bitmap_zncc": (tag * 2e-4).astype(np.float32),
        "options": dict(options),
    }


def _embedded(workspace) -> SfmrReconstruction:
    return SfmrReconstruction.load(workspace).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )


def _measured(readings) -> np.ndarray:
    return ~np.isnan(readings["zncc_self_similarity_ellipse_axes"][:, 0])


def test_find_points_at_infinity_carries_rows_and_adds_unmeasured_ones(
    seoul_bull_workspace,
):
    original = SfmrReconstruction.load(seoul_bull_workspace)
    m = original.observation_count
    with_readings = original.clone_with_changes(observation_readings=_tagged(m))
    result = FindPointsAtInfinityTransform(0.1, 300.0, 2, max_features=1500).apply(
        with_readings
    )
    readings = result.observation_readings
    assert readings is not None
    assert len(readings["plain_bitmap_zncc"]) == result.observation_count
    assert result.observation_count > m
    # The input's rows are where they were, and the new ones are not measured.
    np.testing.assert_array_equal(
        readings["plain_bitmap_zncc"][:m], _tagged(m)["plain_bitmap_zncc"]
    )
    assert not _measured(readings)[m:].any()


def test_localize_keypoints_reads_every_observation_at_its_resolution(
    seoul_bull_workspace,
):
    recon = _embedded(seoul_bull_workspace)
    out = LocalizeKeypointsTransform(resolution=12).apply(recon)
    readings = out.observation_readings
    assert readings["options"]["resolution"] == 12
    assert readings["options"]["sampler"] == "per_view"
    assert _measured(readings).mean() > 0.9
    # No bitmap is stored, so there is nothing to score against.
    assert np.isnan(readings["plain_bitmap_zncc"]).all()


def test_refine_normals_reads_every_observation_again(seoul_bull_workspace):
    recon = _embedded(seoul_bull_workspace)
    out = RefineNormalsTransform(
        resolution=12,
        init_steps=5,
        refine_levels=2,
        sampler="bilinear",
        bitmaps=True,
    ).apply(recon)
    readings = out.observation_readings
    assert readings["options"]["resolution"] == 12
    assert readings["options"]["sampler"] == "bilinear"
    assert readings["options"]["anisotropic_threshold"] is None
    assert _measured(readings).mean() > 0.9
    assert np.isfinite(readings["plain_bitmap_zncc"]).any()


def test_a_bitmap_resolution_or_sampler_change_drops_the_readings(
    seoul_bull_workspace,
):
    # Readings at R = 12, the bitmaps dropped, then added again at another
    # resolution or with another sampler: no row describes those renders.
    recon = _embedded(seoul_bull_workspace)
    refined = RefineKeypointsTransform(resolution=12, max_gn_steps=3).apply(recon)
    assert refined.observation_readings["options"]["resolution"] == 12
    dropped = DropPatchBitmapsTransform().apply(refined)
    assert dropped.observation_readings is not None

    at_8 = AddPatchBitmapsTransform(resolution=8).apply(dropped)
    assert at_8.patch_bitmap_resolution == 8
    assert at_8.observation_readings is None

    bilinear = AddPatchBitmapsTransform(resolution=12, sampler="bilinear").apply(
        dropped
    )
    assert bilinear.observation_readings is None

    # The same resolution and sampler, from the same references: the rows stay
    # as the records they are.
    same = AddPatchBitmapsTransform(resolution=12).apply(dropped)
    kept = same.observation_readings
    assert kept is not None
    np.testing.assert_array_equal(
        kept["zncc_self_similarity_ellipse_axes"],
        refined.observation_readings["zncc_self_similarity_ellipse_axes"],
    )

    # clone_with_changes(patch_bitmaps=...) at another resolution drops them.
    p = dropped.point_count
    other = dropped.clone_with_changes(
        patch_bitmaps=np.zeros((p, 8, 8, 4), dtype=np.uint8)
    )
    assert other.observation_readings is None


def test_replaced_tracks_with_a_moved_reference_clear_its_scores(
    seoul_bull_ground_truth_sfmr,
):
    # The tracks handed back (as a bundle adjustment's readback does) with one
    # point's reference on another observation: that point's scores were read
    # against another bitmap, so they are cleared; every other point's stand.
    base = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    m = base.observation_count
    recon = base.clone_with_changes(observation_readings=_tagged(m))
    pts = np.asarray(recon.track_point_indexes)
    imgs = np.asarray(recon.track_image_indexes)
    kxy = np.asarray(recon.keypoints_xy)
    counts = np.asarray(recon.observation_counts)
    refs = np.asarray(recon.reference_observations).copy()
    p = int(np.flatnonzero(counts >= 3)[0])
    refs[p] = 0 if refs[p] < 0 else (refs[p] + 1) % counts[p]
    out = recon.clone_with_changes(
        track_image_indexes=np.ascontiguousarray(imgs),
        track_feature_indexes=np.zeros(m, dtype=np.uint32),
        track_point_indexes=np.ascontiguousarray(pts),
        keypoints_xy=np.ascontiguousarray(kxy),
        reference_observations=refs.astype(np.int32),
    )
    got = out.observation_readings
    on_p = pts == p
    assert np.isnan(got["plain_bitmap_zncc"][on_p]).all()
    want = _tagged(m)["plain_bitmap_zncc"]
    np.testing.assert_array_equal(got["plain_bitmap_zncc"][~on_p], want[~on_p])
    np.testing.assert_array_equal(
        got["zncc_self_similarity_ellipse_axes"],
        _tagged(m)["zncc_self_similarity_ellipse_axes"],
    )
