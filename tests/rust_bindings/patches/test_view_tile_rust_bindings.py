# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``OrientedPatch.render_view_tile``: one view's ``R x R`` tile of a
patch, rendered as the bench renders an observation's tile, with the per-view
readings the reference-view rule takes on it, and that each stored bitmap the
patch bindings render is the tile of the observation they name. See
``specs/core/patch/reference-view.md``."""

import numpy as np
import pytest

from sfmtool.geometry import CameraIntrinsics, RigidTransform
from sfmtool.patches import OrientedPatch


def _pinhole(w=64, h=48, f=60.0):
    return CameraIntrinsics(
        "PINHOLE",
        w,
        h,
        {
            "focal_length_x": f,
            "focal_length_y": f,
            "principal_point_x": w / 2.0,
            "principal_point_y": h / 2.0,
        },
    )


def _identity():
    return RigidTransform.from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def _photograph(w=64, h=48):
    y, x = np.mgrid[0:h, 0:w]
    grey = (40 + (x * 7 + y * 13) % 150).astype(np.uint8)
    return np.dstack([grey, grey, grey])


def test_a_patch_facing_the_camera_renders_a_whole_tile():
    # Canonical cameras look down -Z; this patch faces back along +Z.
    patch = OrientedPatch([0, 0, -4], [1, 0, 0], [0, 1, 0], [0.4, 0.4])
    tile = patch.render_view_tile(_pinhole(), _identity(), _photograph(), resolution=16)
    assert tile["samples"].shape == (16, 16, 3)
    assert tile["samples"].dtype == np.uint8
    assert tile["valid"].shape == (16, 16) and tile["valid"].all()
    assert tile["coverage"] == 1.0
    assert tile["clipped_share"] == 0.0
    assert tile["viewing_angle_deg"] == pytest.approx(0.0, abs=1e-6)
    assert tile["tilt_direction_deg"] is None
    assert tile["sampler"] in {"bilinear_mip", "anisotropic"}
    assert tile["jacobian"].shape == (2, 2)
    np.testing.assert_allclose(tile["patch_frame"].center, [0, 0, -4])


def test_anchoring_on_a_keypoint_moves_the_tile_and_reads_the_angle_there():
    patch = OrientedPatch([0, 0, -4], [1, 0, 0], [0, 1, 0], [0.4, 0.4])
    tile = patch.render_view_tile(
        _pinhole(), _identity(), _photograph(), keypoint=(42.0, 24.0), resolution=16
    )
    # The slid patch keeps its depth on the plane facing the camera, which is
    # its own here, and its centre projects onto the keypoint, 10 px right of
    # the principal point: 10 / 60 of the depth.
    center = tile["patch_frame"].center
    assert center[2] == pytest.approx(-4.0)
    assert abs(center[0]) == pytest.approx(4.0 * 10.0 / 60.0)
    # The ray to it leans off the normal by atan(10 / 60).
    assert tile["viewing_angle_deg"] == pytest.approx(np.degrees(np.arctan(10 / 60)))
    assert tile["tilt_direction_deg"] is not None


def test_a_blown_out_photograph_is_clipped_and_a_tile_off_its_edge_is_partial():
    patch = OrientedPatch([0, 0, -4], [1, 0, 0], [0, 1, 0], [0.4, 0.4])
    white = np.full((48, 64, 3), 255, np.uint8)
    tile = patch.render_view_tile(_pinhole(), _identity(), white, resolution=16)
    assert tile["clipped_share"] == 1.0
    edge = OrientedPatch([2.1, 0, -4], [1, 0, 0], [0, 1, 0], [0.4, 0.4])
    tile = edge.render_view_tile(_pinhole(), _identity(), _photograph(), resolution=16)
    assert 0.0 < tile["coverage"] < 1.0
    assert tile["coverage"] == pytest.approx(tile["valid"].mean())


def test_a_fixed_sampler_and_bad_arguments():
    patch = OrientedPatch([0, 0, -4], [1, 0, 0], [0, 1, 0], [0.4, 0.4])
    tile = patch.render_view_tile(
        _pinhole(), _identity(), _photograph(), resolution=8, sampler="bilinear"
    )
    assert tile["sampler"] == "bilinear"
    with pytest.raises(ValueError, match="resolution"):
        patch.render_view_tile(_pinhole(), _identity(), _photograph(), resolution=1)
    with pytest.raises(ValueError, match="sampler"):
        patch.render_view_tile(
            _pinhole(), _identity(), _photograph(), sampler="nearest"
        )
    with pytest.raises(ValueError, match="camera"):
        patch.render_view_tile(_pinhole(), _identity(), _photograph(32, 32))
    with pytest.raises(ValueError, match="resolution"):
        patch.render_view_tile(_pinhole(), _identity(), _photograph(), resolution=4096)
    with pytest.raises(ValueError, match="keypoint must be finite"):
        patch.render_view_tile(
            _pinhole(), _identity(), _photograph(), keypoint=(float("nan"), 3.0)
        )
    with pytest.raises(ValueError, match="keypoint must be finite"):
        patch.render_view_tile(
            _pinhole(), _identity(), _photograph(), keypoint=(1.0, float("inf"))
        )
    # An index names a view of an ImagePyramidSet; with one photograph it would
    # be ignored, so it is refused.
    with pytest.raises(ValueError, match="image_index"):
        patch.render_view_tile(_pinhole(), _identity(), _photograph(), image_index=0)


# -- The stored bitmap is the tile of its reference view ----------------------
#
# On the seoul_bull ground truth (an embedded_patches file of 280 points beside
# its 17 photographs), each operation that renders the stored bitmap renders the
# tile of one observation, and names it. These read the file in place.

RESOLUTION = 16


@pytest.fixture(scope="module")
def ground_truth():
    from sfmtool.reconstruction import SfmrReconstruction

    from ...conftest import SEOUL_BULL_GROUND_TRUTH
    from ...patch.conftest import load_images

    recon = SfmrReconstruction.load(str(SEOUL_BULL_GROUND_TRUTH))
    return recon, load_images(recon)


def _track_offsets(recon) -> np.ndarray:
    counts = np.asarray(recon.observation_counts)
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)


def _tile_as_bitmap(recon, images, patch, image, keypoint) -> np.ndarray:
    """The tile of ``patch`` in ``image`` anchored on ``keypoint``, as a stored
    bitmap: its colour, and alpha 255 where it has data and 0 elsewhere."""
    camera = recon.cameras[int(np.asarray(recon.camera_indexes)[image])]
    pose = RigidTransform.from_wxyz_translation(
        np.asarray(recon.quaternions_wxyz)[image].tolist(),
        np.asarray(recon.translations)[image].tolist(),
    )
    tile = patch.render_view_tile(
        camera,
        pose,
        images[image],
        keypoint=[float(c) for c in keypoint],
        resolution=RESOLUTION,
    )
    alpha = np.where(tile["valid"], 255, 0).astype(np.uint8)
    return np.dstack([tile["samples"], alpha])


def test_render_bitmaps_stores_the_tile_of_the_reference_observation(ground_truth):
    recon, images = ground_truth
    cloud = recon.patches
    bitmaps, refs = cloud.render_bitmaps(recon, images, resolution=RESOLUTION)

    n = recon.point_count
    assert bitmaps.shape == (n, RESOLUTION, RESOLUTION, 4)
    assert bitmaps.dtype == np.uint8
    assert refs.shape == (n,)
    assert refs.dtype == np.int32
    counts = np.asarray(recon.observation_counts)
    assert np.all((refs >= -1) & (refs < counts))
    picked = np.flatnonzero(refs >= 0)
    assert len(picked) >= 0.9 * n

    # A picked bitmap is one photograph's tile: its alpha marks data or none.
    assert set(np.unique(bitmaps[picked][..., 3]).tolist()) <= {0, 255}

    offsets = _track_offsets(recon)
    track_images = np.asarray(recon.track_image_indexes)
    keypoints = np.asarray(recon.keypoints_xy)
    by_point = {int(p): i for i, p in enumerate(cloud.point_indexes)}
    for p in picked:
        row = offsets[p] + refs[p]
        expected = _tile_as_bitmap(
            recon, images, cloud[by_point[int(p)]], track_images[row], keypoints[row]
        )
        np.testing.assert_array_equal(bitmaps[p], expected, err_msg=f"point {p}")


def test_refine_keypoints_bitmap_is_the_tile_of_its_reference_image(ground_truth):
    recon, images = ground_truth
    cloud = recon.patches
    results = cloud.refine_keypoints(
        recon, images, resolution=RESOLUTION, render_bitmaps=True
    )
    offsets = _track_offsets(recon)
    counts = np.asarray(recon.observation_counts)
    track_images = np.asarray(recon.track_image_indexes)
    by_point = {int(p): i for i, p in enumerate(cloud.point_indexes)}

    named = 0
    for entry in results:
        p = int(entry["point_index"])
        reference = entry["reference_image"]
        if reference is None:
            continue
        assert entry["bitmap"] is not None
        track = track_images[offsets[p] : offsets[p] + counts[p]]
        assert reference in track, f"point {p} names image {reference}"
        # The tile is rendered at the reference view's final keypoint.
        views = np.asarray(entry["views"])
        k = int(np.flatnonzero(views == reference)[0])
        expected = _tile_as_bitmap(
            recon, images, cloud[by_point[p]], reference, entry["keypoints"][k]
        )
        np.testing.assert_array_equal(entry["bitmap"], expected, err_msg=f"point {p}")
        named += 1
    assert named >= 0.9 * len(results)
