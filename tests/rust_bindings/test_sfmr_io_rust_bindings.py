# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``.sfmr`` dict I/O bindings (``read_sfmr`` / ``write_sfmr``).

The dict API is columnar: ``read_sfmr`` hands back every stored array and
``write_sfmr`` puts them back, so ``write_sfmr(out, read_sfmr(path))`` must
reproduce the file's columns for either observation source — including an
``embedded_patches`` one, whose write validation demands the per-point patch
frame.
"""

import numpy as np
import pytest

from sfmtool._sfmtool.io import read_sfmr, write_sfmr
from sfmtool._sfmtool.patches import PatchCloud
from sfmtool._sfmtool.reconstruction import SfmrReconstruction

# Every numpy-array key `read_sfmr` emits. Optional columns are `None` when the
# file does not carry them.
ARRAY_KEYS = (
    "camera_indexes",
    "quaternions_wxyz",
    "translations_xyz",
    "thumbnails_y_x_rgb",
    "positions_xyzw",
    "colors_rgb",
    "reprojection_errors",
    "normals_xyz",
    "normal_confidence",
    "patch_u_halfvec_xyz",
    "patch_v_halfvec_xyz",
    "patch_bitmaps_y_x_rgba",
    "image_indexes",
    "feature_indexes",
    "keypoints_xy",
    "observation_confidence",
    "point_indexes",
    "observation_counts",
    "observed_depth_histogram_counts",
)


def _assert_columns_identical(lhs: dict, rhs: dict) -> None:
    """Every array column matches bit for bit (and absence matches absence)."""
    for key in ARRAY_KEYS:
        assert key in lhs, key
        assert key in rhs, key
        a, b = lhs[key], rhs[key]
        if a is None or b is None:
            assert a is None and b is None, f"{key}: {a!r} vs {b!r}"
            continue
        assert a.dtype == b.dtype, key
        assert a.shape == b.shape, key
        # Bitwise, so a float column that merely round-trips to within an ulp
        # still fails.
        assert a.tobytes() == b.tobytes(), key


def _embedded_patches_sfmr(recon: SfmrReconstruction, path):
    """Write `recon` (a ``sift_files`` one) out as an ``embedded_patches`` file.

    Carries a per-point patch frame and bitmaps plus both confidence columns,
    with per-row-distinct values so a dropped, zeroed or reordered column is
    visible.
    """
    n_obs = int(np.asarray(recon.track_image_indexes).shape[0])
    n_img = len(recon.image_names)
    n_pts = recon.point_count

    # Keypoints must sit inside the image bounds for read/verify to accept them.
    keypoints = np.arange(n_obs * 2, dtype=np.float32).reshape(n_obs, 2)
    keypoints[:, 0] %= min(c.width for c in recon.cameras)
    keypoints[:, 1] %= min(c.height for c in recon.cameras)

    half = (0.1 + 0.001 * np.arange(n_pts)).astype(np.float32)
    u = np.zeros((n_pts, 3), dtype=np.float32)
    v = np.zeros((n_pts, 3), dtype=np.float32)
    u[:, 0] = half
    v[:, 1] = half
    cloud = PatchCloud.from_halfvec_arrays(
        u, v, np.asarray(recon.positions, dtype=np.float64)
    )

    resolution = 4
    bitmaps = (np.arange(n_pts * resolution * resolution * 4) % 251).astype(np.uint8)
    bitmaps = bitmaps.reshape(n_pts, resolution, resolution, 4)

    embedded = recon.clone_with_changes(
        feature_source="embedded_patches",
        keypoints_xy=keypoints,
        image_file_hashes=[bytes([i % 256] * 16) for i in range(n_img)],
        patches=cloud,
        patch_bitmaps=bitmaps,
        normal_confidence=(np.arange(n_pts) % 255 + 1).astype(np.uint8),
        observation_confidence=(np.arange(n_obs) % 255 + 1).astype(np.uint8),
    )
    embedded.save(path)
    return path


@pytest.fixture
def embedded_patches_sfmr(seoul_bull_sfmr_only, tmp_path):
    recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
    return _embedded_patches_sfmr(recon, tmp_path / "embedded.sfmr")


class TestDictRoundTrip:
    def test_embedded_patches_columns_survive(self, embedded_patches_sfmr, tmp_path):
        data = read_sfmr(embedded_patches_sfmr)

        # The patch frame is what an embedded_patches file is required to carry.
        n_pts = data["positions_xyzw"].shape[0]
        for key in ("patch_u_halfvec_xyz", "patch_v_halfvec_xyz"):
            assert data[key].dtype == np.float32, key
            assert data[key].shape == (n_pts, 3), key
        bitmaps = data["patch_bitmaps_y_x_rgba"]
        assert bitmaps.dtype == np.uint8
        assert bitmaps.shape == (n_pts, 4, 4, 4)
        # Its sift_files counterparts stay absent.
        assert data["feature_indexes"] is None
        assert data["keypoints_xy"] is not None
        # The two confidence columns are optional and independent of the patch
        # frame, and are read back the same way.
        assert data["normal_confidence"].shape == (n_pts,)
        assert data["observation_confidence"].shape == (data["keypoints_xy"].shape[0],)

        out = tmp_path / "roundtrip.sfmr"
        write_sfmr(out, data, skip_recompute_depth_stats=True)
        _assert_columns_identical(data, read_sfmr(out))

    def test_edited_embedded_patches_writes(self, embedded_patches_sfmr, tmp_path):
        """The reported failure: read, edit, write with the default recompute."""
        data = read_sfmr(embedded_patches_sfmr)
        colors = data["colors_rgb"].copy()
        colors[:, 0] = 7
        data["colors_rgb"] = colors

        out = tmp_path / "edited.sfmr"
        write_sfmr(out, data)

        reloaded = read_sfmr(out)
        assert reloaded["metadata"]["feature_source"] == "embedded_patches"
        np.testing.assert_array_equal(reloaded["colors_rgb"], colors)
        for key in (
            "patch_u_halfvec_xyz",
            "patch_v_halfvec_xyz",
            "patch_bitmaps_y_x_rgba",
            "keypoints_xy",
        ):
            assert reloaded[key].tobytes() == data[key].tobytes(), key

    def test_sift_files_columns_survive(self, seoul_bull_sfmr_only, tmp_path):
        data = read_sfmr(seoul_bull_sfmr_only)
        assert data["metadata"]["feature_source"] == "sift_files"
        # A sift_files file carries no patch frame; the keys are present as None.
        for key in (
            "patch_u_halfvec_xyz",
            "patch_v_halfvec_xyz",
            "patch_bitmaps_y_x_rgba",
        ):
            assert data[key] is None, key
        assert data["feature_indexes"] is not None

        out = tmp_path / "roundtrip.sfmr"
        write_sfmr(out, data, skip_recompute_depth_stats=True)
        _assert_columns_identical(data, read_sfmr(out))

    def test_patch_frame_can_be_dropped_by_key(self, embedded_patches_sfmr, tmp_path):
        """An explicit ``None`` clears a column — and the format then refuses the
        embedded_patches file that requires it."""
        data = read_sfmr(embedded_patches_sfmr)
        data["patch_u_halfvec_xyz"] = None
        data["patch_v_halfvec_xyz"] = None
        data["patch_bitmaps_y_x_rgba"] = None

        with pytest.raises(OSError, match="requires patch_u_halfvec_xyz"):
            write_sfmr(tmp_path / "dropped.sfmr", data, skip_recompute_depth_stats=True)


class TestPatchColumnValidation:
    def test_wrong_dtype_names_the_key(self, embedded_patches_sfmr, tmp_path):
        data = read_sfmr(embedded_patches_sfmr)
        data["patch_u_halfvec_xyz"] = data["patch_u_halfvec_xyz"].astype(np.float64)
        with pytest.raises(TypeError, match="'patch_u_halfvec_xyz' must be"):
            write_sfmr(tmp_path / "bad.sfmr", data, skip_recompute_depth_stats=True)

    def test_wrong_shape_names_the_key(self, embedded_patches_sfmr, tmp_path):
        data = read_sfmr(embedded_patches_sfmr)
        n_pts = data["positions_xyzw"].shape[0]
        data["patch_v_halfvec_xyz"] = np.zeros((n_pts, 2), dtype=np.float32)
        with pytest.raises(ValueError, match=r"'patch_v_halfvec_xyz' must have shape"):
            write_sfmr(tmp_path / "bad.sfmr", data, skip_recompute_depth_stats=True)

    def test_wrong_bitmap_shape_names_the_key(self, embedded_patches_sfmr, tmp_path):
        data = read_sfmr(embedded_patches_sfmr)
        n_pts = data["positions_xyzw"].shape[0]
        data["patch_bitmaps_y_x_rgba"] = np.zeros((n_pts, 4, 4, 3), dtype=np.uint8)
        with pytest.raises(
            ValueError, match=r"'patch_bitmaps_y_x_rgba' must have shape"
        ):
            write_sfmr(tmp_path / "bad.sfmr", data, skip_recompute_depth_stats=True)
