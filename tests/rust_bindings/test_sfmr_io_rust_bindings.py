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

from sfmtool._sfmtool.io import (
    POINT_CONSTRAINT_NAMES,
    read_sfmr,
    verify_sfmr,
    write_sfmr,
)
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
    "point_constraints",
    "constraint_distances",
    "constraint_reference_images",
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


# ── Per-point constraints (points3d/point_constraints,
#    /constraint_distances, /constraint_reference_images) ──────────────────

_NO_REFERENCE_IMAGE = np.uint32(0xFFFFFFFF)
# The column is numeric, and this is the numbering every column reaching Python
# is on -- the module constant is what a consumer labels a code with.
_FREE, _RANGED, _HELD = 0, 1, 2


def _constraint_columns(positions_xyzw, n_img):
    """Hold the first finite point and range the second at 12.5 from image 0.

    Returns the triple, plus the two point indexes it constrained, so a caller
    can assert on the rows it wrote rather than on a fixed index.
    """
    n = positions_xyzw.shape[0]
    finite = np.flatnonzero(positions_xyzw[:, 3] != 0.0)
    assert finite.size >= 2 and n_img >= 1
    held, ranged = int(finite[0]), int(finite[1])

    constraints = np.zeros(n, dtype=np.uint8)
    constraints[held] = _HELD
    constraints[ranged] = _RANGED
    distance = np.full(n, np.nan)
    distance[ranged] = 12.5
    reference = np.full(n, _NO_REFERENCE_IMAGE, dtype=np.uint32)
    reference[ranged] = 0
    return (constraints, distance, reference), held, ranged


class TestPointConstraints:
    def test_dict_round_trip(self, seoul_bull_sfmr_only, tmp_path):
        # The three columns go out through `write_sfmr` and come back through
        # `read_sfmr` unchanged, and the file still verifies.
        data = read_sfmr(seoul_bull_sfmr_only)
        assert data["point_constraints"] is None
        (constraints, distance, reference), held, ranged = _constraint_columns(
            data["positions_xyzw"], len(data["image_names"])
        )
        data["point_constraints"] = constraints
        data["constraint_distances"] = distance
        data["constraint_reference_images"] = reference

        out = tmp_path / "constrained.sfmr"
        write_sfmr(out, data, skip_recompute_depth_stats=True)
        reloaded = read_sfmr(out)
        _assert_columns_identical(data, reloaded)

        valid, errors = verify_sfmr(out)
        assert valid, errors
        assert reloaded["constraint_distances"][ranged] == 12.5

        # The column stays a uint8 array, and the module constant is what names
        # its codes.
        codes = reloaded["point_constraints"]
        assert codes.dtype == np.uint8
        assert POINT_CONSTRAINT_NAMES == ("free", "ranged", "held")
        assert POINT_CONSTRAINT_NAMES[codes[held]] == "held"
        assert POINT_CONSTRAINT_NAMES[codes[ranged]] == "ranged"

    def test_all_free_columns_are_dropped(self, seoul_bull_sfmr_only, tmp_path):
        # An all-free set says what carrying no set says, so it is not written.
        data = read_sfmr(seoul_bull_sfmr_only)
        n = data["positions_xyzw"].shape[0]
        data["point_constraints"] = np.zeros(n, dtype=np.uint8)
        data["constraint_distances"] = np.full(n, np.nan)
        data["constraint_reference_images"] = np.full(
            n, _NO_REFERENCE_IMAGE, dtype=np.uint32
        )

        out = tmp_path / "free.sfmr"
        write_sfmr(out, data, skip_recompute_depth_stats=True)
        assert read_sfmr(out)["point_constraints"] is None

    def test_finite_distance_needs_a_real_image(self, seoul_bull_sfmr_only, tmp_path):
        data = read_sfmr(seoul_bull_sfmr_only)
        (constraints, distance, reference), _held, ranged = _constraint_columns(
            data["positions_xyzw"], len(data["image_names"])
        )
        reference[ranged] = 10_000
        data["point_constraints"] = constraints
        data["constraint_distances"] = distance
        data["constraint_reference_images"] = reference
        with pytest.raises(OSError, match="past the"):
            write_sfmr(tmp_path / "bad.sfmr", data, skip_recompute_depth_stats=True)

    def test_columns_survive_a_point_filter(self, seoul_bull_sfmr_only, tmp_path):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        (constraints, distance, reference), held, ranged = _constraint_columns(
            np.asarray(recon.positions_xyzw), len(recon.image_names)
        )
        recon = recon.clone_with_changes(
            point_constraints=constraints,
            constraint_distances=distance,
            constraint_reference_images=reference,
        )

        # Drop everything but the two constrained points, in their own order.
        mask = np.zeros(recon.point_count, dtype=bool)
        mask[[held, ranged]] = True
        out = recon.filter_points_by_mask(mask)

        assert out.point_count == 2
        npt = np.asarray(out.point_constraints)
        assert list(npt) == [_HELD, _RANGED]
        assert np.isnan(np.asarray(out.constraint_distances)[0])
        assert np.asarray(out.constraint_distances)[1] == 12.5
        assert list(np.asarray(out.constraint_reference_images)) == [
            _NO_REFERENCE_IMAGE,
            0,
        ]

    def test_dropping_the_referenced_image_frees_the_point(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        (constraints, distance, reference), held, ranged = _constraint_columns(
            np.asarray(recon.positions_xyzw), len(recon.image_names)
        )
        recon = recon.clone_with_changes(
            point_constraints=constraints,
            constraint_distances=distance,
            constraint_reference_images=reference,
        )
        n_img = len(recon.image_names)

        # Image 0 kept, at a new index: the distance follows it.
        order = np.array(list(range(1, n_img)) + [0], dtype=np.uint32)
        kept = recon.subset_by_image_indices(order, False)
        assert np.asarray(kept.point_constraints)[ranged] == _RANGED
        assert np.asarray(kept.constraint_reference_images)[ranged] == n_img - 1

        # Image 0 gone: nothing is left to measure the distance from, so the
        # point is released.
        dropped = recon.subset_by_image_indices(
            np.arange(1, n_img, dtype=np.uint32), False
        )
        assert np.asarray(dropped.point_constraints)[ranged] == _FREE
        assert np.isnan(np.asarray(dropped.constraint_distances)[ranged])
        assert (
            np.asarray(dropped.constraint_reference_images)[ranged]
            == _NO_REFERENCE_IMAGE
        )
        # A held point names no image, so the same subset leaves it held.
        assert np.asarray(dropped.point_constraints)[held] == _HELD

    def test_the_triple_must_be_passed_together(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        with pytest.raises(ValueError, match="passed together"):
            recon.clone_with_changes(
                point_constraints=np.zeros(recon.point_count, dtype=np.uint8)
            )

    def test_a_code_outside_the_numbering_is_refused(
        self, seoul_bull_sfmr_only, tmp_path
    ):
        # The codes are the canonical numbering, so a column carrying anything
        # else has no name and is not written.
        data = read_sfmr(seoul_bull_sfmr_only)
        (constraints, distance, reference), _held, _ranged = _constraint_columns(
            data["positions_xyzw"], len(data["image_names"])
        )
        constraints[0] = len(POINT_CONSTRAINT_NAMES)
        data["point_constraints"] = constraints
        data["constraint_distances"] = distance
        data["constraint_reference_images"] = reference
        with pytest.raises(OSError, match="its legend gives"):
            write_sfmr(tmp_path / "bad.sfmr", data, skip_recompute_depth_stats=True)


class TestSharedColumnsAreReadOnly:
    """The thumbnail array is a zero-copy view of a buffer that every clone of
    the reconstruction shares, so the view is read-only: a write from Python
    would land in every sharer at once."""

    def test_thumbnails_view_is_read_only(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        view = recon.thumbnails_y_x_rgb
        assert not view.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            view[0, 0, 0, 0] = 1

    def test_a_write_to_a_copy_reaches_no_clone(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        clone = recon.clone_with_changes()
        before = np.asarray(recon.thumbnails_y_x_rgb).copy()
        edited = recon.thumbnails_y_x_rgb.copy()
        edited[0, 0, 0, 0] = (int(edited[0, 0, 0, 0]) + 1) % 256
        assert np.array_equal(np.asarray(recon.thumbnails_y_x_rgb), before)
        assert np.array_equal(np.asarray(clone.thumbnails_y_x_rgb), before)
