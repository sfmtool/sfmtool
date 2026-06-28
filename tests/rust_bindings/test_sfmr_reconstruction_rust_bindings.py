# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the homogeneous-point accessors on the SfmrReconstruction binding."""

import numpy as np
import pytest

from sfmtool._sfmtool import PatchCloud, SfmrReconstruction


class TestHomogeneousPointAccessors:
    def test_finite_reconstruction_accessors(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        m = len(recon.positions)
        assert m > 0

        positions = recon.positions
        positions_xyzw = recon.positions_xyzw
        at_infinity = recon.point_is_at_infinity

        assert positions.shape == (m, 3)
        assert positions_xyzw.shape == (m, 4)
        assert at_infinity.shape == (m,)
        assert at_infinity.dtype == bool

        # A normally solved reconstruction contains only finite points.
        assert not at_infinity.any()
        assert recon.infinity_point_count == 0
        assert np.array_equal(positions_xyzw[:, 3], np.ones(m))
        assert np.array_equal(positions_xyzw[:, :3], positions)

    def test_clone_with_changes_homogeneous_positions(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        positions_xyzw = recon.positions_xyzw.copy()
        assert len(positions_xyzw) >= 2

        # Turn the first point into a point at infinity: direction (0, 0, 2),
        # w = 0. clone_with_changes must normalise the direction to unit length.
        positions_xyzw[0] = [0.0, 0.0, 2.0, 0.0]
        clone = recon.clone_with_changes(positions=positions_xyzw)

        at_infinity = clone.point_is_at_infinity
        assert at_infinity[0]
        assert not at_infinity[1:].any()
        assert clone.infinity_point_count == 1
        assert clone.infinity_point_count == int(at_infinity.sum())

        assert clone.positions_xyzw[0, 3] == 0.0
        np.testing.assert_allclose(clone.positions[0], [0.0, 0.0, 1.0])

    def test_clone_with_changes_rejects_all_zero_homogeneous_coordinate(
        self, seoul_bull_sfmr_only
    ):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        positions_xyzw = recon.positions_xyzw.copy()
        assert len(positions_xyzw) > 0
        # (0, 0, 0, 0) is no point at all: w = 0 with a zero direction.
        positions_xyzw[0] = [0.0, 0.0, 0.0, 0.0]

        with pytest.raises(ValueError, match="all-zero homogeneous coordinate"):
            recon.clone_with_changes(positions=positions_xyzw)

    def test_clone_with_changes_euclidean_positions_stay_finite(
        self, seoul_bull_sfmr_only
    ):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        # An (N, 3) Euclidean array keeps every point finite (w = 1).
        clone = recon.clone_with_changes(positions=recon.positions)
        assert not clone.point_is_at_infinity.any()
        assert np.array_equal(clone.positions_xyzw[:, 3], np.ones(len(clone.positions)))


class TestInfinityConversions:
    def test_classify_preserves_point_count(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        classified = recon.classify_points_at_infinity()
        # Reclassification never adds or drops points or observations.
        assert classified.point_count == recon.point_count
        assert classified.observation_count == recon.observation_count
        assert classified.infinity_point_count == int(
            classified.point_is_at_infinity.sum()
        )

    def test_classify_detects_a_far_point(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        # Pick a well-observed point and push it far away so its observation
        # rays become parallel — its parallax collapses below the noise floor.
        idx = int(np.argmax(recon.observation_counts))
        positions = recon.positions_xyzw.copy()
        positions[idx] = [0.0, 0.0, 1.0e7, 1.0]
        recon = recon.clone_with_changes(positions=positions)

        classified = recon.classify_points_at_infinity()
        assert classified.point_is_at_infinity[idx]
        # The cached count reflects the newly-classified far point.
        assert classified.infinity_point_count >= 1
        assert classified.infinity_point_count == int(
            classified.point_is_at_infinity.sum()
        )
        # A point at infinity stores a unit-length direction.
        np.testing.assert_allclose(
            np.linalg.norm(classified.positions[idx]), 1.0, atol=1e-9
        )

    def test_classify_noise_floor_is_monotone(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        # A larger noise floor can only classify a superset of points.
        strict = recon.classify_points_at_infinity(noise_floor_px=0.01)
        loose = recon.classify_points_at_infinity(noise_floor_px=1.0e4)
        assert loose.point_is_at_infinity.sum() >= strict.point_is_at_infinity.sum()
        assert strict.infinity_point_count == int(strict.point_is_at_infinity.sum())
        assert loose.infinity_point_count == int(loose.point_is_at_infinity.sum())

    def test_materialize_makes_every_point_finite(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        positions = recon.positions_xyzw.copy()
        n_infinity = min(5, len(positions))
        for i in range(n_infinity):
            positions[i] = [0.0, 0.0, 1.0, 0.0]
        recon = recon.clone_with_changes(positions=positions)
        assert recon.point_is_at_infinity[:n_infinity].all()
        # clone_with_changes rebuilds the cache: it counts the new w=0 points.
        assert recon.infinity_point_count == n_infinity

        materialized = recon.materialize_points_at_infinity()
        assert not materialized.point_is_at_infinity.any()
        assert materialized.point_count == recon.point_count
        assert np.all(np.isfinite(materialized.positions))
        assert materialized.infinity_point_count == 0

    def test_materialize_leaves_finite_points_unchanged(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        # No points at infinity — materialise is a no-op on the positions.
        materialized = recon.materialize_points_at_infinity()
        np.testing.assert_array_equal(materialized.positions, recon.positions)
        assert materialized.infinity_point_count == 0


class TestWorldSpaceUnit:
    def test_default_is_none(self, seoul_bull_sfmr_only):
        # A freshly solved reconstruction is in arbitrary units.
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        assert recon.world_space_unit is None

    def test_clone_sets_and_clears_unit(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)

        scaled = recon.clone_with_changes(world_space_unit="m")
        assert scaled.world_space_unit == "m"
        # The original is untouched.
        assert recon.world_space_unit is None

        cleared = scaled.clone_with_changes(world_space_unit=None)
        assert cleared.world_space_unit is None

    def test_unit_survives_save_load_roundtrip(self, seoul_bull_sfmr_only, tmp_path):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        scaled = recon.clone_with_changes(world_space_unit="mm")

        out_path = tmp_path / "scaled.sfmr"
        scaled.save(out_path)

        reloaded = SfmrReconstruction.load(out_path)
        assert reloaded.world_space_unit == "mm"


class TestOptionalNormals:
    def test_default_has_normals(self, seoul_bull_sfmr_only):
        # A solved reconstruction carries (auto-computed) normals.
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        assert recon.has_normals is True

    def test_clear_normals_round_trips(self, seoul_bull_sfmr_only, tmp_path):
        import numpy as np

        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)

        # Opt out of normals entirely.
        no_normals = recon.clone_with_changes(normals=None)
        assert no_normals.has_normals is False
        # The original is untouched.
        assert recon.has_normals is True
        # `normals` is still a valid (all-zero) array.
        n = np.asarray(no_normals.normals)
        assert n.shape == (no_normals.point_count, 3)
        assert not n.any()

        # Absence survives a save/load round trip.
        out_path = tmp_path / "no_normals.sfmr"
        no_normals.save(out_path)
        reloaded = SfmrReconstruction.load(out_path)
        assert reloaded.has_normals is False
        assert not np.asarray(reloaded.normals).any()

        # The normals_xyz entry is absent from the archive.
        import zipfile

        with zipfile.ZipFile(out_path) as zf:
            names = zf.namelist()
        assert not any("normals_xyz" in n for n in names)

    def test_set_normals_marks_present(self, seoul_bull_sfmr_only):
        import numpy as np

        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        cleared = recon.clone_with_changes(normals=None)
        assert cleared.has_normals is False

        # Re-supplying normals marks them present again.
        normals = np.zeros((cleared.point_count, 3), dtype=np.float32)
        normals[:, 2] = 1.0
        restored = cleared.clone_with_changes(normals=normals)
        assert restored.has_normals is True
        np.testing.assert_allclose(np.asarray(restored.normals), normals)


class TestEmbeddedPatches:
    """Format v4 embedded_patches: read accessors and the clone path."""

    def test_sift_files_defaults(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        assert recon.feature_source == "sift_files"
        assert recon.keypoints_xy is None
        assert recon.image_file_hashes is None

    def test_clone_to_embedded_round_trips(self, seoul_bull_sfmr_only, tmp_path):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        n_obs = int(np.asarray(recon.track_image_indexes).shape[0])
        n_img = len(recon.image_names)

        keypoints = np.arange(n_obs * 2, dtype=np.float32).reshape(n_obs, 2)
        # Keep keypoints inside the image bounds so the format's read/verify
        # validation accepts them.
        cam = recon.cameras[0]
        keypoints[:, 0] %= cam.width
        keypoints[:, 1] %= cam.height
        img_hashes = [bytes([i % 256] * 16) for i in range(n_img)]

        # An embedded_patches file requires a per-point patch frame; attach a
        # synthetic one (one patch per point) so the v4 writer accepts it.
        n_pts = recon.point_count
        u = np.tile([0.1, 0.0, 0.0], (n_pts, 1)).astype(np.float32)
        v = np.tile([0.0, 0.1, 0.0], (n_pts, 1)).astype(np.float32)
        cloud = PatchCloud.from_halfvec_arrays(
            u, v, np.asarray(recon.positions, dtype=np.float64)
        )

        embedded = recon.clone_with_changes(
            feature_source="embedded_patches",
            keypoints_xy=keypoints,
            image_file_hashes=img_hashes,
            patches=cloud,
        )
        assert embedded.feature_source == "embedded_patches"
        np.testing.assert_array_equal(np.asarray(embedded.keypoints_xy), keypoints)
        assert embedded.image_file_hashes == img_hashes

        # Survives a save / load through the v4 format.
        out = tmp_path / "embedded.sfmr"
        embedded.save(out)
        reloaded = SfmrReconstruction.load(out)
        assert reloaded.feature_source == "embedded_patches"
        np.testing.assert_array_equal(np.asarray(reloaded.keypoints_xy), keypoints)
        assert reloaded.image_file_hashes == img_hashes

    def test_clone_keypoints_wrong_length_rejected(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        bad = np.zeros((3, 2), dtype=np.float32)  # not observation_count rows
        with pytest.raises(ValueError, match="observation count"):
            recon.clone_with_changes(keypoints_xy=bad)

    def _make_embedded(self, recon):
        """Clone `recon` (a sift_files recon) into an embedded_patches one."""
        n_obs = int(np.asarray(recon.track_image_indexes).shape[0])
        n_img = len(recon.image_names)
        keypoints = np.zeros((n_obs, 2), dtype=np.float32)
        img_hashes = [bytes([i % 256] * 16) for i in range(n_img)]
        return recon.clone_with_changes(
            feature_source="embedded_patches",
            keypoints_xy=keypoints,
            image_file_hashes=img_hashes,
        )

    def test_embedded_sift_only_getters_are_none(self, seoul_bull_sfmr_only):
        # The sift_files-only columns report None (not an empty array) in
        # embedded mode, matching keypoints_xy/image_file_hashes in sift mode.
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        # sift_files mode: these are present, the embedded ones are None.
        assert recon.track_feature_indexes is not None
        assert recon.feature_tool_hashes is not None
        assert recon.sift_content_hashes is not None

        embedded = self._make_embedded(recon)
        assert embedded.track_feature_indexes is None
        assert embedded.feature_tool_hashes is None
        assert embedded.sift_content_hashes is None
        assert embedded.image_file_hashes is not None
        assert embedded.keypoints_xy is not None

    def test_clone_embedded_to_sift_files_rejected(self, seoul_bull_sfmr_only):
        # Embedded → sift_files has no source for per-observation feature
        # indices, so the conversion is refused.
        embedded = self._make_embedded(SfmrReconstruction.load(seoul_bull_sfmr_only))
        n_img = len(embedded.image_names)
        hashes = [bytes(16) for _ in range(n_img)]
        with pytest.raises(
            ValueError, match="converting to sift_files is not supported"
        ):
            embedded.clone_with_changes(
                feature_source="sift_files",
                feature_tool_hashes=hashes,
                sift_content_hashes=hashes,
            )

    def test_clone_to_embedded_requires_image_file_hashes(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        n_obs = int(np.asarray(recon.track_image_indexes).shape[0])
        keypoints = np.zeros((n_obs, 2), dtype=np.float32)
        with pytest.raises(
            ValueError, match="embedded_patches requires image_file_hashes"
        ):
            recon.clone_with_changes(
                feature_source="embedded_patches",
                keypoints_xy=keypoints,
            )

    def test_clone_unknown_feature_source_rejected(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        with pytest.raises(ValueError, match="unknown feature_source"):
            recon.clone_with_changes(feature_source="bogus")

    def test_clone_embedded_track_replacement_without_keypoints_rejected(
        self, seoul_bull_sfmr_only
    ):
        # Replacing the tracks of an embedded recon changes the observation
        # count; without a matching keypoints_xy the columns would desync, so
        # the final validation must reject it.
        embedded = self._make_embedded(SfmrReconstruction.load(seoul_bull_sfmr_only))
        n_new = int(np.asarray(embedded.track_image_indexes).shape[0]) + 2
        img_idx = np.zeros(n_new, dtype=np.uint32)
        feat_idx = np.zeros(n_new, dtype=np.uint32)
        pt_ids = np.zeros(n_new, dtype=np.uint32)
        with pytest.raises(ValueError, match="keypoints_xy"):
            embedded.clone_with_changes(
                track_image_indexes=img_idx,
                track_feature_indexes=feat_idx,
                track_point_indexes=pt_ids,
            )

    def test_clone_embedded_track_and_keypoints_replacement_to_new_size(
        self, seoul_bull_sfmr_only
    ):
        # Replacing tracks AND keypoints together to a new observation count is
        # accepted: the keypoint row count is validated against the *new* track
        # count, not the old one.
        embedded = self._make_embedded(SfmrReconstruction.load(seoul_bull_sfmr_only))
        n_new = int(np.asarray(embedded.track_image_indexes).shape[0]) + 2
        img_idx = np.zeros(n_new, dtype=np.uint32)
        feat_idx = np.zeros(n_new, dtype=np.uint32)
        pt_ids = np.zeros(n_new, dtype=np.uint32)
        keypoints = np.zeros((n_new, 2), dtype=np.float32)
        out = embedded.clone_with_changes(
            track_image_indexes=img_idx,
            track_feature_indexes=feat_idx,
            track_point_indexes=pt_ids,
            keypoints_xy=keypoints,
        )
        assert out.feature_source == "embedded_patches"
        assert np.asarray(out.keypoints_xy).shape == (n_new, 2)

    def test_clone_track_replacement_recomputes_observation_counts(
        self, seoul_bull_sfmr_only
    ):
        # Replacing tracks recomputes observation_counts from the new tracks
        # (grouped by point) instead of leaving the old per-point counts stale.
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        # Point 0 gets 3 observations, point 1 gets 1; all later points get 0.
        img_idx = np.array([0, 1, 2, 0], dtype=np.uint32)
        feat_idx = np.array([0, 1, 2, 3], dtype=np.uint32)
        pt_ids = np.array([0, 0, 0, 1], dtype=np.uint32)
        out = recon.clone_with_changes(
            track_image_indexes=img_idx,
            track_feature_indexes=feat_idx,
            track_point_indexes=pt_ids,
        )
        counts = np.asarray(out.observation_counts)
        assert counts.shape[0] == recon.point_count  # one entry per point
        assert counts[0] == 3
        assert counts[1] == 1
        assert counts[2:].sum() == 0
        assert int(counts.sum()) == out.observation_count == 4

    def test_clone_track_replacement_rejects_out_of_range_point(
        self, seoul_bull_sfmr_only
    ):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        bad_pt = recon.point_count  # one past the last valid point index
        img_idx = np.array([0], dtype=np.uint32)
        feat_idx = np.array([0], dtype=np.uint32)
        pt_ids = np.array([bad_pt], dtype=np.uint32)
        with pytest.raises(ValueError, match="out of range"):
            recon.clone_with_changes(
                track_image_indexes=img_idx,
                track_feature_indexes=feat_idx,
                track_point_indexes=pt_ids,
            )

    def test_clone_multipoint_track_replacement_survives_round_trip(
        self, seoul_bull_sfmr_only, tmp_path
    ):
        # End-to-end: replace the points and tracks with a multi-point,
        # point-grouped layout, then save and reload. The recomputed
        # observation_counts must drive offsets that keep the tracks consistent
        # across the format round trip. (Also exercises the points-resize +
        # track-replacement interaction: the point-index bound is the new count.)
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        # Three points (the format requires every point to be observed) with
        # 2/3/1 observations, grouped contiguously by point.
        positions = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], dtype=np.float64
        )
        img_idx = np.array([0, 1, 0, 1, 2, 0], dtype=np.uint32)
        feat_idx = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
        pt_ids = np.array([0, 0, 1, 1, 1, 2], dtype=np.uint32)
        out = recon.clone_with_changes(
            positions=positions,
            track_image_indexes=img_idx,
            track_feature_indexes=feat_idx,
            track_point_indexes=pt_ids,
        )
        assert out.point_count == 3
        counts = np.asarray(out.observation_counts)
        assert counts.shape[0] == 3
        assert counts[0] == 2
        assert counts[1] == 3
        assert counts[2] == 1

        path = tmp_path / "regrouped.sfmr"
        out.save(path)
        reloaded = SfmrReconstruction.load(path)
        np.testing.assert_array_equal(np.asarray(reloaded.observation_counts), counts)
        np.testing.assert_array_equal(np.asarray(reloaded.track_point_indexes), pt_ids)
        np.testing.assert_array_equal(np.asarray(reloaded.track_image_indexes), img_idx)
