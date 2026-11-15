# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sfm merge CLI command and merge logic."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.reconstruction import RangeExpr
from sfmtool.merge.reconstructions import merge_reconstructions
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.cli import main
from sfmtool.xform import IncludeRangeFilter, apply_transforms

from .conftest import per_point_reprojection_error_stats


def _apply_transforms_to_file(input_path, output_path, transforms):
    """Helper that wraps apply_transforms with file I/O."""
    recon = SfmrReconstruction.load(input_path)
    recon = apply_transforms(recon, transforms)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recon.save(output_path, operation="xform_test")
    return output_path


def _merge_reconstructions_to_file(input_paths, output_path, **kwargs):
    """Helper that wraps merge_reconstructions with file I/O."""
    reconstructions = [SfmrReconstruction.load(p) for p in input_paths]
    merged = merge_reconstructions(reconstructions=reconstructions, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(output_path, operation="merge_test")
    return output_path


# ---------------------------------------------------------------------------
# Unit tests for merge validation
# ---------------------------------------------------------------------------


class TestMergeValidation:
    def test_merge_requires_at_least_two_reconstructions_empty(self):
        with pytest.raises(ValueError, match="Need at least 2 reconstructions"):
            merge_reconstructions(reconstructions=[])

    def test_merge_requires_at_least_two_reconstructions_single(
        self, seoul_bull_workspace
    ):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        with pytest.raises(ValueError, match="Need at least 2 reconstructions"):
            merge_reconstructions(reconstructions=[recon])

    def test_merge_refuses_reconstructions_from_different_feature_folders(self):
        class RecordsFolder:
            """Stands in for a reconstruction: the check reads only this."""

            feature_source = "sift_files"

            def __init__(self, folder):
                self.source_metadata = {
                    "workspace": {"contents": {"feature_prefix_dir": folder}}
                }

        old, new = "features/sift-old", "features/sift-new"
        with pytest.raises(ValueError) as excinfo:
            merge_reconstructions(
                reconstructions=[RecordsFolder(old), RecordsFolder(new)]
            )
        message = str(excinfo.value)
        assert "made from different feature folders" in message
        assert f"input 1: {old}" in message
        assert f"input 2: {new}" in message


# ---------------------------------------------------------------------------
# Merge core tests
# ---------------------------------------------------------------------------


class TestMergeSubsets:
    def test_merge_two_overlapping_subsets(self, seoul_bull_workspace, tmp_path):
        """Split into 1-10 and 6-17, merge, verify image count and names match."""
        original_path = seoul_bull_workspace
        original = SfmrReconstruction.load(original_path)

        subset_a_path = tmp_path / "subset_a.sfmr"
        subset_b_path = tmp_path / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(RangeExpr("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(RangeExpr("6-17"))],
        )

        subset_a = SfmrReconstruction.load(subset_a_path)
        subset_b = SfmrReconstruction.load(subset_b_path)
        assert subset_a.image_count == 10
        assert subset_b.image_count == 12

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=[subset_a_path, subset_b_path],
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(merged_path)

        assert merged.image_count == original.image_count
        original_names = {Path(n).name for n in original.image_names}
        merged_names = {Path(n).name for n in merged.image_names}
        assert original_names == merged_names

    def test_subsets_and_merge_store_the_rms_of_their_tracks(
        self, seoul_bull_workspace, tmp_path
    ):
        """An image subset and a merge change tracks, so each scores its points
        again: the stored errors are the RMS over the tracks written."""
        subset_paths = []
        for name, images in [("a", "1-10"), ("b", "6-17")]:
            path = tmp_path / f"subset_{name}.sfmr"
            _apply_transforms_to_file(
                seoul_bull_workspace, path, [IncludeRangeFilter(RangeExpr(images))]
            )
            subset_paths.append(path)
        for path in subset_paths:
            subset = SfmrReconstruction.load(path)
            _, rms = per_point_reprojection_error_stats(subset)
            np.testing.assert_allclose(
                subset.rms_reprojection_errors, rms, rtol=1e-5, atol=1e-5
            )

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=subset_paths, output_path=merged_path
        )
        merged = SfmrReconstruction.load(merged_path)
        _, rms = per_point_reprojection_error_stats(merged)
        np.testing.assert_allclose(
            merged.rms_reprojection_errors, rms, rtol=1e-5, atol=1e-5
        )

    def test_merge_three_overlapping_subsets(self, seoul_bull_workspace, tmp_path):
        """Merge three overlapping subsets (1-7, 5-12, 10-17)."""
        original_path = seoul_bull_workspace
        original = SfmrReconstruction.load(original_path)

        subset_paths = []
        ranges = ["1-7", "5-12", "10-17"]

        for i, range_str in enumerate(ranges):
            subset_path = tmp_path / f"subset_{i}.sfmr"
            range_expr = RangeExpr(range_str)
            _apply_transforms_to_file(
                original_path, subset_path, [IncludeRangeFilter(range_expr)]
            )
            subset_paths.append(subset_path)

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=subset_paths,
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(merged_path)

        assert merged.image_count == original.image_count
        original_names = {Path(n).name for n in original.image_names}
        merged_names = {Path(n).name for n in merged.image_names}
        assert original_names == merged_names


class TestMergeQuality:
    def test_merge_preserves_camera_intrinsics(self, seoul_bull_workspace, tmp_path):
        """Camera parameters should be preserved through merge."""
        original_path = seoul_bull_workspace
        original = SfmrReconstruction.load(original_path)

        subset_a_path = tmp_path / "subset_a.sfmr"
        subset_b_path = tmp_path / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(RangeExpr("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(RangeExpr("6-17"))],
        )

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=[subset_a_path, subset_b_path],
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(merged_path)

        assert len(merged.cameras) == len(original.cameras)
        for orig_cam, merged_cam in zip(original.cameras, merged.cameras):
            assert orig_cam.model == merged_cam.model
            assert orig_cam.width == merged_cam.width
            assert orig_cam.height == merged_cam.height
            for key in orig_cam.parameters:
                assert np.isclose(
                    orig_cam.parameters[key], merged_cam.parameters[key], rtol=1e-5
                )

    def test_merge_preserves_point_count(self, seoul_bull_workspace, tmp_path):
        """Merging subsets that together cover all images recovers the original
        points: none are lost, and only a handful of boundary tracks (whose
        observations straddle the subset gap without sharing an image) split,
        so the merged count is the original plus a small surplus."""
        original_path = seoul_bull_workspace
        original = SfmrReconstruction.load(original_path)

        subset_a_path = tmp_path / "subset_a.sfmr"
        subset_b_path = tmp_path / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(RangeExpr("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(RangeExpr("6-17"))],
        )

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=[subset_a_path, subset_b_path],
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(merged_path)
        assert merged.point_count >= original.point_count
        assert merged.point_count <= int(original.point_count * 1.05)

    def test_merge_preserves_point_positions(self, seoul_bull_workspace, tmp_path):
        """After merge, point positions should match original via feature mapping."""
        original_path = seoul_bull_workspace
        original = SfmrReconstruction.load(original_path)

        subset_a_path = tmp_path / "subset_a.sfmr"
        subset_b_path = tmp_path / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(RangeExpr("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(RangeExpr("6-17"))],
        )

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=[subset_a_path, subset_b_path],
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(merged_path)

        def build_feature_point_map(recon):
            result = {}
            image_names = recon.image_names
            for img_idx, feat_idx, pt_id in zip(
                recon.track_image_indexes,
                recon.track_feature_indexes,
                recon.track_point_indexes,
            ):
                img_name = Path(image_names[img_idx]).name
                result[(img_name, int(feat_idx))] = int(pt_id)
            return result

        original_feat_map = build_feature_point_map(original)
        merged_feat_map = build_feature_point_map(merged)

        original_to_merged = {}
        for key, orig_pt_id in original_feat_map.items():
            if key in merged_feat_map:
                merged_pt_id = merged_feat_map[key]
                if orig_pt_id not in original_to_merged:
                    original_to_merged[orig_pt_id] = merged_pt_id

        assert len(original_to_merged) == original.point_count

        original_positions = original.positions
        merged_positions = merged.positions
        for orig_pt_id, merged_pt_id in original_to_merged.items():
            orig_pos = original_positions[orig_pt_id]
            merged_pos = merged_positions[merged_pt_id]
            assert np.allclose(orig_pos, merged_pos, atol=1e-5)

    def test_merge_camera_poses_close(self, seoul_bull_workspace, tmp_path):
        """After merge, at least 85% of camera poses should be close to original."""
        original_path = seoul_bull_workspace
        original = SfmrReconstruction.load(original_path)

        subset_a_path = tmp_path / "subset_a.sfmr"
        subset_b_path = tmp_path / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(RangeExpr("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(RangeExpr("6-17"))],
        )

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=[subset_a_path, subset_b_path],
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(merged_path)

        original_name_to_idx = {
            Path(n).name: i for i, n in enumerate(original.image_names)
        }
        merged_name_to_idx = {Path(n).name: i for i, n in enumerate(merged.image_names)}

        close_count = 0
        original_names = {Path(n).name for n in original.image_names}
        for name in original_names:
            orig_idx = original_name_to_idx[name]
            merged_idx = merged_name_to_idx[name]

            orig_quat = original.quaternions_wxyz[orig_idx]
            merged_quat = merged.quaternions_wxyz[merged_idx]
            quat_close = np.allclose(orig_quat, merged_quat, atol=0.05) or np.allclose(
                orig_quat, -merged_quat, atol=0.05
            )

            orig_trans = original.translations[orig_idx]
            merged_trans = merged.translations[merged_idx]
            trans_close = np.allclose(orig_trans, merged_trans, atol=0.1)

            if quat_close and trans_close:
                close_count += 1

        close_ratio = close_count / len(original_names)
        assert close_ratio >= 0.85, (
            f"Only {close_count}/{len(original_names)} ({close_ratio:.0%}) images have "
            f"close poses after merge (expected >= 85%)"
        )

    def test_merge_no_duplicate_points(self, seoul_bull_workspace, tmp_path):
        """Each (image, feature) pair should map to exactly one point."""
        original_path = seoul_bull_workspace
        original = SfmrReconstruction.load(original_path)

        subset_a_path = tmp_path / "subset_a.sfmr"
        subset_b_path = tmp_path / "subset_b.sfmr"

        # Large overlap: 1-12 and 6-17
        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(RangeExpr("1-12"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(RangeExpr("6-17"))],
        )

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=[subset_a_path, subset_b_path],
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(merged_path)

        seen_observations = set()
        for img_idx, feat_idx in zip(
            merged.track_image_indexes, merged.track_feature_indexes
        ):
            obs_key = (int(img_idx), int(feat_idx))
            assert obs_key not in seen_observations, (
                f"Duplicate observation: image {img_idx}, feature {feat_idx}"
            )
            seen_observations.add(obs_key)

        # No points are lost; a few boundary tracks split across the subset gap.
        assert merged.point_count >= original.point_count
        assert merged.point_count <= int(original.point_count * 1.05)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestMergeCLI:
    def test_merge_cli_basic(self, seoul_bull_workspace, tmp_path):
        """Basic CLI merge succeeds and produces output."""
        original_path = seoul_bull_workspace
        workspace = original_path.parent

        subset_a_path = workspace / "subset_a.sfmr"
        subset_b_path = workspace / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(RangeExpr("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(RangeExpr("6-17"))],
        )

        output_path = tmp_path / "merged.sfmr"

        result = CliRunner().invoke(
            main,
            [
                "merge",
                str(subset_a_path),
                str(subset_b_path),
                "-o",
                str(output_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Merge complete!" in result.output
        assert "Saved to:" in result.output
        assert output_path.exists()

    def test_merge_cli_non_sfmr_rejected(self, tmp_path):
        """Non-.sfmr files should be rejected."""
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.sfmr"
        p1.touch()
        p2.touch()
        result = CliRunner().invoke(
            main, ["merge", str(p1), str(p2), "-o", str(tmp_path / "out.sfmr")]
        )
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_merge_cli_non_sfmr_output_rejected(self, tmp_path):
        """Non-.sfmr output path should be rejected."""
        p1 = tmp_path / "a.sfmr"
        p2 = tmp_path / "b.sfmr"
        p1.touch()
        p2.touch()
        result = CliRunner().invoke(
            main, ["merge", str(p1), str(p2), "-o", str(tmp_path / "out.txt")]
        )
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_merge_cli_single_file_rejected(self, tmp_path):
        """Single file should be rejected."""
        p = tmp_path / "a.sfmr"
        p.touch()
        result = CliRunner().invoke(
            main, ["merge", str(p), "-o", str(tmp_path / "out.sfmr")]
        )
        assert result.exit_code != 0
        assert "2 reconstructions" in result.output

    def test_merge_cli_with_rig_frame_data(self, seoul_bull_workspace, tmp_path):
        """Merge should succeed when reconstructions have rig_frame_data."""
        original_path = seoul_bull_workspace
        recon = SfmrReconstruction.load(original_path)

        recon_with_rig = _with_rigs(recon, [2], _pairs(recon.image_count))
        rig_path = original_path.parent / "with_rig.sfmr"
        recon_with_rig.save(rig_path, operation="add_rig")

        # Create two overlapping subsets
        workspace = original_path.parent
        subset_a_path = workspace / "rig_subset_a.sfmr"
        subset_b_path = workspace / "rig_subset_b.sfmr"

        _apply_transforms_to_file(
            rig_path,
            subset_a_path,
            [IncludeRangeFilter(RangeExpr("1-10"))],
        )
        _apply_transforms_to_file(
            rig_path,
            subset_b_path,
            [IncludeRangeFilter(RangeExpr("6-17"))],
        )

        # Merge should succeed (save without shape mismatch error)
        output_path = tmp_path / "merged_rig.sfmr"
        result = CliRunner().invoke(
            main,
            [
                "merge",
                str(subset_a_path),
                str(subset_b_path),
                "-o",
                str(output_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output_path.exists()

        merged = SfmrReconstruction.load(output_path)
        assert merged.image_count == recon.image_count

    def test_merge_cli_with_percentile(self, seoul_bull_workspace, tmp_path):
        """Merge with custom percentile option."""
        original_path = seoul_bull_workspace
        workspace = original_path.parent

        subset_a_path = workspace / "subset_a.sfmr"
        subset_b_path = workspace / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(RangeExpr("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(RangeExpr("6-17"))],
        )

        output_path = tmp_path / "merged.sfmr"

        result = CliRunner().invoke(
            main,
            [
                "merge",
                str(subset_a_path),
                str(subset_b_path),
                "-o",
                str(output_path),
                "--merge-percentile",
                "99",
            ],
        )
        assert result.exit_code == 0, result.output
        assert output_path.exists()


# ---------------------------------------------------------------------------
# Points at infinity
# ---------------------------------------------------------------------------


def _subsets_with_points_at_infinity(original_path, tmp_path):
    """Split the reconstruction into images 1-10 and 6-17 after moving some
    points seen in both halves to infinity.

    Returns ``(original, subset_a, subset_b, infinite_ids)``, where
    ``original`` is the reconstruction with the points at infinity and
    ``infinite_ids`` are their point ids in it.
    """
    original = SfmrReconstruction.load(original_path)
    image_indexes = np.asarray(original.track_image_indexes)
    point_indexes = np.asarray(original.track_point_indexes)
    # Points observed in at least two of images 6-10 (indexes 5-9) appear in
    # both subsets with tracks long enough to survive the split.
    in_overlap = (image_indexes >= 5) & (image_indexes <= 9)
    overlap_counts = np.bincount(
        point_indexes[in_overlap], minlength=original.point_count
    )
    infinite_ids = np.flatnonzero(overlap_counts >= 2)[:10]
    assert len(infinite_ids) == 10

    xyzw = np.array(original.positions_xyzw, dtype=np.float64)
    directions = xyzw[infinite_ids, :3]
    xyzw[infinite_ids, :3] = directions / np.linalg.norm(
        directions, axis=1, keepdims=True
    )
    xyzw[infinite_ids, 3] = 0.0
    original = original.clone_with_changes(positions=xyzw)
    assert original.point_is_at_infinity.sum() == len(infinite_ids)

    original_path = tmp_path / "with_infinity.sfmr"
    original.save(original_path, operation="infinity_test")
    subsets = []
    for name, range_expr in (("subset_a", "1-10"), ("subset_b", "6-17")):
        path = tmp_path / f"{name}.sfmr"
        _apply_transforms_to_file(
            original_path, path, [IncludeRangeFilter(RangeExpr(range_expr))]
        )
        subset = SfmrReconstruction.load(path)
        assert subset.point_is_at_infinity.any()
        subsets.append(subset)
    return original, subsets[0], subsets[1], infinite_ids


def _merged_point_ids(original, merged, point_ids):
    """Map each original point id to the merged point that holds its
    observations, keyed on (image name, feature index)."""
    merged_names = merged.image_names
    original_names = original.image_names
    merged_by_obs = {
        (Path(merged_names[i]).name, int(f)): int(p)
        for i, f, p in zip(
            merged.track_image_indexes,
            merged.track_feature_indexes,
            merged.track_point_indexes,
        )
    }
    result = {}
    for i, f, p in zip(
        original.track_image_indexes,
        original.track_feature_indexes,
        original.track_point_indexes,
    ):
        if int(p) in point_ids and int(p) not in result:
            result[int(p)] = merged_by_obs[(Path(original_names[i]).name, int(f))]
    return result


class TestMergePointsAtInfinity:
    def test_correspondence_groups_exclude_points_at_infinity(
        self, seoul_bull_workspace, tmp_path
    ):
        """The pairwise step drops pairs with a point at infinity, so no
        correspondence group holds one."""
        from sfmtool.merge.correspondences import find_point_correspondences

        _, subset_a, subset_b, _ = _subsets_with_points_at_infinity(
            seoul_bull_workspace, tmp_path
        )
        recons = [subset_a, subset_b]
        image_mapping = {}
        for recon_idx, recon in enumerate(recons):
            for img_idx, name in enumerate(recon.image_names):
                image_mapping.setdefault(Path(name).as_posix(), []).append(
                    (recon_idx, img_idx)
                )

        groups = find_point_correspondences(recons, image_mapping, 100.0)

        assert len(groups) > 0
        for group in groups:
            for recon_idx, point_id in group:
                assert not recons[recon_idx].point_is_at_infinity[point_id]

    def test_merge_keeps_points_at_infinity(self, seoul_bull_workspace, tmp_path):
        """Both halves hold the same points at infinity, so the merge keeps each
        at infinity along its direction and makes no other point one."""
        original, subset_a, subset_b, infinite_ids = _subsets_with_points_at_infinity(
            seoul_bull_workspace, tmp_path
        )

        merged = merge_reconstructions(reconstructions=[subset_a, subset_b])

        mapping = _merged_point_ids(original, merged, set(infinite_ids.tolist()))
        assert len(mapping) == len(infinite_ids)
        assert merged.point_is_at_infinity.sum() == len(infinite_ids)
        for orig_id, merged_id in mapping.items():
            assert merged.point_is_at_infinity[merged_id]
            assert np.allclose(
                merged.positions[merged_id], original.positions[orig_id], atol=1e-6
            )


# ---------------------------------------------------------------------------
# Inputs the merge refuses, before any step
# ---------------------------------------------------------------------------


def _embedded_copy(sift_path):
    """Save an ``embedded_patches`` conversion of ``sift_path`` beside it."""
    from sfmtool.xform import ToEmbeddedPatchesTransform

    embedded = ToEmbeddedPatchesTransform().apply(SfmrReconstruction.load(sift_path))
    path = sift_path.parent / "embedded.sfmr"
    embedded.save(path, operation="test")
    return path


def _pairs(first, stop=None):
    """Frames of rig 0 holding images ``first, first + 1``, ``first + 2,
    first + 3`` and so on below ``stop``, the last one alone when the count
    is odd; ``_pairs(n)`` covers images ``0..n``."""
    if stop is None:
        first, stop = 0, first
    return [(0, list(range(i, min(i + 2, stop)))) for i in range(first, stop, 2)]


def _with_rigs(recon, sensor_counts, frames, sensor_cameras=None):
    """``recon`` with rigs of ``sensor_counts`` sensors and the ``frames``,
    each ``(rig, images)`` with the images in sensor order. The sensors sit on
    camera 0 unless ``sensor_cameras`` names theirs."""
    offsets = np.concatenate([[0], np.cumsum(sensor_counts)[:-1]]).astype(int).tolist()
    sensor_total = int(sum(sensor_counts))
    image_sensors = np.zeros(recon.image_count, dtype=np.uint32)
    image_frames = np.zeros(recon.image_count, dtype=np.uint32)
    for frame, (rig, images) in enumerate(frames):
        for position, image in enumerate(images):
            image_sensors[image] = offsets[rig] + position
            image_frames[image] = frame
    translations = np.zeros((sensor_total, 3))
    for rig, count in enumerate(sensor_counts):
        translations[offsets[rig] : offsets[rig] + count, 0] = 0.1 * np.arange(count)
    return recon.clone_with_changes(
        rig_frame_data={
            "rigs_metadata": {
                "rig_count": len(sensor_counts),
                "sensor_count": sensor_total,
                "rigs": [
                    {
                        "name": f"rig{rig}",
                        "sensor_count": count,
                        "sensor_offset": offsets[rig],
                        "ref_sensor_name": "sensor0",
                        "sensor_names": [f"sensor{k}" for k in range(count)],
                    }
                    for rig, count in enumerate(sensor_counts)
                ],
            },
            "sensor_camera_indexes": np.array(
                sensor_cameras or [0] * sensor_total, dtype=np.uint32
            ),
            "sensor_quaternions_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (sensor_total, 1)),
            "sensor_translations_xyz": translations,
            "frames_metadata": {"frame_count": len(frames)},
            "rig_indexes": np.array([rig for rig, _ in frames], dtype=np.uint32),
            "image_sensor_indexes": image_sensors,
            "image_frame_indexes": image_frames,
        }
    )


def _rig_and_single_images(recon):
    """A solve of rig frames and single images: rig 0 pairs images 1-14,
    and images 15-17 are each a frame of the single-sensor rig 1."""
    return _with_rigs(recon, [2, 1], _pairs(14) + [(1, [i]) for i in (14, 15, 16)])


def _rig_frames_alone(recon):
    """A solve of the rig frames alone: images 1-14 in rig 0."""
    first_14 = apply_transforms(recon, [IncludeRangeFilter(RangeExpr("1-14"))])
    return _with_rigs(first_14, [2], _pairs(14))


def _assert_refused_before_any_step(result, output_path):
    assert result.exit_code == 1, result.output
    assert "Merging" not in result.output
    assert "[1/7]" not in result.output
    assert not output_path.exists()


class TestMergeRefusesInputs:
    def test_cli_refuses_embedded_patches_inputs(self, seoul_bull_workspace, tmp_path):
        embedded = _embedded_copy(seoul_bull_workspace)
        other = embedded.with_name("embedded_b.sfmr")
        other.write_bytes(embedded.read_bytes())
        output_path = tmp_path / "merged.sfmr"

        result = CliRunner().invoke(
            main, ["merge", str(embedded), str(other), "-o", str(output_path)]
        )

        _assert_refused_before_any_step(result, output_path)
        assert (
            "Input 1 (embedded.sfmr) and input 2 (embedded_b.sfmr) are "
            "embedded_patches reconstructions" in result.output
        )
        assert "every input must be a sift_files reconstruction" in result.output
        assert "sfm xform --to-embedded-patches" in result.output

    def test_cli_refuses_an_embedded_patches_input_beside_sift_files(
        self, seoul_bull_workspace, tmp_path
    ):
        embedded = _embedded_copy(seoul_bull_workspace)
        output_path = tmp_path / "merged.sfmr"

        result = CliRunner().invoke(
            main,
            ["merge", str(seoul_bull_workspace), str(embedded), "-o", str(output_path)],
        )

        _assert_refused_before_any_step(result, output_path)
        assert (
            "Input 2 (embedded.sfmr) is an embedded_patches reconstruction"
            in result.output
        )
        assert seoul_bull_workspace.name not in result.output.split("Error:")[1]

    def test_library_refuses_embedded_patches_with_the_cli_message(
        self, seoul_bull_workspace, capsys
    ):
        embedded = SfmrReconstruction.load(_embedded_copy(seoul_bull_workspace))
        sift = SfmrReconstruction.load(seoul_bull_workspace)
        capsys.readouterr()

        with pytest.raises(ValueError) as excinfo:
            merge_reconstructions([sift, embedded], input_names=["a.sfmr", "b.sfmr"])

        assert str(excinfo.value).startswith(
            "Input 2 (b.sfmr) is an embedded_patches reconstruction."
        )
        assert capsys.readouterr().out == ""

    def test_cli_refuses_rig_frames_in_some_inputs_only(
        self, seoul_bull_workspace, tmp_path
    ):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        rig_path = seoul_bull_workspace.parent / "with_rig.sfmr"
        _with_rigs(recon, [2], _pairs(recon.image_count)).save(
            rig_path, operation="add_rig"
        )
        subset_path = seoul_bull_workspace.parent / "rig_subset.sfmr"
        _apply_transforms_to_file(
            rig_path, subset_path, [IncludeRangeFilter(RangeExpr("1-10"))]
        )
        output_path = tmp_path / "merged.sfmr"

        result = CliRunner().invoke(
            main,
            [
                "merge",
                str(subset_path),
                str(seoul_bull_workspace),
                "-o",
                str(output_path),
            ],
        )

        _assert_refused_before_any_step(result, output_path)
        assert (
            f"Input 1 (rig_subset.sfmr) has rig frames and input 2 "
            f"({seoul_bull_workspace.name}) has none" in result.output
        )

    def test_merges_rig_frames_beside_single_images_with_rig_frames_alone(
        self, seoul_bull_workspace, tmp_path
    ):
        """The reference defines rig 0 and a single-sensor rig 1 for its single
        images; the other input defines rig 0 alone, the same rig."""
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        workspace = seoul_bull_workspace.parent
        both_path = workspace / "both.sfmr"
        _rig_and_single_images(recon).save(both_path, operation="add_rig")
        later_path = workspace / "later.sfmr"
        _apply_transforms_to_file(
            both_path, later_path, [IncludeRangeFilter(RangeExpr("6-17"))]
        )
        alone_path = workspace / "alone.sfmr"
        _rig_frames_alone(recon).save(alone_path, operation="add_rig")
        output_path = tmp_path / "merged.sfmr"

        result = CliRunner().invoke(
            main,
            ["merge", str(later_path), str(alone_path), "-o", str(output_path)],
        )

        assert result.exit_code == 0, result.output
        merged = SfmrReconstruction.load(output_path)
        assert merged.image_count == recon.image_count
        rfd = merged.rig_frame_data
        assert rfd["rigs_metadata"]["rig_count"] == 2
        sensors = np.asarray(rfd["image_sensor_indexes"])
        assert sorted(sensors.tolist()) == [0] * 7 + [1] * 7 + [2] * 3
        cameras = np.asarray(merged.camera_indexes)
        assert (np.asarray(rfd["sensor_camera_indexes"])[sensors] == cameras).all()

    def test_refuses_an_input_with_a_rig_the_reference_lacks(
        self, seoul_bull_workspace, capsys
    ):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        alone = _rig_frames_alone(recon)
        capsys.readouterr()

        with pytest.raises(ValueError) as excinfo:
            merge_reconstructions([alone, _rig_and_single_images(recon)])

        assert str(excinfo.value).startswith(
            "Input 2 has images in rig 1, and input 1, the first input with "
            "rigs, whose rigs the merge keeps, defines 1 rig."
        )
        assert capsys.readouterr().out == ""

    def test_refuses_an_input_whose_rig_has_other_sensors(
        self, seoul_bull_workspace, capsys
    ):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        pairs = _with_rigs(recon, [2], _pairs(recon.image_count))
        # Image 1 alone in a one-sensor rig 0, the rest in pairs in rig 1.
        other = _with_rigs(
            recon,
            [1, 2],
            [(0, [0])] + [(1, images) for _, images in _pairs(1, recon.image_count)],
        )

        with pytest.raises(ValueError) as excinfo:
            merge_reconstructions([pairs, other])

        assert str(excinfo.value).startswith(
            "Rig 0 of input 2 has 1 sensor from sensor index 0, with reference "
            "sensor 0, and rig 0 of input 1, the first input with rigs, whose "
            "rigs the merge keeps, has 2 sensors from sensor index 0"
        )
        assert capsys.readouterr().out == ""

    def test_refuses_a_sensor_on_a_camera_the_reference_sensor_is_not_on(
        self, seoul_bull_workspace, capsys
    ):
        """Input 2 puts its single-image sensor on a second camera, which no
        image shares with input 1, so it stays a camera of its own."""
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        reference = _with_rigs(
            apply_transforms(recon, [IncludeRangeFilter(RangeExpr("1-14"))]),
            [2, 1],
            _pairs(14),
        )
        two_cameras = recon.clone_with_changes(
            cameras=recon.cameras * 2,
            camera_indexes=np.array([0] * 14 + [1] * 3, dtype=np.uint32),
            per_image_adjustments=recon.per_image_adjustments * 2,
            rolling_shutter_scanlines=recon.rolling_shutter_scanlines * 2,
        )
        other = _with_rigs(
            two_cameras,
            [2, 1],
            _pairs(14) + [(1, [i]) for i in (14, 15, 16)],
            sensor_cameras=[0, 0, 1],
        )
        name = recon.image_names[14]
        capsys.readouterr()

        with pytest.raises(ValueError) as excinfo:
            merge_reconstructions([reference, other])

        assert str(excinfo.value).startswith(
            f"Image {name} of input 2 was taken by sensor 2 through a camera "
            "that merges into camera 1, and sensor 2 of input 1"
        )
        assert capsys.readouterr().out == ""

    def test_refuses_cameras_that_vary_different_parameters_before_any_step(
        self, seoul_bull_workspace, capsys
    ):
        plain = SfmrReconstruction.load(seoul_bull_workspace)
        flagged = plain.clone_with_changes(per_image_adjustments=[["focal"]])

        with pytest.raises(ValueError) as excinfo:
            merge_reconstructions([plain, flagged])

        assert "camera 0 of input 1, which varies nothing" in str(excinfo.value)
        assert "camera 0 of input 2, which varies focal" in str(excinfo.value)
        assert capsys.readouterr().out == ""
