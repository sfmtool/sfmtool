# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sfm merge CLI command and merge logic."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from openjd.model import IntRangeExpr

from sfmtool._merge import merge_reconstructions
from sfmtool._sfmtool import SfmrReconstruction
from sfmtool.cli import main
from sfmtool.xform import IncludeRangeFilter, apply_transforms


def _apply_transforms_to_file(input_path, output_path, transforms):
    """Helper that wraps apply_transforms with file I/O."""
    recon = SfmrReconstruction.load(str(input_path))
    recon = apply_transforms(recon, transforms)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recon.save(str(output_path), operation="xform_test")
    return output_path


def _merge_reconstructions_to_file(input_paths, output_path, **kwargs):
    """Helper that wraps merge_reconstructions with file I/O."""
    reconstructions = [SfmrReconstruction.load(str(p)) for p in input_paths]
    merged = merge_reconstructions(reconstructions=reconstructions, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(str(output_path), operation="merge_test")
    return output_path


# ---------------------------------------------------------------------------
# Unit tests for merge validation
# ---------------------------------------------------------------------------


class TestMergeValidation:
    def test_merge_requires_at_least_two_reconstructions_empty(self):
        with pytest.raises(ValueError, match="Need at least 2 reconstructions"):
            merge_reconstructions(reconstructions=[])

    def test_merge_requires_at_least_two_reconstructions_single(
        self, sfmrfile_reconstruction_with_17_images
    ):
        recon = SfmrReconstruction.load(str(sfmrfile_reconstruction_with_17_images))
        with pytest.raises(ValueError, match="Need at least 2 reconstructions"):
            merge_reconstructions(reconstructions=[recon])


# ---------------------------------------------------------------------------
# Merge core tests
# ---------------------------------------------------------------------------


class TestMergeSubsets:
    def test_merge_two_overlapping_subsets(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Split into 1-10 and 6-17, merge, verify image count and names match."""
        original_path = sfmrfile_reconstruction_with_17_images
        original = SfmrReconstruction.load(str(original_path))

        subset_a_path = tmp_path / "subset_a.sfmr"
        subset_b_path = tmp_path / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("6-17"))],
        )

        subset_a = SfmrReconstruction.load(str(subset_a_path))
        subset_b = SfmrReconstruction.load(str(subset_b_path))
        assert subset_a.image_count == 10
        assert subset_b.image_count == 12

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=[subset_a_path, subset_b_path],
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(str(merged_path))

        assert merged.image_count == original.image_count
        original_names = {Path(n).name for n in original.image_names}
        merged_names = {Path(n).name for n in merged.image_names}
        assert original_names == merged_names

    def test_merge_three_overlapping_subsets(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Merge three overlapping subsets (1-7, 5-12, 10-17)."""
        original_path = sfmrfile_reconstruction_with_17_images
        original = SfmrReconstruction.load(str(original_path))

        subset_paths = []
        ranges = ["1-7", "5-12", "10-17"]

        for i, range_str in enumerate(ranges):
            subset_path = tmp_path / f"subset_{i}.sfmr"
            range_expr = IntRangeExpr.from_str(range_str)
            _apply_transforms_to_file(
                original_path, subset_path, [IncludeRangeFilter(range_expr)]
            )
            subset_paths.append(subset_path)

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=subset_paths,
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(str(merged_path))

        assert merged.image_count == original.image_count
        original_names = {Path(n).name for n in original.image_names}
        merged_names = {Path(n).name for n in merged.image_names}
        assert original_names == merged_names


class TestMergeQuality:
    def test_merge_preserves_camera_intrinsics(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Camera parameters should be preserved through merge."""
        original_path = sfmrfile_reconstruction_with_17_images
        original = SfmrReconstruction.load(str(original_path))

        subset_a_path = tmp_path / "subset_a.sfmr"
        subset_b_path = tmp_path / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("6-17"))],
        )

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=[subset_a_path, subset_b_path],
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(str(merged_path))

        assert len(merged.cameras) == len(original.cameras)
        for orig_cam, merged_cam in zip(original.cameras, merged.cameras):
            assert orig_cam.model == merged_cam.model
            assert orig_cam.width == merged_cam.width
            assert orig_cam.height == merged_cam.height
            for key in orig_cam.parameters:
                assert np.isclose(
                    orig_cam.parameters[key], merged_cam.parameters[key], rtol=1e-5
                )

    def test_merge_preserves_point_count(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Merged point count should match original when subsets cover all images."""
        original_path = sfmrfile_reconstruction_with_17_images
        original = SfmrReconstruction.load(str(original_path))

        subset_a_path = tmp_path / "subset_a.sfmr"
        subset_b_path = tmp_path / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("6-17"))],
        )

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=[subset_a_path, subset_b_path],
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(str(merged_path))
        assert merged.point_count == original.point_count

    def test_merge_preserves_point_positions(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """After merge, point positions should match original via feature mapping."""
        original_path = sfmrfile_reconstruction_with_17_images
        original = SfmrReconstruction.load(str(original_path))

        subset_a_path = tmp_path / "subset_a.sfmr"
        subset_b_path = tmp_path / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("6-17"))],
        )

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=[subset_a_path, subset_b_path],
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(str(merged_path))

        def build_feature_point_map(recon):
            result = {}
            for img_idx, feat_idx, pt_id in zip(
                recon.track_image_indexes,
                recon.track_feature_indexes,
                recon.track_point_ids,
            ):
                img_name = Path(recon.image_names[img_idx]).name
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

        for orig_pt_id, merged_pt_id in original_to_merged.items():
            orig_pos = original.positions[orig_pt_id]
            merged_pos = merged.positions[merged_pt_id]
            assert np.allclose(orig_pos, merged_pos, atol=1e-5)

    def test_merge_camera_poses_close(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """After merge, at least 85% of camera poses should be close to original."""
        original_path = sfmrfile_reconstruction_with_17_images
        original = SfmrReconstruction.load(str(original_path))

        subset_a_path = tmp_path / "subset_a.sfmr"
        subset_b_path = tmp_path / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("6-17"))],
        )

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=[subset_a_path, subset_b_path],
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(str(merged_path))

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

    def test_merge_no_duplicate_points(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Each (image, feature) pair should map to exactly one point."""
        original_path = sfmrfile_reconstruction_with_17_images
        original = SfmrReconstruction.load(str(original_path))

        subset_a_path = tmp_path / "subset_a.sfmr"
        subset_b_path = tmp_path / "subset_b.sfmr"

        # Large overlap: 1-12 and 6-17
        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("1-12"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("6-17"))],
        )

        merged_path = tmp_path / "merged.sfmr"
        _merge_reconstructions_to_file(
            input_paths=[subset_a_path, subset_b_path],
            output_path=merged_path,
        )

        merged = SfmrReconstruction.load(str(merged_path))

        seen_observations = set()
        for img_idx, feat_idx in zip(
            merged.track_image_indexes, merged.track_feature_indexes
        ):
            obs_key = (int(img_idx), int(feat_idx))
            assert obs_key not in seen_observations, (
                f"Duplicate observation: image {img_idx}, feature {feat_idx}"
            )
            seen_observations.add(obs_key)

        assert merged.point_count == original.point_count


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestMergeCLI:
    def test_merge_cli_basic(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Basic CLI merge succeeds and produces output."""
        original_path = sfmrfile_reconstruction_with_17_images
        workspace = original_path.parent

        subset_a_path = workspace / "subset_a.sfmr"
        subset_b_path = workspace / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("6-17"))],
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

    def test_merge_cli_with_percentile(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Merge with custom percentile option."""
        original_path = sfmrfile_reconstruction_with_17_images
        workspace = original_path.parent

        subset_a_path = workspace / "subset_a.sfmr"
        subset_b_path = workspace / "subset_b.sfmr"

        _apply_transforms_to_file(
            original_path,
            subset_a_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("1-10"))],
        )
        _apply_transforms_to_file(
            original_path,
            subset_b_path,
            [IncludeRangeFilter(IntRangeExpr.from_str("6-17"))],
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
