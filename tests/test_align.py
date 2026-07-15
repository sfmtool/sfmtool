# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sfm align CLI command and multi-alignment logic."""

import numpy as np
from click.testing import CliRunner

from sfmtool import RangeExpr
from sfmtool.align.multi import (
    _build_connectivity_graph,
    _find_shared_images,
    _get_reconstruction_images,
    align_reconstructions,
)
from sfmtool._sfmtool import SfmrReconstruction
from sfmtool._sfmtool.geometry import Se3Transform
from sfmtool.cli import main
from sfmtool.xform import IncludeRangeFilter, SimilarityTransform, apply_transforms


def _apply_transforms_to_file(input_path, output_path, transforms):
    """Helper that wraps apply_transforms with file I/O."""
    recon = SfmrReconstruction.load(input_path)
    recon = apply_transforms(recon, transforms)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recon.save(output_path, operation="xform_test")
    return output_path


# ---------------------------------------------------------------------------
# estimate_alignment binding (Rust least-squares fit + trimming, via PyO3)
# ---------------------------------------------------------------------------


class TestEstimateAlignment:
    """Exercise the estimate_alignment wrapper and its PyO3 params.

    The default (single-shot similarity) path is covered indirectly by the
    align CLI tests; these drive the trim/rigid keyword arguments through the
    binding to pin the ``core.py`` <-> ``core.rs`` signature.
    """

    @staticmethod
    def _make_similarity(n, scale, translation, seed):
        # 90-degree rotation about Z, applied as target = scale * R @ src + t.
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rng = np.random.default_rng(seed)
        source = rng.uniform(-5.0, 5.0, size=(n, 3))
        target = scale * (source @ rot.T) + np.asarray(translation)
        return source, target, rot

    def test_default_is_single_shot_similarity(self):
        from sfmtool.align.core import estimate_alignment

        source, target, _ = self._make_similarity(12, 1.7, [2.0, -3.0, 1.0], seed=1)
        t = estimate_alignment(source, target)
        assert np.isclose(t.scale, 1.7, atol=1e-9)
        np.testing.assert_allclose(t.apply_to_points(source), target, atol=1e-8)

    def test_trimming_rejects_outliers(self):
        from sfmtool.align.core import estimate_alignment

        source, target, _ = self._make_similarity(30, 1.7, [2.0, -3.0, 1.0], seed=2)
        clean = target.copy()
        # Corrupt ~20% of correspondences with large target offsets.
        target[::5] += np.array([40.0, -35.0, 30.0])

        # A plain fit is dragged off by the outliers; trimming recovers.
        plain = estimate_alignment(source, target)
        trimmed = estimate_alignment(source, target, rounds=5, keep_fraction=0.7)
        assert not np.isclose(plain.scale, 1.7, atol=1e-3)
        assert np.isclose(trimmed.scale, 1.7, atol=1e-6)
        # Inlier rows map back onto their clean targets under the trimmed fit.
        recovered = trimmed.apply_to_points(source)
        inliers = np.ones(len(source), dtype=bool)
        inliers[::5] = False
        np.testing.assert_allclose(recovered[inliers], clean[inliers], atol=1e-5)

    def test_rigid_fit_keeps_scale_one(self):
        from sfmtool.align.core import estimate_alignment

        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        target = source * 3.0  # a scaled target...
        rigid = estimate_alignment(source, target, estimate_scale=False)
        assert np.isclose(rigid.scale, 1.0, atol=1e-12)  # ...but rigid pins scale=1


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_get_reconstruction_images(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        images = _get_reconstruction_images(recon)
        assert len(images) == 17
        assert all(isinstance(v, str) for v in images.values())

    def test_find_shared_images(self):
        a = {0: "img1.jpg", 1: "img2.jpg", 2: "img3.jpg"}
        b = {0: "img2.jpg", 1: "img4.jpg"}
        shared = _find_shared_images(a, b)
        assert shared == {"img2.jpg"}

    def test_find_shared_images_none(self):
        a = {0: "img1.jpg"}
        b = {0: "img2.jpg"}
        shared = _find_shared_images(a, b)
        assert shared == set()

    def test_build_connectivity_graph(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        graph = _build_connectivity_graph([recon, recon])
        assert 1 in graph[0]
        assert 0 in graph[1]
        assert len(graph[0][1]) == 17


# ---------------------------------------------------------------------------
# Alignment core tests
# ---------------------------------------------------------------------------


class TestAlignReconstructionsPoints:
    """Test align_reconstructions with point-based method."""

    def test_align_transformed_recovery(self, seoul_bull_sfmr_only):
        """Alignment should recover the inverse of an applied transform."""
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        transform = Se3Transform(translation=[5, 0, 0], scale=2.0)
        transformed = apply_transforms(recon, [SimilarityTransform(transform)])

        result = align_reconstructions(
            reference=recon,
            to_align=[transformed],
            method="points",
        )

        assert result.aligned[0] is not None
        assert result.total_shared_images == 17

    def test_align_identical(self, seoul_bull_sfmr_only):
        """Aligning identical reconstructions should succeed with near-zero error."""
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        result = align_reconstructions(
            reference=recon,
            to_align=[recon],
            method="points",
        )
        assert result.aligned[0] is not None

    def test_align_no_shared_images(self, seoul_bull_sfmr_only):
        """Aligning reconstructions with no shared images should fail gracefully."""
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        # Filter to disjoint image sets
        subset1 = apply_transforms(recon, [IncludeRangeFilter(RangeExpr("1-5"))])
        subset2 = apply_transforms(recon, [IncludeRangeFilter(RangeExpr("10-17"))])

        result = align_reconstructions(
            reference=subset1,
            to_align=[subset2],
            method="points",
        )
        assert result.aligned[0] is None


class TestAlignPointsAtInfinity:
    """Point alignment must ignore points at infinity (directions, not metric
    locations) so they cannot corrupt the similarity fit."""

    def test_correspondences_exclude_points_at_infinity(self, seoul_bull_sfmr_only):
        """find_point_correspondences drops pairs where either point is w=0."""
        from sfmtool._point_correspondence import find_point_correspondences

        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        shared = [(i, i) for i in range(recon.image_count)]

        corr_all, _, _ = find_point_correspondences(recon, recon, shared)

        # Mark the first five points as at infinity.
        xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
        xyzw[:5, 3] = 0.0
        inf_recon = recon.clone_with_changes(positions=xyzw)
        inf_ids = set(
            np.flatnonzero(np.asarray(inf_recon.point_is_at_infinity)).tolist()
        )
        assert len(inf_ids) >= 5

        corr_inf, src_pos, tgt_pos = find_point_correspondences(
            inf_recon, inf_recon, shared
        )

        # No infinity point survives as a correspondence (either side).
        assert inf_ids.isdisjoint(corr_inf.keys())
        assert inf_ids.isdisjoint(corr_inf.values())
        # Returned positions are all finite.
        assert np.all(np.isfinite(src_pos))
        assert np.all(np.isfinite(tgt_pos))
        # Exactly the infinity points were removed vs. the all-finite run.
        assert len(corr_inf) == len(corr_all) - len(inf_ids & set(corr_all.keys()))

    def test_align_succeeds_with_shared_infinity_points(self, seoul_bull_sfmr_only):
        """Point alignment runs cleanly when shared w=0 points are present."""
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
        xyzw[:20, 3] = 0.0
        inf_recon = recon.clone_with_changes(positions=xyzw)

        transform = Se3Transform(translation=[5, 0, 0], scale=2.0)
        transformed = apply_transforms(inf_recon, [SimilarityTransform(transform)])

        result = align_reconstructions(
            reference=inf_recon, to_align=[transformed], method="points"
        )
        assert result.aligned[0] is not None
        assert result.total_shared_images == 17


class TestAlignReconstructionsCameras:
    """Test align_reconstructions with camera-based method."""

    def test_align_transformed_recovery(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        transform = Se3Transform(translation=[5, 0, 0], scale=2.0)
        transformed = apply_transforms(recon, [SimilarityTransform(transform)])

        result = align_reconstructions(
            reference=recon,
            to_align=[transformed],
            method="cameras",
        )
        assert result.aligned[0] is not None
        assert result.total_shared_images == 17


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestAlignCLI:
    def test_align_points_default(self, seoul_bull_sfmr_only, tmp_path):
        """Default align (points) succeeds and produces output."""
        ref = seoul_bull_sfmr_only
        workspace = ref.parent

        transform = Se3Transform(translation=[3, 0, 0], scale=1.5)
        target_path = _apply_transforms_to_file(
            ref,
            workspace / "target_for_align.sfmr",
            [SimilarityTransform(transform)],
        )
        output_dir = tmp_path / "aligned"

        result = CliRunner().invoke(
            main,
            [
                "align",
                str(ref),
                str(target_path),
                "-o",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Successfully aligned: 1/1" in result.output
        assert "Results saved to:" in result.output
        assert (output_dir / ref.name).exists()
        assert (output_dir / target_path.name).exists()

    def test_align_cameras_method(self, seoul_bull_sfmr_only, tmp_path):
        ref = seoul_bull_sfmr_only
        workspace = ref.parent

        transform = Se3Transform(translation=[2, 1, 0], scale=1.2)
        target_path = _apply_transforms_to_file(
            ref,
            workspace / "target_cam.sfmr",
            [SimilarityTransform(transform)],
        )
        output_dir = tmp_path / "aligned_cam"

        result = CliRunner().invoke(
            main,
            [
                "align",
                str(ref),
                str(target_path),
                "-o",
                str(output_dir),
                "--method",
                "cameras",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Using camera-based alignment" in result.output
        assert "Successfully aligned: 1/1" in result.output

    def test_align_non_sfmr_reference(self, tmp_path):
        p = tmp_path / "ref.txt"
        p.touch()
        target = tmp_path / "target.sfmr"
        target.touch()
        result = CliRunner().invoke(
            main, ["align", str(p), str(target), "-o", str(tmp_path / "out")]
        )
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_align_non_sfmr_target(self, tmp_path):
        ref = tmp_path / "ref.sfmr"
        ref.touch()
        target = tmp_path / "target.txt"
        target.touch()
        result = CliRunner().invoke(
            main, ["align", str(ref), str(target), "-o", str(tmp_path / "out")]
        )
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_align_ransac_options_with_cameras_rejected(self, tmp_path):
        ref = tmp_path / "ref.sfmr"
        ref.touch()
        target = tmp_path / "target.sfmr"
        target.touch()
        result = CliRunner().invoke(
            main,
            [
                "align",
                str(ref),
                str(target),
                "-o",
                str(tmp_path / "out"),
                "--method",
                "cameras",
                "--no-ransac",
            ],
        )
        assert result.exit_code != 0
        assert "RANSAC" in result.output

    def test_align_basename_collision_rejected(self, tmp_path):
        """Inputs sharing a basename are rejected rather than silently overwritten."""
        ref = tmp_path / "a" / "recon.sfmr"
        other = tmp_path / "b" / "recon.sfmr"
        ref.parent.mkdir()
        other.parent.mkdir()
        ref.touch()
        other.touch()
        result = CliRunner().invoke(
            main, ["align", str(ref), str(other), "-o", str(tmp_path / "out")]
        )
        assert result.exit_code != 0
        assert "recon.sfmr" in result.output
        assert "overwrite each other" in result.output
        # Nothing should have been written to the output directory.
        assert not (tmp_path / "out").exists()

    def test_align_same_file_twice_rejected(self, tmp_path):
        """Passing the same file as reference and target is rejected as a collision."""
        ref = tmp_path / "recon.sfmr"
        ref.touch()
        result = CliRunner().invoke(
            main, ["align", str(ref), str(ref), "-o", str(tmp_path / "out")]
        )
        assert result.exit_code != 0
        assert "overwrite each other" in result.output

    def test_aligned_positions_close_to_reference(self, seoul_bull_sfmr_only, tmp_path):
        """After alignment, camera positions should be close to the reference."""
        ref = seoul_bull_sfmr_only
        workspace = ref.parent

        transform = Se3Transform(translation=[5, 0, 0], scale=2.0)
        target_path = _apply_transforms_to_file(
            ref,
            workspace / "target_verify.sfmr",
            [SimilarityTransform(transform)],
        )
        output_dir = tmp_path / "aligned_verify"

        result = CliRunner().invoke(
            main,
            ["align", str(ref), str(target_path), "-o", str(output_dir)],
        )
        assert result.exit_code == 0, result.output

        # Load reference and aligned, compare positions
        ref_recon = SfmrReconstruction.load(output_dir / ref.name)
        aligned_recon = SfmrReconstruction.load(output_dir / target_path.name)

        # Positions should be very close after alignment
        ref_positions = ref_recon.positions
        aligned_positions = aligned_recon.positions
        assert ref_positions.shape == aligned_positions.shape

        distances = np.linalg.norm(aligned_positions - ref_positions, axis=1)
        assert np.mean(distances) < 0.01
