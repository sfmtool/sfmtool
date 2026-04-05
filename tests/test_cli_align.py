# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sfm align CLI command and multi-alignment logic."""

import numpy as np
from click.testing import CliRunner
from openjd.model import IntRangeExpr

from sfmtool._multi_align import (
    _build_connectivity_graph,
    _find_shared_images,
    _get_reconstruction_images,
    align_reconstructions,
)
from sfmtool._sfmtool import Se3Transform, SfmrReconstruction
from sfmtool.cli import main
from sfmtool.xform import IncludeRangeFilter, SimilarityTransform, apply_transforms


def _apply_transforms_to_file(input_path, output_path, transforms):
    """Helper that wraps apply_transforms with file I/O."""
    recon = SfmrReconstruction.load(str(input_path))
    recon = apply_transforms(recon, transforms)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recon.save(str(output_path), operation="xform_test")
    return output_path


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_get_reconstruction_images(self, sfmrfile_reconstruction_with_17_images):
        recon = SfmrReconstruction.load(str(sfmrfile_reconstruction_with_17_images))
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

    def test_build_connectivity_graph(self, sfmrfile_reconstruction_with_17_images):
        recon = SfmrReconstruction.load(str(sfmrfile_reconstruction_with_17_images))
        graph = _build_connectivity_graph([recon, recon])
        assert 1 in graph[0]
        assert 0 in graph[1]
        assert len(graph[0][1]) == 17


# ---------------------------------------------------------------------------
# Alignment core tests
# ---------------------------------------------------------------------------


class TestAlignReconstructionsPoints:
    """Test align_reconstructions with point-based method."""

    def test_align_transformed_recovery(self, sfmrfile_reconstruction_with_17_images):
        """Alignment should recover the inverse of an applied transform."""
        recon = SfmrReconstruction.load(str(sfmrfile_reconstruction_with_17_images))
        transform = Se3Transform(translation=[5, 0, 0], scale=2.0)
        transformed = apply_transforms(recon, [SimilarityTransform(transform)])

        result = align_reconstructions(
            reference=recon,
            to_align=[transformed],
            method="points",
        )

        assert result.aligned[0] is not None
        assert result.total_shared_images == 17

    def test_align_identical(self, sfmrfile_reconstruction_with_17_images):
        """Aligning identical reconstructions should succeed with near-zero error."""
        recon = SfmrReconstruction.load(str(sfmrfile_reconstruction_with_17_images))
        result = align_reconstructions(
            reference=recon,
            to_align=[recon],
            method="points",
        )
        assert result.aligned[0] is not None

    def test_align_no_shared_images(self, sfmrfile_reconstruction_with_17_images):
        """Aligning reconstructions with no shared images should fail gracefully."""
        recon = SfmrReconstruction.load(str(sfmrfile_reconstruction_with_17_images))
        # Filter to disjoint image sets
        subset1 = apply_transforms(
            recon, [IncludeRangeFilter(IntRangeExpr.from_str("1-5"))]
        )
        subset2 = apply_transforms(
            recon, [IncludeRangeFilter(IntRangeExpr.from_str("10-17"))]
        )

        result = align_reconstructions(
            reference=subset1,
            to_align=[subset2],
            method="points",
        )
        assert result.aligned[0] is None


class TestAlignReconstructionsCameras:
    """Test align_reconstructions with camera-based method."""

    def test_align_transformed_recovery(self, sfmrfile_reconstruction_with_17_images):
        recon = SfmrReconstruction.load(str(sfmrfile_reconstruction_with_17_images))
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
    def test_align_points_default(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Default align (points) succeeds and produces output."""
        ref = sfmrfile_reconstruction_with_17_images
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

    def test_align_cameras_method(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        ref = sfmrfile_reconstruction_with_17_images
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

    def test_aligned_positions_close_to_reference(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """After alignment, camera positions should be close to the reference."""
        ref = sfmrfile_reconstruction_with_17_images
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
        ref_recon = SfmrReconstruction.load(str(output_dir / ref.name))
        aligned_recon = SfmrReconstruction.load(str(output_dir / target_path.name))

        # Positions should be very close after alignment
        ref_positions = ref_recon.positions
        aligned_positions = aligned_recon.positions
        assert ref_positions.shape == aligned_positions.shape

        distances = np.linalg.norm(aligned_positions - ref_positions, axis=1)
        assert np.mean(distances) < 0.01
