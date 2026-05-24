# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `sfm panorama` command."""

import cv2
import pytest
from click.testing import CliRunner

from sfmtool._filenames import number_from_filename
from sfmtool._panorama import select_source_indices
from sfmtool._sfmtool import SfmrReconstruction
from sfmtool.cli import main


@pytest.fixture
def recon_17(sfmrfile_reconstruction_with_17_images):
    return SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)


class TestPanoramaCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["panorama", "--help"])
        assert result.exit_code == 0
        assert "equirectangular panorama" in result.output

    def test_non_sfmr_rejected(self, tmp_path):
        recon = tmp_path / "recon.txt"
        recon.write_bytes(b"fake")
        runner = CliRunner()
        result = runner.invoke(
            main, ["panorama", str(recon), "-o", str(tmp_path / "pano.png")]
        )
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_missing_reconstruction_errors(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["panorama", str(tmp_path / "nope.sfmr"), "-o", str(tmp_path / "pano.png")],
        )
        assert result.exit_code != 0

    def test_output_required(self):
        runner = CliRunner()
        result = runner.invoke(main, ["panorama", "dummy.sfmr"])
        assert result.exit_code != 0

    def test_odd_width_rejected(self, tmp_path):
        recon = tmp_path / "recon.sfmr"
        recon.write_bytes(b"fake")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "panorama",
                str(recon),
                "-o",
                str(tmp_path / "pano.png"),
                "--equirect-width",
                "255",
            ],
        )
        assert result.exit_code != 0
        assert "even" in result.output

    def test_near_count_requires_near_image(self, tmp_path):
        recon = tmp_path / "recon.sfmr"
        recon.write_bytes(b"fake")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "panorama",
                str(recon),
                "-o",
                str(tmp_path / "p.png"),
                "--near-count",
                "5",
            ],
        )
        assert result.exit_code != 0
        assert "--near-image" in result.output

    def test_near_image_requires_count_or_radius(self, tmp_path):
        recon = tmp_path / "recon.sfmr"
        recon.write_bytes(b"fake")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "panorama",
                str(recon),
                "-o",
                str(tmp_path / "p.png"),
                "--near-image",
                "frame.jpg",
            ],
        )
        assert result.exit_code != 0
        assert "--near-count" in result.output


class TestSelectSourceIndices:
    """Unit tests for the range / near-image source selection."""

    def test_no_filter_returns_none(self, recon_17):
        assert select_source_indices(recon_17) is None

    def test_range_subsets_by_file_number(self, recon_17):
        names = list(recon_17.image_names)
        nums = sorted(
            {number_from_filename(n) for n in names if number_from_filename(n)}
        )
        wanted = nums[:3]
        expr = ",".join(str(x) for x in wanted)
        idx = select_source_indices(recon_17, range_expr=expr)
        kept = {number_from_filename(names[i]) for i in idx}
        assert kept == set(wanted)

    def test_range_no_match_raises(self, recon_17):
        with pytest.raises(ValueError, match="range"):
            select_source_indices(recon_17, range_expr="999999")

    def test_near_count_keeps_n_including_reference(self, recon_17):
        names = list(recon_17.image_names)
        ref = names[0]
        idx = select_source_indices(recon_17, near_image=ref, near_count=5)
        assert len(idx) == 5
        assert names.index(ref) in set(int(i) for i in idx)

    def test_near_count_caps_at_available(self, recon_17):
        idx = select_source_indices(
            recon_17, near_image=list(recon_17.image_names)[0], near_count=10_000
        )
        assert len(idx) == recon_17.image_count

    def test_near_radius_tiny_keeps_only_reference(self, recon_17):
        names = list(recon_17.image_names)
        idx = select_source_indices(recon_17, near_image=names[0], near_radius=1e-9)
        assert list(idx) == [names.index(names[0])]

    def test_near_image_no_match_raises(self, recon_17):
        with pytest.raises(ValueError, match="did not match"):
            select_source_indices(
                recon_17, near_image="nonexistent_image.jpg", near_count=3
            )


class TestPanoramaE2E:
    """End-to-end render using the Seoul Bull dataset."""

    def test_renders_panorama(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        sfmr_path = sfmrfile_reconstruction_with_17_images
        output_path = tmp_path / "pano.png"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "panorama",
                str(sfmr_path),
                "-o",
                str(output_path),
                "--equirect-width",
                "256",
                "--n-tiles",
                "64",
            ],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert output_path.exists()
        img = cv2.imread(str(output_path))
        assert img is not None
        # height = width / 2
        assert img.shape[:2] == (128, 256)

    def test_renders_with_range(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon = SfmrReconstruction.load(sfmr_path)
        nums = sorted(
            {
                number_from_filename(n)
                for n in recon.image_names
                if number_from_filename(n)
            }
        )
        expr = ",".join(str(x) for x in nums[:4])
        output_path = tmp_path / "pano_range.png"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "panorama",
                str(sfmr_path),
                "-o",
                str(output_path),
                "--equirect-width",
                "256",
                "--n-tiles",
                "64",
                "--range",
                expr,
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Selected 4 of 17" in result.output
        assert output_path.exists()

    def test_renders_near_image(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon = SfmrReconstruction.load(sfmr_path)
        ref = list(recon.image_names)[0]
        output_path = tmp_path / "pano_near.png"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "panorama",
                str(sfmr_path),
                "-o",
                str(output_path),
                "--equirect-width",
                "256",
                "--n-tiles",
                "64",
                "--near-image",
                ref,
                "--near-count",
                "6",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Selected 6 of 17" in result.output
        assert output_path.exists()
