# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for COLMAP interop CLI commands."""

from click.testing import CliRunner

from sfmtool.cli import main


# =============================================================================
# to-colmap-bin CLI tests
# =============================================================================


class TestToColmapBinCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["to-colmap-bin", "--help"])
        assert result.exit_code == 0
        assert "colmap" in result.output.lower()
        assert "cameras.bin" in result.output

    def test_rejects_non_sfmr(self, tmp_path):
        fake = tmp_path / "test.txt"
        fake.write_text("not sfmr")
        runner = CliRunner()
        result = runner.invoke(
            main, ["to-colmap-bin", str(fake), str(tmp_path / "out")]
        )
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_missing_input(self):
        runner = CliRunner()
        result = runner.invoke(main, ["to-colmap-bin", "nonexistent.sfmr", "out/"])
        assert result.exit_code != 0


# =============================================================================
# to-colmap-db CLI tests
# =============================================================================


class TestToColmapDbCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["to-colmap-db", "--help"])
        assert result.exit_code == 0
        assert "colmap" in result.output.lower()
        assert "--out-db" in result.output

    def test_rejects_non_sfmr_non_matches(self, tmp_path):
        fake = tmp_path / "test.txt"
        fake.write_text("not valid")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["to-colmap-db", str(fake), "--out-db", str(tmp_path / "db.db")],
        )
        assert result.exit_code != 0
        assert ".sfmr" in result.output or ".matches" in result.output

    def test_missing_input(self):
        runner = CliRunner()
        result = runner.invoke(
            main, ["to-colmap-db", "nonexistent.sfmr", "--out-db", "db.db"]
        )
        assert result.exit_code != 0

    def test_out_db_required(self, tmp_path):
        fake = tmp_path / "test.sfmr"
        fake.write_bytes(b"fake")
        runner = CliRunner()
        result = runner.invoke(main, ["to-colmap-db", str(fake)])
        assert result.exit_code != 0


# =============================================================================
# from-colmap-bin CLI tests
# =============================================================================


class TestFromColmapBinCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["from-colmap-bin", "--help"])
        assert result.exit_code == 0
        assert "colmap" in result.output.lower()
        assert "--image-dir" in result.output

    def test_rejects_non_sfmr_output(self, tmp_path):
        colmap_dir = tmp_path / "colmap"
        colmap_dir.mkdir()
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "from-colmap-bin",
                str(colmap_dir),
                "--image-dir",
                str(image_dir),
                "-o",
                str(tmp_path / "output.txt"),
            ],
        )
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_missing_input(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "from-colmap-bin",
                "nonexistent_dir/",
                "--image-dir",
                "images/",
                "-o",
                "out.sfmr",
            ],
        )
        assert result.exit_code != 0

    def test_image_dir_required(self, tmp_path):
        colmap_dir = tmp_path / "colmap"
        colmap_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["from-colmap-bin", str(colmap_dir), "-o", "out.sfmr"],
        )
        assert result.exit_code != 0


# =============================================================================
# to-colmap-bin E2E test
# =============================================================================


class TestToColmapBinE2E:
    def test_export_reconstruction(
        self, tmp_path, sfmrfile_reconstruction_with_17_images
    ):
        """Export a .sfmr to COLMAP binary format."""
        sfmr_path = sfmrfile_reconstruction_with_17_images

        output_dir = tmp_path / "colmap_output"
        runner = CliRunner()
        result = runner.invoke(main, ["to-colmap-bin", str(sfmr_path), str(output_dir)])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert (output_dir / "cameras.bin").exists()
        assert (output_dir / "images.bin").exists()
        assert (output_dir / "points3D.bin").exists()
