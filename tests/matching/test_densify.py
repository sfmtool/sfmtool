# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the densify command and its image-pair pruning."""

from click.testing import CliRunner

from sfmtool._densify import prune_image_pairs
from sfmtool.cli import main


class TestPruneImagePairs:
    def test_empty_input(self):
        assert prune_image_pairs([]) == []

    def test_close_pairs_kept(self):
        pairs = [(0, 1, 10), (1, 2, 8), (2, 3, 6)]
        result = prune_image_pairs(pairs, close_pair_threshold=4, max_distant_pairs=0)
        assert len(result) == 3

    def test_max_close_pairs_limits(self):
        pairs = [(0, 1, 10), (1, 2, 8), (2, 3, 6)]
        result = prune_image_pairs(
            pairs, close_pair_threshold=4, max_close_pairs=2, max_distant_pairs=0
        )
        assert len(result) == 2

    def test_distant_pairs_sorted_by_score(self):
        pairs = [
            (0, 100, 5),
            (0, 200, 10),
            (0, 300, 1),
        ]
        result = prune_image_pairs(
            pairs,
            close_pair_threshold=4,
            max_distant_pairs=2,
            distant_pair_search_multiplier=10,
        )
        assert len(result) == 2
        # Best scores should be kept
        assert (0, 200) in result
        assert (0, 100) in result

    def test_mixed_close_and_distant(self):
        pairs = [(0, 1, 10), (0, 2, 8), (0, 100, 20), (0, 200, 15)]
        result = prune_image_pairs(pairs, close_pair_threshold=4, max_distant_pairs=1)
        # 2 close + 1 best distant
        close = [(i, j) for i, j in result if abs(j - i) <= 4]
        distant = [(i, j) for i, j in result if abs(j - i) > 4]
        assert len(close) == 2
        assert len(distant) == 1


# ===== Densify CLI Tests =====


class TestDensifyCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["densify", "--help"])
        assert result.exit_code == 0
        assert "Densify matches" in result.output

    def test_non_sfmr_input_rejected(self, tmp_path):
        runner = CliRunner()
        input_file = tmp_path / "input.txt"
        input_file.write_text("hello")
        output_file = tmp_path / "output.sfmr"
        result = runner.invoke(main, ["densify", str(input_file), str(output_file)])
        assert result.exit_code != 0
        assert (
            "sfmr" in result.output.lower() or "sfmr" in str(result.exception).lower()
        )

    def test_non_sfmr_output_rejected(self, tmp_path):
        runner = CliRunner()
        input_file = tmp_path / "input.sfmr"
        input_file.write_text("hello")
        output_file = tmp_path / "output.txt"
        result = runner.invoke(main, ["densify", str(input_file), str(output_file)])
        assert result.exit_code != 0

    def test_nonexistent_input_rejected(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "densify",
                str(tmp_path / "nonexistent.sfmr"),
                str(tmp_path / "output.sfmr"),
            ],
        )
        assert result.exit_code != 0


# ===== End-to-end Densify Test =====


class TestDensifyE2E:
    def test_densify_reconstruction(self, seoul_bull_workspace):
        """Test that densify produces a valid reconstruction with more points."""
        from sfmtool._densify import densify_reconstruction
        from sfmtool._sfmtool.reconstruction import SfmrReconstruction

        sfmr_path = seoul_bull_workspace
        recon = SfmrReconstruction.load(sfmr_path)

        result = densify_reconstruction(
            recon=recon,
            max_features=512,
            sweep_window_size=30,
        )

        assert result.image_count == recon.image_count
        assert result.camera_count >= 1
        assert result.point_count > 0
        # Densified should generally have more points
        # (but don't strictly require it since filtering can reduce count)

    def test_densify_cli_e2e(self, seoul_bull_workspace, tmp_path):
        """Test densify CLI end-to-end."""
        runner = CliRunner()
        input_path = seoul_bull_workspace
        output_path = tmp_path / "densified.sfmr"

        result = runner.invoke(
            main,
            [
                "densify",
                str(input_path),
                str(output_path),
                "--max-features",
                "512",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output_path.exists()

        from sfmtool._sfmtool.reconstruction import SfmrReconstruction

        densified = SfmrReconstruction.load(output_path)
        assert densified.image_count > 0
        assert densified.point_count > 0
