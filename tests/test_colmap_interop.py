# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for COLMAP interop CLI commands."""

import numpy as np
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
        assert "OUTPUT_DB_PATH" in result.output

    def test_rejects_non_sfmr_non_matches(self, tmp_path):
        fake = tmp_path / "test.txt"
        fake.write_text("not valid")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["to-colmap-db", str(fake), str(tmp_path / "db.db")],
        )
        assert result.exit_code != 0
        assert ".sfmr" in result.output or ".matches" in result.output

    def test_missing_input(self):
        runner = CliRunner()
        result = runner.invoke(main, ["to-colmap-db", "nonexistent.sfmr", "db.db"])
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
    def test_export_reconstruction(self, tmp_path, seoul_bull_workspace):
        """Export a .sfmr to COLMAP binary format."""
        sfmr_path = seoul_bull_workspace

        output_dir = tmp_path / "colmap_output"
        runner = CliRunner()
        result = runner.invoke(main, ["to-colmap-bin", str(sfmr_path), str(output_dir)])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert (output_dir / "cameras.bin").exists()
        assert (output_dir / "images.bin").exists()
        assert (output_dir / "points3D.bin").exists()

    def test_points_at_infinity_materialized(self, tmp_path, seoul_bull_workspace):
        """COLMAP binary export materialises w=0 points to finite landmarks."""
        from sfmtool.colmap.io import save_colmap_binary
        from sfmtool.reconstruction import SfmrReconstruction
        from sfmtool.fileio import read_colmap_binary

        recon = SfmrReconstruction.load(seoul_bull_workspace)
        positions_xyzw = recon.positions_xyzw.copy()
        assert len(positions_xyzw) > 0
        # Turn the first point into a point at infinity (w = 0).
        positions_xyzw[0] = [0.0, 0.0, 1.0, 0.0]
        recon = recon.clone_with_changes(positions=positions_xyzw)

        output_dir = tmp_path / "colmap_output"
        save_colmap_binary(recon, output_dir)

        # The exported reconstruction holds only finite COLMAP points.
        data = read_colmap_binary(output_dir)
        positions = np.asarray(data["positions_xyz"])
        assert len(positions) == len(positions_xyzw)
        assert np.all(np.isfinite(positions))

    def test_range_subsets_images_and_keeps_all_points(
        self, tmp_path, seoul_bull_workspace
    ):
        """--range keeps only the requested images but retains every 3D point."""
        from sfmtool.reconstruction import SfmrReconstruction
        from sfmtool.fileio import read_colmap_binary

        sfmr_path = seoul_bull_workspace
        original = SfmrReconstruction.load(sfmr_path)
        original_point_count = len(original.positions)

        output_dir = tmp_path / "colmap_output"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["to-colmap-bin", str(sfmr_path), str(output_dir), "-r", "1-5"],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        data = read_colmap_binary(output_dir)
        # Five image file numbers (1..5) were requested and all exist.
        assert len(data["image_names"]) == 5
        # All 3D points are kept even though most observations were dropped.
        assert len(data["positions_xyz"]) == original_point_count
        # At least some points have zero surviving observations.
        assert any(c == 0 for c in data["observation_counts"])

    def test_range_with_filter_points_drops_orphans(
        self, tmp_path, seoul_bull_workspace
    ):
        """--filter-points removes 3D points left with no observations."""
        from sfmtool.reconstruction import SfmrReconstruction
        from sfmtool.fileio import read_colmap_binary

        sfmr_path = seoul_bull_workspace
        original = SfmrReconstruction.load(sfmr_path)
        original_point_count = len(original.positions)

        output_dir = tmp_path / "colmap_output"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "to-colmap-bin",
                str(sfmr_path),
                str(output_dir),
                "-r",
                "1-5",
                "--filter-points",
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        data = read_colmap_binary(output_dir)
        assert len(data["image_names"]) == 5
        # Point cloud shrunk and every surviving point has at least one obs.
        assert len(data["positions_xyz"]) < original_point_count
        assert all(c >= 1 for c in data["observation_counts"])

    def test_filter_points_without_range_errors(self, tmp_path, seoul_bull_workspace):
        """--filter-points without --range is a usage error."""
        sfmr_path = seoul_bull_workspace
        output_dir = tmp_path / "colmap_output"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["to-colmap-bin", str(sfmr_path), str(output_dir), "--filter-points"],
        )
        assert result.exit_code != 0
        assert "--filter-points" in result.output
        assert "--range" in result.output

    def test_range_with_no_matches_errors(self, tmp_path, seoul_bull_workspace):
        """A range that matches no image lists the available file numbers."""
        sfmr_path = seoul_bull_workspace
        output_dir = tmp_path / "colmap_output"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["to-colmap-bin", str(sfmr_path), str(output_dir), "-r", "100-200"],
        )
        assert result.exit_code != 0
        assert "100-200" in result.output
        assert "Available file numbers" in result.output


# =============================================================================
# COLMAP import stores the RMS error
# =============================================================================


class TestColmapImportErrors:
    """A COLMAP import stores each point's RMS reprojection error.

    COLMAP's per-point error is the mean of its observation errors. The model
    written here carries a sentinel in place of it, so a test that read COLMAP's
    value would see the sentinel.
    """

    SENTINEL = 99.0

    def _model_with_sentinel_errors(self, tmp_path, sfmr_path):
        import pycolmap

        from sfmtool.colmap.io import save_colmap_binary
        from sfmtool.reconstruction import SfmrReconstruction

        model_dir = tmp_path / "colmap_model"
        save_colmap_binary(SfmrReconstruction.load(sfmr_path), model_dir)
        model = pycolmap.Reconstruction(str(model_dir))
        for point in model.points3D.values():
            point.error = self.SENTINEL
        model.write(str(model_dir))
        return model_dir, model

    def _metadata(self, sfmr_path, model):
        from sfmtool import load_workspace_config
        from sfmtool.colmap.io import build_metadata

        workspace_dir = sfmr_path.parent
        return build_metadata(
            workspace_dir=workspace_dir,
            output_path=workspace_dir / "imported.sfmr",
            workspace_config=load_workspace_config(workspace_dir),
            operation="import",
            tool_name="colmap",
            image_count=model.num_reg_images(),
            point_count=len(model.points3D),
            observation_count=sum(
                len(p.track.elements) for p in model.points3D.values()
            ),
            camera_count=len(model.cameras),
        )

    def _assert_rms(self, recon):
        from .conftest import per_point_reprojection_error_stats

        stored = np.asarray(recon.rms_reprojection_errors, dtype=np.float64)
        mean, rms = per_point_reprojection_error_stats(recon)
        assert not np.any(stored == np.float32(self.SENTINEL))
        np.testing.assert_allclose(stored, rms, rtol=1e-5, atol=1e-5)
        # The model is one where the two statistics differ, so the check above
        # tells the RMS from the mean.
        assert np.mean(np.abs(rms - mean) > 1e-3) > 0.5

    def test_binary_import_stores_the_rms(self, tmp_path, seoul_bull_workspace):
        from sfmtool.colmap.io import colmap_binary_to_rust_sfmr

        model_dir, model = self._model_with_sentinel_errors(
            tmp_path, seoul_bull_workspace
        )
        recon = colmap_binary_to_rust_sfmr(
            model_dir,
            seoul_bull_workspace.parent / "test_17_image",
            self._metadata(seoul_bull_workspace, model),
            classify_infinity=False,
        )
        self._assert_rms(recon)

    def test_pycolmap_import_stores_the_rms(self, tmp_path, seoul_bull_workspace):
        from sfmtool.colmap.io import pycolmap_to_rust_sfmr

        _, model = self._model_with_sentinel_errors(tmp_path, seoul_bull_workspace)
        recon = pycolmap_to_rust_sfmr(
            model,
            seoul_bull_workspace.parent / "test_17_image",
            self._metadata(seoul_bull_workspace, model),
            classify_infinity=False,
        )
        self._assert_rms(recon)


# =============================================================================
# COLMAP import drops repeat observations
# =============================================================================


class TestColmapImportRepeatObservations:
    """A COLMAP import keeps one observation of each point per image.

    The model written here gives one point a second observation in an image it
    is already seen in, on the unused feature of that image farthest from the
    point's own, so the import keeps the point's own observation.
    """

    def _model(self, tmp_path, sfmr_path):
        import pycolmap

        from sfmtool.colmap.io import save_colmap_binary
        from sfmtool.reconstruction import SfmrReconstruction

        model_dir = tmp_path / "colmap_model"
        save_colmap_binary(SfmrReconstruction.load(sfmr_path), model_dir)
        return pycolmap.Reconstruction(str(model_dir))

    def _import(self, model, sfmr_path):
        from sfmtool import load_workspace_config
        from sfmtool.colmap.io import build_metadata, pycolmap_to_rust_sfmr

        workspace_dir = sfmr_path.parent
        metadata = build_metadata(
            workspace_dir=workspace_dir,
            output_path=workspace_dir / "imported.sfmr",
            workspace_config=load_workspace_config(workspace_dir),
            operation="import",
            tool_name="glomap",
            image_count=model.num_reg_images(),
            point_count=len(model.points3D),
            observation_count=sum(
                len(p.track.elements) for p in model.points3D.values()
            ),
            camera_count=len(model.cameras),
        )
        return pycolmap_to_rust_sfmr(
            model,
            sfmr_path.parent / "test_17_image",
            metadata,
            classify_infinity=False,
        )

    def test_a_repeat_observation_is_dropped_on_import(
        self, tmp_path, seoul_bull_workspace, capsys
    ):
        model = self._model(tmp_path, seoul_bull_workspace)
        clean = self._import(model, seoul_bull_workspace)
        assert "repeat observation" not in capsys.readouterr().out

        pid = min(model.points3D.keys())
        point = model.points3D[pid]
        element = point.track.elements[0]
        image = model.images[element.image_id]
        used = {
            te.point2D_idx
            for p in model.points3D.values()
            for te in p.track.elements
            if te.image_id == element.image_id
        }
        xy = np.array([p2d.xy for p2d in image.points2D])
        distance = np.linalg.norm(xy - xy[element.point2D_idx], axis=1)
        distance[list(used)] = -1.0
        extra = int(np.argmax(distance))
        assert distance[extra] > 10.0
        track = point.track
        track.add_element(element.image_id, extra)
        point.track = track
        assert len(model.points3D[pid].track.elements) == len(point.track.elements)

        recon = self._import(model, seoul_bull_workspace)
        out = capsys.readouterr().out
        assert "Dropped 1 repeat observation(s) from 1 point(s)" in out

        # The import is the clean one: the extra observation went, the point's
        # own stayed, and every column agrees.
        assert recon.observation_count == clean.observation_count
        assert recon.point_count == clean.point_count
        np.testing.assert_array_equal(
            recon.track_image_indexes, clean.track_image_indexes
        )
        np.testing.assert_array_equal(
            recon.track_feature_indexes, clean.track_feature_indexes
        )
        np.testing.assert_array_equal(
            recon.track_point_indexes, clean.track_point_indexes
        )
        np.testing.assert_array_equal(recon.keypoints_xy, clean.keypoints_xy)
        np.testing.assert_allclose(
            recon.rms_reprojection_errors, clean.rms_reprojection_errors
        )
