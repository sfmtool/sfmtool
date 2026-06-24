# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `sfm panorama` command."""

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._filenames import number_from_filename
from sfmtool.rig.panorama import load_panorama_rig, select_source_indices
from sfmtool._sfmtool import SfmrReconstruction
from sfmtool._sfmtool.io import write_camrig
from sfmtool._sfmtool.spherical import SphericalTileRig
from sfmtool.cli import main


def _write_spherical_tiles_camrig(path, *, n=48, equirect_width=256, seed=1):
    """Build a small spherical-tile rig and write it to ``path``."""
    rig = SphericalTileRig(n=n, arc_per_pixel=2 * np.pi / equirect_width, seed=seed)
    rig.write_camrig(str(path))
    return n


def _write_generic_camrig(path):
    """Write a non-``spherical_tiles`` (generic) rig — not a valid panorama rig."""
    write_camrig(
        path=str(path),
        name=path.stem,
        rig_type="generic",
        cameras=[
            {
                "model": "PINHOLE",
                "width": 640,
                "height": 480,
                "parameters": {
                    "focal_length_x": 500.0,
                    "focal_length_y": 500.0,
                    "principal_point_x": 320.0,
                    "principal_point_y": 240.0,
                },
            }
        ],
        sensor_image_patterns=["imgs/*.jpg"],
        camera_indexes=[0],
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        translations_xyz=np.zeros((1, 3)),
    )


@pytest.fixture
def recon_17(seoul_bull_workspace):
    return SfmrReconstruction.load(seoul_bull_workspace)


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

    def test_camrig_wrong_suffix_rejected(self, tmp_path):
        recon = tmp_path / "recon.sfmr"
        recon.write_bytes(b"fake")
        bad = tmp_path / "rig.txt"
        bad.write_bytes(b"not a rig")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "panorama",
                str(recon),
                "-o",
                str(tmp_path / "p.png"),
                "--camrig",
                str(bad),
            ],
        )
        assert result.exit_code != 0
        assert ".camrig" in result.output


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


class TestLoadPanoramaRig:
    """The loaded rig keeps the saved layout but sizes its patch to the width."""

    def test_patch_size_tracks_output_width(self, tmp_path):
        # Build (and persist) a rig at a high resolution → large stored patch.
        rig_path = tmp_path / "tiles.camrig"
        SphericalTileRig(n=48, arc_per_pixel=2 * np.pi / 4096, seed=1).write_camrig(
            str(rig_path)
        )
        stored = SphericalTileRig.read_camrig(str(rig_path)).patch_size

        small = load_panorama_rig(rig_path, 256)
        large = load_panorama_rig(rig_path, 4096)

        # Tile count (layout) is unchanged regardless of width.
        assert small.n == large.n == 48
        # Patch is sized to the width: smaller output → smaller patch, and it
        # can be much smaller than the rig's stored patch_size.
        assert small.patch_size < large.patch_size
        assert small.patch_size < stored
        # Both are powers of two (atlas packer requirement).
        for ps in (small.patch_size, large.patch_size):
            assert ps & (ps - 1) == 0


class TestPanoramaE2E:
    """End-to-end render using the Seoul Bull dataset."""

    def test_renders_panorama(self, seoul_bull_workspace, tmp_path):
        sfmr_path = seoul_bull_workspace
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

    def test_renders_with_range(self, seoul_bull_workspace, tmp_path):
        sfmr_path = seoul_bull_workspace
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

    def test_renders_near_image(self, seoul_bull_workspace, tmp_path):
        sfmr_path = seoul_bull_workspace
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

    def test_renders_with_camrig(self, seoul_bull_workspace, tmp_path):
        sfmr_path = seoul_bull_workspace
        camrig_path = tmp_path / "tiles.camrig"
        n_tiles = _write_spherical_tiles_camrig(camrig_path, n=48, equirect_width=128)
        output_path = tmp_path / "pano_camrig.png"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "panorama",
                str(sfmr_path),
                "-o",
                str(output_path),
                "--camrig",
                str(camrig_path),
                # Output resolution is decoupled from the rig's tile density.
                "--equirect-width",
                "256",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Loaded rig" in result.output
        assert f"tiles={n_tiles}" in result.output
        assert "tile_fx=" in result.output
        assert output_path.exists()
        img = cv2.imread(str(output_path))
        assert img is not None
        assert img.shape[:2] == (128, 256)

    def test_camrig_takes_precedence_over_n_tiles(self, seoul_bull_workspace, tmp_path):
        sfmr_path = seoul_bull_workspace
        camrig_path = tmp_path / "tiles.camrig"
        n_tiles = _write_spherical_tiles_camrig(camrig_path, n=40, equirect_width=128)
        output_path = tmp_path / "pano_precedence.png"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "panorama",
                str(sfmr_path),
                "-o",
                str(output_path),
                "--camrig",
                str(camrig_path),
                "--n-tiles",
                "999",
                "--equirect-width",
                "256",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "--n-tiles is ignored" in result.output
        assert f"tiles={n_tiles}" in result.output
        assert "tiles=999" not in result.output

    def test_non_spherical_camrig_rejected(self, seoul_bull_workspace, tmp_path):
        sfmr_path = seoul_bull_workspace
        camrig_path = tmp_path / "generic.camrig"
        _write_generic_camrig(camrig_path)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "panorama",
                str(sfmr_path),
                "-o",
                str(tmp_path / "p.png"),
                "--camrig",
                str(camrig_path),
                "--equirect-width",
                "256",
            ],
        )
        assert result.exit_code != 0
