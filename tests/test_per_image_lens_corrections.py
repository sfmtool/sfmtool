# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-image radial corrections and rolling-shutter rates in the Python package
and the exports.

Each test refits a real reconstruction's cameras to ``SFMTOOL_PINHOLE``, gives
some of its images a radial correction or rolling-shutter rates, and checks
that a consumer reads each image through them, carries them, or refuses them
naming the camera and the correction. See ``specs/formats/sfmr-file-format.md``
§ "Per-image intrinsic adjustments" and the specs of each export it links.
"""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import read_sfmr, verify_sfmr, write_sfmr
from sfmtool.geometry import CameraIntrinsics
from sfmtool.reconstruction import SfmrReconstruction

CORRECTIONS = ["radial_bspline", "rolling_shutter"]
# Image 2's radial correction, and image 3's rolling-shutter rates on rows, in
# radians per short side of readout.
RADIAL = (0.004, -0.006)
RATES = (2e-2, -1.5e-2, 1e-2)


def _refit(recon):
    """``recon`` with every camera refitted to ``SFMTOOL_PINHOLE``."""
    return recon.clone_with_changes(
        cameras=[c.refit("SFMTOOL_PINHOLE")[0] for c in recon.cameras]
    )


def _corrected(recon, radial_images=(2,), rate_images=(3,), *, scanlines="ROWS"):
    """``recon``, whose cameras are ``SFMTOOL_PINHOLE``, with ``RADIAL`` on
    ``radial_images``, ``RATES`` on ``rate_images``, and every camera varying
    both corrections on ``scanlines``."""
    n = recon.image_count
    radial = np.zeros((n, 2))
    rates = np.zeros((n, 3))
    for i in radial_images:
        radial[i] = RADIAL
    for i in rate_images:
        rates[i] = RATES
    return recon.clone_with_changes(
        per_image_adjustments=[CORRECTIONS for _ in recon.cameras],
        rolling_shutter_scanlines=[scanlines for _ in recon.cameras],
        radial_bspline_adjustments=radial,
        rolling_shutter_rates_xyz=rates,
    )


def _no_distortion(camera):
    """``camera``, an ``SFMTOOL_PINHOLE``, with every spline coefficient zero:
    a camera with no lens distortion whose images can still carry the
    corrections."""
    params = {
        k: (0.0 if k.startswith("bspline_c") and k != "bspline_coeff_count" else v)
        for k, v in camera.parameters.items()
    }
    return CameraIntrinsics(camera.model, camera.width, camera.height, params)


class TestTheReconstruction:
    def test_clone_with_changes_a_file_and_the_dicts_keep_them(
        self, seoul_bull_sfmr_only, tmp_path
    ):
        plain = _refit(SfmrReconstruction.load(seoul_bull_sfmr_only))
        recon = _corrected(plain)
        camera = recon.cameras[0]
        assert recon.per_image_adjustments == [CORRECTIONS]
        assert recon.rolling_shutter_scanlines == ["ROWS"]
        assert recon.image_intrinsics(2) == camera.with_adjustment(
            radial_bspline=RADIAL
        )
        assert recon.image_intrinsics(3).rolling_shutter == ("ROWS", RATES)
        assert recon.image_intrinsics(0) == camera

        # A file keeps them: the guard at the file boundary is gone.
        out = tmp_path / "corrected.sfmr"
        recon.save(out)
        ok, errors = verify_sfmr(out)
        assert ok, errors
        loaded = SfmrReconstruction.load(out)
        assert loaded.per_image_adjustments == [CORRECTIONS]
        assert loaded.rolling_shutter_scanlines == ["ROWS"]
        for i in range(recon.image_count):
            assert loaded.image_intrinsics(i) == recon.image_intrinsics(i)

        # The dicts round-trip them bit for bit.
        data = read_sfmr(out)
        assert (
            data["radial_bspline_adjustments"].tobytes()
            == recon.radial_bspline_adjustments.tobytes()
        )
        assert (
            data["rolling_shutter_rates_xyz"].tobytes()
            == recon.rolling_shutter_rates_xyz.tobytes()
        )
        again = tmp_path / "again.sfmr"
        write_sfmr(again, data)
        assert read_sfmr(again)["content_hash"] == data["content_hash"]

        # Passing None clears them.
        cleared = recon.clone_with_changes(
            radial_bspline_adjustments=None,
            rolling_shutter_rates_xyz=None,
            per_image_adjustments=None,
            rolling_shutter_scanlines=None,
        )
        assert cleared.radial_bspline_adjustments is None
        assert cleared.rolling_shutter_rates_xyz is None
        assert cleared.rolling_shutter_scanlines == [None]
        assert cleared.image_intrinsics(3) == cleared.cameras[0]

    def test_clone_with_changes_checks_each_image_against_its_camera(
        self, seoul_bull_sfmr_only
    ):
        plain = SfmrReconstruction.load(seoul_bull_sfmr_only)
        recon = _corrected(_refit(plain))
        # Rates with no scanlines setting to read them on.
        with pytest.raises(ValueError, match="image 3"):
            recon.clone_with_changes(rolling_shutter_scanlines=None)
        # A rate that is not finite.
        rates = np.array(recon.rolling_shutter_rates_xyz)
        rates[3, 0] = np.nan
        with pytest.raises(ValueError, match="not finite"):
            recon.clone_with_changes(rolling_shutter_rates_xyz=rates)
        # A radial correction on a camera of another model.
        radial = np.zeros((plain.image_count, 2))
        radial[2] = RADIAL
        with pytest.raises(ValueError, match="SFMTOOL_PINHOLE"):
            plain.clone_with_changes(radial_bspline_adjustments=radial)

    def test_inspect_names_the_corrections_and_the_scanlines(
        self, seoul_bull_sfmr_only, tmp_path
    ):
        out = tmp_path / "corrected.sfmr"
        _corrected(_refit(SfmrReconstruction.load(seoul_bull_sfmr_only))).save(out)
        result = CliRunner().invoke(main, ["inspect", str(out), "--verbose"])
        assert result.exit_code == 0, result.output
        assert "Adjusted per image: radial_bspline, rolling_shutter" in result.output
        assert "Rolling-shutter scanlines: ROWS" in result.output

    @pytest.mark.parametrize(
        "scanlines, edge", [("ROWS", "bottom"), ("COLUMNS", "right")]
    )
    def test_inspect_gives_the_corrections_sizes_in_pixels(
        self, seoul_bull_sfmr_only, tmp_path, scanlines, edge
    ):
        """The radial correction is ``f_i·e_i1`` at the spline domain end, and
        the rolling shutter's displacement at the edge centre read farthest from
        the principal point's scanline matches the model's first-order formula,
        the velocity at the undistorted point times the readout position."""
        recon = _corrected(
            _refit(SfmrReconstruction.load(seoul_bull_sfmr_only)), scanlines=scanlines
        )
        out = tmp_path / "corrected.sfmr"
        recon.save(out)
        result = CliRunner().invoke(main, ["inspect", str(out), "--verbose"])
        assert result.exit_code == 0, result.output
        n = recon.image_count

        f_2 = recon.image_intrinsics(2).focal_lengths[0]
        assert (
            "    Radial correction at the spline domain end: median 0.00 px, max "
            f"{abs(f_2 * RADIAL[1]):.2f} px over {n} images"
        ) in result.output

        # The image's pixel at the edge centre, its ray, and the displacement
        # the rates give it by the formula in the draft's "The map for image i".
        camera = recon.image_intrinsics(3)
        f = camera.focal_lengths[0]
        cx, cy = camera.principal_point
        m = min(camera.width, camera.height)
        if scanlines == "ROWS":
            assert cy <= camera.height / 2
            u, v = camera.width / 2, camera.height
            t = (v - cy) / m
        else:
            assert cx <= camera.width / 2
            u, v = camera.width, camera.height / 2
            t = (u - cx) / m
        without = recon.cameras[0]
        ray = without.pixel_to_ray(u, v)
        x, y = ray[0] / -ray[2], ray[1] / -ray[2]
        wx, wy, wz = RATES
        xdot = wy * (1 + x * x) - wx * x * y + wz * y
        ydot = -wx * (1 + y * y) + wy * x * y - wz * x
        expected = f * t * np.hypot(xdot, ydot)
        assert (
            f"    Rolling-shutter displacement at the {edge} edge centre: median "
            f"0.00 px, max {expected:.2f} px over {n} images"
        ) in result.output

        from sfmtool.analyze._correction_sizes import image_correction_sizes

        sizes = image_correction_sizes(recon)
        assert sizes["rolling_shutter"][3] == pytest.approx(expected, rel=1e-3)
        assert sizes["radial_bspline"][2] == pytest.approx(f_2 * RADIAL[1])

    def test_inspect_gives_no_sizes_without_the_corrections(self, seoul_bull_sfmr_only):
        result = CliRunner().invoke(
            main, ["inspect", str(seoul_bull_sfmr_only), "--verbose"]
        )
        assert result.exit_code == 0, result.output
        assert "Radial correction at" not in result.output
        assert "Rolling-shutter displacement" not in result.output

    def test_analyze_metrics_lists_each_images_corrections(
        self, seoul_bull_sfmr_only, tmp_path
    ):
        recon = _corrected(_refit(SfmrReconstruction.load(seoul_bull_sfmr_only)))
        out = tmp_path / "corrected.sfmr"
        recon.save(out)
        result = CliRunner().invoke(main, ["analyze", str(out), "--metrics"])
        assert result.exit_code == 0, result.output
        header = next(x for x in result.output.splitlines() if x.startswith("Image ("))
        assert header.split()[-3:] == ["Radial", "RShut", "Flag"]
        name = recon.image_names[2].split("/")[-1]
        row = next(x for x in result.output.splitlines() if name in x).split()
        f_2 = recon.image_intrinsics(2).focal_lengths[0]
        assert float(row[6]) == pytest.approx(f_2 * RADIAL[1], abs=0.005)
        assert "RShut = the rolling-shutter displacement" in result.output

        plain = CliRunner().invoke(
            main, ["analyze", str(seoul_bull_sfmr_only), "--metrics"]
        )
        assert plain.exit_code == 0, plain.output
        assert "Radial" not in plain.output and "RShut" not in plain.output


class TestColmapExport:
    """COLMAP has no form for either correction, so each COLMAP writer refuses
    them before it writes anything, rather than dropping them."""

    def test_the_binary_export_refuses_them(self, seoul_bull_workspace, tmp_path):
        from sfmtool.colmap.io import save_colmap_binary

        recon = _corrected(_refit(SfmrReconstruction.load(seoul_bull_workspace)))
        with pytest.raises(ValueError, match="camera 0 carry a radial correction"):
            save_colmap_binary(recon, tmp_path / "colmap")
        assert not (tmp_path / "colmap").exists()

        rates_only = _corrected(
            _refit(SfmrReconstruction.load(seoul_bull_workspace)), radial_images=()
        )
        with pytest.raises(ValueError, match=r"rolling-shutter rates \(rolling"):
            save_colmap_binary(rates_only, tmp_path / "colmap")

    def test_the_database_export_refuses_them(self, seoul_bull_workspace, tmp_path):
        from sfmtool.colmap.db_export import create_colmap_db_from_reconstruction

        recon = _corrected(_refit(SfmrReconstruction.load(seoul_bull_workspace)))
        db_path = tmp_path / "out.db"
        with pytest.raises(ValueError, match="radial correction"):
            create_colmap_db_from_reconstruction(
                recon, db_path, populate_two_view_geometries=False
            )
        assert not db_path.exists()


class TestOtherWriters:
    def test_nerfstudio_refuses_them_before_writing(
        self, seoul_bull_workspace, tmp_path
    ):
        from sfmtool._to_nerfstudio import export_to_nerfstudio

        refit = _refit(SfmrReconstruction.load(seoul_bull_workspace))
        # A camera with no lens distortion passes the distortion check, and its
        # images' corrections are refused on their own.
        undistorted = refit.clone_with_changes(
            cameras=[_no_distortion(c) for c in refit.cameras]
        )
        assert not undistorted.cameras[0].has_distortion
        recon = _corrected(undistorted, radial_images=())
        out = tmp_path / "nerfstudio"
        with pytest.raises(ValueError, match="rolling-shutter rates"):
            export_to_nerfstudio(recon, out)
        assert not out.exists()

    def test_camrig_cp_says_it_copies_the_shared_camera(
        self, seoul_bull_workspace, tmp_path
    ):
        source = seoul_bull_workspace.parent / "corrected.sfmr"
        _corrected(_refit(SfmrReconstruction.load(seoul_bull_workspace))).save(source)
        out = tmp_path / "cam.camrig"
        result = CliRunner().invoke(main, ["camrig", "cp", str(source), str(out)])
        assert result.exit_code == 0, result.output
        assert "camera 0 copied as the shared camera" in result.output
        assert "per-image intrinsic adjustments" in result.output
        assert (
            "camera 0 copied without the rolling_shutter_scanlines setting"
            in result.output
        )

    def test_densify_refuses_them(self, seoul_bull_sfmr_only):
        from sfmtool._densify import densify_reconstruction

        recon = _corrected(_refit(SfmrReconstruction.load(seoul_bull_sfmr_only)))
        with pytest.raises(
            ValueError, match=r"per-image intrinsic adjustments \(radial_bspline"
        ):
            densify_reconstruction(recon)


def _undistort(recon, output_dir):
    from sfmtool._undistort_images import undistort_reconstruction_images

    _, _, sfmr_path = undistort_reconstruction_images(
        recon=recon, output_dir=output_dir, resampling_filter="bilinear"
    )
    return SfmrReconstruction.load(Path(sfmr_path))


def _retake(recon, i, adjusted):
    """Rewrite image ``i``'s pixels and ``.sift`` keypoints in ``recon``'s
    workspace as if its camera had taken it through ``adjusted``, the image's
    intrinsics with its corrections, so that a reconstruction carrying them
    describes the image exactly."""
    from sfmtool.fileio import read_image_rgb, write_image_rgb
    from sfmtool.flow import WarpMap
    from sfmtool.sift.file import SiftReader, get_sift_path_from_recon, write_sift

    camera = recon.cameras[int(recon.camera_indexes[i])]
    name = recon.image_names[i]

    image_path = Path(recon.workspace_dir) / name
    image = read_image_rgb(image_path)
    warp = WarpMap.from_cameras(src=camera, dst=adjusted)
    write_image_rgb(image_path, warp.remap_bilinear(image))

    sift_path = get_sift_path_from_recon(recon, name)
    with SiftReader(sift_path) as reader:
        feature_tool_metadata = reader.feature_tool_metadata
        metadata = reader.metadata
        positions = reader.read_positions().astype(np.float64)
        affine_shapes = reader.read_affine_shapes()
        descriptors = reader.read_descriptors()
        thumbnail = reader.read_thumbnail()
    positions = adjusted.project_batch(camera.unproject_batch(positions))
    write_sift(
        sift_path,
        feature_tool_metadata,
        metadata,
        positions.astype(np.float32),
        affine_shapes,
        descriptors,
        thumbnail,
    )


class TestUndistort:
    """Undistortion removes each image's radial correction and rolling shutter
    along with its distortion: the warp and the keypoint remap read the image
    through its own intrinsics, whose ``project`` and ``unproject`` apply both,
    and every image of a camera is written through the camera's one pinhole."""

    def test_the_output_has_no_corrections(self, seoul_bull_workspace, tmp_path):
        recon = _corrected(_refit(SfmrReconstruction.load(seoul_bull_workspace)))
        out = _undistort(recon, tmp_path / "undistorted")

        assert out.per_image_adjustments == [[]]
        assert out.radial_bspline_adjustments is None
        assert out.rolling_shutter_rates_xyz is None
        assert out.rolling_shutter_scanlines == [None]
        assert out.image_intrinsics(3) == out.cameras[0]
        assert out.cameras[0].rolling_shutter is None

    def test_a_corrected_image_is_warped_into_the_shared_pinhole(
        self, seoul_bull_workspace, tmp_path
    ):
        from sfmtool.fileio import read_image_rgb
        from sfmtool.flow import WarpMap
        from sfmtool.sift.file import SiftReader, get_sift_path_from_recon

        plain = _refit(SfmrReconstruction.load(seoul_bull_workspace))
        plain_out = _undistort(plain, tmp_path / "plain")

        # Image 2 retaken with both corrections, and a reconstruction that
        # says so: its undistortion should be the plain image's.
        recon = _corrected(plain, radial_images=(2,), rate_images=(2,))
        adjusted = recon.image_intrinsics(2)
        assert adjusted.rolling_shutter == ("ROWS", RATES)
        _retake(plain, 2, adjusted)
        out = _undistort(recon, tmp_path / "corrected")

        assert out.cameras == plain_out.cameras
        assert out.per_image_adjustments == [[]]

        # Each keypoint of image 2 lands where the plain run put it.
        name = recon.image_names[2]
        with SiftReader(get_sift_path_from_recon(out, name)) as reader:
            positions = reader.read_positions()
        with SiftReader(get_sift_path_from_recon(plain_out, name)) as reader:
            plain_positions = reader.read_positions()
        assert positions.shape == plain_positions.shape
        np.testing.assert_allclose(positions, plain_positions, atol=1e-3)

        # So every observation reprojects as it did in the plain run.
        def by_feature(errors):
            return errors[np.argsort(errors[:, 0], kind="stable")]

        for i in range(out.image_count):
            np.testing.assert_allclose(
                by_feature(out.compute_observation_reprojection_errors(i)),
                by_feature(plain_out.compute_observation_reprojection_errors(i)),
                atol=1e-3,
            )

        # Image 2's pixels match the plain run's, away from the border the
        # retake left black. Warping it as if it had no corrections puts its
        # content a few pixels off.
        pixels = read_image_rgb(Path(out.workspace_dir) / name).astype(np.float64)
        expected = read_image_rgb(Path(plain_out.workspace_dir) / name).astype(
            np.float64
        )
        retaken = read_image_rgb(Path(recon.workspace_dir) / name)
        camera = plain.cameras[0]
        misread = (
            WarpMap.from_cameras(src=camera, dst=out.cameras[0])
            .remap_bilinear(retaken)
            .astype(np.float64)
        )
        h, w = pixels.shape[:2]
        inner = (slice(h // 8, h - h // 8), slice(w // 8, w - w // 8))
        difference = np.mean(np.abs(pixels[inner] - expected[inner]))
        misread_difference = np.mean(np.abs(misread[inner] - expected[inner]))
        # About 2.3 against 12.5 grey levels here.
        assert difference < 0.3 * misread_difference


class TestMerge:
    def test_merge_carries_each_images_corrections(
        self, seoul_bull_workspace, tmp_path
    ):
        from sfmtool.merge.reconstructions import merge_reconstructions
        from sfmtool.reconstruction import RangeExpr
        from sfmtool.xform import IncludeRangeFilter, apply_transforms

        original = _corrected(
            _refit(SfmrReconstruction.load(seoul_bull_workspace)),
            radial_images=(2, 12),
            rate_images=(3, 13),
        )
        subsets = [
            apply_transforms(original, [IncludeRangeFilter(RangeExpr(r))])
            for r in ("1-10", "6-17")
        ]
        merged = merge_reconstructions(subsets)

        assert merged.per_image_adjustments == [CORRECTIONS]
        assert merged.rolling_shutter_scanlines == ["ROWS"]
        index_of = {Path(n).name: i for i, n in enumerate(original.image_names)}
        for i, name in enumerate(merged.image_names):
            j = index_of[Path(name).name]
            assert tuple(merged.radial_bspline_adjustments[i]) == tuple(
                original.radial_bspline_adjustments[j]
            )
            assert tuple(merged.rolling_shutter_rates_xyz[i]) == tuple(
                original.rolling_shutter_rates_xyz[j]
            )
        assert np.count_nonzero(merged.radial_bspline_adjustments[:, 0]) == 2
        assert np.count_nonzero(merged.rolling_shutter_rates_xyz[:, 0]) == 2

    def test_cameras_with_different_scanlines_are_not_merged(
        self, seoul_bull_sfmr_only
    ):
        from sfmtool.merge.reconstructions import merge_reconstructions

        plain = SfmrReconstruction.load(seoul_bull_sfmr_only)
        rows = plain.clone_with_changes(rolling_shutter_scanlines=["ROWS"])
        with pytest.raises(ValueError, match="same scanlines setting"):
            merge_reconstructions([plain, rows])


class TestBundleAdjust:
    def test_sfmtools_adjustment_holds_them(self, seoul_bull_sfmr_only, capsys):
        from sfmtool.xform import BundleAdjustTransform

        recon = _corrected(_refit(SfmrReconstruction.load(seoul_bull_sfmr_only)))
        result = BundleAdjustTransform().apply(recon)
        assert "sfmtool; a camera has a spline model" in capsys.readouterr().out
        # The corrections are held fixed and written back, with the setting.
        assert result.per_image_adjustments == [CORRECTIONS]
        assert result.rolling_shutter_scanlines == ["ROWS"]
        assert (
            result.radial_bspline_adjustments.tobytes()
            == recon.radial_bspline_adjustments.tobytes()
        )
        assert (
            result.rolling_shutter_rates_xyz.tobytes()
            == recon.rolling_shutter_rates_xyz.tobytes()
        )

    def test_pycolmaps_path_refuses_them(self, seoul_bull_sfmr_only):
        from sfmtool.xform import BundleAdjustTransform

        recon = _corrected(_refit(SfmrReconstruction.load(seoul_bull_sfmr_only)))
        with pytest.raises(ValueError, match="radial correction"):
            BundleAdjustTransform()._apply_pycolmap(recon)
