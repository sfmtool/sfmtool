# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-image intrinsic adjustments in the Python package and the exports.

Each test gives some images of a real reconstruction a non-zero adjustment of
their camera's focal and principal point, and checks that a consumer reads, writes
or carries each image's own intrinsics. See ``specs/formats/sfmr-file-format.md``
§ "Per-image intrinsic adjustments" and the specs of each export it links.
"""

from pathlib import Path

import numpy as np
import pycolmap
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.reconstruction import SfmrReconstruction

from .conftest import KERRY_PARK_GROUND_TRUTH

BOTH = ["focal", "principal_point"]
# Image 2's adjustment: a focal 1.5% longer, and the principal point moved by
# 1% of the short side right and 0.5% up.
FOCAL = 0.015
PRINCIPAL_POINT = (0.01, -0.005)


def _adjusted(recon, adjusted_images=(2,), *, split_camera_at=None):
    """``recon`` with ``adjusted_images`` adjusted by ``FOCAL`` and
    ``PRINCIPAL_POINT`` and every camera varying both parameters.

    With ``split_camera_at``, the images from that index on move to a copy of
    camera 0, so that a camera with an adjusted image sits beside one without.
    """
    n = recon.image_count
    focal = np.zeros(n)
    principal_point = np.zeros((n, 2))
    for i in adjusted_images:
        focal[i] = FOCAL
        principal_point[i] = PRINCIPAL_POINT
    kwargs = {}
    cameras = list(recon.cameras)
    if split_camera_at is not None:
        cameras = [cameras[0], cameras[0]]
        camera_indexes = np.zeros(n, dtype=np.uint32)
        camera_indexes[split_camera_at:] = 1
        kwargs = {"cameras": cameras, "camera_indexes": camera_indexes}
    return recon.clone_with_changes(
        **kwargs,
        per_image_adjustments=[BOTH for _ in cameras],
        focal_adjustments=focal,
        principal_point_adjustments_xy=principal_point,
    )


def _expected(camera):
    """``camera`` with the test adjustment applied, by the formula."""
    short_side = min(camera.width, camera.height)
    fx, fy = camera.focal_lengths
    cx, cy = camera.principal_point
    return (
        (fx * (1 + FOCAL), fy * (1 + FOCAL)),
        (cx + PRINCIPAL_POINT[0] * short_side, cy + PRINCIPAL_POINT[1] * short_side),
    )


class TestTheReconstruction:
    def test_getters_and_clone_with_changes(self, seoul_bull_sfmr_only, tmp_path):
        plain = SfmrReconstruction.load(seoul_bull_sfmr_only)
        assert plain.focal_adjustments is None
        assert plain.principal_point_adjustments_xy is None
        assert plain.per_image_adjustments == [[] for _ in plain.cameras]

        recon = _adjusted(plain)
        assert recon.per_image_adjustments == [BOTH]
        assert recon.focal_adjustments[2] == FOCAL
        assert tuple(recon.principal_point_adjustments_xy[2]) == PRINCIPAL_POINT
        focals, principal_point = _expected(recon.cameras[0])
        image = recon.image_intrinsics(2)
        assert image.focal_lengths == pytest.approx(focals)
        assert image.principal_point == pytest.approx(principal_point)
        assert image == recon.cameras[0].with_adjustment(
            focal=FOCAL, principal_point=PRINCIPAL_POINT
        )

        # A file keeps them.
        out = tmp_path / "adjusted.sfmr"
        recon.save(out)
        loaded = SfmrReconstruction.load(out)
        assert loaded.image_intrinsics(2) == image
        assert loaded.per_image_adjustments == [BOTH]

        # Passing None clears them.
        cleared = recon.clone_with_changes(
            focal_adjustments=None,
            principal_point_adjustments_xy=None,
            per_image_adjustments=None,
        )
        assert cleared.focal_adjustments is None
        assert cleared.image_intrinsics(2) == cleared.cameras[0]

    def test_inspect_names_what_a_camera_adjusts(self, seoul_bull_sfmr_only, tmp_path):
        out = tmp_path / "adjusted.sfmr"
        _adjusted(SfmrReconstruction.load(seoul_bull_sfmr_only)).save(out)
        result = CliRunner().invoke(main, ["inspect", str(out), "--verbose"])
        assert result.exit_code == 0, result.output
        assert "Adjusted per image: focal, principal_point" in result.output


def _read_colmap(directory: Path) -> tuple:
    """The COLMAP model in ``directory``, and each image's camera id by name."""
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read_binary(str(directory))
    return reconstruction, {
        image.name: image.camera_id for image in reconstruction.images.values()
    }


class TestColmapExport:
    def test_an_adjusted_camera_is_written_once_per_image(
        self, seoul_bull_workspace, tmp_path
    ):
        from sfmtool.colmap.io import save_colmap_binary

        recon = _adjusted(
            SfmrReconstruction.load(seoul_bull_workspace), split_camera_at=9
        )
        save_colmap_binary(recon, tmp_path / "colmap")
        reconstruction, camera_of = _read_colmap(tmp_path / "colmap")

        # Camera 0, with image 2 adjusted, is one camera per image of it (9);
        # camera 1 has no adjusted image and is written once.
        assert len(reconstruction.cameras) == 10
        names = recon.image_names
        assert len({camera_of[n] for n in names[:9]}) == 9
        assert len({camera_of[n] for n in names[9:]}) == 1
        for i, name in enumerate(names):
            camera = reconstruction.cameras[camera_of[name]]
            expected = recon.image_intrinsics(i)
            assert camera.model.name == expected.model
            assert camera.focal_length == pytest.approx(expected.focal_lengths[0])
            assert (camera.principal_point_x, camera.principal_point_y) == (
                pytest.approx(expected.principal_point)
            )
        adjusted = reconstruction.cameras[camera_of[names[2]]]
        focals, _ = _expected(recon.cameras[0])
        assert adjusted.focal_length == pytest.approx(focals[0])

    def test_a_split_rig_sensor_camera_is_refused(self, tmp_path):
        from sfmtool.colmap.io import save_colmap_binary

        recon = SfmrReconstruction.load(KERRY_PARK_GROUND_TRUTH)
        assert recon.rig_frame_data is not None
        recon = _adjusted(recon, adjusted_images=(0,))
        camera = int(recon.camera_indexes[0])
        with pytest.raises(ValueError, match=f"Camera {camera} is the camera of a rig"):
            save_colmap_binary(recon, tmp_path / "colmap")
        assert not (tmp_path / "colmap").exists()

    def test_the_database_export_follows_the_same_rule(
        self, seoul_bull_workspace, tmp_path
    ):
        from sfmtool.colmap.db_export import create_colmap_db_from_reconstruction

        recon = _adjusted(
            SfmrReconstruction.load(seoul_bull_workspace), split_camera_at=9
        )
        db_path = tmp_path / "out.db"
        create_colmap_db_from_reconstruction(
            recon, db_path, populate_two_view_geometries=False
        )
        with pycolmap.Database.open(str(db_path)) as db:
            cameras = {c.camera_id: c for c in db.read_all_cameras()}
            camera_of = {
                Path(image.name).name: image.camera_id for image in db.read_all_images()
            }
        assert len(cameras) == 10
        for i, name in enumerate(recon.image_names):
            camera = cameras[camera_of[Path(name).name]]
            expected = recon.image_intrinsics(i)
            assert camera.focal_length == pytest.approx(expected.focal_lengths[0])
            assert (camera.principal_point_x, camera.principal_point_y) == (
                pytest.approx(expected.principal_point)
            )


class TestOtherWriters:
    def test_nerfstudio_writes_each_frames_intrinsics(self, seoul_bull_sfmr_only):
        from sfmtool._to_nerfstudio import build_transforms_json

        recon = _adjusted(SfmrReconstruction.load(seoul_bull_sfmr_only))
        basenames = [Path(n).name for n in recon.image_names]
        transforms = build_transforms_json(recon, basenames)

        # Nerfstudio reads an intrinsic from the top level or from every frame.
        assert "fl_x" not in transforms
        for i, frame in enumerate(transforms["frames"]):
            image = recon.image_intrinsics(i)
            assert (frame["fl_x"], frame["fl_y"]) == image.focal_lengths
            assert (frame["cx"], frame["cy"]) == image.principal_point
        focals, principal_point = _expected(recon.cameras[0])
        assert transforms["frames"][2]["fl_x"] == pytest.approx(focals[0])
        assert transforms["frames"][2]["cx"] == pytest.approx(principal_point[0])
        assert transforms["frames"][0]["fl_x"] == recon.cameras[0].focal_lengths[0]

    def test_camrig_cp_says_it_copies_the_shared_camera(
        self, seoul_bull_workspace, tmp_path
    ):
        source = seoul_bull_workspace.parent / "adjusted.sfmr"
        _adjusted(SfmrReconstruction.load(seoul_bull_workspace)).save(source)
        out = tmp_path / "cam.camrig"
        result = CliRunner().invoke(main, ["camrig", "cp", str(source), str(out)])
        assert result.exit_code == 0, result.output
        assert "camera 0 copied as the shared camera" in result.output
        assert "per-image intrinsic adjustments" in result.output

        plain = tmp_path / "plain.camrig"
        result = CliRunner().invoke(
            main, ["camrig", "cp", str(seoul_bull_workspace), str(plain)]
        )
        assert result.exit_code == 0, result.output
        assert "per-image intrinsic adjustments" not in result.output

    def test_densify_refuses_adjusted_images(self, seoul_bull_sfmr_only):
        from sfmtool._densify import densify_reconstruction

        recon = _adjusted(SfmrReconstruction.load(seoul_bull_sfmr_only))
        with pytest.raises(ValueError, match="per-image intrinsic adjustments"):
            densify_reconstruction(recon)


def _undistort(recon, output_dir):
    from sfmtool._undistort_images import undistort_reconstruction_images

    _, _, sfmr_path = undistort_reconstruction_images(
        recon=recon, output_dir=output_dir, resampling_filter="bilinear"
    )
    return SfmrReconstruction.load(Path(sfmr_path))


def _retake_with_adjustment(recon, i):
    """Rewrite image ``i``'s pixels and ``.sift`` keypoints in ``recon``'s
    workspace as if its camera had taken it with the test adjustment, and
    return the image's camera and adjusted intrinsics.

    Both go from the camera's intrinsics to the adjusted ones, so the image is
    one that the adjusted reconstruction describes exactly.
    """
    from sfmtool.fileio import read_image_rgb, write_image_rgb
    from sfmtool.flow import WarpMap
    from sfmtool.sift.file import get_sift_path_from_recon
    from sfmtool.fileio import SiftFile, write_sift

    camera = recon.cameras[int(recon.camera_indexes[i])]
    adjusted = camera.with_adjustment(focal=FOCAL, principal_point=PRINCIPAL_POINT)
    name = recon.image_names[i]

    image_path = Path(recon.workspace_dir) / name
    image = read_image_rgb(image_path)
    warp = WarpMap.from_cameras(src=camera, dst=adjusted)
    write_image_rgb(image_path, warp.remap_bilinear(image))

    sift_path = get_sift_path_from_recon(recon, name)
    with SiftFile(sift_path) as reader:
        feature_tool_metadata = reader.feature_tool_metadata
        metadata = reader.metadata
        positions = reader.read_positions().astype(np.float64)
        affine_shapes = reader.read_affine_shapes()
        descriptors = reader.read_descriptors()
        thumbnail = reader.read_thumbnail()
    positions = adjusted.project_batch(camera.unproject_batch(positions))
    write_sift(
        sift_path,
        {
            "feature_tool_metadata": feature_tool_metadata,
            "metadata": metadata,
            "positions_xy": positions.astype(np.float32),
            "affine_shapes": affine_shapes,
            "descriptors": descriptors,
            "thumbnail_y_x_rgb": thumbnail,
        },
    )
    return camera, adjusted


class TestUndistort:
    """Undistortion removes each image's adjustment along with its distortion:
    every image of a camera is written through the camera's one pinhole."""

    def test_the_output_has_no_adjustments(self, seoul_bull_workspace, tmp_path):
        recon = _adjusted(SfmrReconstruction.load(seoul_bull_workspace))
        out = _undistort(recon, tmp_path / "undistorted")

        assert out.per_image_adjustments == [[]]
        assert out.focal_adjustments is None
        assert out.principal_point_adjustments_xy is None
        assert out.image_intrinsics(2) == out.cameras[0]

        # Image 2's keypoints were read through its own intrinsics, which do
        # not describe them here (the image was not retaken), so each
        # observation stays as far from its point as it was: about the offset,
        # 2.7 px on this 270 px short side. Read through the shared camera they
        # would sit at the unadjusted error instead.
        before = np.nanmedian(recon.compute_observation_reprojection_errors(2)[:, 1])
        after = np.nanmedian(out.compute_observation_reprojection_errors(2)[:, 1])
        assert before > 1.0
        assert after == pytest.approx(before, abs=0.3)

    def test_an_adjusted_image_is_warped_into_the_shared_pinhole(
        self, seoul_bull_workspace, tmp_path
    ):
        from sfmtool.fileio import read_image_rgb
        from sfmtool.flow import WarpMap
        from sfmtool.sift.file import get_sift_path_from_recon
        from sfmtool.fileio import SiftFile

        plain = SfmrReconstruction.load(seoul_bull_workspace)
        plain_out = _undistort(plain, tmp_path / "plain")

        # Image 2 retaken with the adjustment, and a reconstruction that says
        # so: its undistortion should be the plain image's.
        camera, _ = _retake_with_adjustment(plain, 2)
        recon = _adjusted(plain)
        out = _undistort(recon, tmp_path / "adjusted")

        assert out.cameras == plain_out.cameras
        assert out.per_image_adjustments == [[]]
        assert out.focal_adjustments is None

        # Each keypoint of image 2 lands where the shared pinhole projects it,
        # which is where the plain run put it.
        name = recon.image_names[2]
        with SiftFile(get_sift_path_from_recon(out, name)) as reader:
            positions = reader.read_positions()
        with SiftFile(get_sift_path_from_recon(plain_out, name)) as reader:
            plain_positions = reader.read_positions()
        assert positions.shape == plain_positions.shape
        np.testing.assert_allclose(positions, plain_positions, atol=1e-3)

        # So every observation reprojects through its camera's pinhole as it
        # did in the plain run.
        def by_feature(errors):
            return errors[np.argsort(errors[:, 0], kind="stable")]

        for i in range(out.image_count):
            errors = by_feature(out.compute_observation_reprojection_errors(i))
            plain_errors = by_feature(
                plain_out.compute_observation_reprojection_errors(i)
            )
            np.testing.assert_allclose(errors, plain_errors, atol=1e-3)

        # Image 2's pixels match the plain run's, away from the border the
        # retake left black. Warping it as if it had no adjustment puts its
        # content a few pixels off.
        pixels = read_image_rgb(Path(out.workspace_dir) / name).astype(np.float64)
        expected = read_image_rgb(Path(plain_out.workspace_dir) / name).astype(
            np.float64
        )
        retaken = read_image_rgb(Path(recon.workspace_dir) / name)
        unadjusted_warp = WarpMap.from_cameras(src=camera, dst=out.cameras[0])
        misread = unadjusted_warp.remap_bilinear(retaken).astype(np.float64)
        h, w = pixels.shape[:2]
        inner = (slice(h // 8, h - h // 8), slice(w // 8, w - w // 8))
        difference = np.mean(np.abs(pixels[inner] - expected[inner]))
        misread_difference = np.mean(np.abs(misread[inner] - expected[inner]))
        # About 2.5 against 14 grey levels here.
        assert difference < 0.3 * misread_difference


class TestMerge:
    def test_merge_carries_each_images_adjustment(self, seoul_bull_workspace, tmp_path):
        from sfmtool.merge.reconstructions import merge_reconstructions
        from sfmtool.reconstruction import RangeExpr
        from sfmtool.xform import IncludeRangeFilter, apply_transforms

        original = _adjusted(
            SfmrReconstruction.load(seoul_bull_workspace), adjusted_images=(2, 12)
        )
        subsets = [
            apply_transforms(original, [IncludeRangeFilter(RangeExpr(r))])
            for r in ("1-10", "6-17")
        ]
        merged = merge_reconstructions(subsets)

        assert merged.per_image_adjustments == [BOTH]
        index_of = {Path(n).name: i for i, n in enumerate(original.image_names)}
        for i, name in enumerate(merged.image_names):
            j = index_of[Path(name).name]
            assert merged.focal_adjustments[i] == original.focal_adjustments[j]
            assert tuple(merged.principal_point_adjustments_xy[i]) == tuple(
                original.principal_point_adjustments_xy[j]
            )
        assert np.count_nonzero(merged.focal_adjustments) == 2

    def test_cameras_that_vary_different_parameters_are_not_merged(
        self, seoul_bull_sfmr_only
    ):
        from sfmtool.merge.reconstructions import merge_reconstructions

        plain = SfmrReconstruction.load(seoul_bull_sfmr_only)
        flagged = plain.clone_with_changes(per_image_adjustments=[["focal"]])
        with pytest.raises(ValueError, match="vary the same parameters per image"):
            merge_reconstructions([plain, flagged])


class TestBundleAdjust:
    def test_adjusted_images_go_through_sfmtools_adjustment(
        self, seoul_bull_sfmr_only, capsys
    ):
        from sfmtool.xform import BundleAdjustTransform

        recon = _adjusted(SfmrReconstruction.load(seoul_bull_sfmr_only))
        # SIMPLE_RADIAL is a model whose focal the adjustment cannot release.
        result = BundleAdjustTransform(release_focal=False).apply(recon)
        assert "per-image intrinsic adjustment" in capsys.readouterr().out
        # sfmtool's adjustment holds each image's adjustment and writes it back.
        assert result.per_image_adjustments == [BOTH]
        assert result.focal_adjustments.tobytes() == recon.focal_adjustments.tobytes()
        assert (
            result.principal_point_adjustments_xy.tobytes()
            == recon.principal_point_adjustments_xy.tobytes()
        )
