# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `_camera_config.py` and the camera-config-aware paths in
`_camera_setup.py`."""

import json
import shutil
from pathlib import Path

import pytest

from sfmtool._camera_config import (
    CameraConfigError,
    CameraConfigResolver,
    find_camera_config_for_directory,
    load_camera_config,
)

TEST_DATA_DIR = Path(__file__).parent.parent / "test-data"
TEST_IMAGE = (
    TEST_DATA_DIR / "images" / "seoul_bull_sculpture" / "seoul_bull_sculpture_01.jpg"
)


def _write_config(path: Path, body: dict) -> None:
    path.write_text(json.dumps(body))


def _full_opencv_block(width: int, height: int) -> dict:
    return {
        "model": "OPENCV",
        "width": width,
        "height": height,
        "parameters": {
            "focal_length_x": 100.0,
            "focal_length_y": 101.0,
            "principal_point_x": width / 2,
            "principal_point_y": height / 2,
            "radial_distortion_k1": -0.01,
            "radial_distortion_k2": 0.005,
            "tangential_distortion_p1": 0.0001,
            "tangential_distortion_p2": -0.0002,
        },
    }


# ---------------------------------------------------------------------------
# load_camera_config
# ---------------------------------------------------------------------------


class TestLoadCameraConfig:
    def test_full_block_roundtrip(self, tmp_path: Path):
        cfg_path = tmp_path / "camera_config.json"
        block = _full_opencv_block(100, 80)
        _write_config(cfg_path, {"version": 1, "camera_intrinsics": block})
        result = load_camera_config(cfg_path)
        assert result == block

    def test_model_only(self, tmp_path: Path):
        cfg_path = tmp_path / "camera_config.json"
        _write_config(
            cfg_path,
            {"version": 1, "camera_intrinsics": {"model": "OPENCV_FISHEYE"}},
        )
        result = load_camera_config(cfg_path)
        assert result == {"model": "OPENCV_FISHEYE"}

    def test_distortion_only_no_size_required(self, tmp_path: Path):
        cfg_path = tmp_path / "camera_config.json"
        block = {
            "model": "OPENCV",
            "parameters": {
                "radial_distortion_k1": -0.02,
                "radial_distortion_k2": 0.01,
            },
        }
        _write_config(cfg_path, {"version": 1, "camera_intrinsics": block})
        result = load_camera_config(cfg_path)
        assert result == block

    def test_missing_intrinsics_returns_none(self, tmp_path: Path):
        cfg_path = tmp_path / "camera_config.json"
        _write_config(cfg_path, {"version": 1})
        assert load_camera_config(cfg_path) is None

    def test_bad_version(self, tmp_path: Path):
        cfg_path = tmp_path / "camera_config.json"
        _write_config(
            cfg_path, {"version": 2, "camera_intrinsics": {"model": "PINHOLE"}}
        )
        with pytest.raises(CameraConfigError, match="version"):
            load_camera_config(cfg_path)

    def test_invalid_json(self, tmp_path: Path):
        cfg_path = tmp_path / "camera_config.json"
        cfg_path.write_text("{not json")
        with pytest.raises(CameraConfigError, match="Invalid JSON"):
            load_camera_config(cfg_path)

    def test_unknown_top_level_key(self, tmp_path: Path):
        cfg_path = tmp_path / "camera_config.json"
        _write_config(
            cfg_path,
            {"version": 1, "camera_intrinsics": {"model": "PINHOLE"}, "extra": True},
        )
        with pytest.raises(CameraConfigError, match="unknown top-level keys"):
            load_camera_config(cfg_path)

    def test_unknown_intrinsics_key(self, tmp_path: Path):
        cfg_path = tmp_path / "camera_config.json"
        _write_config(
            cfg_path,
            {
                "version": 1,
                "camera_intrinsics": {"model": "PINHOLE", "frob": 1},
            },
        )
        with pytest.raises(
            CameraConfigError, match="unknown keys in 'camera_intrinsics'"
        ):
            load_camera_config(cfg_path)

    def test_unknown_model(self, tmp_path: Path):
        cfg_path = tmp_path / "camera_config.json"
        _write_config(
            cfg_path,
            {"version": 1, "camera_intrinsics": {"model": "BOGUS_MODEL"}},
        )
        with pytest.raises(CameraConfigError, match="unknown camera model"):
            load_camera_config(cfg_path)

    def test_missing_model(self, tmp_path: Path):
        cfg_path = tmp_path / "camera_config.json"
        _write_config(
            cfg_path,
            {"version": 1, "camera_intrinsics": {"width": 10, "height": 10}},
        )
        with pytest.raises(
            CameraConfigError, match="'camera_intrinsics.model' is required"
        ):
            load_camera_config(cfg_path)

    def test_missing_size_with_focal(self, tmp_path: Path):
        cfg_path = tmp_path / "camera_config.json"
        _write_config(
            cfg_path,
            {
                "version": 1,
                "camera_intrinsics": {
                    "model": "PINHOLE",
                    "parameters": {"focal_length_x": 100.0},
                },
            },
        )
        with pytest.raises(
            CameraConfigError, match="'width' and 'height' are required"
        ):
            load_camera_config(cfg_path)

    def test_unknown_parameter_for_model(self, tmp_path: Path):
        cfg_path = tmp_path / "camera_config.json"
        _write_config(
            cfg_path,
            {
                "version": 1,
                "camera_intrinsics": {
                    "model": "PINHOLE",
                    "width": 10,
                    "height": 10,
                    "parameters": {"focal_length_x": 1.0, "wat": 2.0},
                },
            },
        )
        with pytest.raises(CameraConfigError, match="unknown parameters"):
            load_camera_config(cfg_path)

    def test_width_without_height_rejected(self, tmp_path: Path):
        cfg_path = tmp_path / "camera_config.json"
        _write_config(
            cfg_path,
            {
                "version": 1,
                "camera_intrinsics": {"model": "PINHOLE", "width": 10},
            },
        )
        with pytest.raises(
            CameraConfigError, match="'width' and 'height' must be set together"
        ):
            load_camera_config(cfg_path)


# ---------------------------------------------------------------------------
# find_camera_config_for_directory
# ---------------------------------------------------------------------------


class TestFindCameraConfigForDirectory:
    def test_at_workspace_root(self, tmp_path: Path):
        block = _full_opencv_block(100, 80)
        _write_config(
            tmp_path / "camera_config.json", {"version": 1, "camera_intrinsics": block}
        )
        result = find_camera_config_for_directory(tmp_path, tmp_path)
        assert result is not None
        path, intrinsics = result
        assert path == tmp_path / "camera_config.json"
        assert intrinsics == block

    def test_walks_up_one_level(self, tmp_path: Path):
        block = _full_opencv_block(100, 80)
        _write_config(
            tmp_path / "camera_config.json", {"version": 1, "camera_intrinsics": block}
        )
        sub = tmp_path / "photos"
        sub.mkdir()
        result = find_camera_config_for_directory(sub, tmp_path)
        assert result is not None
        assert result[0] == tmp_path / "camera_config.json"

    def test_walks_up_two_levels(self, tmp_path: Path):
        block = _full_opencv_block(100, 80)
        _write_config(
            tmp_path / "camera_config.json", {"version": 1, "camera_intrinsics": block}
        )
        sub = tmp_path / "photos" / "set_a"
        sub.mkdir(parents=True)
        result = find_camera_config_for_directory(sub, tmp_path)
        assert result is not None
        assert result[0] == tmp_path / "camera_config.json"

    def test_nested_override_wins(self, tmp_path: Path):
        outer = _full_opencv_block(100, 80)
        inner = _full_opencv_block(200, 160)
        _write_config(
            tmp_path / "camera_config.json", {"version": 1, "camera_intrinsics": outer}
        )
        sub = tmp_path / "gopro"
        sub.mkdir()
        _write_config(
            sub / "camera_config.json", {"version": 1, "camera_intrinsics": inner}
        )
        result = find_camera_config_for_directory(sub, tmp_path)
        assert result is not None
        path, intrinsics = result
        assert path == sub / "camera_config.json"
        assert intrinsics["width"] == 200

    def test_no_config_returns_none(self, tmp_path: Path):
        sub = tmp_path / "photos"
        sub.mkdir()
        assert find_camera_config_for_directory(sub, tmp_path) is None

    def test_outside_workspace_returns_none(self, tmp_path: Path):
        ws = tmp_path / "ws"
        ws.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        block = _full_opencv_block(100, 80)
        _write_config(
            elsewhere / "camera_config.json", {"version": 1, "camera_intrinsics": block}
        )
        # Even though there's a config in `elsewhere`, it shouldn't be returned
        # for a directory outside the workspace.
        assert find_camera_config_for_directory(elsewhere, ws) is None


# ---------------------------------------------------------------------------
# CameraConfigResolver
# ---------------------------------------------------------------------------


class TestCameraConfigResolver:
    def test_caches_by_directory(self, tmp_path: Path):
        block = _full_opencv_block(100, 80)
        _write_config(
            tmp_path / "camera_config.json", {"version": 1, "camera_intrinsics": block}
        )
        sub = tmp_path / "photos"
        sub.mkdir()

        resolver = CameraConfigResolver(tmp_path)
        r1 = resolver.resolve_for_directory(sub)
        # Mutate disk to prove cache is being used
        (tmp_path / "camera_config.json").unlink()
        r2 = resolver.resolve_for_directory(sub)
        assert r1 == r2

    def test_resolve_for_image(self, tmp_path: Path):
        block = _full_opencv_block(100, 80)
        _write_config(
            tmp_path / "camera_config.json", {"version": 1, "camera_intrinsics": block}
        )
        img = tmp_path / "img.jpg"
        img.touch()
        resolver = CameraConfigResolver(tmp_path)
        result = resolver.resolve_for_image(img)
        assert result is not None
        assert result[1] == block

    def test_no_workspace(self, tmp_path: Path):
        resolver = CameraConfigResolver(None)
        assert resolver.resolve_for_directory(tmp_path) is None

    def test_distinct_dirs_resolve_independently(self, tmp_path: Path):
        outer = _full_opencv_block(100, 80)
        inner = _full_opencv_block(200, 160)
        _write_config(
            tmp_path / "camera_config.json", {"version": 1, "camera_intrinsics": outer}
        )
        gopro = tmp_path / "gopro"
        nikon = tmp_path / "nikon"
        gopro.mkdir()
        nikon.mkdir()
        _write_config(
            gopro / "camera_config.json", {"version": 1, "camera_intrinsics": inner}
        )

        resolver = CameraConfigResolver(tmp_path)
        gopro_r = resolver.resolve_for_directory(gopro)
        nikon_r = resolver.resolve_for_directory(nikon)
        assert gopro_r is not None
        assert nikon_r is not None
        assert gopro_r[1]["width"] == 200
        assert nikon_r[1]["width"] == 100


# ---------------------------------------------------------------------------
# build_intrinsics_from_camera_config — Phase 2
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_test_image(tmp_path: Path) -> Path:
    """Copy a test image into an isolated tmp_path."""
    dest = tmp_path / "img.jpg"
    shutil.copy(TEST_IMAGE, dest)
    return dest


def _intrinsics_to_dict(intrinsics) -> dict:
    return intrinsics.to_dict()


class TestBuildIntrinsicsFromCameraConfig:
    def test_none_config_calls_infer(self, isolated_test_image: Path):
        from sfmtool._camera_setup import build_intrinsics_from_camera_config

        intrinsics, prior = build_intrinsics_from_camera_config(
            None, isolated_test_image, camera_model_override=None
        )
        assert prior is False
        d = _intrinsics_to_dict(intrinsics)
        # EXIF inference produces a real model name
        assert d["model"] in {
            "SIMPLE_RADIAL",
            "SIMPLE_PINHOLE",
            "PINHOLE",
            "OPENCV",
            "RADIAL",
        }

    def test_model_only_overrides_model(self, isolated_test_image: Path):
        from sfmtool._camera_setup import build_intrinsics_from_camera_config

        intrinsics, prior = build_intrinsics_from_camera_config(
            {"model": "OPENCV"}, isolated_test_image, camera_model_override=None
        )
        assert prior is False
        d = _intrinsics_to_dict(intrinsics)
        assert d["model"] == "OPENCV"

    def test_distortion_only_overlays(self, isolated_test_image: Path):
        from sfmtool._camera_setup import build_intrinsics_from_camera_config

        block = {
            "model": "OPENCV",
            "parameters": {
                "radial_distortion_k1": -0.02,
                "radial_distortion_k2": 0.01,
            },
        }
        intrinsics, prior = build_intrinsics_from_camera_config(
            block, isolated_test_image, camera_model_override=None
        )
        assert prior is False
        d = _intrinsics_to_dict(intrinsics)
        assert d["model"] == "OPENCV"
        assert d["parameters"]["radial_distortion_k1"] == pytest.approx(-0.02)
        assert d["parameters"]["radial_distortion_k2"] == pytest.approx(0.01)
        # Unspecified distortion coefs zero out
        assert d["parameters"]["tangential_distortion_p1"] == 0.0
        assert d["parameters"]["tangential_distortion_p2"] == 0.0

    def test_full_block_at_native_resolution(self, isolated_test_image: Path):
        import cv2

        from sfmtool._camera_setup import build_intrinsics_from_camera_config

        img = cv2.imread(
            str(isolated_test_image),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        actual_h, actual_w = img.shape[:2]

        block = {
            "model": "OPENCV",
            "width": actual_w,
            "height": actual_h,
            "parameters": {
                "focal_length_x": 314.0,
                "focal_length_y": 313.0,
                "principal_point_x": actual_w / 2.0,
                "principal_point_y": actual_h / 2.0,
                "radial_distortion_k1": -0.02,
                "radial_distortion_k2": 0.01,
                "tangential_distortion_p1": 0.0,
                "tangential_distortion_p2": 0.0,
            },
        }
        intrinsics, prior = build_intrinsics_from_camera_config(
            block, isolated_test_image, camera_model_override=None
        )
        assert prior is True
        d = _intrinsics_to_dict(intrinsics)
        assert d["model"] == "OPENCV"
        assert d["width"] == actual_w
        assert d["height"] == actual_h
        assert d["parameters"]["focal_length_x"] == pytest.approx(314.0)
        assert d["parameters"]["focal_length_y"] == pytest.approx(313.0)

    def test_full_block_uniform_downscale(self, isolated_test_image: Path):
        import cv2

        from sfmtool._camera_setup import build_intrinsics_from_camera_config

        img = cv2.imread(
            str(isolated_test_image),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        actual_h, actual_w = img.shape[:2]

        # Calibration at 4x the actual size
        calib_w = actual_w * 4
        calib_h = actual_h * 4
        block = {
            "model": "OPENCV",
            "width": calib_w,
            "height": calib_h,
            "parameters": {
                "focal_length_x": 1200.0,
                "focal_length_y": 1196.0,
                "principal_point_x": calib_w / 2.0,
                "principal_point_y": calib_h / 2.0,
                "radial_distortion_k1": -0.02,
                "radial_distortion_k2": 0.01,
                "tangential_distortion_p1": 0.0,
                "tangential_distortion_p2": 0.0,
            },
        }
        intrinsics, prior = build_intrinsics_from_camera_config(
            block, isolated_test_image, camera_model_override=None
        )
        assert prior is True
        d = _intrinsics_to_dict(intrinsics)
        # Scale s = actual_w / calib_w = 1/4
        assert d["parameters"]["focal_length_x"] == pytest.approx(300.0)
        assert d["parameters"]["focal_length_y"] == pytest.approx(299.0)
        assert d["parameters"]["principal_point_x"] == pytest.approx(actual_w / 2.0)
        assert d["parameters"]["principal_point_y"] == pytest.approx(actual_h / 2.0)
        # Distortion unchanged
        assert d["parameters"]["radial_distortion_k1"] == pytest.approx(-0.02)
        assert d["parameters"]["radial_distortion_k2"] == pytest.approx(0.01)

    def test_aspect_mismatch_raises(self, isolated_test_image: Path):
        import cv2

        from sfmtool._camera_setup import build_intrinsics_from_camera_config

        img = cv2.imread(
            str(isolated_test_image),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        actual_h, actual_w = img.shape[:2]

        # Calibrated with different aspect
        calib_w = actual_h * 2
        calib_h = actual_w * 2
        block = {
            "model": "OPENCV",
            "width": calib_w,
            "height": calib_h,
            "parameters": {
                "focal_length_x": 100.0,
                "focal_length_y": 100.0,
                "principal_point_x": calib_w / 2.0,
                "principal_point_y": calib_h / 2.0,
                "radial_distortion_k1": 0.0,
                "radial_distortion_k2": 0.0,
                "tangential_distortion_p1": 0.0,
                "tangential_distortion_p2": 0.0,
            },
        }
        with pytest.raises(CameraConfigError, match="aspect"):
            build_intrinsics_from_camera_config(
                block, isolated_test_image, camera_model_override=None
            )


# ---------------------------------------------------------------------------
# _check_camera_model_conflict — Phase 4
# ---------------------------------------------------------------------------


class TestCheckCameraModelConflict:
    def test_no_camera_model_no_conflict(self, tmp_path: Path):
        from sfmtool._camera_setup import _check_camera_model_conflict

        block = _full_opencv_block(100, 80)
        _write_config(
            tmp_path / "camera_config.json",
            {"version": 1, "camera_intrinsics": block},
        )
        img = tmp_path / "img.jpg"
        img.touch()
        resolver = CameraConfigResolver(tmp_path)
        # Should not raise
        _check_camera_model_conflict([img], resolver, None)

    def test_no_camera_config_no_conflict(self, tmp_path: Path):
        from sfmtool._camera_setup import _check_camera_model_conflict

        img = tmp_path / "img.jpg"
        img.touch()
        resolver = CameraConfigResolver(tmp_path)
        # Should not raise — no camera_config.json present
        _check_camera_model_conflict([img], resolver, "OPENCV")

    def test_conflict_raises(self, tmp_path: Path):
        import click

        from sfmtool._camera_setup import _check_camera_model_conflict

        block = _full_opencv_block(100, 80)
        _write_config(
            tmp_path / "camera_config.json",
            {"version": 1, "camera_intrinsics": block},
        )
        img = tmp_path / "img.jpg"
        img.touch()
        resolver = CameraConfigResolver(tmp_path)
        with pytest.raises(click.UsageError, match="--camera-model cannot be used"):
            _check_camera_model_conflict([img], resolver, "OPENCV")

    def test_no_resolver_no_conflict(self, tmp_path: Path):
        from sfmtool._camera_setup import _check_camera_model_conflict

        img = tmp_path / "img.jpg"
        img.touch()
        # No resolver (e.g., no workspace) — flag works as before
        _check_camera_model_conflict([img], None, "OPENCV")


# ---------------------------------------------------------------------------
# CLI integration — Phase 3 + 4
# ---------------------------------------------------------------------------


def _seoul_bull_native_intrinsics() -> dict:
    """A full OPENCV calibration at the seoul_bull_sculpture native resolution."""
    return {
        "model": "OPENCV",
        "width": 270,
        "height": 480,
        "parameters": {
            "focal_length_x": 350.5,
            "focal_length_y": 351.5,
            "principal_point_x": 135.0,
            "principal_point_y": 240.0,
            "radial_distortion_k1": -0.05,
            "radial_distortion_k2": 0.02,
            "tangential_distortion_p1": 0.0,
            "tangential_distortion_p2": 0.0,
        },
    }


def test_solve_uses_camera_config(isolated_seoul_bull_17_images):
    """End-to-end: drop a camera_config.json, run sfm solve, verify intrinsics
    in the resulting .sfmr."""
    from click.testing import CliRunner

    from sfmtool._sfmtool import SfmrReconstruction
    from sfmtool.cli import main

    workspace_dir = isolated_seoul_bull_17_images[0].parent

    runner = CliRunner()
    result = runner.invoke(main, ["init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    block = _seoul_bull_native_intrinsics()
    _write_config(
        workspace_dir / "camera_config.json",
        {"version": 1, "camera_intrinsics": block},
    )

    result = runner.invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    output_path = workspace_dir / "test_solve_with_camera_config.sfmr"
    result = runner.invoke(
        main,
        ["solve", "-i", "--output", str(output_path), str(workspace_dir)],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()

    recon = SfmrReconstruction.load(output_path)
    assert recon.camera_count >= 1
    cam = recon.cameras[0]
    cam_dict = cam.to_dict()
    # Intrinsics start from the configured values; bundle adjustment may
    # refine them, but the model + dimensions must match exactly.
    assert cam_dict["model"] == "OPENCV"
    assert cam_dict["width"] == 270
    assert cam_dict["height"] == 480


def test_solve_rejects_camera_model_with_camera_config(isolated_seoul_bull_17_images):
    from click.testing import CliRunner

    from sfmtool.cli import main

    workspace_dir = isolated_seoul_bull_17_images[0].parent

    runner = CliRunner()
    result = runner.invoke(main, ["init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    _write_config(
        workspace_dir / "camera_config.json",
        {"version": 1, "camera_intrinsics": _seoul_bull_native_intrinsics()},
    )

    result = runner.invoke(
        main,
        [
            "solve",
            "-i",
            "--camera-model",
            "PINHOLE",
            "--output",
            str(workspace_dir / "out.sfmr"),
            str(workspace_dir),
        ],
    )
    assert result.exit_code != 0
    assert "--camera-model cannot be used" in result.output
    # Must fail before any expensive work begins.
    assert not (workspace_dir / "out.sfmr").exists()


def test_match_rejects_camera_model_with_camera_config(isolated_seoul_bull_17_images):
    from click.testing import CliRunner

    from sfmtool.cli import main

    workspace_dir = isolated_seoul_bull_17_images[0].parent

    runner = CliRunner()
    result = runner.invoke(main, ["init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    _write_config(
        workspace_dir / "camera_config.json",
        {"version": 1, "camera_intrinsics": _seoul_bull_native_intrinsics()},
    )

    result = runner.invoke(
        main,
        [
            "match",
            "--exhaustive",
            "--camera-model",
            "PINHOLE",
            str(workspace_dir),
        ],
    )
    assert result.exit_code != 0
    assert "--camera-model cannot be used" in result.output


def test_to_colmap_db_rejects_camera_model_with_camera_config(
    isolated_seoul_bull_17_images, tmp_path: Path
):
    """For .matches input, --camera-model must be rejected when a camera_config
    resolves for any image referenced by the matches."""
    from click.testing import CliRunner

    from sfmtool.cli import main

    workspace_dir = isolated_seoul_bull_17_images[0].parent

    runner = CliRunner()
    result = runner.invoke(main, ["init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Produce a .matches file (no camera_config yet, so this should succeed)
    matches_path = workspace_dir / "test.matches"
    result = runner.invoke(
        main,
        [
            "match",
            "--exhaustive",
            "--output",
            str(matches_path),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert matches_path.exists()

    # Now drop a camera_config.json and invoke to-colmap-db with --camera-model
    _write_config(
        workspace_dir / "camera_config.json",
        {"version": 1, "camera_intrinsics": _seoul_bull_native_intrinsics()},
    )

    out_db = tmp_path / "out.db"
    result = runner.invoke(
        main,
        [
            "to-colmap-db",
            str(matches_path),
            "--out-db",
            str(out_db),
            "--camera-model",
            "PINHOLE",
        ],
    )
    assert result.exit_code != 0
    assert "--camera-model cannot be used" in result.output
    assert not out_db.exists()
