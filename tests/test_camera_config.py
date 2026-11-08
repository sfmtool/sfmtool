# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `camera/config.py` and the camera-config-aware paths in
`camera/setup.py`. End-to-end CLI coverage lives in `test_camera_config_cli.py`."""

import json
import shutil
from pathlib import Path

import pytest

from sfmtool.camera.config import (
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


def test_read_image_size_is_header_only(isolated_test_image: Path):
    # `_read_image_size` reads dimensions from the header alone (via the Rust
    # `image_dimensions` binding) — the seoul_bull_sculpture images are
    # 270x480.
    from sfmtool.camera.setup import _read_image_size

    assert _read_image_size(isolated_test_image) == (270, 480)


def test_image_dimensions_binding_rejects_non_image(tmp_path: Path):
    from sfmtool.fileio import image_dimensions

    bogus = tmp_path / "notes.txt"
    bogus.write_text("not an image")
    with pytest.raises(Exception):
        image_dimensions(str(bogus))


class TestBuildIntrinsicsFromCameraConfig:
    def test_none_config_calls_infer(self, isolated_test_image: Path):
        from sfmtool.camera.setup import build_intrinsics_from_camera_config

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
        from sfmtool.camera.setup import build_intrinsics_from_camera_config

        intrinsics, prior = build_intrinsics_from_camera_config(
            {"model": "OPENCV"}, isolated_test_image, camera_model_override=None
        )
        assert prior is False
        d = _intrinsics_to_dict(intrinsics)
        assert d["model"] == "OPENCV"

    def test_distortion_only_overlays(self, isolated_test_image: Path):
        from sfmtool.camera.setup import build_intrinsics_from_camera_config

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
        from sfmtool.fileio import read_image_rgb

        from sfmtool.camera.setup import build_intrinsics_from_camera_config

        img = read_image_rgb(isolated_test_image)
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
        from sfmtool.fileio import read_image_rgb

        from sfmtool.camera.setup import build_intrinsics_from_camera_config

        img = read_image_rgb(isolated_test_image)
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
        from sfmtool.fileio import read_image_rgb

        from sfmtool.camera.setup import build_intrinsics_from_camera_config

        img = read_image_rgb(isolated_test_image)
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

    def test_model_with_no_colmap_representation_raises(
        self, isolated_test_image: Path
    ):
        """A `.camrig` camera of an sfmtool model with no COLMAP carrier, such
        as `EQUIRECTANGULAR`, is a clean error naming the model."""
        from sfmtool.fileio import read_image_rgb

        from sfmtool.camera.setup import build_intrinsics_from_camera_config

        img = read_image_rgb(isolated_test_image)
        actual_h, actual_w = img.shape[:2]
        block = {
            "model": "EQUIRECTANGULAR",
            "width": actual_w,
            "height": actual_h,
            "parameters": {
                "focal_length_x": 100.0,
                "focal_length_y": 100.0,
                "principal_point_x": actual_w / 2.0,
                "principal_point_y": actual_h / 2.0,
            },
        }
        with pytest.raises(
            CameraConfigError, match="EQUIRECTANGULAR has no COLMAP representation"
        ):
            build_intrinsics_from_camera_config(
                block, isolated_test_image, camera_model_override=None
            )


# ---------------------------------------------------------------------------
# intrinsics_for_image with --camera-model: the camera_config.json camera is
# refit to the model
# ---------------------------------------------------------------------------


def _seoul_simple_radial_block() -> dict:
    """A SIMPLE_RADIAL calibration at seoul_bull_sculpture's 270x480."""
    return {
        "model": "SIMPLE_RADIAL",
        "width": 270,
        "height": 480,
        "parameters": {
            "focal_length": 344.0,
            "principal_point_x": 136.0,
            "principal_point_y": 241.0,
            "radial_distortion_k1": 0.035,
        },
    }


class TestIntrinsicsForImageCameraModel:
    def test_camera_config_is_refit_to_camera_model(
        self, isolated_test_image: Path, capsys
    ):
        from sfmtool.camera.setup import intrinsics_for_image

        config_path = isolated_test_image.parent / "camera_config.json"
        _write_config(
            config_path,
            {"version": 1, "camera_intrinsics": _seoul_simple_radial_block()},
        )
        resolver = CameraConfigResolver(isolated_test_image.parent)
        intrinsics, prior = intrinsics_for_image(
            isolated_test_image, resolver, "opencv"
        )

        d = intrinsics.to_dict()
        assert d["model"] == "OPENCV"
        assert (d["width"], d["height"]) == (270, 480)
        params = d["parameters"]
        # The refit keeps the file's focal and principal point and carries
        # its k1 over; nothing was inferred from EXIF.
        assert params["focal_length_x"] == pytest.approx(344.0, rel=1e-6)
        assert params["focal_length_y"] == pytest.approx(344.0, rel=1e-6)
        assert params["principal_point_x"] == 136.0
        assert params["principal_point_y"] == 241.0
        assert params["radial_distortion_k1"] == pytest.approx(0.035, rel=1e-4)
        assert prior is True

        out = capsys.readouterr().out
        assert f"The camera of {config_path}: SIMPLE_RADIAL -> OPENCV" in out
        assert "rms 0.000 px" in out

    def test_without_camera_model_the_file_is_used_as_is(
        self, isolated_test_image: Path, capsys
    ):
        from sfmtool.camera.setup import intrinsics_for_image

        _write_config(
            isolated_test_image.parent / "camera_config.json",
            {"version": 1, "camera_intrinsics": _seoul_simple_radial_block()},
        )
        resolver = CameraConfigResolver(isolated_test_image.parent)
        intrinsics, prior = intrinsics_for_image(isolated_test_image, resolver, None)

        assert intrinsics.to_dict() == _seoul_simple_radial_block()
        assert prior is True
        assert "->" not in capsys.readouterr().out

    def test_refused_refit_names_the_camera(self, isolated_test_image: Path):
        from sfmtool.camera.setup import intrinsics_for_image

        # An equidistant-like fisheye whose image corner is past 90 degrees,
        # which a perspective model cannot reach.
        block = {
            "model": "SIMPLE_RADIAL_FISHEYE",
            "width": 270,
            "height": 480,
            "parameters": {
                "focal_length": 100.0,
                "principal_point_x": 135.0,
                "principal_point_y": 240.0,
                "radial_distortion_k1": 0.0,
            },
        }
        config_path = isolated_test_image.parent / "camera_config.json"
        _write_config(config_path, {"version": 1, "camera_intrinsics": block})
        resolver = CameraConfigResolver(isolated_test_image.parent)
        with pytest.raises(ValueError) as excinfo:
            intrinsics_for_image(isolated_test_image, resolver, "PINHOLE")
        message = str(excinfo.value)
        assert f"cannot switch the camera of {config_path}" in message
        assert "SIMPLE_RADIAL_FISHEYE to PINHOLE" in message
        assert "90" in message
