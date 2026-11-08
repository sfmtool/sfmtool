# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CLI tests for camera-config-aware commands.

These exercise `camera_config.json` and `.camrig` intrinsics through the full
`sfm` CLI: that a configured calibration flows into the solved `.sfmr`, that
`--camera-model` refits the camera a `camera_config.json` or `.camrig`
supplies into the COLMAP database, and that a refused refit is a clean error.
The `camera/config.py` and `camera/setup.py` unit tests live in
`test_camera_config.py`.
"""

import json
from pathlib import Path

import numpy as np
import pytest


def _write_config(path: Path, body: dict) -> None:
    path.write_text(json.dumps(body))


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

    from sfmtool.reconstruction import SfmrReconstruction
    from sfmtool.cli import main

    workspace_dir = isolated_seoul_bull_17_images[0].parent

    runner = CliRunner()
    result = runner.invoke(main, ["ws", "init", str(workspace_dir)])
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


def _db_cameras(db_path: Path) -> list:
    import pycolmap

    with pycolmap.Database.open(db_path) as db:
        return list(db.read_all_cameras())


def test_to_colmap_db_refits_camera_config_camera(
    seoul_bull_workspace: Path, tmp_path: Path
):
    """`--camera-model` refits the camera_config.json camera (SIMPLE_RADIAL,
    focal 344, k1 0.035) to OPENCV, keeping its calibration."""
    from click.testing import CliRunner

    from sfmtool.cli import main

    workspace_dir = seoul_bull_workspace.parent
    matches_path = workspace_dir / "matches" / "recon-clusters.matches"
    out_db = tmp_path / "out.db"
    result = CliRunner().invoke(
        main,
        ["to-colmap-db", str(matches_path), str(out_db), "--camera-model", "OPENCV"],
    )
    assert result.exit_code == 0, result.output

    (camera,) = _db_cameras(out_db)
    assert camera.model.name == "OPENCV"
    fx, fy, cx, cy, k1, k2, p1, p2 = (float(v) for v in camera.params)
    assert fx == pytest.approx(344.0, rel=1e-6)
    assert fy == pytest.approx(344.0, rel=1e-6)
    assert (cx, cy) == (135.0, 240.0)
    assert k1 == pytest.approx(0.035, rel=1e-4)
    assert camera.has_prior_focal_length
    config_path = workspace_dir / "camera_config.json"
    assert f"The camera of {config_path}: SIMPLE_RADIAL -> OPENCV" in result.output


def test_to_colmap_db_refits_camrig_camera(seoul_bull_workspace: Path, tmp_path: Path):
    """`--camera-model` refits a covering single-sensor `.camrig`'s
    SIMPLE_PINHOLE camera to SIMPLE_RADIAL at k1 = 0, and leaves the file as
    it was."""
    from click.testing import CliRunner

    from sfmtool.cli import main
    from sfmtool.fileio import write_camrig

    workspace_dir = seoul_bull_workspace.parent
    camrig_file = workspace_dir / "test_17_image" / "cam.camrig"
    write_camrig(
        path=str(camrig_file),
        name="cam",
        rig_type="generic",
        cameras=[
            {
                "model": "SIMPLE_PINHOLE",
                "width": 270,
                "height": 480,
                "parameters": {
                    "focal_length": 330.0,
                    "principal_point_x": 135.0,
                    "principal_point_y": 240.0,
                },
            }
        ],
        sensor_image_patterns=["seoul_bull_sculpture_%02d.jpg"],
        camera_indexes=[0],
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        translations_xyz=np.zeros((1, 3)),
    )
    before = camrig_file.read_bytes()

    matches_path = workspace_dir / "matches" / "recon-clusters.matches"
    out_db = tmp_path / "out.db"
    result = CliRunner().invoke(
        main,
        [
            "to-colmap-db",
            str(matches_path),
            str(out_db),
            "--camera-model",
            "SIMPLE_RADIAL",
        ],
    )
    assert result.exit_code == 0, result.output

    (camera,) = _db_cameras(out_db)
    assert camera.model.name == "SIMPLE_RADIAL"
    assert [float(v) for v in camera.params] == [330.0, 135.0, 240.0, 0.0]
    assert camera.has_prior_focal_length
    assert camrig_file.read_bytes() == before
    assert (
        f"Camera 0 of {camrig_file.resolve()}: SIMPLE_PINHOLE -> SIMPLE_RADIAL"
        in result.output
    )


def test_solve_refused_refit_is_a_clean_error(isolated_seoul_bull_17_images):
    """A `.camrig` camera that `--camera-model` cannot fit (an equidistant
    fisheye reaching past 90 degrees, as a perspective model) stops the solve
    with an error naming the camera, before any feature work."""
    from click.testing import CliRunner

    from sfmtool.cli import main
    from sfmtool.fileio import write_camrig

    workspace_dir = isolated_seoul_bull_17_images[0].parent
    runner = CliRunner()
    result = runner.invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    camrig_file = workspace_dir / "cam.camrig"
    write_camrig(
        path=str(camrig_file),
        name="cam",
        rig_type="generic",
        cameras=[
            {
                "model": "EQUIDISTANT_FISHEYE",
                "width": 270,
                "height": 480,
                "parameters": {
                    "focal_length": 150.0,
                    "principal_point_x": 135.0,
                    "principal_point_y": 240.0,
                },
            }
        ],
        sensor_image_patterns=["seoul_bull_sculpture_%02d.jpg"],
        camera_indexes=[0],
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        translations_xyz=np.zeros((1, 3)),
    )

    result = runner.invoke(
        main,
        ["solve", "-g", "--camera-model", "PINHOLE", str(workspace_dir)],
    )
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert (
        f"cannot switch camera 0 of {camrig_file.resolve()} from "
        "EQUIDISTANT_FISHEYE to PINHOLE"
    ) in result.output
    assert "Traceback" not in result.output
    assert not list(workspace_dir.rglob("*.sift"))
