# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CLI tests for camera-config-aware commands.

These exercise `camera_config.json` through the full `sfm` CLI — that a
configured calibration flows into the solved `.sfmr`, and that
`--camera-model` is rejected up front by every command that resolves a
camera config (`solve`, `match`, `to-colmap-db`). The `camera/config.py`
and `camera/setup.py` unit tests live in `test_camera_config.py`.
"""

import json
from pathlib import Path


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

    from sfmtool._sfmtool.reconstruction import SfmrReconstruction
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


def test_solve_rejects_camera_model_with_camera_config(isolated_seoul_bull_17_images):
    from click.testing import CliRunner

    from sfmtool.cli import main

    workspace_dir = isolated_seoul_bull_17_images[0].parent

    runner = CliRunner()
    result = runner.invoke(main, ["ws", "init", str(workspace_dir)])
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
    result = runner.invoke(main, ["ws", "init", str(workspace_dir)])
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
    result = runner.invoke(main, ["ws", "init", str(workspace_dir)])
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
            str(out_db),
            "--camera-model",
            "PINHOLE",
        ],
    )
    assert result.exit_code != 0
    assert "--camera-model cannot be used" in result.output
    assert not out_db.exists()


def test_solve_from_matches_rejects_camera_model_with_camera_config(
    isolated_seoul_bull_17_images, tmp_path: Path
):
    """Counterpart to test_solve_rejects_camera_model_with_camera_config for the
    `.matches` input branch: --camera-model must also be rejected when a
    camera_config resolves for an image referenced by the matches.

    The check fires inside `_setup_for_sfm_from_matches` (db_setup.py); this
    test pins that the chain still raises a usage-error-shaped click error
    before the SfM run actually produces output."""
    from click.testing import CliRunner

    from sfmtool.cli import main

    workspace_dir = isolated_seoul_bull_17_images[0].parent

    runner = CliRunner()
    result = runner.invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    matches_path = workspace_dir / "solve_in.matches"
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

    _write_config(
        workspace_dir / "camera_config.json",
        {"version": 1, "camera_intrinsics": _seoul_bull_native_intrinsics()},
    )

    out_sfmr = tmp_path / "out.sfmr"
    result = runner.invoke(
        main,
        [
            "solve",
            "-i",
            "--camera-model",
            "PINHOLE",
            "--output",
            str(out_sfmr),
            str(matches_path),
        ],
    )
    assert result.exit_code != 0
    assert "--camera-model cannot be used" in result.output
    assert not out_sfmr.exists()
