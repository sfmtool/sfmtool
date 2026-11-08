# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Fisheye rig tests built on the kerry_park dataset.

The kerry_park dataset is a 24-frame back-to-back OPENCV_FISHEYE stereo
rig captured at 480x480, with a calibrated ``rig_config.json`` shipped
alongside the images. These tests exercise the parts of sfmtool that
diverge from the rectilinear / single-camera case: rig pose loading,
fisheye-aware focal-length inference, and a full global SfM solve.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sfmtool.reconstruction import SfmrReconstruction

# --------------------------------------------------------------------------- #
# Cheap fixture-based tests: no SfM solve required.
# --------------------------------------------------------------------------- #


def test_rig_config_parses_back_to_back_fisheye(isolated_kerry_park_rig: Path) -> None:
    """The kerry_park rig_config.json should declare two OPENCV_FISHEYE sensors
    in COLMAP rig_configurator format, with the right sensor flipped 180° about Y.
    """
    rig_config_path = isolated_kerry_park_rig / "rig_config.json"
    rig_configs = json.loads(rig_config_path.read_text())

    assert len(rig_configs) == 1
    rig = rig_configs[0]
    # COLMAP format: no rig-level intrinsics block; each sensor is calibrated.
    assert "camera_intrinsics" not in rig

    cameras = rig["cameras"]
    assert len(cameras) == 2
    left, right = cameras
    # Both sensors carry the calibrated OPENCV_FISHEYE intrinsics inline:
    # camera_params order is fx, fy, cx, cy, k1, k2, k3, k4.
    for cam in cameras:
        assert cam["camera_model_name"] == "OPENCV_FISHEYE"
        assert len(cam["camera_params"]) == 8

    assert left["image_prefix"] == "fisheye_left/"
    assert left["ref_sensor"] is True
    # The reference sensor must not carry an explicit pose.
    assert "cam_from_rig_rotation" not in left

    assert right["image_prefix"] == "fisheye_right/"
    assert right.get("ref_sensor", False) is False
    # 180° about Y in WXYZ is [0, 0, 1, 0]. Translation is a ~3 cm baseline
    # along the rig's -Z direction.
    assert right["cam_from_rig_rotation"] == [0, 0, 1, 0]
    tx, ty, tz = right["cam_from_rig_translation"]
    assert (tx, ty) == (0, 0)
    assert abs(tz) == pytest.approx(0.0307, abs=1e-4)


def test_sensor_from_rig_pose_wxyz_to_xyzw() -> None:
    """``_sensor_from_rig_pose`` must convert WXYZ → XYZW for pycolmap.

    This guards the convention documented in
    ``specs/workspace/rig-config.md``.
    """
    pycolmap = pytest.importorskip("pycolmap")
    from sfmtool.rig.config import _sensor_from_rig_pose

    # WXYZ = [0, 0, 1, 0] is a 180° rotation about Y.
    cam_cfg = {
        "cam_from_rig_rotation": [0.0, 0.0, 1.0, 0.0],
        "cam_from_rig_translation": [0.0, 0.0, -0.0307],
    }
    pose = _sensor_from_rig_pose(cam_cfg)
    assert isinstance(pose, pycolmap.Rigid3d)

    # Apply to +Z (the optical axis). 180°-about-Y maps (0, 0, 1) -> (0, 0, -1)
    # plus the translation.
    out = pose * np.array([0.0, 0.0, 1.0])
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    assert out[1] == pytest.approx(0.0, abs=1e-9)
    assert out[2] == pytest.approx(-1.0 - 0.0307, abs=1e-9)


def test_sensor_from_rig_pose_missing_rotation_returns_none() -> None:
    """Reference sensors (no rotation field) should yield ``None``."""
    from sfmtool.rig.config import _sensor_from_rig_pose

    assert _sensor_from_rig_pose({"ref_sensor": True}) is None


def test_fisheye_focal_length_inference(isolated_kerry_park_rig: Path) -> None:
    """``_infer_camera`` should seed OPENCV_FISHEYE focal length to ~size/π,
    not the EXIF-derived 1.2× image-size guess used for rectilinear lenses.
    """
    from sfmtool.camera.setup import _infer_camera

    image = isolated_kerry_park_rig / "fisheye_left" / "frame_01.jpg"
    cam = _infer_camera(str(image), "OPENCV_FISHEYE")

    assert cam.model.name == "OPENCV_FISHEYE"
    assert cam.width == 480 and cam.height == 480
    fx, fy = float(cam.params[0]), float(cam.params[1])
    expected = 480 / math.pi
    assert fx == pytest.approx(expected, rel=1e-4)
    assert fy == pytest.approx(expected, rel=1e-4)
    # And, crucially, far away from the rectilinear default of ~576.
    assert fx < 200


def test_rig_intrinsics_used_directly_when_fully_specified(
    isolated_kerry_park_rig: Path,
) -> None:
    """When a rig sensor entry carries ``camera_model_name`` + ``camera_params``,
    ``_camera_from_sensor_entry`` should use those parameters directly rather
    than falling back to pycolmap inference.
    """
    from sfmtool.colmap.db_builders import _camera_from_sensor_entry

    rig_config = json.loads((isolated_kerry_park_rig / "rig_config.json").read_text())[
        0
    ]
    sensor = rig_config["cameras"][0]
    image = isolated_kerry_park_rig / "fisheye_left" / "frame_01.jpg"

    cam = _camera_from_sensor_entry(sensor, image, camera_model_override=None)

    assert cam.model.name == "OPENCV_FISHEYE"
    assert cam.width == 480 and cam.height == 480
    assert cam.has_prior_focal_length is True

    expected_focal_x = sensor["camera_params"][0]
    assert float(cam.params[0]) == pytest.approx(expected_focal_x, rel=1e-9)


def test_rig_intrinsics_refit_to_camera_model(
    isolated_kerry_park_rig: Path, capsys
) -> None:
    """With ``--camera-model``, a sensor's ``camera_params`` calibration is refit
    to that model rather than replaced by an inferred camera."""
    from sfmtool.colmap.db_builders import _camera_from_sensor_entry

    rig_config = json.loads((isolated_kerry_park_rig / "rig_config.json").read_text())[
        0
    ]
    sensor = rig_config["cameras"][0]
    image = isolated_kerry_park_rig / "fisheye_left" / "frame_01.jpg"

    cam = _camera_from_sensor_entry(
        sensor, image, camera_model_override="THIN_PRISM_FISHEYE"
    )

    assert cam.model.name == "THIN_PRISM_FISHEYE"
    assert cam.has_prior_focal_length is True
    # THIN_PRISM_FISHEYE: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1.
    fx, fy, cx, cy, k1, k2, _p1, _p2, k3, k4, _sx1, _sy1 = (
        float(v) for v in cam.params
    )
    expected = sensor["camera_params"]
    # THIN_PRISM_FISHEYE holds every OPENCV_FISHEYE term, so the fit carries
    # the calibration over instead of the ~152.8 px an inference would give.
    assert fx == pytest.approx(expected[0], rel=1e-6)
    assert fy == pytest.approx(expected[1], rel=1e-6)
    assert (cx, cy) == (expected[2], expected[3])
    assert k1 == pytest.approx(expected[4], abs=1e-6)
    assert k4 == pytest.approx(expected[7], abs=1e-6)
    out = capsys.readouterr().out
    assert (
        "The camera of rig_config.json sensor 'fisheye_left/': "
        "OPENCV_FISHEYE -> THIN_PRISM_FISHEYE"
    ) in out


def test_rig_intrinsics_wrong_length_rejected(
    isolated_kerry_park_rig: Path,
) -> None:
    """``camera_params`` with the wrong length for its model must be rejected,
    not silently fed to pycolmap.
    """
    from sfmtool.colmap.db_builders import _camera_from_sensor_entry

    image = isolated_kerry_park_rig / "fisheye_left" / "frame_01.jpg"
    sensor = {"camera_model_name": "OPENCV_FISHEYE", "camera_params": [1.0, 2.0, 3.0]}

    with pytest.raises(ValueError, match="expects 8 values, got 3"):
        _camera_from_sensor_entry(sensor, image, camera_model_override=None)


def test_rig_params_without_model_rejected(isolated_kerry_park_rig: Path) -> None:
    """``camera_params`` is positional and meaningless without a model name;
    a sensor entry carrying params but no ``camera_model_name`` is malformed.
    """
    from sfmtool.colmap.db_builders import _camera_from_sensor_entry

    image = isolated_kerry_park_rig / "fisheye_left" / "frame_01.jpg"
    sensor = {"camera_params": [1.0, 2.0, 3.0, 4.0]}

    with pytest.raises(ValueError, match="requires camera_model_name"):
        _camera_from_sensor_entry(sensor, image, camera_model_override=None)


# --------------------------------------------------------------------------- #
# Full-solve tests: depend on the session-scoped reconstruction fixture.
# --------------------------------------------------------------------------- #


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotation_angle_deg(R: np.ndarray) -> float:
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def test_kerry_park_solve_registers_all_frames(
    kerry_park_solve_once: Path,
) -> None:
    """The global solve should register every image of the 8-frame prefix
    subset (16 images) and produce a substantive point cloud. Bounds are
    forgiving: GLOMAP output drifts slightly across pycolmap versions and isn't
    seed-deterministic.
    """
    recon = SfmrReconstruction.load(kerry_park_solve_once)
    assert recon.image_count == 16
    assert recon.camera_count == 2  # one camera per sensor
    assert recon.point_count >= 150
    # Median reprojection error should be small for a well-conditioned solve.
    errors = recon.rms_reprojection_errors
    assert errors.size > 0
    assert float(np.median(errors)) < 1.5


def test_kerry_park_rig_rotation_recovered(
    kerry_park_solve_once: Path,
) -> None:
    """For every rig frame in which both sensors are registered, the
    recovered right-from-left rotation should be a ~180° flip — the rig's
    spec ``cam_from_rig_rotation = [0, 0, 1, 0]`` (180° about Y) means
    ``right_from_left = R_y(180°)`` regardless of which way the rig as a
    whole points in world space.

    Translation is intentionally not checked: at 480x480 with 24 frames
    the rig's ~3 cm baseline is too small relative to the solved scene
    scale to be meaningful.
    """
    recon = SfmrReconstruction.load(kerry_park_solve_once)
    names = recon.image_names
    qs = recon.quaternions_wxyz  # cam_from_world per image, (N, 4)

    # Build a map: frame_key -> {sensor: (qvec_wxyz)}.
    frames: dict[str, dict[str, np.ndarray]] = {}
    for idx, name in enumerate(names):
        for sensor in ("fisheye_left", "fisheye_right"):
            prefix = f"{sensor}/"
            if name.startswith(prefix):
                frame_key = name[len(prefix) :]
                frames.setdefault(frame_key, {})[sensor] = qs[idx]
                break

    paired_frames = [
        f
        for f, sensors in frames.items()
        if "fisheye_left" in sensors and "fisheye_right" in sensors
    ]
    assert len(paired_frames) >= 6, (
        f"Expected almost every rig frame to have both sensors registered, "
        f"got only {len(paired_frames)} of {len(frames)} frames paired."
    )

    angles_from_180 = []
    for frame_key in paired_frames:
        # cam_from_world for left and right.
        R_left = _quat_wxyz_to_rotmat(frames[frame_key]["fisheye_left"])
        R_right = _quat_wxyz_to_rotmat(frames[frame_key]["fisheye_right"])
        # right_from_left = R_right @ R_left^T   (since both are cam_from_world)
        R_rel = R_right @ R_left.T
        theta_deg = _rotation_angle_deg(R_rel)
        angles_from_180.append(abs(theta_deg - 180.0))

    # Loose per-frame tolerance, tighter median tolerance — global SfM
    # gives a noisy per-image estimate but the median is reliable.
    assert max(angles_from_180) < 15.0, (
        f"At least one rig frame deviates {max(angles_from_180):.1f}° from "
        f"a 180° flip; expected the recovered right-from-left rotation to "
        f"be close to R_y(180°)."
    )
    assert float(np.median(angles_from_180)) < 5.0


# --------------------------------------------------------------------------- #
# Multi-sensor .camrig: the same rig described by a kerry_park.camrig file.
# --------------------------------------------------------------------------- #


def test_kerry_park_camrig_parses_as_fisheye_360(
    isolated_kerry_park_camrig: Path,
) -> None:
    """The kerry_park.camrig should be a 2-sensor ``fisheye_360`` rig sharing
    one OPENCV_FISHEYE camera, with the right sensor flipped 180° about Y.
    """
    from sfmtool.fileio import read_camrig

    data = read_camrig(str(isolated_kerry_park_camrig / "kerry_park.camrig"))
    meta = data["metadata"]
    assert meta["sensor_count"] == 2
    assert meta["camera_count"] == 1
    assert meta["rig_type"] == "fisheye_360"
    assert data["cameras"][0]["model"] == "OPENCV_FISHEYE"
    assert data["sensor_image_patterns"] == [
        "fisheye_left/frame_%02d.jpg",
        "fisheye_right/frame_%02d.jpg",
    ]
    # Sensor 0 identity, sensor 1 = 180° about Y. In the canonical convention
    # this is the S-conjugated quaternion (WXYZ [0, 0, -1, 0]); S·Ry(180°)·S is
    # the same rotation as Ry(180°) but the unit quaternion picks the -Y sign.
    np.testing.assert_allclose(data["quaternions_wxyz"][0], [1, 0, 0, 0], atol=1e-15)
    np.testing.assert_allclose(data["quaternions_wxyz"][1], [0, 0, -1, 0], atol=1e-15)


def test_kerry_park_camrig_resolves_to_rig(
    isolated_kerry_park_camrig: Path,
) -> None:
    """``resolve_camrig_for_solve`` should pair the 48 images into 24 frames,
    each frame carrying both sensors.
    """
    from collections import Counter

    from sfmtool.camrig.resolver import resolve_camrig_for_solve

    image_paths = []
    for sensor in ("fisheye_left", "fisheye_right"):
        image_paths.extend(sorted((isolated_kerry_park_camrig / sensor).glob("*.jpg")))

    result = resolve_camrig_for_solve(image_paths, isolated_kerry_park_camrig, None)
    assert result is not None and result.is_multi_sensor
    (rig,) = result.rigs
    assignments = rig.assignments
    assert len(assignments) == 48

    sensors = Counter(s for s, _f in assignments.values())
    frames = Counter(f for _s, f in assignments.values())
    assert dict(sensors) == {0: 24, 1: 24}
    assert set(frames) == set(range(1, 25))
    assert all(count == 2 for count in frames.values())


def test_kerry_park_camrig_solve_registers_all_frames(
    kerry_park_camrig_workspace: Path,
) -> None:
    """A global solve driven by the multi-sensor ``.camrig`` should register
    every rig image of the 8-frame prefix subset (16 images) with one camera
    per sensor — the same outcome as the ``rig_config.json`` path. The
    fixture's solve also holds single images, which are left out here.
    """
    recon = SfmrReconstruction.load(kerry_park_camrig_workspace)
    rig_cameras = {
        int(camera)
        for name, camera in zip(recon.image_names, recon.camera_indexes)
        if not name.startswith("solo/")
    }
    rig_images = [n for n in recon.image_names if not n.startswith("solo/")]
    assert len(rig_images) == 16
    assert len(rig_cameras) == 2  # one camera per sensor
    assert recon.point_count >= 150
    errors = recon.rms_reprojection_errors
    assert errors.size > 0
    assert float(np.median(errors)) < 1.5


def test_kerry_park_solve_mixes_rig_frames_and_single_images(
    kerry_park_camrig_workspace: Path,
) -> None:
    """The ``.camrig`` fixture's global solve of rig frames beside single
    images: ``kerry_park.camrig`` covers frames 1-8 of both lenses, and frames
    9-12 of the left lens, in ``solo/``, are single images under a
    single-sensor ``.camrig`` holding the left lens's camera. The ``.sfmr``
    holds the rig and a one-sensor rig for the single images, and
    ``sfm inspect`` and ``sfm motion`` read it.
    """
    from click.testing import CliRunner

    from sfmtool.cli import main

    recon = SfmrReconstruction.load(kerry_park_camrig_workspace)
    assert recon.image_count == 20
    assert recon.camera_count == 3  # the two lenses of the rig and solo.camrig
    camera_of = dict(zip(recon.image_names, (int(c) for c in recon.camera_indexes)))
    assert len({camera_of[f"solo/frame_{n:02d}.jpg"] for n in range(9, 13)}) == 1
    assert len(set(camera_of.values())) == 3

    data = recon.rig_frame_data
    rigs = data["rigs_metadata"]["rigs"]
    assert sorted(r["sensor_count"] for r in rigs) == [1, 2]
    frame_rigs = [int(r) for r in data["rig_indexes"]]
    frame_of = dict(
        zip(recon.image_names, (int(f) for f in data["image_frame_indexes"]))
    )
    # Each rig frame holds the two lenses' images of one frame number, in the
    # two-sensor rig; each single image is a frame of its own, in the other.
    two_sensor = next(i for i, r in enumerate(rigs) if r["sensor_count"] == 2)
    for n in range(1, 9):
        frame = frame_of[f"fisheye_left/frame_{n:02d}.jpg"]
        assert frame_of[f"fisheye_right/frame_{n:02d}.jpg"] == frame
        assert frame_rigs[frame] == two_sensor
    solo_frames = [frame_of[f"solo/frame_{n:02d}.jpg"] for n in range(9, 13)]
    assert len(set(solo_frames)) == 4
    assert all(frame_rigs[f] != two_sensor for f in solo_frames)
    assert len(frame_rigs) == 12

    runner = CliRunner()
    inspect = runner.invoke(main, ["inspect", str(kerry_park_camrig_workspace)])
    assert inspect.exit_code == 0, inspect.output
    assert "Cameras:        3" in inspect.output
    motion = runner.invoke(main, ["motion", str(kerry_park_camrig_workspace)])
    assert motion.exit_code == 0, motion.output
    assert "solo" in motion.output and "fisheye_right" in motion.output
