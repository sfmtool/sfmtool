# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""``sfmtool._sfmtool.geometry.resect_images`` — the held-out resection of an
image set at the Python boundary.

The fixtures mirror the core tests
(``crates/sfmtool-core/src/geometry/resect_images/tests.rs``): an
``embedded_patches`` reconstruction of a camera ring over a deterministic point
cloud, so the 2D observations are inline and no ``.sift`` companion has to
exist on disk. What this module pins is the binding's own share — the
name-to-index lookup, the report dict and its per-image list, the
refusal-is-not-an-exception contract, and that the input reconstruction
survives the call untouched. See ``specs/gui/gui-resect-image.md``.
"""

import math

import numpy as np
import pytest

from sfmtool._sfmtool.geometry import CameraIntrinsics, RotQuaternion, resect_images
from sfmtool._sfmtool.reconstruction import SfmrReconstruction

WIDTH, HEIGHT = 640, 480


def _pinhole(focal: float) -> CameraIntrinsics:
    return CameraIntrinsics(
        "PINHOLE",
        WIDTH,
        HEIGHT,
        {
            "focal_length_x": focal,
            "focal_length_y": focal,
            "principal_point_x": WIDTH / 2.0,
            "principal_point_y": HEIGHT / 2.0,
        },
    )


def _look_at(eye: np.ndarray, at: np.ndarray) -> RotQuaternion:
    """World-to-camera rotation of a camera at ``eye`` looking at ``at``, in the
    canonical convention (-Z forward, +Y up)."""
    forward = at - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    m = np.ascontiguousarray(np.stack([right, up, -forward]))
    return RotQuaternion.from_rotation_matrix(m)


def _image_at(name: str, eye: np.ndarray, at: np.ndarray):
    """One posed image as ``(name, world-to-camera rotation, translation)``."""
    q = _look_at(eye, at)
    r = np.asarray(q.to_rotation_matrix(), dtype=np.float64)
    return name, q, r @ (-eye)


def _ring(count: int, radius: float):
    """``count`` cameras on a circle of ``radius`` about the origin, each looking
    at it, rising in z so the views are not coplanar."""
    images = []
    for i in range(count):
        angle = i * 2.0 * math.pi / count
        eye = np.array([radius * math.cos(angle), radius * math.sin(angle), 0.4 * i])
        images.append(_image_at(f"frames/{i:03d}.jpg", eye, np.zeros(3)))
    return images


def _cloud(count: int) -> list[np.ndarray]:
    """A deterministic low-discrepancy scatter inside a ball (a test that only
    passes for one random seed is a test of the seed)."""
    points = []
    for i in range(count):
        phi = ((i + 0.5) / count) * math.pi
        theta = i * 2.399963229728653
        r = 0.4 + 0.6 * ((i % 7) / 7.0)
        points.append(
            np.array(
                [
                    r * math.sin(phi) * math.cos(theta),
                    r * math.sin(phi) * math.sin(theta),
                    r * math.cos(phi),
                ]
            )
        )
    return points


def _build(images, positions, focal, workspace_dir) -> SfmrReconstruction:
    """An ``embedded_patches`` reconstruction of ``images`` observing
    ``positions``, with every observation the camera model admits."""
    camera = _pinhole(focal)
    rotations = [
        np.asarray(q.to_rotation_matrix(), dtype=np.float64) for _, q, _ in images
    ]
    xyzw, image_indexes, point_indexes, keypoints, counts = [], [], [], [], []
    for position in positions:
        seen = []
        for i, (_, _, t) in enumerate(images):
            local = rotations[i] @ position + t
            uv = camera.ray_to_pixel([local[0], local[1], local[2]])
            if uv is None:
                continue
            u, v = uv
            if 0.0 <= u < WIDTH and 0.0 <= v < HEIGHT:
                seen.append((i, (u, v)))
        if len(seen) < 2:
            continue
        point_index = len(xyzw)
        for i, uv in seen:
            image_indexes.append(i)
            point_indexes.append(point_index)
            keypoints.append(uv)
        counts.append(len(seen))
        xyzw.append([position[0], position[1], position[2], 1.0])

    n_img, n_pt, n_obs = len(images), len(xyzw), len(image_indexes)
    data = {
        "metadata": {
            "version": 6,
            "operation": "test",
            "tool": "sfmtool",
            "tool_version": "0",
            "tool_options": {},
            "workspace": {
                "absolute_path": str(workspace_dir),
                "relative_path": ".",
                "contents": {
                    "feature_tool": "none",
                    "feature_type": "sift",
                    "feature_options": {},
                    "feature_prefix_dir": "",
                },
            },
            "timestamp": "",
            "image_count": n_img,
            "point_count": n_pt,
            "infinity_point_count": 0,
            "observation_count": n_obs,
            "camera_count": 1,
            "feature_source": "embedded_patches",
        },
        "cameras": [camera],
        "image_names": [name for name, _, _ in images],
        "camera_indexes": np.zeros(n_img, np.uint32),
        "quaternions_wxyz": np.stack(
            [np.asarray(q.to_wxyz_array()) for _, q, _ in images]
        ),
        "translations_xyz": np.stack([t for _, _, t in images]).astype(np.float64),
        "positions_xyzw": np.asarray(xyzw, np.float64).reshape(n_pt, 4),
        "colors_rgb": np.full((n_pt, 3), 128, np.uint8),
        "reprojection_errors": np.zeros(n_pt, np.float32),
        "image_indexes": np.asarray(image_indexes, np.uint32),
        "point_indexes": np.asarray(point_indexes, np.uint32),
        "observation_counts": np.asarray(counts, np.uint32),
        "keypoints_xy": np.asarray(keypoints, np.float32).reshape(n_obs, 2),
        "image_file_hashes": [bytes(16)] * n_img,
        "thumbnails_y_x_rgb": np.zeros((n_img, 1, 1, 3), np.uint8),
    }
    return SfmrReconstruction.from_data(workspace_dir, data)


@pytest.fixture(scope="module")
def orbit(tmp_path_factory) -> SfmrReconstruction:
    """Eight ring cameras over a 200-point ball — a well-conditioned finite
    fixture, built once for the whole module (nothing here mutates it)."""
    workspace = tmp_path_factory.mktemp("resect_image")
    return _build(_ring(8, 4.0), _cloud(200), 800.0, workspace)


def _perturb(recon: SfmrReconstruction, index: int) -> SfmrReconstruction:
    """``recon`` with one image's pose wrong by tens of degrees and a
    substantial fraction of the scene — the disagreement resection exists to
    show."""
    quaternions = np.asarray(recon.quaternions_wxyz, np.float64).copy()
    translations = np.asarray(recon.translations, np.float64).copy()
    spun = RotQuaternion.from_axis_angle(
        [0.0, 1.0, 0.0], 0.35
    ) * RotQuaternion.from_wxyz_array(quaternions[index])
    quaternions[index] = np.asarray(spun.to_wxyz_array())
    translations[index] = translations[index] + np.array([0.6, -0.3, 0.2])
    return recon.clone_with_changes(
        quaternions_wxyz=quaternions, translations=translations
    )


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two world-to-camera rotations given as wxyz rows."""
    qa = RotQuaternion.from_wxyz_array(np.asarray(a, np.float64))
    qb = RotQuaternion.from_wxyz_array(np.asarray(b, np.float64))
    return math.degrees(abs((qa.inverse() * qb).angle()))


def _camera_centers(recon: SfmrReconstruction) -> np.ndarray:
    quaternions = np.asarray(recon.quaternions_wxyz, np.float64)
    translations = np.asarray(recon.translations, np.float64)
    centers = []
    for q, t in zip(quaternions, translations):
        r = np.asarray(
            RotQuaternion.from_wxyz_array(q).to_rotation_matrix(), np.float64
        )
        centers.append(-r.T @ t)
    return np.stack(centers)


def _resect_one(recon: SfmrReconstruction, image_name: str, **kwargs):
    """The set form on a one-element set, with that image's own report."""
    derived, report = resect_images(recon, [image_name], **kwargs)
    assert report["targets"] == 1
    assert len(report["images"]) == 1
    only = report["images"][0]
    # The totals of a one-element set are that element's, point counts aside
    # (the totals count distinct points, the report counts the target's).
    assert report["accepted"] == int(only["accepted"])
    assert report["refused"] == int(only["refused"])
    assert report["correspondences"] == only["correspondences"]
    assert report["inliers"] == only["inliers"]
    assert report["scene_scale"] == only["scene_scale"]
    return derived, only


def _corrupt_observations(
    recon: SfmrReconstruction, indexes: list[int]
) -> SfmrReconstruction:
    """``recon`` with the given images' inline keypoints replaced by junk — the
    rows the hold-out is supposed to be blind to."""
    keypoints = np.asarray(recon.keypoints_xy, np.float32).copy()
    images = np.asarray(recon.track_image_indexes)
    for index in indexes:
        rows = np.flatnonzero(images == index)
        n = np.arange(len(rows))
        keypoints[rows, 0] = (n % WIDTH).astype(np.float32)
        keypoints[rows, 1] = ((n * 7) % HEIGHT).astype(np.float32)
    return recon.clone_with_changes(keypoints_xy=keypoints)


class TestFinitePath:
    def test_perturbed_pose_is_recovered(self, orbit):
        """The estimate walks back to the truth the other seven cameras imply."""
        truth_q = np.asarray(orbit.quaternions_wxyz, np.float64)[0].copy()
        truth_center = _camera_centers(orbit)[0].copy()
        source = _perturb(orbit, 0)

        derived, report = _resect_one(source, "frames/000.jpg")

        assert report["accepted"], report["refusal"]
        assert report["refused"] is False
        assert report["refusal"] is None
        assert report["image_index"] == 0
        assert report["image_name"] == "frames/000.jpg"
        assert report["source"] == "observations"
        assert report["rotation_only"] is False
        assert report["correspondences"] >= 100
        assert report["inliers"] <= report["correspondences"]
        assert report["inlier_fraction"] == pytest.approx(
            report["inliers"] / report["correspondences"]
        )
        assert report["held_out_points"] > 100
        assert report["retriangulated"] > 0
        # The report describes the move away from the *stored* (perturbed) pose.
        assert report["rotation_deg"] > 15.0
        assert report["translation"] > 0.5
        assert report["scene_scale"] > 0.0
        assert report["translation_scene"] == pytest.approx(
            report["translation"] / report["scene_scale"]
        )

        fitted_q = np.asarray(derived.quaternions_wxyz, np.float64)[0]
        assert _angle_deg(fitted_q, truth_q) < 0.1
        assert np.linalg.norm(_camera_centers(derived)[0] - truth_center) < 0.01

    def test_input_reconstruction_is_not_modified(self, orbit):
        """The core clones; the binding must not hand the caller an alias."""
        source = _perturb(orbit, 0)
        before_q = np.asarray(source.quaternions_wxyz, np.float64).copy()
        before_t = np.asarray(source.translations, np.float64).copy()
        before_p = np.asarray(source.positions_xyzw, np.float64).copy()

        derived, report = _resect_one(source, "frames/000.jpg")

        np.testing.assert_array_equal(np.asarray(source.quaternions_wxyz), before_q)
        np.testing.assert_array_equal(np.asarray(source.translations), before_t)
        np.testing.assert_array_equal(np.asarray(source.positions_xyzw), before_p)
        # And the derived one actually moved, so the comparison above means
        # something.
        assert report["accepted"]
        assert not np.array_equal(np.asarray(derived.quaternions_wxyz), before_q)

    def test_same_input_gives_the_same_answer(self, orbit):
        """Deterministic: the RANSAC is seeded as a pure function of its input."""
        source = _perturb(orbit, 2)
        one, report_one = _resect_one(source, "frames/002.jpg")
        two, report_two = _resect_one(source, "frames/002.jpg")

        np.testing.assert_array_equal(
            np.asarray(one.quaternions_wxyz), np.asarray(two.quaternions_wxyz)
        )
        np.testing.assert_array_equal(
            np.asarray(one.translations), np.asarray(two.translations)
        )
        np.testing.assert_array_equal(
            np.asarray(one.positions_xyzw), np.asarray(two.positions_xyzw)
        )
        assert report_one == report_two


class TestRefusal:
    def test_a_refused_estimate_comes_back_as_a_report_not_an_exception(self, orbit):
        """Junk observations and a junk pose: nothing the target says is usable,
        so the estimate is refused — and the caller still gets the derived
        reconstruction, with the stored pose retained."""
        corrupted = _corrupt_observations(_perturb(orbit, 0), [0])

        derived, report = _resect_one(corrupted, "frames/000.jpg")

        assert report["refused"] is True
        assert report["accepted"] is False
        assert isinstance(report["refusal"], str) and report["refusal"]
        assert report["retriangulated"] == 0
        assert report["held_out_points"] > 100
        # The stored pose is what the derived node keeps on a refusal.
        np.testing.assert_array_equal(
            np.asarray(derived.quaternions_wxyz)[0],
            np.asarray(corrupted.quaternions_wxyz)[0],
        )
        np.testing.assert_array_equal(
            np.asarray(derived.translations)[0], np.asarray(corrupted.translations)[0]
        )

    def test_an_unknown_image_name_raises(self, orbit):
        with pytest.raises(ValueError, match="no image named"):
            _resect_one(orbit, "frames/nope.jpg")

    def test_too_few_other_posed_images_raises(self, tmp_path):
        """Three cameras total leaves two non-targets — below the floor at which
        'the rest of the reconstruction' is a reconstruction."""
        source = _build(_ring(3, 4.0), _cloud(120), 800.0, tmp_path)
        with pytest.raises(ValueError, match="non-target posed image"):
            _resect_one(source, "frames/000.jpg")


class TestSet:
    def test_two_targets_are_held_out_together_and_both_recover(self, orbit):
        """A set is one question about a group: the shared structure is
        re-triangulated from neither target, and each pose is fit against it."""
        names = ["frames/000.jpg", "frames/001.jpg"]
        truth_q = np.asarray(orbit.quaternions_wxyz, np.float64)[:2].copy()
        truth_centers = _camera_centers(orbit)[:2].copy()
        source = _perturb(_perturb(orbit, 0), 1)

        derived, report = resect_images(source, names)

        assert report["targets"] == 2
        assert report["accepted"] == 2, report["images"]
        assert report["refused"] == 0
        assert [r["image_name"] for r in report["images"]] == names
        assert [r["image_index"] for r in report["images"]] == [0, 1]
        assert report["correspondences"] == sum(
            r["correspondences"] for r in report["images"]
        )
        assert report["inliers"] == sum(r["inliers"] for r in report["images"])
        assert report["inlier_fraction"] == pytest.approx(
            report["inliers"] / report["correspondences"]
        )
        # Each point is counted once in the totals, in both reports below.
        assert report["held_out_points"] > 100
        assert report["retriangulated"] > 0

        fitted_q = np.asarray(derived.quaternions_wxyz, np.float64)[:2]
        fitted_centers = _camera_centers(derived)[:2]
        for i in range(2):
            assert _angle_deg(fitted_q[i], truth_q[i]) < 0.1, names[i]
            assert np.linalg.norm(fitted_centers[i] - truth_centers[i]) < 0.01

    def test_the_hold_out_ignores_every_target_not_just_one(self, orbit):
        """Both targets' observations are junk. A hold-out that dropped only the
        image being estimated would read the other target's corrupted rows for
        the points they share; one that drops the whole set reads the six honest
        cameras and lands on the truth."""
        names = ["frames/000.jpg", "frames/001.jpg"]
        truth_positions = np.asarray(orbit.positions_xyzw, np.float64).copy()
        source = _corrupt_observations(orbit, [0, 1])

        derived, report = resect_images(source, names)

        assert report["accepted"] == 0, report["images"]
        assert report["refused"] == 2
        assert report["retriangulated"] == 0
        assert report["held_out_points"] > 100
        # The fixture's points are all seen by every camera, so the hold-out
        # places every one of them and nothing is dropped.
        assert report["removed_points"] == 0
        for r in report["images"]:
            assert r["refused"] is True
            assert isinstance(r["refusal"], str) and r["refusal"]

        held_out = np.asarray(derived.positions_xyzw, np.float64)
        # Not zero: the fixture stores its keypoints as float32, so a position
        # re-triangulated from them lands within a pixel's quantization of the
        # truth. What matters is that corrupting the targets moved nothing.
        assert np.abs(held_out - truth_positions).max() < 1e-5

    def test_an_empty_target_list_raises(self, orbit):
        with pytest.raises(ValueError, match="no target images"):
            resect_images(orbit, [])

    def test_a_target_named_twice_raises(self, orbit):
        with pytest.raises(ValueError, match="twice"):
            resect_images(orbit, ["frames/002.jpg", "frames/002.jpg"])
