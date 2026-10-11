# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Epipolar geometry visualization for SfM reconstructions."""

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from ..camera.cameras import colmap_camera_from_intrinsics, get_intrinsic_matrix
from .._sfmtool.fileio import read_image_rgb, write_image_rgb
from ..sift.file import SiftReader, get_sift_path_for_image
from .._sfmtool.analysis import epipolar_curves
from .._sfmtool.geometry import RotQuaternion
from ._common import get_color_palette
from ._rectification import compute_stereo_rectification


def _draw_epipolar_line(
    image: np.ndarray,
    line: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw an epipolar line on an image."""
    h, w = image.shape[:2]
    a, b, c = line

    points = []

    if abs(b) > 1e-10:
        y = -c / b
        if 0 <= y < h:
            points.append((0, int(round(y))))
    if abs(b) > 1e-10:
        y = -(a * (w - 1) + c) / b
        if 0 <= y < h:
            points.append((w - 1, int(round(y))))
    if abs(a) > 1e-10:
        x = -c / a
        if 0 <= x < w:
            points.append((int(round(x)), 0))
    if abs(a) > 1e-10:
        x = -(b * (h - 1) + c) / a
        if 0 <= x < w:
            points.append((int(round(x)), h - 1))

    points = list(dict.fromkeys(points))
    if len(points) >= 2:
        cv2.line(image, points[0], points[1], color, thickness)


def _curve_anchor_depths(
    recon,
    track_point_indexes: np.ndarray,
    R_from: np.ndarray,
    t_from: np.ndarray,
    R_other: np.ndarray,
    t_other: np.ndarray,
) -> np.ndarray:
    """Per-feature seed depth for epipolar-curve sampling, in `R_from` coords.

    Uses the reconstructed track depth when a triangulated 3D point is
    available (and is in front of the camera); otherwise the baseline length
    `‖C_other − C_from‖`. `track_point_indexes` is one entry per feature pair with
    `-1` marking unmatched features. See specs/core/camera/epipolar-curves.md
    ("Caller-side seeding strategy") for the rationale.
    """
    c_from = -R_from.T @ t_from
    c_other = -R_other.T @ t_other
    baseline = float(np.linalg.norm(c_other - c_from))
    if baseline < 1e-9:
        baseline = 1.0  # degenerate; Rust returns empty anyway.

    depths = np.full(len(track_point_indexes), baseline, dtype=np.float64)
    valid = track_point_indexes >= 0
    if not valid.any():
        return depths
    pids = track_point_indexes[valid]
    points = np.asarray(recon.positions)[pids]
    # Canonical cameras look down -Z, so depth (positive in front) is -z, where
    # z = R_from[2, :] · X + t_from[2] is the camera-space z of the track point.
    track_depths = -(points @ R_from[2, :] + t_from[2])
    in_front = track_depths > 0
    if in_front.any():
        valid_idx = np.where(valid)[0]
        depths[valid_idx[in_front]] = track_depths[in_front]
    return depths


def _draw_polyline(
    image: np.ndarray,
    polyline: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw an epipolar curve on an image.

    The Rust sampler returns vertices that are already inside the image
    rectangle, so no clipping is needed here — just round and call polylines.
    """
    if polyline is None or len(polyline) < 2:
        return
    pts = np.round(np.asarray(polyline, dtype=np.float64)).astype(np.int32)
    pts = pts.reshape(-1, 1, 2)
    cv2.polylines(image, [pts], False, color, thickness)


@dataclass
class _PairPoses:
    """The two images' `cam_from_world` poses, canonical and OpenCV-frame.

    The canonical (.sfmr) poses look down -Z and drive the epipolar-curve
    renderer and the depth seeding. The OpenCV/COLMAP-frame poses are their
    S-flips (S-only; §1 invariant), for the pixel-space consumers (pycolmap's
    epipolar-guided matcher and `cv2.stereoRectify`), which expect +Z-forward
    camera frames.
    """

    quat1: np.ndarray
    t1: np.ndarray
    R1: np.ndarray
    quat2: np.ndarray
    t2: np.ndarray
    R2: np.ndarray
    q1_cv: np.ndarray
    t1_cv: np.ndarray
    R1_cv: np.ndarray
    q2_cv: np.ndarray
    t2_cv: np.ndarray
    R2_cv: np.ndarray

    @classmethod
    def from_recon(cls, recon, image1_idx: int, image2_idx: int) -> "_PairPoses":
        from ..colmap.convention import flip_camera_pose_s

        quaternions = recon.quaternions_wxyz
        translations = recon.translations
        quat1 = quaternions[image1_idx]
        quat2 = quaternions[image2_idx]
        t1 = np.asarray(translations[image1_idx], dtype=np.float64)
        t2 = np.asarray(translations[image2_idx], dtype=np.float64)
        q1_cv, t1_cv = flip_camera_pose_s(quat1, t1)
        q2_cv, t2_cv = flip_camera_pose_s(quat2, t2)
        return cls(
            quat1=quat1,
            t1=t1,
            R1=RotQuaternion.from_wxyz_array(quat1).to_rotation_matrix(),
            quat2=quat2,
            t2=t2,
            R2=RotQuaternion.from_wxyz_array(quat2).to_rotation_matrix(),
            q1_cv=q1_cv,
            t1_cv=t1_cv,
            R1_cv=RotQuaternion.from_wxyz_array(q1_cv).to_rotation_matrix(),
            q2_cv=q2_cv,
            t2_cv=t2_cv,
            R2_cv=RotQuaternion.from_wxyz_array(q2_cv).to_rotation_matrix(),
        )

    def relative_pose_cv(self) -> tuple[np.ndarray, np.ndarray]:
        """Image 2's OpenCV-frame pose relative to image 1's, as `(R, t)`."""
        R_rel = self.R2_cv @ self.R1_cv.T
        t_rel = self.t2_cv - R_rel @ self.t1_cv
        return R_rel, t_rel


@dataclass
class _UndistortedPair:
    """Both images and their cameras, before and after removing lens distortion."""

    cam1: pycolmap.Camera
    cam2: pycolmap.Camera
    bitmap1: pycolmap.Bitmap
    bitmap2: pycolmap.Bitmap
    undist_cam1: pycolmap.Camera
    undist_cam2: pycolmap.Camera

    def rgb_images(self) -> tuple[np.ndarray, np.ndarray]:
        """The undistorted images as RGB arrays of their own, ready to draw on."""
        img1 = np.array(self.bitmap1.to_array())
        img2 = np.array(self.bitmap2.to_array())
        return img1, img2


def _find_image_index(image_names, query: str) -> int | None:
    """Index of `query` in `image_names`, preferring an exact full-path match.

    An exact match wins so that datasets with the same basename under several
    subdirectories (e.g. rig sensors with `fisheye_left/frame_01.jpg` and
    `fisheye_right/frame_01.jpg`) resolve unambiguously; otherwise the first
    image with the same basename is returned.
    """
    exact = None
    basename_match = None
    query_basename = Path(query).name
    for idx, name in enumerate(image_names):
        if name == query:
            exact = idx
            break
        if Path(name).name == query_basename and basename_match is None:
            basename_match = idx
    return exact if exact is not None else basename_match


def _require_sift_path(
    image_path: Path,
    image_name: str,
    feature_tool: str | None,
    feature_options: dict | None,
) -> Path:
    """The image's `.sift` path, raising `FileNotFoundError` if it is missing."""
    sift_path = get_sift_path_for_image(
        image_path,
        feature_tool=feature_tool,
        feature_options=feature_options,
    )
    if not sift_path.exists():
        raise FileNotFoundError(
            f"SIFT file not found for {image_name}. Expected at: {sift_path}."
        )
    return sift_path


def _read_and_undistort(
    recon, image1_idx: int, image2_idx: int, image1_path: Path, image2_path: Path
) -> _UndistortedPair:
    """Read both images, build their pycolmap cameras, and undistort them."""
    bitmap1 = pycolmap.Bitmap.read(str(image1_path), as_rgb=True)
    bitmap2 = pycolmap.Bitmap.read(str(image2_path), as_rgb=True)
    if bitmap1 is None or bitmap2 is None:
        raise ValueError("Could not read image files as Bitmaps.")

    h1, w1 = bitmap1.to_array().shape[:2]
    h2, w2 = bitmap2.to_array().shape[:2]

    cameras = recon.cameras
    camera_indexes = recon.camera_indexes
    cam1 = colmap_camera_from_intrinsics(
        cameras[camera_indexes[image1_idx]], width=w1, height=h1
    )
    cam2 = colmap_camera_from_intrinsics(
        cameras[camera_indexes[image2_idx]], width=w2, height=h2
    )

    options = pycolmap.UndistortCameraOptions()
    options.blank_pixels = 0
    undist_bitmap1, undist_cam1 = pycolmap.undistort_image(options, bitmap1, cam1)
    undist_bitmap2, undist_cam2 = pycolmap.undistort_image(options, bitmap2, cam2)
    return _UndistortedPair(
        cam1=cam1,
        cam2=cam2,
        bitmap1=undist_bitmap1,
        bitmap2=undist_bitmap2,
        undist_cam1=undist_cam1,
        undist_cam2=undist_cam2,
    )


def _compute_rectification(undistorted: _UndistortedPair, poses: _PairPoses):
    """The stereo rectification of an undistorted image pair."""
    R_rel, t_rel = poses.relative_pose_cv()
    return compute_stereo_rectification(
        undistorted.cam1,
        undistorted.cam2,
        undistorted.undist_cam1,
        undistorted.undist_cam2,
        R_rel,
        t_rel,
    )


def _subsample_evenly(items: list, max_count: int | None) -> np.ndarray | None:
    """Indexes of `max_count` evenly spaced items, or `None` to keep them all."""
    if max_count is None or len(items) <= max_count:
        return None
    return np.linspace(0, len(items) - 1, max_count, dtype=int)


def _shared_track_pairs(
    recon,
    image1_idx: int,
    image2_idx: int,
    image1_name: str,
    image2_name: str,
    max_features: int | None,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Feature pairs that observe the same 3D point in both images.

    Returns the `(feature in image 1, feature in image 2)` pairs and, for each
    pair, the index of the 3D point they observe.
    """
    observations = np.column_stack(
        [recon.track_image_indexes, recon.track_feature_indexes]
    )
    point_ids = recon.track_point_indexes

    obs1_mask = observations[:, 0] == image1_idx
    obs2_mask = observations[:, 0] == image2_idx

    obs1 = observations[obs1_mask]
    obs2 = observations[obs2_mask]

    point_ids1 = point_ids[obs1_mask]
    point_ids2 = point_ids[obs2_mask]

    shared_point_ids = np.intersect1d(point_ids1, point_ids2)

    if len(shared_point_ids) == 0:
        raise ValueError(
            f"No shared features found between '{image1_name}' and '{image2_name}'"
        )

    feature_pairs = []
    track_point_indexes = []
    for point_id in shared_point_ids:
        feat1_idx = obs1[point_ids1 == point_id, 1][0]
        feat2_idx = obs2[point_ids2 == point_id, 1][0]
        feature_pairs.append((feat1_idx, feat2_idx))
        track_point_indexes.append(point_id)

    indices = _subsample_evenly(feature_pairs, max_features)
    if indices is not None:
        feature_pairs = [feature_pairs[i] for i in indices]
        track_point_indexes = [track_point_indexes[i] for i in indices]
    return feature_pairs, np.asarray(track_point_indexes, dtype=np.int64)


@dataclass
class _SweepResult:
    """Sort-and-sweep matches between the two images.

    `rectification` and the rectified feature positions are `None` when the
    epipole lies in the frame, in which case the matcher swept in polar
    coordinates and no rectified view exists.
    """

    feature_pairs: list[tuple[int, int]]
    positions1: np.ndarray
    positions2: np.ndarray
    undistorted: _UndistortedPair
    rectification: object | None
    rect_pts1: np.ndarray | None
    rect_pts2: np.ndarray | None


def _sweep_matches(
    recon,
    image1_idx: int,
    image2_idx: int,
    image1_path: Path,
    image2_path: Path,
    sift1_path: Path,
    sift2_path: Path,
    poses: _PairPoses,
    sweep_max_features: int,
    sweep_window_size: int,
    max_features: int | None,
) -> _SweepResult:
    """Match the first `sweep_max_features` features of each image by sort-and-sweep."""
    with SiftReader(sift1_path) as reader:
        positions1 = reader.read_positions(count=sweep_max_features)
        descriptors1 = reader.read_descriptors(count=sweep_max_features)

    with SiftReader(sift2_path) as reader:
        positions2 = reader.read_positions(count=sweep_max_features)
        descriptors2 = reader.read_descriptors(count=sweep_max_features)

    undistorted = _read_and_undistort(
        recon, image1_idx, image2_idx, image1_path, image2_path
    )
    cam1 = undistorted.cam1
    cam2 = undistorted.cam2

    from ..feature_match._geometry import check_rectification_safe

    rectification_safe = check_rectification_safe(
        get_intrinsic_matrix(cam1),
        poses.R1,
        poses.t1,
        get_intrinsic_matrix(cam2),
        poses.R2,
        poses.t2,
        width=cam1.width,
        height=cam1.height,
        margin=50,
    )

    # pycolmap poses in the OpenCV frame for the epipolar-guided matcher.
    pose1 = pycolmap.Rigid3d(pycolmap.Rotation3d(np.roll(poses.q1_cv, -1)), poses.t1_cv)
    pose2 = pycolmap.Rigid3d(pycolmap.Rotation3d(np.roll(poses.q2_cv, -1)), poses.t2_cv)

    from ..feature_match import match_image_pair

    mutual_matches = match_image_pair(
        pose1,
        pose2,
        cam1,
        cam2,
        positions1,
        descriptors1,
        positions2,
        descriptors2,
        window_size=sweep_window_size,
    )

    if rectification_safe:
        match_method = "rectified sweep"
        rectification = _compute_rectification(undistorted, poses)
        rect_pts1 = rectification.rectify_points_1(positions1)
        rect_pts2 = rectification.rectify_points_2(positions2)
    else:
        match_method = "polar sweep (in-frame epipole)"
        rectification = None
        rect_pts1 = None
        rect_pts2 = None

    if not mutual_matches:
        raise ValueError(f"No mutual matches found using {match_method}.")

    indices = _subsample_evenly(mutual_matches, max_features)
    if indices is not None:
        mutual_matches = [mutual_matches[i] for i in indices]

    print(f"Found {len(mutual_matches)} matches using {match_method}")
    return _SweepResult(
        feature_pairs=[(m[0], m[1]) for m in mutual_matches],
        positions1=positions1,
        positions2=positions2,
        undistorted=undistorted,
        rectification=rectification,
        rect_pts1=rect_pts1,
        rect_pts2=rect_pts2,
    )


def _draw_rectified(
    img1: np.ndarray,
    img2: np.ndarray,
    rect_pts1: np.ndarray,
    rect_pts2: np.ndarray,
    colors: list[tuple[int, int, int]],
    feature_size: int,
    draw_lines: bool,
) -> None:
    """Draw features on a rectified pair, with a shared scanline per pair.

    In a rectified pair the epipolar lines are image rows, so each pair's line
    is the row halfway between its two features' rows.
    """
    for i in range(len(rect_pts1)):
        color = colors[i]
        p1 = rect_pts1[i]
        p2 = rect_pts2[i]

        cv2.circle(img1, (int(p1[0]), int(p1[1])), feature_size, color, -1)
        cv2.circle(img2, (int(p2[0]), int(p2[1])), feature_size, color, -1)

        if draw_lines:
            y = int((p1[1] + p2[1]) / 2)
            cv2.line(img1, (0, y), (img1.shape[1], y), color, 1)
            cv2.line(img2, (0, y), (img2.shape[1], y), color, 1)


def _draw_undistorted(
    img1: np.ndarray,
    img2: np.ndarray,
    undistorted: _UndistortedPair,
    poses: _PairPoses,
    pts1: np.ndarray,
    pts2: np.ndarray,
    colors: list[tuple[int, int, int]],
    feature_size: int,
    line_thickness: int,
    draw_lines: bool,
) -> None:
    """Draw features and straight epipolar lines on an undistorted pair.

    `pts1` and `pts2` are feature positions in the original (distorted) images;
    they are moved into the undistorted images first.
    """
    cam1, cam2 = undistorted.cam1, undistorted.cam2
    undist_cam1, undist_cam2 = undistorted.undist_cam1, undistorted.undist_cam2

    undist_pts1 = np.array([cam1.cam_from_img(p) for p in pts1])
    undist_pts2 = np.array([cam2.cam_from_img(p) for p in pts2])

    undist_pts1 = np.array([undist_cam1.img_from_cam(p) for p in undist_pts1])
    undist_pts2 = np.array([undist_cam2.img_from_cam(p) for p in undist_pts2])

    from ..feature_match._geometry import get_fundamental_matrix

    K1_undist = get_intrinsic_matrix(undist_cam1)
    K2_undist = get_intrinsic_matrix(undist_cam2)
    F_undist = get_fundamental_matrix(
        K1_undist, poses.R1, poses.t1, K2_undist, poses.R2, poses.t2
    )

    for i in range(len(undist_pts1)):
        color = colors[i]
        pos1 = undist_pts1[i]
        pos2 = undist_pts2[i]

        pt1 = (int(round(pos1[0])), int(round(pos1[1])))
        cv2.circle(img1, pt1, feature_size, color, -1)

        if draw_lines:
            p1_homog = np.array([pos1[0], pos1[1], 1.0])
            epipolar_line2 = F_undist @ p1_homog
            _draw_epipolar_line(img2, epipolar_line2, color, line_thickness)

        pt2 = (int(round(pos2[0])), int(round(pos2[1])))
        cv2.circle(img2, pt2, feature_size, color, -1)

        if draw_lines:
            p2_homog = np.array([pos2[0], pos2[1], 1.0])
            epipolar_line1 = F_undist.T @ p2_homog
            _draw_epipolar_line(img1, epipolar_line1, color, line_thickness)


def _draw_on_original(
    img1: np.ndarray,
    img2: np.ndarray,
    recon,
    cam1_intrinsics,
    cam2_intrinsics,
    poses: _PairPoses,
    pts1: np.ndarray,
    pts2: np.ndarray,
    track_point_indexes: np.ndarray,
    colors: list[tuple[int, int, int]],
    feature_size: int,
    line_thickness: int,
    draw_lines: bool,
) -> None:
    """Draw features and epipolar curves on the original (distorted) images.

    The epipolar geometry goes through the full camera model, so the "lines"
    are curves for fisheye and wide-FOV cameras. See
    specs/core/camera/epipolar-curves.md.
    """
    if draw_lines:
        R1, t1, quat1 = poses.R1, poses.t1, poses.quat1
        R2, t2, quat2 = poses.R2, poses.t2, poses.quat2
        anchors_from_cam1 = _curve_anchor_depths(
            recon, track_point_indexes, R1, t1, R2, t2
        )
        anchors_from_cam2 = _curve_anchor_depths(
            recon, track_point_indexes, R2, t2, R1, t1
        )
        curves_in_2 = epipolar_curves(
            pts1,
            anchors_from_cam1,
            cam1_intrinsics,
            quat1,
            t1,
            cam2_intrinsics,
            quat2,
            t2,
        )
        curves_in_1 = epipolar_curves(
            pts2,
            anchors_from_cam2,
            cam2_intrinsics,
            quat2,
            t2,
            cam1_intrinsics,
            quat1,
            t1,
        )

    for i, color in enumerate(colors):
        p1 = pts1[i]
        p2 = pts2[i]
        cv2.circle(
            img1, (int(round(p1[0])), int(round(p1[1]))), feature_size, color, -1
        )
        cv2.circle(
            img2, (int(round(p2[0])), int(round(p2[1]))), feature_size, color, -1
        )

        if draw_lines:
            _draw_polyline(img2, curves_in_2[i], color, line_thickness)
            _draw_polyline(img1, curves_in_1[i], color, line_thickness)


def _save_output(
    img1: np.ndarray,
    img2: np.ndarray,
    output_path: Path,
    side_by_side: bool,
    save_which: str,
) -> None:
    """Write the drawn images as one side-by-side image or as separate files."""
    output_path.parent.mkdir(exist_ok=True, parents=True)

    if side_by_side:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        max_height = max(h1, h2)

        output = np.zeros((max_height, w1 + w2, 3), dtype=np.uint8)
        output[:h1, :w1] = img1
        output[:h2, w1 : w1 + w2] = img2

        write_image_rgb(output_path, output)
        print(f"Visualized pairs to: {output_path} (side-by-side)")
    elif save_which == "both":
        stem = output_path.stem
        ext = output_path.suffix
        output_path_other = output_path.with_name(f"{stem}_other{ext}")

        write_image_rgb(output_path, img1)
        write_image_rgb(output_path_other, img2)
        print(f"Visualized pairs to: {output_path} and {output_path_other}")
    elif save_which == "first":
        write_image_rgb(output_path, img1)
        print(f"Visualized pairs to: {output_path}")
    elif save_which == "second":
        write_image_rgb(output_path, img2)
        print(f"Visualized pairs to: {output_path}")
    else:
        raise ValueError(
            f"Invalid save_which value: {save_which}. Must be 'both', 'first', or 'second'."
        )


def draw_epipolar_visualization(
    recon,
    image1_name: str,
    image2_name: str,
    output_path: str | Path,
    max_features: int | None = None,
    line_thickness: int = 1,
    feature_size: int = 3,
    rectify: bool = False,
    undistort: bool = False,
    draw_lines: bool = True,
    side_by_side: bool = False,
    feature_tool: str | None = None,
    feature_options: dict | None = None,
    sweep_max_features: int | None = None,
    sweep_window_size: int = 30,
    save_which: str = "both",
) -> None:
    """Create a visualization of features and epipolar lines between two images.

    The feature pairs come from the reconstruction's shared tracks, or from
    sort-and-sweep matching when `sweep_max_features` is set. They are drawn
    on rectified images (`rectify`), on undistorted images (`undistort`), or
    on the original images with epipolar curves through the full camera model.

    Args:
        recon: SfmrReconstruction containing camera parameters, poses, and tracks
        image1_name: Filename of first image
        image2_name: Filename of second image
        output_path: Path where visualization should be saved
        max_features: Maximum number of shared features to visualize (default: all)
        line_thickness: Thickness of epipolar lines in pixels
        feature_size: Size of feature point markers in pixels
        rectify: Whether to rectify images
        undistort: Whether to remove lens distortion
        side_by_side: Whether to combine images side-by-side or save separately
        feature_tool: Feature extraction tool name
        feature_options: Optional dict of options for feature tool
        sweep_max_features: If set, runs sort-and-sweep matching with this many features
        sweep_window_size: Window size for sort-and-sweep matching
        save_which: Which image(s) to save - "both", "first", or "second"
    """
    workspace_dir = Path(recon.workspace_dir)
    output_path = Path(output_path)
    image_names = recon.image_names

    image1_idx = _find_image_index(image_names, image1_name)
    image2_idx = _find_image_index(image_names, image2_name)

    from .._sfmr_naming import get_image_hint_message

    if image1_idx is None:
        raise ValueError(get_image_hint_message(recon, image1_name))
    if image2_idx is None:
        raise ValueError(get_image_hint_message(recon, image2_name))
    if image1_idx == image2_idx:
        raise ValueError("Cannot visualize epipolar geometry with the same image")

    cam1_intrinsics = recon.cameras[recon.camera_indexes[image1_idx]]
    cam2_intrinsics = recon.cameras[recon.camera_indexes[image2_idx]]

    if rectify and (
        "FISHEYE" in cam1_intrinsics.model or "FISHEYE" in cam2_intrinsics.model
    ):
        raise ValueError(
            "--rectify is not supported for fisheye cameras (no global rectifying "
            "homography exists). Use the default visualization (epipolar curves on "
            "the original images) or --undistort."
        )

    poses = _PairPoses.from_recon(recon, image1_idx, image2_idx)

    image1_path = workspace_dir / image_names[image1_idx]
    image2_path = workspace_dir / image_names[image2_idx]
    sift1_path = _require_sift_path(
        image1_path, image1_name, feature_tool, feature_options
    )
    sift2_path = _require_sift_path(
        image2_path, image2_name, feature_tool, feature_options
    )

    sweep = None
    if sweep_max_features is not None:
        sweep = _sweep_matches(
            recon,
            image1_idx,
            image2_idx,
            image1_path,
            image2_path,
            sift1_path,
            sift2_path,
            poses,
            sweep_max_features,
            sweep_window_size,
            max_features,
        )
        feature_pairs = sweep.feature_pairs
        positions1 = sweep.positions1
        positions2 = sweep.positions2
        # Sweep matches aren't tied to triangulated 3D points, so per-feature
        # track depth is unavailable; -1 sentinel routes the caller to the
        # baseline-length fallback.
        track_point_indexes = np.full(len(feature_pairs), -1, dtype=np.int64)
    else:
        feature_pairs, track_point_indexes = _shared_track_pairs(
            recon, image1_idx, image2_idx, image1_name, image2_name, max_features
        )
        with SiftReader(sift1_path) as reader:
            positions1 = reader.read_positions()
        with SiftReader(sift2_path) as reader:
            positions2 = reader.read_positions()

    colors = get_color_palette(len(feature_pairs))
    feat1_indices = [f[0] for f in feature_pairs]
    feat2_indices = [f[1] for f in feature_pairs]

    if rectify and sweep is not None and sweep.rectification is None:
        print(
            "Warning: Rectified visualization not available for in-frame epipole cases."
        )
        print("         Falling back to standard epipolar visualization.")
        rectify = False

    if rectify:
        if sweep is not None:
            undistorted = sweep.undistorted
            rectification = sweep.rectification
            rect_pts1 = sweep.rect_pts1[feat1_indices]
            rect_pts2 = sweep.rect_pts2[feat2_indices]
        else:
            undistorted = _read_and_undistort(
                recon, image1_idx, image2_idx, image1_path, image2_path
            )
            rectification = _compute_rectification(undistorted, poses)
            rect_pts1 = rectification.rectify_points_1(positions1[feat1_indices])
            rect_pts2 = rectification.rectify_points_2(positions2[feat2_indices])
        img1, img2 = undistorted.rgb_images()
        img1 = rectification.rectify_image_1(img1)
        img2 = rectification.rectify_image_2(img2)
        _draw_rectified(
            img1, img2, rect_pts1, rect_pts2, colors, feature_size, draw_lines
        )
    elif undistort:
        undistorted = _read_and_undistort(
            recon, image1_idx, image2_idx, image1_path, image2_path
        )
        img1, img2 = undistorted.rgb_images()
        _draw_undistorted(
            img1,
            img2,
            undistorted,
            poses,
            positions1[feat1_indices],
            positions2[feat2_indices],
            colors,
            feature_size,
            line_thickness,
            draw_lines,
        )
    else:
        try:
            img1 = read_image_rgb(image1_path)
            img2 = read_image_rgb(image2_path)
        except OSError as e:
            raise ValueError(f"Failed to load image files: {e}") from e
        _draw_on_original(
            img1,
            img2,
            recon,
            cam1_intrinsics,
            cam2_intrinsics,
            poses,
            np.ascontiguousarray(positions1[feat1_indices, :2], dtype=np.float64),
            np.ascontiguousarray(positions2[feat2_indices, :2], dtype=np.float64),
            track_point_indexes,
            colors,
            feature_size,
            line_thickness,
            draw_lines,
        )

    _save_output(img1, img2, output_path, side_by_side, save_which)

    if sweep is not None:
        print(f"Visualized {len(feature_pairs)} sweep matches")
    else:
        print(f"Visualized {len(feature_pairs)} shared features")
    print(f"  Image 1: {os.path.basename(image1_name)}")
    print(f"  Image 2: {os.path.basename(image2_name)}")
