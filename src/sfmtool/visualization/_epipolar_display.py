# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Epipolar geometry visualization for SfM reconstructions."""

import colorsys
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from ..camera.cameras import colmap_camera_from_intrinsics, get_intrinsic_matrix
from .._rectification import compute_stereo_rectification
from ..sift.file import SiftReader, get_sift_path_for_image
from .._sfmtool import RotQuaternion, epipolar_curves


def _get_color_palette(n_colors: int) -> list:
    """Generate a cycling color palette with distinct colors randomized.

    Returns:
        List of (B, G, R) tuples for use with cv2
    """
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    random.Random(42).shuffle(colors)
    return colors


def _compute_fundamental_matrix(
    K1: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    K2: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
) -> np.ndarray:
    """Compute fundamental matrix from two camera poses.

    F relates corresponding points: x2^T F x1 = 0.
    """
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    t_skew = np.array(
        [[0, -t_rel[2], t_rel[1]], [t_rel[2], 0, -t_rel[0]], [-t_rel[1], t_rel[0], 0]]
    )
    E = t_skew @ R_rel
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F


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
    track_point_ids: np.ndarray,
    R_from: np.ndarray,
    t_from: np.ndarray,
    R_other: np.ndarray,
    t_other: np.ndarray,
) -> np.ndarray:
    """Per-feature seed depth for epipolar-curve sampling, in `R_from` coords.

    Uses the reconstructed track depth when a triangulated 3D point is
    available (and is in front of the camera); otherwise the baseline length
    `‖C_other − C_from‖`. `track_point_ids` is one entry per feature pair with
    `-1` marking unmatched features. See specs/core/epipolar-curves.md
    ("Caller-side seeding strategy") for the rationale.
    """
    c_from = -R_from.T @ t_from
    c_other = -R_other.T @ t_other
    baseline = float(np.linalg.norm(c_other - c_from))
    if baseline < 1e-9:
        baseline = 1.0  # degenerate; Rust returns empty anyway.

    depths = np.full(len(track_point_ids), baseline, dtype=np.float64)
    valid = track_point_ids >= 0
    if not valid.any():
        return depths
    pids = track_point_ids[valid]
    points = np.asarray(recon.positions)[pids]
    track_depths = points @ R_from[2, :] + t_from[2]
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
    camera_indexes = recon.camera_indexes
    quaternions = recon.quaternions_wxyz
    translations = recon.translations
    cameras = recon.cameras
    image_indexes = recon.track_image_indexes
    feature_indexes = recon.track_feature_indexes
    point_ids = recon.track_point_ids

    observations = np.column_stack([image_indexes, feature_indexes])

    # Find image indices in reconstruction. Prefer exact full-path matches so
    # that datasets with the same basename under multiple subdirectories (e.g.
    # rig sensors with `fisheye_left/frame_01.jpg` and
    # `fisheye_right/frame_01.jpg`) resolve unambiguously.
    def _find_index(query: str) -> int | None:
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

    image1_idx = _find_index(image1_name)
    image2_idx = _find_index(image2_name)

    from .._sfm_filenames import get_image_hint_message

    if image1_idx is None:
        raise ValueError(get_image_hint_message(recon, image1_name))
    if image2_idx is None:
        raise ValueError(get_image_hint_message(recon, image2_name))
    if image1_idx == image2_idx:
        raise ValueError("Cannot visualize epipolar geometry with the same image")

    cam1_intrinsics = cameras[camera_indexes[image1_idx]]
    cam2_intrinsics = cameras[camera_indexes[image2_idx]]

    if rectify and (
        "FISHEYE" in cam1_intrinsics.model or "FISHEYE" in cam2_intrinsics.model
    ):
        raise ValueError(
            "--rectify is not supported for fisheye cameras (no global rectifying "
            "homography exists). Use the default visualization (epipolar curves on "
            "the original images) or --undistort."
        )

    # Get camera poses (cam_from_world)
    quat1 = quaternions[image1_idx]
    quat2 = quaternions[image2_idx]
    R1 = RotQuaternion.from_wxyz_array(quat1).to_rotation_matrix()
    t1 = translations[image1_idx]
    R2 = RotQuaternion.from_wxyz_array(quat2).to_rotation_matrix()
    t2 = translations[image2_idx]

    # Load SIFT features
    image1_path = workspace_dir / image_names[image1_idx]
    image2_path = workspace_dir / image_names[image2_idx]

    sift1_path = get_sift_path_for_image(
        image1_path,
        feature_tool=feature_tool,
        feature_options=feature_options,
    )
    sift2_path = get_sift_path_for_image(
        image2_path,
        feature_tool=feature_tool,
        feature_options=feature_options,
    )

    if not sift1_path.exists():
        raise FileNotFoundError(
            f"SIFT file not found for {image1_name}. Expected at: {sift1_path}."
        )
    if not sift2_path.exists():
        raise FileNotFoundError(
            f"SIFT file not found for {image2_name}. Expected at: {sift2_path}."
        )

    # Track rectification state for later branches
    rectification = None

    if sweep_max_features is not None:
        # Sort and Sweep matching
        with SiftReader(sift1_path) as reader:
            positions1 = reader.read_positions(count=sweep_max_features)
            descriptors1 = reader.read_descriptors(count=sweep_max_features)

        with SiftReader(sift2_path) as reader:
            positions2 = reader.read_positions(count=sweep_max_features)
            descriptors2 = reader.read_descriptors(count=sweep_max_features)

        bitmap1 = pycolmap.Bitmap.read(str(image1_path), as_rgb=True)
        bitmap2 = pycolmap.Bitmap.read(str(image2_path), as_rgb=True)
        h1, w1 = bitmap1.to_array().shape[:2]
        h2, w2 = bitmap2.to_array().shape[:2]

        cam1 = colmap_camera_from_intrinsics(
            cameras[camera_indexes[image1_idx]], width=w1, height=h1
        )
        cam2 = colmap_camera_from_intrinsics(
            cameras[camera_indexes[image2_idx]], width=w2, height=h2
        )

        from ..feature_match._geometry import check_rectification_safe

        K1_check = get_intrinsic_matrix(cam1)
        K2_check = get_intrinsic_matrix(cam2)
        rectification_safe = check_rectification_safe(
            K1_check, R1, t1, K2_check, R2, t2, width=w1, height=h1, margin=50
        )

        # Create pycolmap poses
        quat1_wxyz = quaternions[image1_idx]
        quat1_xyzw = np.roll(quat1_wxyz, -1)
        pose1 = pycolmap.Rigid3d(pycolmap.Rotation3d(quat1_xyzw), t1)

        quat2_wxyz = quaternions[image2_idx]
        quat2_xyzw = np.roll(quat2_wxyz, -1)
        pose2 = pycolmap.Rigid3d(pycolmap.Rotation3d(quat2_xyzw), t2)

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
            options = pycolmap.UndistortCameraOptions()
            undist_bmp1_obj, undist_cam1 = pycolmap.undistort_image(
                options, bitmap1, cam1
            )
            undist_bmp2_obj, undist_cam2 = pycolmap.undistort_image(
                options, bitmap2, cam2
            )

            R_rel = R2 @ R1.T
            t_rel = t2 - R_rel @ t1

            rectification = compute_stereo_rectification(
                cam1, cam2, undist_cam1, undist_cam2, R_rel, t_rel
            )
            rect_pts1 = rectification.rectify_points_1(positions1)
            rect_pts2 = rectification.rectify_points_2(positions2)
        else:
            match_method = "polar sweep (in-frame epipole)"
            options = pycolmap.UndistortCameraOptions()
            undist_bmp1_obj, undist_cam1 = pycolmap.undistort_image(
                options, bitmap1, cam1
            )
            undist_bmp2_obj, undist_cam2 = pycolmap.undistort_image(
                options, bitmap2, cam2
            )

            R_rel = R2 @ R1.T
            t_rel = t2 - R_rel @ t1
            rect_pts1 = None
            rect_pts2 = None

        if not mutual_matches:
            raise ValueError(f"No mutual matches found using {match_method}.")

        if max_features is not None and len(mutual_matches) > max_features:
            indices = np.linspace(0, len(mutual_matches) - 1, max_features, dtype=int)
            mutual_matches = [mutual_matches[i] for i in indices]

        feature_pairs = [(m[0], m[1]) for m in mutual_matches]
        # Sweep matches aren't tied to triangulated 3D points, so per-feature
        # track depth is unavailable; -1 sentinel routes the caller to the
        # baseline-length fallback.
        feature_track_point_ids = np.full(len(feature_pairs), -1, dtype=np.int64)
        print(f"Found {len(mutual_matches)} matches using {match_method}")

    else:
        # Standard track-based visualization
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
        feature_track_point_ids = []
        for point_id in shared_point_ids:
            feat1_idx = obs1[point_ids1 == point_id, 1][0]
            feat2_idx = obs2[point_ids2 == point_id, 1][0]
            feature_pairs.append((feat1_idx, feat2_idx))
            feature_track_point_ids.append(point_id)

        if max_features is not None and len(feature_pairs) > max_features:
            indices = np.linspace(0, len(feature_pairs) - 1, max_features, dtype=int)
            feature_pairs = [feature_pairs[i] for i in indices]
            feature_track_point_ids = [feature_track_point_ids[i] for i in indices]
        feature_track_point_ids = np.asarray(feature_track_point_ids, dtype=np.int64)

        with SiftReader(sift1_path) as reader:
            positions1 = reader.read_positions()

        with SiftReader(sift2_path) as reader:
            positions2 = reader.read_positions()

    # Generate color palette
    colors = _get_color_palette(len(feature_pairs))

    # Handle rectification fallback for in-frame epipole
    if rectify and sweep_max_features is not None and rectification is None:
        print(
            "Warning: Rectified visualization not available for in-frame epipole cases."
        )
        print("         Falling back to standard epipolar visualization.")
        rectify = False

    if rectify:
        if sweep_max_features is None:
            bitmap1 = pycolmap.Bitmap.read(str(image1_path), as_rgb=True)
            bitmap2 = pycolmap.Bitmap.read(str(image2_path), as_rgb=True)
            if bitmap1 is None or bitmap2 is None:
                raise ValueError("Could not read image files as Bitmaps.")

            h1, w1 = bitmap1.to_array().shape[:2]
            h2, w2 = bitmap2.to_array().shape[:2]

            cam1 = colmap_camera_from_intrinsics(
                cameras[camera_indexes[image1_idx]], width=w1, height=h1
            )
            cam2 = colmap_camera_from_intrinsics(
                cameras[camera_indexes[image2_idx]], width=w2, height=h2
            )

            options = pycolmap.UndistortCameraOptions()
            options.blank_pixels = 0
            undist_bmp1_obj, undist_cam1 = pycolmap.undistort_image(
                options, bitmap1, cam1
            )
            undist_bmp2_obj, undist_cam2 = pycolmap.undistort_image(
                options, bitmap2, cam2
            )

            R_rel = R2 @ R1.T
            t_rel = t2 - R_rel @ t1

            rectification = compute_stereo_rectification(
                cam1, cam2, undist_cam1, undist_cam2, R_rel, t_rel
            )

        img1 = cv2.cvtColor(undist_bmp1_obj.to_array(), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(undist_bmp2_obj.to_array(), cv2.COLOR_RGB2BGR)

        img1 = rectification.rectify_image_1(img1)
        img2 = rectification.rectify_image_2(img2)

        if sweep_max_features is not None:
            rect_pts1_disp = rect_pts1[[f[0] for f in feature_pairs]]
            rect_pts2_disp = rect_pts2[[f[1] for f in feature_pairs]]
        else:
            pts1 = positions1[[f[0] for f in feature_pairs]]
            pts2 = positions2[[f[1] for f in feature_pairs]]
            rect_pts1_disp = rectification.rectify_points_1(pts1)
            rect_pts2_disp = rectification.rectify_points_2(pts2)

        for i in range(len(rect_pts1_disp)):
            color = colors[i]
            p1 = rect_pts1_disp[i]
            p2 = rect_pts2_disp[i]

            cv2.circle(img1, (int(p1[0]), int(p1[1])), feature_size, color, -1)
            cv2.circle(img2, (int(p2[0]), int(p2[1])), feature_size, color, -1)

            if draw_lines:
                y = int((p1[1] + p2[1]) / 2)
                cv2.line(img1, (0, y), (img1.shape[1], y), color, 1)
                cv2.line(img2, (0, y), (img2.shape[1], y), color, 1)

    elif undistort:
        bitmap1 = pycolmap.Bitmap.read(str(image1_path), as_rgb=True)
        bitmap2 = pycolmap.Bitmap.read(str(image2_path), as_rgb=True)
        if bitmap1 is None or bitmap2 is None:
            raise ValueError("Could not read image files as Bitmaps.")

        h1, w1 = bitmap1.to_array().shape[:2]
        h2, w2 = bitmap2.to_array().shape[:2]

        cam1 = colmap_camera_from_intrinsics(
            cameras[camera_indexes[image1_idx]], width=w1, height=h1
        )
        cam2 = colmap_camera_from_intrinsics(
            cameras[camera_indexes[image2_idx]], width=w2, height=h2
        )

        options = pycolmap.UndistortCameraOptions()
        options.blank_pixels = 0
        undist_bmp1_obj, undist_cam1 = pycolmap.undistort_image(options, bitmap1, cam1)
        undist_bmp2_obj, undist_cam2 = pycolmap.undistort_image(options, bitmap2, cam2)

        img1 = cv2.cvtColor(undist_bmp1_obj.to_array(), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(undist_bmp2_obj.to_array(), cv2.COLOR_RGB2BGR)

        pts1 = positions1[[f[0] for f in feature_pairs]]
        pts2 = positions2[[f[1] for f in feature_pairs]]

        undist_pts1 = np.array([cam1.cam_from_img(p) for p in pts1])
        undist_pts2 = np.array([cam2.cam_from_img(p) for p in pts2])

        undist_pts1 = np.array([undist_cam1.img_from_cam(p) for p in undist_pts1])
        undist_pts2 = np.array([undist_cam2.img_from_cam(p) for p in undist_pts2])

        K1_undist = get_intrinsic_matrix(undist_cam1)
        K2_undist = get_intrinsic_matrix(undist_cam2)
        F_undist = _compute_fundamental_matrix(K1_undist, R1, t1, K2_undist, R2, t2)

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

    else:
        # Standard visualization on the original (distorted) images. The
        # epipolar geometry is drawn directly through the full camera model, so
        # the "lines" are curves for fisheye / wide-FOV cameras. See
        # specs/core/epipolar-curves.md.
        img1 = cv2.imread(
            str(image1_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img2 = cv2.imread(
            str(image2_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        if img1 is None or img2 is None:
            raise ValueError("Failed to load image files.")

        feat1_indices = [f[0] for f in feature_pairs]
        feat2_indices = [f[1] for f in feature_pairs]
        pts1 = np.ascontiguousarray(positions1[feat1_indices, :2], dtype=np.float64)
        pts2 = np.ascontiguousarray(positions2[feat2_indices, :2], dtype=np.float64)

        if draw_lines:
            anchors_from_cam1 = _curve_anchor_depths(
                recon, feature_track_point_ids, R1, t1, R2, t2
            )
            anchors_from_cam2 = _curve_anchor_depths(
                recon, feature_track_point_ids, R2, t2, R1, t1
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

    # Output
    output_path.parent.mkdir(exist_ok=True, parents=True)

    if side_by_side:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        max_height = max(h1, h2)

        output = np.zeros((max_height, w1 + w2, 3), dtype=np.uint8)
        output[:h1, :w1] = img1
        output[:h2, w1 : w1 + w2] = img2

        cv2.imwrite(str(output_path), output)
        print(f"Visualized pairs to: {output_path} (side-by-side)")
    else:
        if save_which == "both":
            stem = output_path.stem
            ext = output_path.suffix
            output_path_other = output_path.with_name(f"{stem}_other{ext}")

            cv2.imwrite(str(output_path), img1)
            cv2.imwrite(str(output_path_other), img2)
            print(f"Visualized pairs to: {output_path} and {output_path_other}")
        elif save_which == "first":
            cv2.imwrite(str(output_path), img1)
            print(f"Visualized pairs to: {output_path}")
        elif save_which == "second":
            cv2.imwrite(str(output_path), img2)
            print(f"Visualized pairs to: {output_path}")
        else:
            raise ValueError(
                f"Invalid save_which value: {save_which}. Must be 'both', 'first', or 'second'."
            )

    if sweep_max_features is not None:
        print(f"Visualized {len(feature_pairs)} sweep matches")
    else:
        print(f"Visualized {len(feature_pairs)} shared features")
    print(f"  Image 1: {os.path.basename(image1_name)}")
    print(f"  Image 2: {os.path.basename(image2_name)}")
