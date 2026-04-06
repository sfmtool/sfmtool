# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Stereo rectification utilities for calibrated camera pairs."""

from dataclasses import dataclass

import cv2
import numpy as np
import pycolmap

from ._cameras import get_intrinsic_matrix


@dataclass
class StereoRectification:
    """Stereo rectification parameters for a calibrated camera pair.

    Attributes:
        K1, K2: Intrinsic matrices (3x3)
        R1_rect, R2_rect: Rectification rotations (3x3)
        P1, P2: Rectification projection matrices (3x4)
        Q: Disparity-to-depth mapping matrix (4x4)
        valid_region_1, valid_region_2: (x, y, width, height) tuples
        cam1_distorted, cam2_distorted: Original cameras with distortion
        cam1_undistorted, cam2_undistorted: Undistorted PINHOLE cameras
    """

    K1: np.ndarray
    K2: np.ndarray
    R1_rect: np.ndarray
    R2_rect: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    valid_region_1: tuple[int, int, int, int]
    valid_region_2: tuple[int, int, int, int]
    cam1_distorted: pycolmap.Camera
    cam2_distorted: pycolmap.Camera
    cam1_undistorted: pycolmap.Camera
    cam2_undistorted: pycolmap.Camera

    def rectify_image_1(self, image: np.ndarray) -> np.ndarray:
        """Rectify an image from camera 1."""
        return rectify_image(
            image, self.cam1_undistorted, self.K1, self.R1_rect, self.P1
        )

    def rectify_image_2(self, image: np.ndarray) -> np.ndarray:
        """Rectify an image from camera 2."""
        return rectify_image(
            image, self.cam2_undistorted, self.K2, self.R2_rect, self.P2
        )

    def rectify_points_1(self, points: np.ndarray) -> np.ndarray:
        """Rectify points from camera 1: distorted -> rectified."""
        return undistort_and_rectify_points(
            points,
            self.cam1_distorted,
            self.cam1_undistorted,
            self.K1,
            self.R1_rect,
            self.P1,
        )

    def rectify_points_2(self, points: np.ndarray) -> np.ndarray:
        """Rectify points from camera 2: distorted -> rectified."""
        return undistort_and_rectify_points(
            points,
            self.cam2_distorted,
            self.cam2_undistorted,
            self.K2,
            self.R2_rect,
            self.P2,
        )


def compute_stereo_rectification(
    cam1_distorted: pycolmap.Camera,
    cam2_distorted: pycolmap.Camera,
    cam1_undistorted: pycolmap.Camera,
    cam2_undistorted: pycolmap.Camera,
    R_rel: np.ndarray,
    t_rel: np.ndarray,
) -> StereoRectification:
    """Compute stereo rectification parameters for undistorted PINHOLE cameras.

    Args:
        cam1_distorted: Original camera 1 with distortion
        cam2_distorted: Original camera 2 with distortion
        cam1_undistorted: Undistorted PINHOLE camera 1
        cam2_undistorted: Undistorted PINHOLE camera 2
        R_rel: Relative rotation from cam1 to cam2 (3x3)
        t_rel: Relative translation from cam1 to cam2 (3x1)

    Returns:
        StereoRectification object with all parameters and convenience methods
    """
    K1 = get_intrinsic_matrix(cam1_undistorted)
    K2 = get_intrinsic_matrix(cam2_undistorted)

    img_size1 = (cam1_undistorted.width, cam1_undistorted.height)
    D = np.zeros(5)

    R1_rect, R2_rect, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1,
        D,
        K2,
        D,
        img_size1,
        R_rel,
        t_rel,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1.0,
    )

    return StereoRectification(
        K1=K1,
        K2=K2,
        R1_rect=R1_rect,
        R2_rect=R2_rect,
        P1=P1,
        P2=P2,
        Q=Q,
        valid_region_1=tuple(roi1),
        valid_region_2=tuple(roi2),
        cam1_distorted=cam1_distorted,
        cam2_distorted=cam2_distorted,
        cam1_undistorted=cam1_undistorted,
        cam2_undistorted=cam2_undistorted,
    )


def rectify_image(
    image: np.ndarray,
    cam_undistorted: pycolmap.Camera,
    K_rect: np.ndarray,
    R_rect: np.ndarray,
    P_rect: np.ndarray,
) -> np.ndarray:
    """Rectify a single undistorted pinhole image."""
    img_size = (cam_undistorted.width, cam_undistorted.height)
    D = np.zeros(5)

    map1, map2 = cv2.initUndistortRectifyMap(
        K_rect, D, R_rect, P_rect, img_size, cv2.CV_32FC1
    )

    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)


def rectify_points(
    points: np.ndarray,
    K_rect: np.ndarray,
    R_rect: np.ndarray,
    P_rect: np.ndarray,
) -> np.ndarray:
    """Transform 2D keypoints from undistorted pinhole image to rectified image.

    Args:
        points: (N, 2) array of points in the undistorted pinhole image
        K_rect: Intrinsic matrix of the input pinhole camera
        R_rect: Rectification rotation matrix
        P_rect: Rectification projection matrix

    Returns:
        (N, 2) array of rectified points
    """
    D = np.zeros(5)
    pts_reshaped = points.reshape(-1, 1, 2)
    rect_pts = cv2.undistortPoints(pts_reshaped, K_rect, D, R=R_rect, P=P_rect)
    return rect_pts.reshape(-1, 2)


def undistort_and_rectify_points(
    points: np.ndarray,
    cam_distorted: pycolmap.Camera,
    cam_undistorted: pycolmap.Camera,
    K_undist: np.ndarray,
    R_rect: np.ndarray,
    P_rect: np.ndarray,
) -> np.ndarray:
    """Undistort and rectify points: distorted -> normalized -> undistorted -> rectified.

    Args:
        points: (N, 2) points in original distorted image
        cam_distorted: Original camera with distortion
        cam_undistorted: Undistorted PINHOLE camera
        K_undist: Intrinsic matrix of undistorted camera
        R_rect: Rectification rotation matrix
        P_rect: Rectification projection matrix

    Returns:
        (N, 2) points in rectified image coordinates
    """
    normalized = cam_distorted.cam_from_img(points)
    normalized_h = np.hstack([normalized, np.ones((len(normalized), 1))])
    undist_pts = (K_undist @ normalized_h.T).T[:, :2]
    rect_pts = rectify_points(undist_pts, K_undist, R_rect, P_rect)
    return rect_pts
