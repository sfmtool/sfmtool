# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for pose-aware WarpMap constructors.

Covers :py:meth:`WarpMap.from_cameras_with_rotation` and
:py:meth:`WarpMap.from_cameras_with_pose`, including validation against
real ``.sfmr`` reconstructions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfmtool._sfmtool import (
    CameraIntrinsics,
    RigidTransform,
    RotQuaternion,
    SfmrReconstruction,
    WarpMap,
)


# =============================================================================
# Helpers
# =============================================================================


def _pinhole(width: int, height: int, focal: float) -> CameraIntrinsics:
    return CameraIntrinsics(
        "PINHOLE",
        width,
        height,
        {
            "focal_length_x": focal,
            "focal_length_y": focal,
            "principal_point_x": width / 2.0,
            "principal_point_y": height / 2.0,
        },
    )


def _simple_radial_fisheye(
    width: int, height: int, focal: float, k1: float
) -> CameraIntrinsics:
    return CameraIntrinsics(
        "SIMPLE_RADIAL_FISHEYE",
        width,
        height,
        {
            "focal_length": focal,
            "principal_point_x": width / 2.0,
            "principal_point_y": height / 2.0,
            "radial_distortion_k1": k1,
        },
    )


def _equirect(width: int, height: int) -> CameraIntrinsics:
    return CameraIntrinsics(
        "EQUIRECTANGULAR",
        width,
        height,
        {
            "focal_length_x": width / (2.0 * np.pi),
            "focal_length_y": height / np.pi,
            "principal_point_x": width / 2.0,
            "principal_point_y": height / 2.0,
        },
    )


def _max_diff(a: WarpMap, b: WarpMap) -> float:
    """Max pixel distance between two warp maps at pixels valid in both."""
    ax, ay = a.to_numpy()
    bx, by = b.to_numpy()
    valid = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(bx) & np.isfinite(by)
    if not np.any(valid):
        return 0.0
    dx = ax[valid] - bx[valid]
    dy = ay[valid] - by[valid]
    return float(np.max(np.hypot(dx, dy)))


# =============================================================================
# Synthetic tests (don't need a reconstruction)
# =============================================================================


class TestRotationOnlyIdentity:
    """``from_cameras_with_rotation(identity)`` must equal ``from_cameras``."""

    def test_pinhole(self):
        cam = _pinhole(64, 48, 100.0)
        baseline = WarpMap.from_cameras(src=cam, dst=cam)
        rotated = WarpMap.from_cameras_with_rotation(
            src=cam, dst=cam, rot_src_from_dst=RotQuaternion.identity()
        )
        assert _max_diff(baseline, rotated) < 1e-4

    def test_fisheye_to_equirect(self):
        src = _simple_radial_fisheye(200, 200, 100.0, 0.0)
        dst = _equirect(400, 200)
        baseline = WarpMap.from_cameras(src=src, dst=dst)
        rotated = WarpMap.from_cameras_with_rotation(
            src=src, dst=dst, rot_src_from_dst=RotQuaternion.identity()
        )
        assert _max_diff(baseline, rotated) < 1e-3


class TestPoseInfinityShortCircuit:
    """``depth=inf`` must equal the rotation-only path for the same R_sd."""

    def test_pinhole(self):
        cam = _pinhole(64, 48, 100.0)

        src_rot = RotQuaternion.from_axis_angle([0.0, 1.0, 0.0], 0.2)
        src_pose = RigidTransform(src_rot, [0.5, -0.2, 3.0])
        dst_pose = RigidTransform(None, [-0.1, 0.3, 2.5])

        # Relative rotation src-from-dst = R_sw * R_dw^T.
        r_sw = src_pose.rotation.to_rotation_matrix()
        r_dw = dst_pose.rotation.to_rotation_matrix()
        r_sd = RotQuaternion.from_rotation_matrix(r_sw @ r_dw.T)

        pose_inf = WarpMap.from_cameras_with_pose(
            src=cam,
            dst=cam,
            src_from_world=src_pose,
            dst_from_world=dst_pose,
            depth=float("inf"),
        )
        rot_only = WarpMap.from_cameras_with_rotation(
            src=cam, dst=cam, rot_src_from_dst=r_sd
        )
        assert _max_diff(pose_inf, rot_only) < 1e-3


class TestPoseCoincident:
    """If src and dst share a pose, from_cameras_with_pose must equal from_cameras."""

    def test_pinhole(self):
        cam = _pinhole(64, 48, 100.0)
        pose = RigidTransform(
            RotQuaternion.from_axis_angle([1.0, 0.5, 0.2], 0.4),
            [2.0, -1.0, 0.5],
        )
        baseline = WarpMap.from_cameras(src=cam, dst=cam)
        posed = WarpMap.from_cameras_with_pose(
            src=cam,
            dst=cam,
            src_from_world=pose,
            dst_from_world=pose,
            depth=5.0,
        )
        assert _max_diff(baseline, posed) < 1e-2


class TestKnownDepthSphere:
    """A sphere of radius r around dst must align under depth=r."""

    def test_pinhole(self):
        src = _pinhole(160, 120, 200.0)
        dst = _pinhole(160, 120, 200.0)

        src_pose = RigidTransform(
            RotQuaternion.from_axis_angle([0.1, 1.0, 0.05], 0.15),
            [0.5, 0.0, 0.0],
        )
        dst_pose = RigidTransform(
            RotQuaternion.from_axis_angle([0.0, 1.0, 0.1], -0.05),
            [0.0, 0.0, 0.0],
        )
        radius = 10.0

        warp = WarpMap.from_cameras_with_pose(
            src=src,
            dst=dst,
            src_from_world=src_pose,
            dst_from_world=dst_pose,
            depth=radius,
        )
        mx, my = warp.to_numpy()

        # For each pixel, manually project a sphere-point and compare.
        r_sw = src_pose.rotation.to_rotation_matrix()
        r_dw = dst_pose.rotation.to_rotation_matrix()
        t_sw = src_pose.translation
        t_dw = dst_pose.translation

        h, w = mx.shape
        checked = 0
        max_err = 0.0
        rng = np.random.default_rng(seed=1234)
        sample_cols = rng.integers(w // 10, 9 * w // 10, size=200)
        sample_rows = rng.integers(h // 10, 9 * h // 10, size=200)
        for col, row in zip(sample_cols, sample_rows):
            if not np.isfinite(mx[row, col]):
                continue
            u = col + 0.5
            v = row + 0.5
            d_dst = np.array(dst.pixel_to_ray(u, v))
            p_dst = radius * d_dst
            p_world = r_dw.T @ (p_dst - t_dw)
            p_src = r_sw @ p_world + t_sw
            exp = src.ray_to_pixel(p_src.tolist())
            assert exp is not None
            err = np.hypot(mx[row, col] - exp[0], my[row, col] - exp[1])
            max_err = max(max_err, err)
            checked += 1
        assert checked > 50
        assert max_err < 1e-2, f"max alignment error {max_err} px"


class TestEquirectDstWithPose:
    """Equirectangular dst should still produce a well-formed warp."""

    def test_center_valid(self):
        src = _pinhole(200, 200, 100.0)
        dst = _equirect(400, 200)
        pose = RigidTransform.identity()
        warp = WarpMap.from_cameras_with_pose(
            src=src,
            dst=dst,
            src_from_world=pose,
            dst_from_world=pose,
            depth=1e6,
        )
        mx, my = warp.to_numpy()
        # Centre of the equirect (forward direction) must be valid in src.
        cx = warp.width // 2
        cy = warp.height // 2
        assert np.isfinite(mx[cy, cx])
        assert np.isfinite(my[cy, cx])


# =============================================================================
# Tests against a real reconstruction (seoul_bull, 17 images)
# =============================================================================


def _pose_from_image(recon: SfmrReconstruction, idx: int) -> RigidTransform:
    qs = recon.quaternions_wxyz
    ts = recon.translations
    return RigidTransform.from_wxyz_translation(qs[idx].tolist(), ts[idx].tolist())


def _project_world_point(
    recon: SfmrReconstruction, img_idx: int, p_world: np.ndarray
) -> tuple[tuple[float, float] | None, float]:
    """Project a world point through image ``img_idx`` and return
    ``((pixel_x, pixel_y) | None, radial_depth)``."""
    cam_idx = recon.camera_indexes[img_idx]
    cam = recon.cameras[cam_idx]
    pose = _pose_from_image(recon, img_idx)
    p_cam = pose.transform_point(p_world.tolist())
    depth = float(np.linalg.norm(p_cam))
    pixel = cam.ray_to_pixel(np.asarray(p_cam).tolist())
    return pixel, depth


class TestRealReconstruction:
    """Validate the pose-aware warp on a real seoul_bull reconstruction.

    For each co-visible 3D point ``P`` between two images we can compute:
    - the pixel where it projects in dst: ``(u_B, v_B)``
    - its radial distance from the dst camera: ``r``
    - the pixel where it projects in src: ``(u_A, v_A)``

    Building the warp with ``depth=r`` should land ``(u_B, v_B)`` at
    ``(u_A, v_A)`` to within one pixel.
    """

    def _setup(
        self, sfmrfile_reconstruction_with_17_images
    ) -> tuple[SfmrReconstruction, int, int, np.ndarray]:
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        assert recon.image_count == 17
        # Use neighbouring images — they typically share many 3D points.
        return recon, 0, 1, recon.positions

    def test_per_point_reprojection(self, sfmrfile_reconstruction_with_17_images):
        recon, src_idx, dst_idx, positions = self._setup(
            sfmrfile_reconstruction_with_17_images
        )
        src_cam = recon.cameras[recon.camera_indexes[src_idx]]
        dst_cam = recon.cameras[recon.camera_indexes[dst_idx]]
        src_pose = _pose_from_image(recon, src_idx)
        dst_pose = _pose_from_image(recon, dst_idx)

        # Collect 3D points that project inside both images.
        samples = []
        for point_id, p in enumerate(positions):
            a_uv, _ = _project_world_point(recon, src_idx, p)
            b_uv, r = _project_world_point(recon, dst_idx, p)
            if a_uv is None or b_uv is None:
                continue
            if not (0 <= a_uv[0] < src_cam.width and 0 <= a_uv[1] < src_cam.height):
                continue
            if not (0 <= b_uv[0] < dst_cam.width and 0 <= b_uv[1] < dst_cam.height):
                continue
            if r <= 0:
                continue
            samples.append((a_uv, b_uv, r))

        assert len(samples) > 20, (
            f"not enough shared 3D points between images "
            f"{src_idx}/{dst_idx}: {len(samples)}"
        )

        # For each sample, build a warp at that point's radial depth and
        # check that the warp at the dst pixel maps to the src pixel.
        errors = []
        # Sample up to 10 points to keep the test fast.
        for a_uv, b_uv, r in samples[:10]:
            warp = WarpMap.from_cameras_with_pose(
                src=src_cam,
                dst=dst_cam,
                src_from_world=src_pose,
                dst_from_world=dst_pose,
                depth=r,
            )
            mx, my = warp.to_numpy()
            col = int(b_uv[0])
            row = int(b_uv[1])
            # Bilinear sample at the fractional pixel for a cleaner read.
            x = b_uv[0] - 0.5
            y = b_uv[1] - 0.5
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = x0 + 1, y0 + 1
            x0 = max(0, min(mx.shape[1] - 1, x0))
            x1 = max(0, min(mx.shape[1] - 1, x1))
            y0 = max(0, min(mx.shape[0] - 1, y0))
            y1 = max(0, min(mx.shape[0] - 1, y1))
            ax = x - np.floor(x)
            ay = y - np.floor(y)

            def sample(m):
                return (
                    (1 - ax) * (1 - ay) * m[y0, x0]
                    + ax * (1 - ay) * m[y0, x1]
                    + (1 - ax) * ay * m[y1, x0]
                    + ax * ay * m[y1, x1]
                )

            if not (
                np.isfinite(mx[y0, x0])
                and np.isfinite(mx[y1, x1])
                and np.isfinite(mx[y0, x1])
                and np.isfinite(mx[y1, x0])
            ):
                # Fall back to nearest pixel if any corner is NaN.
                sx, sy = mx[row, col], my[row, col]
            else:
                sx = sample(mx)
                sy = sample(my)
            err = np.hypot(sx - a_uv[0], sy - a_uv[1])
            errors.append(err)

        errors = np.array(errors)
        # The warp formulation is exact for a point on the sphere of radius
        # ``r`` — remaining error comes from bilinear sampling of the warp
        # at a sub-pixel location, which should be well under 1 px.
        assert np.max(errors) < 1.0, (
            f"max per-point reprojection error: {np.max(errors):.3f} px; errors={errors}"
        )

    def test_rotation_only_vs_pose_diverge_for_nearby_scene(
        self, sfmrfile_reconstruction_with_17_images
    ):
        """At scene-comparable baselines, the rotation-only approximation
        disagrees with the exact pose-aware formulation by many pixels."""
        recon, src_idx, dst_idx, positions = self._setup(
            sfmrfile_reconstruction_with_17_images
        )
        src_cam = recon.cameras[recon.camera_indexes[src_idx]]
        dst_cam = recon.cameras[recon.camera_indexes[dst_idx]]
        src_pose = _pose_from_image(recon, src_idx)
        dst_pose = _pose_from_image(recon, dst_idx)

        # Median radial depth of co-visible points.
        rs = []
        for p in positions:
            _, r = _project_world_point(recon, dst_idx, p)
            if r > 0:
                rs.append(r)
        assert rs
        r_med = float(np.median(rs))

        # Baseline between the two cameras.
        src_center = np.asarray(src_pose.inverse_translation_origin())
        dst_center = np.asarray(dst_pose.inverse_translation_origin())
        baseline = float(np.linalg.norm(src_center - dst_center))

        # On seoul_bull baseline (~0.6) is comparable to scene depth
        # (~6) — r/B ≈ 10 — so a rotation-only approximation is already
        # off by tens of pixels. Use depth = scene-median radial distance:
        # this is the "best" depth for a sphere approximation, and the
        # rotation-only formula ignoring translation still disagrees with
        # the exact formula by many pixels across the frame.
        exact = WarpMap.from_cameras_with_pose(
            src=src_cam,
            dst=dst_cam,
            src_from_world=src_pose,
            dst_from_world=dst_pose,
            depth=r_med,
        )
        r_sd = RotQuaternion.from_rotation_matrix(
            src_pose.rotation.to_rotation_matrix()
            @ dst_pose.rotation.to_rotation_matrix().T
        )
        approx = WarpMap.from_cameras_with_rotation(
            src=src_cam,
            dst=dst_cam,
            rot_src_from_dst=r_sd,
        )
        diff = _max_diff(exact, approx)
        assert diff > 10.0, (
            f"rotation-only vs exact should disagree by many pixels; "
            f"got max diff {diff}, baseline {baseline}, r_med {r_med}"
        )

    def test_remap_real_image(
        self,
        sfmrfile_reconstruction_with_17_images,
    ):
        """Build a warp between two real images and remap the source image.

        Validates:
          - ``remap_bilinear`` executes on a pose-aware map without error
          - the output image has the expected shape
          - a non-trivial fraction of output pixels are non-black (i.e. the
            warp lands many dst pixels inside the src image)
        """
        import cv2

        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        src_idx, dst_idx = 0, 1

        src_cam = recon.cameras[recon.camera_indexes[src_idx]]
        dst_cam = recon.cameras[recon.camera_indexes[dst_idx]]
        src_pose = _pose_from_image(recon, src_idx)
        dst_pose = _pose_from_image(recon, dst_idx)

        # Median radial depth of co-visible points — a natural "scene depth".
        rs = [
            r
            for p in recon.positions
            for r in [_project_world_point(recon, dst_idx, p)[1]]
            if r > 0
        ]
        depth = float(np.median(rs))

        workspace = Path(sfmrfile_reconstruction_with_17_images).parent
        image_name = recon.image_names[src_idx]
        image_path = workspace / image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        assert image is not None

        warp = WarpMap.from_cameras_with_pose(
            src=src_cam,
            dst=dst_cam,
            src_from_world=src_pose,
            dst_from_world=dst_pose,
            depth=depth,
        )
        out = warp.remap_bilinear(image)
        assert out.shape[0] == dst_cam.height
        assert out.shape[1] == dst_cam.width
        # Most of the dst frame should be filled (non-zero).
        non_zero_frac = (
            float(np.mean(out.sum(axis=-1) > 0))
            if out.ndim == 3
            else float(np.mean(out > 0))
        )
        assert non_zero_frac > 0.5, (
            f"expected more than half of the warped frame to fall inside the "
            f"src image, got {non_zero_frac:.2f}"
        )

    def test_equirect_destination_from_pinhole(
        self, sfmrfile_reconstruction_with_17_images
    ):
        """Render a pinhole source image into an equirectangular destination
        at a specific pose, using a scene-comparable depth. At least one
        back-projection lane (the src camera's FOV) should land valid
        pixels; everything outside that FOV is NaN.
        """
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        src_idx = 0
        src_cam = recon.cameras[recon.camera_indexes[src_idx]]
        src_pose = _pose_from_image(recon, src_idx)

        # Build an equirect dst co-located with src so the src camera's
        # forward direction corresponds to the equirect centre.
        dst_cam = _equirect(720, 360)
        dst_pose = src_pose  # same pose — equirect camera sits where src is

        # Scene depth is irrelevant when src and dst share a pose (the
        # translation term t_sd is zero), so we use a mild finite depth.
        warp = WarpMap.from_cameras_with_pose(
            src=src_cam,
            dst=dst_cam,
            src_from_world=src_pose,
            dst_from_world=dst_pose,
            depth=10.0,
        )
        mx, my = warp.to_numpy()
        # Centre of the equirect frame looks along +Z; forward of a
        # reconstruction camera should project into the src image.
        cy = warp.height // 2
        cx = warp.width // 2
        assert np.isfinite(mx[cy, cx])
        assert np.isfinite(my[cy, cx])
        # And a non-trivial patch around the centre should be valid.
        patch = np.isfinite(mx[cy - 20 : cy + 20, cx - 20 : cx + 20])
        assert patch.sum() > 400
