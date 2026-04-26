# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for SphericalTileRig (Rust binding) and the equirect resampling helper."""

import numpy as np
import pytest

from sfmtool import resample_atlas_to_equirect
from sfmtool._sfmtool import (
    CameraIntrinsics,
    RotQuaternion,
    Se3Transform,
    SphericalTileRig,
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


def _smooth_pattern(dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """Direction-only smooth pattern, evaluated at a unit world direction.

    Used both to fill an atlas (per tile-pixel world ray) and to build the
    equirect reference (per equirect-pixel world direction), so the atlas →
    equirect resample is exercised against a ground truth that depends only
    on direction, not on atlas packing.
    """
    return 0.5 + 0.2 * dx + 0.15 * dy * dz


def _atlas_pattern(rig: SphericalTileRig) -> np.ndarray:
    """Fill an atlas by evaluating ``_smooth_pattern`` at each tile pixel's
    in-tile world ray."""
    aw, ah = rig.atlas_size
    img = np.zeros((ah, aw), dtype=np.float32)
    cam = rig.tile_camera()
    fx, fy = cam.focal_lengths
    cx, cy = cam.principal_point
    p = rig.patch_size
    for idx in range(len(rig)):
        ox, oy = rig.tile_atlas_origin(idx)
        rot_mat = rig.tile_rotation(idx)  # (3, 3) row-major
        # In-tile pixel grid: (in_x + 0.5, in_y + 0.5).
        ix = np.arange(p) + 0.5
        iy = np.arange(p) + 0.5
        u, v = np.meshgrid(ix, iy, indexing="xy")
        # Tile-frame rays.
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = np.ones_like(x)
        rays = np.stack([x, y, z], axis=-1)  # (p, p, 3)
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        rays /= norms
        # Tile → world: R @ ray.
        world = rays @ rot_mat.T
        wx, wy, wz = world[..., 0], world[..., 1], world[..., 2]
        vals = _smooth_pattern(wx, wy, wz)
        img[oy : oy + p, ox : ox + p] = vals.astype(np.float32)
    return img


def _equirect_pattern_reference(width: int, height: int) -> np.ndarray:
    """Evaluate ``_smooth_pattern`` at every equirect pixel direction."""
    cols = np.arange(width) + 0.5
    rows = np.arange(height) + 0.5
    lon = (cols - width / 2.0) * (2.0 * np.pi / width)
    lat = -(rows - height / 2.0) * (np.pi / height)
    cos_lat = np.cos(lat)[:, None]
    sin_lat = np.sin(lat)[:, None]
    sin_lon = np.sin(lon)[None, :]
    cos_lon = np.cos(lon)[None, :]
    dx = sin_lon * cos_lat
    dy = np.broadcast_to(sin_lat, (height, width))
    dz = cos_lon * cos_lat
    return _smooth_pattern(dx, dy, dz).astype(np.float32)


class TestSphericalTileRigBasics:
    def test_construction_returns_expected_attributes(self):
        rig = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, seed=42)
        assert len(rig) == 80
        assert rig.n == 80
        assert rig.patch_size > 0
        aw, ah = rig.atlas_size
        assert aw == rig.atlas_cols * rig.patch_size
        assert ah == rig.atlas_rows * rig.patch_size
        assert rig.half_fov_rad > 0
        assert rig.measured_max_coverage_angle > 0

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            SphericalTileRig(n=1, arc_per_pixel=2 * np.pi / 256)

    def test_invalid_arc_per_pixel_raises(self):
        with pytest.raises(ValueError):
            SphericalTileRig(n=80, arc_per_pixel=0.0)

    def test_invalid_overlap_factor_raises(self):
        with pytest.raises(ValueError):
            SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, overlap_factor=0.5)

    def test_seed_makes_construction_deterministic(self):
        a = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, seed=7)
        b = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, seed=7)
        np.testing.assert_array_equal(a.directions(), b.directions())
        assert a.half_fov_rad == b.half_fov_rad
        assert a.patch_size == b.patch_size

    def test_directions_are_unit_norm(self):
        rig = SphericalTileRig(n=200, arc_per_pixel=2 * np.pi / 512, seed=1)
        dirs = rig.directions()
        assert dirs.shape == (200, 3)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_basis_is_orthonormal_and_right_handed(self):
        rig = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, seed=2)
        for i in range(len(rig)):
            er, eu = rig.basis(i)
            d = rig.direction(i)
            er = np.array(er)
            eu = np.array(eu)
            d = np.array(d)
            assert abs(np.dot(er, eu)) < 1e-9
            assert abs(np.dot(er, d)) < 1e-9
            assert abs(np.dot(eu, d)) < 1e-9
            np.testing.assert_allclose(np.cross(er, eu), d, atol=1e-9)

    def test_tile_atlas_origin_matches_packing(self):
        rig = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, seed=3)
        cols = rig.atlas_cols
        p = rig.patch_size
        for i in range(len(rig)):
            ox, oy = rig.tile_atlas_origin(i)
            assert ox == (i % cols) * p
            assert oy == (i // cols) * p

    def test_apply_transform_translates_centre(self):
        rig = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, seed=4)
        t = Se3Transform(
            rotation=RotQuaternion.identity(),
            translation=[1.0, 2.0, 3.0],
            scale=1.0,
        )
        rig.apply_transform(t)
        np.testing.assert_allclose(rig.centre, [1.0, 2.0, 3.0])

    def test_warp_to_atlas_shape_matches_atlas_size(self):
        rig = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, seed=5)
        equirect = _equirect(256, 128)
        warp = rig.warp_to_atlas_with_rotation(equirect, RotQuaternion.identity())
        aw, ah = rig.atlas_size
        assert warp.width == aw
        assert warp.height == ah

    def test_warp_from_atlas_shape_matches_dst(self):
        rig = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, seed=6)
        equirect = _equirect(256, 128)
        warp = rig.warp_from_atlas_with_rotation(equirect, RotQuaternion.identity())
        assert warp.width == 256
        assert warp.height == 128


class TestResampleAtlasToEquirect:
    def test_smooth_pattern_round_trip_k1(self):
        rig = SphericalTileRig(n=320, arc_per_pixel=2 * np.pi / 512, seed=11)
        atlas = _atlas_pattern(rig)
        out = resample_atlas_to_equirect(rig, atlas, 512, 256, k=1)
        assert out.shape == (256, 512)
        assert out.dtype == np.float32

        ref = _equirect_pattern_reference(512, 256)

        # Closest-tile sampling has visible Voronoi seams; mean error should
        # be small even with k=1.
        mae = float(np.mean(np.abs(out - ref)))
        assert mae < 1e-2, f"k=1 mean abs error {mae} too high"

    def test_k3_blend_runs_and_is_close_to_pattern(self):
        rig = SphericalTileRig(n=320, arc_per_pixel=2 * np.pi / 512, seed=12)
        atlas = _atlas_pattern(rig)
        out = resample_atlas_to_equirect(rig, atlas, 256, 128, k=3)
        assert out.shape == (128, 256)
        assert out.dtype == np.float32

        ref = _equirect_pattern_reference(256, 128)

        # k=3 blends adjacent tile values with non-zero weight even deep in
        # a Voronoi cell, so it has more error than k=1 for a perfectly
        # direction-only pattern. It should still track the pattern within
        # ~1.5% (≈ 4/255 on u8).
        mae = float(np.mean(np.abs(out - ref)))
        assert mae < 1.5e-2, f"k=3 mean abs error {mae} unexpectedly high"

    def test_atlas_shape_mismatch_raises(self):
        rig = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, seed=13)
        bogus = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="atlas_image shape"):
            resample_atlas_to_equirect(rig, bogus, 128, 64)

    def test_atlas_dtype_mismatch_raises(self):
        rig = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, seed=14)
        aw, ah = rig.atlas_size
        bogus = np.zeros((ah, aw), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            resample_atlas_to_equirect(rig, bogus, 128, 64)

    def test_k_zero_raises(self):
        rig = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, seed=15)
        aw, ah = rig.atlas_size
        atlas = np.zeros((ah, aw), dtype=np.float32)
        with pytest.raises(ValueError, match="k must be >= 1"):
            resample_atlas_to_equirect(rig, atlas, 128, 64, k=0)
