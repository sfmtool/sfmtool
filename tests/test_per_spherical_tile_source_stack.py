# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PerSphericalTileSourceStack Rust binding."""

from pathlib import Path

import numpy as np
import pytest

from sfmtool._sfmtool import (
    CameraIntrinsics,
    PerSphericalTileSourceStack,
    RotQuaternion,
    SfmrReconstruction,
    SphericalTileRig,
)


def _pinhole(w: int, h: int, fov_deg: float) -> CameraIntrinsics:
    half_fov = np.deg2rad(fov_deg) * 0.5
    f = (w / 2.0) / np.tan(half_fov)
    return CameraIntrinsics(
        "PINHOLE",
        w,
        h,
        {
            "focal_length_x": f,
            "focal_length_y": f,
            "principal_point_x": w / 2.0,
            "principal_point_y": h / 2.0,
        },
    )


def _render_synthetic(
    intrinsics: CameraIntrinsics, r_src_from_world: RotQuaternion
) -> np.ndarray:
    """Render a smooth direction-only RGB pattern through ``intrinsics``."""
    w = intrinsics.width
    h = intrinsics.height
    cols = np.arange(w) + 0.5
    rows = np.arange(h) + 0.5
    u, v = np.meshgrid(cols, rows, indexing="xy")
    fx, fy = intrinsics.focal_lengths
    cx, cy = intrinsics.principal_point
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)
    rays = np.stack([x, y, z], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    # Rotate camera-frame rays back into world frame.
    r_inv = r_src_from_world.inverse().to_rotation_matrix()
    world = rays @ r_inv.T
    wx, wy, wz = world[..., 0], world[..., 1], world[..., 2]
    rgb = np.stack(
        [
            np.clip(0.5 + 0.4 * wx, 0.0, 1.0),
            np.clip(0.5 + 0.4 * wy, 0.0, 1.0),
            np.clip(0.5 + 0.4 * wz, 0.0, 1.0),
        ],
        axis=-1,
    )
    return (rgb * 255.0).round().astype(np.uint8)


def _make_pow2_rig(n: int, w_equiv: int, target_patch: int) -> SphericalTileRig:
    rig = SphericalTileRig(n=n, arc_per_pixel=2 * np.pi / w_equiv, seed=99)
    rig.set_patch_size(target_patch)
    return rig


class TestPerSphericalTileSourceStackBasics:
    def test_build_succeeds_after_set_patch_size(self):
        rig = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, seed=1)
        rig.set_patch_size(_next_pow2(rig.patch_size))
        stack = PerSphericalTileSourceStack.build_rotation_only(rig, [])
        assert stack.n_tiles == 80
        assert len(stack) == 80
        assert stack.base_patch_size == rig.patch_size
        assert stack.pyramid_levels == int(np.log2(rig.patch_size)) + 1
        for t in range(stack.n_tiles):
            assert stack.n_contributors(t) == 0

    def test_build_fails_on_non_power_of_two_patch_size(self):
        rig = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 256, seed=2)
        # Constructor's natural value is ~21, not a power of two.
        assert (rig.patch_size & (rig.patch_size - 1)) != 0
        with pytest.raises(ValueError, match="not a power of two"):
            PerSphericalTileSourceStack.build_rotation_only(rig, [])

    def test_pyramid_level_count_and_sizes(self):
        for b in (8, 16, 32, 64):
            rig = _make_pow2_rig(20, 256, b)
            stack = PerSphericalTileSourceStack.build_rotation_only(rig, [])
            assert stack.base_patch_size == b
            assert stack.pyramid_levels == int(np.log2(b)) + 1
            for li in range(stack.pyramid_levels):
                assert stack.level_size(li) == b >> li
            assert stack.level_size(stack.pyramid_levels - 1) == 1

    def test_mixed_channel_sources_rejected(self):
        rig = _make_pow2_rig(20, 256, 16)
        cam = _pinhole(64, 64, 60.0)
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        gray = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="channel count"):
            PerSphericalTileSourceStack.build_rotation_only(
                rig,
                [
                    (cam, RotQuaternion.identity(), rgb),
                    (cam, RotQuaternion.identity(), gray),
                ],
            )


class TestPerSphericalTileSourceStackSynthetic:
    def test_buffer_shapes_match_spec(self):
        rig = _make_pow2_rig(40, 256, 16)
        cam = _pinhole(128, 128, 60.0)
        sources = []
        for i in range(3):
            q = RotQuaternion.from_axis_angle([0.0, 1.0, 0.0], i * (np.pi / 4))
            img = _render_synthetic(cam, q)
            sources.append((cam, q, img))
        stack = PerSphericalTileSourceStack.build_rotation_only(rig, sources)
        assert stack.dtype == "uint8"
        for t in range(stack.n_tiles):
            k = stack.n_contributors(t)
            for li in range(stack.pyramid_levels):
                s = stack.level_size(li)
                patches = stack.patches_for_tile(t, li)
                valid = stack.valid_for_tile(t, li)
                assert valid.dtype == np.bool_
                if k == 0:
                    # Empty tiles: shape (0, s, s, 3) and (0, s, s).
                    assert patches.shape == (0, s, s, 3)
                    assert valid.shape == (0, s, s)
                else:
                    assert patches.shape == (k, s, s, 3)
                    assert patches.dtype == np.uint8
                    assert valid.shape == (k, s, s)

    def test_src_indices_are_strictly_ascending(self):
        rig = _make_pow2_rig(40, 256, 16)
        cam = _pinhole(128, 128, 60.0)
        n_src = 5
        sources = []
        for i in range(n_src):
            q = RotQuaternion.from_axis_angle([0.0, 1.0, 0.0], i * (2 * np.pi / n_src))
            sources.append((cam, q, _render_synthetic(cam, q)))
        stack = PerSphericalTileSourceStack.build_rotation_only(rig, sources)
        for t in range(stack.n_tiles):
            indices = stack.src_indices_for_tile(t)
            assert indices.dtype == np.uint32
            if len(indices) > 1:
                assert np.all(np.diff(indices) > 0)
            for i in indices:
                assert 0 <= int(i) < n_src

    def test_pyramid_downsample_round_trip_pure_python(self):
        """Recompute the pyramid for one (source, tile) pair from level 0 in
        pure Python and compare with the stack's stored levels."""
        rig = _make_pow2_rig(40, 256, 16)
        cam = _pinhole(128, 128, 60.0)
        q = RotQuaternion.from_axis_angle([0.0, 1.0, 0.0], np.pi / 6)
        img = _render_synthetic(cam, q)
        sources = [(cam, q, img)]
        stack = PerSphericalTileSourceStack.build_rotation_only(rig, sources)

        # Find a tile with the source contributing.
        chosen = None
        for t in range(stack.n_tiles):
            if stack.n_contributors(t) >= 1:
                chosen = t
                break
        assert chosen is not None, "expected at least one kept tile"

        # Pull level-0 patch + valid for the only contributor (pos=0).
        l0_patches = stack.patches_for_tile(chosen, 0)
        l0_valid = stack.valid_for_tile(chosen, 0)
        assert l0_patches.shape[0] == 1
        cur_p = l0_patches[0].copy()
        cur_v = l0_valid[0].copy()
        for li in range(1, stack.pyramid_levels):
            # 2x box-filter downsample of cur_p, all-four AND of cur_v.
            new_size = cur_p.shape[0] // 2
            blocks_p = cur_p.reshape(new_size, 2, new_size, 2, 3).astype(np.uint16)
            new_p = ((blocks_p.sum(axis=(1, 3)) + 2) // 4).astype(np.uint8)
            blocks_v = cur_v.reshape(new_size, 2, new_size, 2)
            new_v = blocks_v.all(axis=(1, 3))

            stored_p = stack.patches_for_tile(chosen, li)[0]
            stored_v = stack.valid_for_tile(chosen, li)[0]
            np.testing.assert_array_equal(stored_p, new_p)
            np.testing.assert_array_equal(stored_v, new_v)

            cur_p = new_p
            cur_v = new_v

        # Final 1×1 level.
        assert stack.patches_for_tile(chosen, stack.pyramid_levels - 1).shape[1:] == (
            1,
            1,
            3,
        )

    def test_index_out_of_range_raises(self):
        rig = _make_pow2_rig(20, 256, 8)
        stack = PerSphericalTileSourceStack.build_rotation_only(rig, [])
        with pytest.raises(IndexError):
            stack.n_contributors(rig.n)
        with pytest.raises(IndexError):
            stack.patches_for_tile(0, stack.pyramid_levels)
        with pytest.raises(IndexError):
            stack.valid_for_tile(rig.n, 0)


class TestPerSphericalTileSourceStackCsr:
    """CSR-flat layout invariants exposed at the Python level."""

    def test_tile_offsets_and_ids_consistent(self):
        rig = _make_pow2_rig(40, 256, 16)
        cam = _pinhole(128, 128, 60.0)
        sources = []
        for i in range(4):
            q = RotQuaternion.from_axis_angle([0.0, 1.0, 0.0], i * (np.pi / 4))
            sources.append((cam, q, _render_synthetic(cam, q)))
        stack = PerSphericalTileSourceStack.build_rotation_only(rig, sources)

        offsets = stack.tile_offsets()
        tile_id = stack.tile_id()
        src_id = stack.src_id()
        assert offsets.dtype == np.uint32
        assert tile_id.dtype == np.uint32
        assert src_id.dtype == np.uint32
        assert offsets.shape == (stack.n_tiles + 1,)
        assert tile_id.shape == (stack.total_contrib_rows,)
        assert src_id.shape == (stack.total_contrib_rows,)
        assert offsets[0] == 0
        assert offsets[-1] == stack.total_contrib_rows
        assert np.all(np.diff(offsets) >= 0)

        # tile_id[r] == t for every r in tile t's range.
        for t in range(stack.n_tiles):
            start, end = int(offsets[t]), int(offsets[t + 1])
            assert np.all(tile_id[start:end] == t)
            # src_id slice equals src_indices_for_tile.
            np.testing.assert_array_equal(
                src_id[start:end], stack.src_indices_for_tile(t)
            )

    def test_whole_level_buffers_match_per_tile_concatenation(self):
        rig = _make_pow2_rig(40, 256, 16)
        cam = _pinhole(128, 128, 60.0)
        sources = []
        for i in range(3):
            q = RotQuaternion.from_axis_angle([0.0, 1.0, 0.0], i * (np.pi / 5))
            sources.append((cam, q, _render_synthetic(cam, q)))
        stack = PerSphericalTileSourceStack.build_rotation_only(rig, sources)

        for li in range(stack.pyramid_levels):
            whole_p = stack.level_patches(li)
            whole_v = stack.level_valid(li)
            assert whole_p.shape[0] == stack.total_contrib_rows
            assert whole_v.shape[0] == stack.total_contrib_rows

            # Concatenate per-tile slices and compare.
            tile_ps = []
            tile_vs = []
            for t in range(stack.n_tiles):
                if stack.n_contributors(t) > 0:
                    tile_ps.append(stack.patches_for_tile(t, li))
                    tile_vs.append(stack.valid_for_tile(t, li))
            if tile_ps:
                np.testing.assert_array_equal(np.concatenate(tile_ps, axis=0), whole_p)
                np.testing.assert_array_equal(np.concatenate(tile_vs, axis=0), whole_v)


class TestPerSphericalTileSourceStackFloat32:
    """The dtype='float32' build path."""

    def test_dtype_float32_level_zero_byte_equivalent_to_uint8(self):
        rig = _make_pow2_rig(40, 256, 16)
        cam = _pinhole(128, 128, 60.0)
        sources = []
        for i in range(3):
            q = RotQuaternion.from_axis_angle([0.0, 1.0, 0.0], i * (np.pi / 6))
            sources.append((cam, q, _render_synthetic(cam, q)))

        stack_u8 = PerSphericalTileSourceStack.build_rotation_only(rig, sources)
        stack_f32 = PerSphericalTileSourceStack.build_rotation_only(
            rig, sources, dtype="float32"
        )

        assert stack_u8.dtype == "uint8"
        assert stack_f32.dtype == "float32"
        assert stack_u8.total_contrib_rows == stack_f32.total_contrib_rows

        l0_u8 = stack_u8.level_patches(0)
        l0_f32 = stack_f32.level_patches(0)
        assert l0_u8.dtype == np.uint8
        assert l0_f32.dtype == np.float32
        # Range-preserving conversion: f32[i] == u8[i] as f32.
        np.testing.assert_array_equal(l0_f32, l0_u8.astype(np.float32))

        # Valid masks must be identical.
        np.testing.assert_array_equal(stack_f32.level_valid(0), stack_u8.level_valid(0))

    def test_invalid_dtype_rejected(self):
        rig = _make_pow2_rig(20, 256, 8)
        with pytest.raises(ValueError, match="dtype"):
            PerSphericalTileSourceStack.build_rotation_only(rig, [], dtype="float64")


class TestPerSphericalTileSourceStackOnReconstruction:
    """Smoke test on a real bundled reconstruction."""

    def test_smoke_on_seoul_bull_reconstruction(
        self, sfmrfile_reconstruction_with_17_images: Path
    ):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        cameras = recon.cameras
        camera_indexes = recon.camera_indexes
        quats = recon.quaternions_wxyz
        image_names = recon.image_names
        workspace_dir = sfmrfile_reconstruction_with_17_images.parent
        # The fixture stores images under either workspace_dir/test_17_image
        # (the session fixture) or workspace_dir/<name> directly.
        candidates = [workspace_dir / "test_17_image", workspace_dir]
        image_dir = next(c for c in candidates if (c / image_names[0]).exists())

        # Use the rig centred at the median camera centre — its exact value
        # doesn't matter for a rotation-only build, but we still want a
        # plausible position so the scene is in front of the cameras.
        rig = SphericalTileRig(n=320, arc_per_pixel=2 * np.pi / 512, seed=1234)
        rig.set_patch_size(_next_pow2(rig.patch_size))

        import cv2

        sources = []
        for i, name in enumerate(image_names):
            cam = cameras[camera_indexes[i]]
            q = RotQuaternion(quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3])
            bgr = cv2.imread(str(image_dir / name), cv2.IMREAD_COLOR)
            assert bgr is not None, f"failed to read {image_dir / name}"
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            sources.append((cam, q, img))

        stack = PerSphericalTileSourceStack.build_rotation_only(rig, sources)
        assert stack.n_tiles == 320
        assert stack.base_patch_size == rig.patch_size
        assert stack.pyramid_levels == int(np.log2(rig.patch_size)) + 1

        total_kept = sum(stack.n_contributors(t) for t in range(stack.n_tiles))
        # At least *some* tile should pick up *some* source; otherwise the
        # cull is broken.
        assert total_kept > 0

        # Sanity-check buffer sizing for a few non-empty tiles.
        checked = 0
        for t in range(stack.n_tiles):
            k = stack.n_contributors(t)
            if k == 0:
                continue
            indices = stack.src_indices_for_tile(t)
            assert len(indices) == k
            assert np.all((indices >= 0) & (indices < len(image_names)))
            for li in range(stack.pyramid_levels):
                s = stack.level_size(li)
                patches = stack.patches_for_tile(t, li)
                valid = stack.valid_for_tile(t, li)
                assert patches.shape == (k, s, s, 3)
                assert valid.shape == (k, s, s)
            checked += 1
            if checked >= 3:
                break
        assert checked > 0


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()
