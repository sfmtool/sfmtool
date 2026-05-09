# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the photometric RANSAC refinement Rust binding.

End-to-end behaviour of ``refine_photometric_ransac`` driven via PyO3
against a real reconstruction. Uses the shared
``sfmrfile_reconstruction_with_17_images`` fixture; COLMAP's incremental
SfM is non-deterministic even with a fixed seed (multi-threaded BA), so
assertions here are self-relative or generous lower bounds rather than
absolute numerical targets.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from sfmtool._sfmtool import (
    PerSphericalTileSourceStack,
    RotQuaternion,
    SfmrReconstruction,
    SphericalTileRig,
    refine_photometric_ransac,
)


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _per_tile_lum_mad(
    row_lum: np.ndarray,
    tile_offsets: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    min_inliers: int = 2,
) -> np.ndarray:
    """Per-tile MAD over `row_lum`, restricted to rows where `mask` is True
    when provided. Tiles below `min_inliers` get NaN."""
    n_tiles = len(tile_offsets) - 1
    out = np.full(n_tiles, np.nan, dtype=np.float32)
    for t in range(n_tiles):
        a, b = int(tile_offsets[t]), int(tile_offsets[t + 1])
        if mask is None:
            vals = row_lum[a:b]
        else:
            vals = row_lum[a:b][mask[a:b]]
        if len(vals) >= min_inliers:
            med = float(np.median(vals))
            out[t] = float(np.median(np.abs(vals - med)))
    return out


def _build_seoul_bull_stack(
    sfmr_path: Path,
    *,
    # 160 tiles balances rig coverage against per-tile contributor count
    # for a 17-image dataset: fewer tiles is coarser than the algorithm
    # was designed for, many more tiles starves most tiles of the ≥2
    # contributors needed for a defined MAD.
    n_tiles: int = 160,
    arc_pixels: int = 384,
    seed: int = 1234,
) -> tuple[PerSphericalTileSourceStack, np.ndarray]:
    """Build an f32 stack from the seoul_bull reconstruction.

    Returns `(stack, src_index_array)` for downstream metric computations.
    """
    import cv2  # local import — heavy module, only needed by integration

    recon = SfmrReconstruction.load(sfmr_path)
    cameras = recon.cameras
    camera_indexes = recon.camera_indexes
    quats = recon.quaternions_wxyz
    image_names = recon.image_names
    workspace_dir = sfmr_path.parent
    candidates = [workspace_dir / "test_17_image", workspace_dir]
    image_dir = next(c for c in candidates if (c / image_names[0]).exists())

    rig = SphericalTileRig(n=n_tiles, arc_per_pixel=2 * np.pi / arc_pixels, seed=seed)
    rig.set_patch_size(_next_pow2(rig.patch_size))

    sources = []
    for i, name in enumerate(image_names):
        cam = cameras[camera_indexes[i]]
        q = RotQuaternion(quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3])
        bgr = cv2.imread(str(image_dir / name), cv2.IMREAD_COLOR)
        assert bgr is not None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sources.append((cam, q, rgb))

    stack = PerSphericalTileSourceStack.build_rotation_only(
        rig, sources, dtype="float32"
    )
    return stack, np.asarray(stack.src_id())


def _row_mean_lum_at_level(
    stack: PerSphericalTileSourceStack, target_size: int, scoring_size: int
) -> np.ndarray:
    """Recompute the per-row mean luminance over the central
    `scoring_size × scoring_size` sub-patch at the pyramid level whose patch
    side equals `target_size`. Mirrors the reduction the algorithm performs
    internally so the test can compute its own pre-correction baseline to
    compare against the post-correction result."""
    levels = stack.pyramid_levels
    chosen = None
    for li in range(levels):
        if stack.level_size(li) == target_size:
            chosen = li
            break
    assert chosen is not None
    patches = stack.level_patches(chosen).astype(np.float32)  # (R, s, s, C)
    valid = stack.level_valid(chosen)  # (R, s, s) bool
    s = patches.shape[1]
    half = (s - scoring_size) // 2
    centre_p = patches[:, half : half + scoring_size, half : half + scoring_size, :]
    centre_v = valid[:, half : half + scoring_size, half : half + scoring_size].astype(
        np.float32
    )
    # Saturation mask (default 254): drop pixels where any channel >= 254.
    sat_mask = (centre_p.max(axis=-1) < 254.0).astype(np.float32)
    centre_v = centre_v * sat_mask
    valid_b = centre_v[..., None]
    num = (centre_p * valid_b).sum(axis=(1, 2, 3))
    denom = valid_b.sum(axis=(1, 2, 3)) * centre_p.shape[-1]
    return num / np.clip(denom, 1.0, None)


@pytest.fixture
def seoul_bull_stack(sfmrfile_reconstruction_with_17_images: Path):
    """Build the seoul_bull stack from the shared 17-image reconstruction."""
    return _build_seoul_bull_stack(sfmrfile_reconstruction_with_17_images)


class TestRefinePhotometricRansacSeoulBull:
    """Integration tests on `seoul_bull_sculpture` (17 images) — covers
    validation-plan items 10, 11, 12, 13."""

    def test_rejects_uint8_stack(self):
        # The documented "must be float32" gate. A trivial empty stack
        # exercises the rejection path without needing a reconstruction.
        rig = SphericalTileRig(n=10, arc_per_pixel=2 * np.pi / 256, seed=1)
        rig.set_patch_size(_next_pow2(rig.patch_size))
        u8_stack = PerSphericalTileSourceStack.build_rotation_only(rig, [])
        with pytest.raises(ValueError, match="float32"):
            refine_photometric_ransac(u8_stack)

    def test_primary_more_compact_than_all_rows(self, seoul_bull_stack):
        """The primary cluster is, by construction, the subset of rows the
        algorithm decided agree photometrically; outliers go to the
        secondary / rejected groups. So the primary-cluster median MAD must
        be strictly lower than the post-correction all-rows median MAD —
        any reconstruction where that fails means the cluster split isn't
        actually separating agreeing from disagreeing rows."""
        stack, src_index = seoul_bull_stack
        target_size = 4
        scoring_size = 2
        row_lum_pre = _row_mean_lum_at_level(stack, target_size, scoring_size)
        tile_offsets = np.asarray(stack.tile_offsets())
        active = np.diff(tile_offsets) >= 2

        out = refine_photometric_ransac(
            stack,
            target_patch_size=target_size,
            scoring_patch_size=scoring_size,
        )

        # All-rows post-correction MAD (apply gains then recompute).
        gain_per_row = np.exp(out.log_gain[src_index]).astype(np.float32)
        row_lum_post = row_lum_pre * gain_per_row
        all_rows_mad = _per_tile_lum_mad(row_lum_post, tile_offsets)[active]
        all_rows_mad = all_rows_mad[~np.isnan(all_rows_mad)]

        primary_mad = np.asarray(out.tile_primary_lum_mad)[active]
        primary_mad = primary_mad[~np.isnan(primary_mad)]

        median_all = float(np.median(all_rows_mad))
        median_primary = float(np.median(primary_mad))
        assert median_primary < median_all, (
            f"primary-cluster MAD {median_primary:.3f} should be strictly "
            f"lower than all-rows MAD {median_all:.3f}"
        )

    def test_algorithm_recovers_nontrivial_gains(self, seoul_bull_stack):
        """Sanity check: the algorithm should produce non-zero per-source
        gains on a real dataset. Pre-correction `log_gain` is all zeros;
        if the algorithm leaves it that way, something is broken upstream
        (e.g., LSQ never fired, or every primary cluster was empty)."""
        stack, _ = seoul_bull_stack
        out = refine_photometric_ransac(stack)
        # Pre-correction `log_gain` is all zeros, so any real recovery
        # produces non-zero per-source spread. 0.005 is small enough to
        # absorb reconstruction-quality variance while still rejecting
        # the degenerate all-zeros case (LSQ never fired, every primary
        # cluster empty, etc.).
        assert out.log_gain.std() > 0.005, (
            f"log_gain std {out.log_gain.std():.4f} suggests no recovery"
        )

    def test_inter_seed_gain_stability(self, seoul_bull_stack):
        """RANSAC determinism: re-running on the same stack with different
        seeds should land on essentially the same gains. The bound is set
        well below the magnitude of the gains the algorithm typically
        recovers (per-source `log_gain` runs in the 0.05-0.15 range), so a
        mean inter-seed std under 0.02 means seed choice never flips the
        consensus enough to matter for downstream luminance correction."""
        stack, _ = seoul_bull_stack
        gains = []
        for seed in range(8):
            out = refine_photometric_ransac(stack, seed=seed)
            gains.append(np.asarray(out.log_gain, dtype=np.float64))
        gains = np.stack(gains, axis=0)  # (8, n_sources)
        per_source_std = gains.std(axis=0, ddof=0)
        mean_std = float(per_source_std.mean())
        assert mean_std < 0.02, f"mean inter-seed log_gain std {mean_std:.4f} >= 0.02"

    def test_wallclock_under_one_second(self, seoul_bull_stack):
        """End-to-end algorithm time on a 17-image stack should stay well
        under 1 s so the refinement is cheap enough to run interactively
        and inside iterative pipelines. The bound is wall-clock on the
        binding alone; stack construction is not included."""
        stack, _ = seoul_bull_stack
        # Warm-up call to amortise any first-call init cost.
        refine_photometric_ransac(stack)
        t0 = time.perf_counter()
        out = refine_photometric_ransac(stack)
        elapsed = time.perf_counter() - t0
        assert out.outer_iters >= 1
        assert elapsed < 1.0, f"refine_photometric_ransac took {elapsed:.3f} s >= 1.0"


class TestRefinePhotometricRansacAPISmoke:
    """Lightweight smoke checks for the binding's surface — shape, dtype,
    error paths."""

    def test_returns_expected_shapes_and_dtypes(self, seoul_bull_stack):
        stack, _ = seoul_bull_stack
        out = refine_photometric_ransac(stack)
        n_tiles = stack.n_tiles
        r = stack.total_contrib_rows
        # log_gain length matches max(src_id) + 1.
        n_sources = int(np.asarray(stack.src_id()).max()) + 1
        assert out.log_gain.shape == (n_sources,)
        assert out.log_gain.dtype == np.float32
        assert out.primary_mask.shape == (r,)
        assert out.primary_mask.dtype == np.bool_
        assert out.secondary_mask.shape == (r,)
        assert out.secondary_mask.dtype == np.bool_
        assert out.tile_primary_count.shape == (n_tiles,)
        assert out.tile_primary_count.dtype == np.int32
        assert out.tile_secondary_count.shape == (n_tiles,)
        assert out.tile_secondary_count.dtype == np.int32
        assert out.tile_primary_lum_mad.shape == (n_tiles,)
        assert out.tile_primary_lum_mad.dtype == np.float32
        assert out.tile_secondary_lum_mad.shape == (n_tiles,)
        assert out.tile_secondary_lum_mad.dtype == np.float32
        assert isinstance(out.outer_iters, int)
        assert out.mask_change_history.shape == (out.outer_iters,)
        assert out.mask_change_history.dtype == np.uint32
        # Mean-zero log_gain check.
        assert abs(float(out.log_gain.mean())) < 1e-5
        # Cluster invariants: primary and secondary disjoint.
        assert not np.any(out.primary_mask & out.secondary_mask)
