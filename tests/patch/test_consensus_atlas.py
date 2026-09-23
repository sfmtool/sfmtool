# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tile-batched consensus atlas Rust binding.

``render_consensus_atlas`` composites a panorama consensus atlas in tile
batches so peak memory is bounded by the heaviest single batch rather than the
whole source set. Its headline guarantee is that the result is byte-identical
to the monolithic ``build_rotation_only`` -> ``refine_photometric_ransac`` ->
``primary_consensus_atlas`` path for *any* ``batch_size``.

Driven via PyO3 against the shared seoul_bull 17-image reconstruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool._sfmtool.patches import refine_photometric_ransac, render_consensus_atlas
from sfmtool._sfmtool.geometry import RotQuaternion
from sfmtool._sfmtool.spherical import (
    PerSphericalTileSourceStack,
    SphericalTileRig,
)


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _seoul_bull_rig_and_sources(
    sfmr_path: Path,
    *,
    n_tiles: int = 64,
    arc_pixels: int = 384,
    seed: int = 1234,
):
    """Build a ``(rig, sources)`` pair from the seoul_bull reconstruction.

    ``sources`` is the list of ``(CameraIntrinsics, RotQuaternion, ndarray)``
    tuples both ``build_rotation_only`` and ``render_consensus_atlas`` accept.
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

    return rig, sources


@pytest.fixture
def seoul_bull_rig_and_sources(seoul_bull_workspace: Path):
    return _seoul_bull_rig_and_sources(seoul_bull_workspace)


def _monolithic_atlas(rig, sources, dtype: str) -> np.ndarray:
    """The atlas produced by the un-batched build -> RANSAC -> consensus trio."""
    stack = PerSphericalTileSourceStack.build_rotation_only(rig, sources, dtype=dtype)
    out = refine_photometric_ransac(stack)
    return stack.primary_consensus_atlas(rig, out.primary_mask)


class TestRenderConsensusAtlasInvariance:
    """Validation plan #1: batch-size invariance."""

    def test_f32_batched_matches_monolithic(self, seoul_bull_rig_and_sources):
        """For float32 storage, every ``batch_size`` yields an atlas that is
        bitwise-equal to the others and to the monolithic path."""
        rig, sources = seoul_bull_rig_and_sources
        reference = _monolithic_atlas(rig, sources, "float32")

        for batch_size in (1, 3, 7, rig.n):
            atlas, *_ = render_consensus_atlas(
                rig, sources, batch_size=batch_size, dtype="float32"
            )
            assert atlas.shape == reference.shape
            assert np.array_equal(atlas, reference, equal_nan=True), (
                f"batch_size={batch_size}: atlas differs from the monolithic path"
            )

    def test_f16_batched_runs_agree(self, seoul_bull_rig_and_sources):
        """For float16 storage, the batched runs agree with each other
        (not necessarily with the float32 run)."""
        rig, sources = seoul_bull_rig_and_sources
        reference, *_ = render_consensus_atlas(
            rig, sources, batch_size=rig.n, dtype="float16"
        )
        for batch_size in (1, 3, 7):
            atlas, *_ = render_consensus_atlas(
                rig, sources, batch_size=batch_size, dtype="float16"
            )
            assert np.array_equal(atlas, reference, equal_nan=True), (
                f"batch_size={batch_size}: f16 atlas differs across batch sizes"
            )

    def test_per_tile_arrays_match_monolithic(self, seoul_bull_rig_and_sources):
        """The four per-tile report arrays equal the monolithic
        ``RansacPhotometricOutput`` regardless of batch size."""
        rig, sources = seoul_bull_rig_and_sources
        stack = PerSphericalTileSourceStack.build_rotation_only(
            rig, sources, dtype="float32"
        )
        out = refine_photometric_ransac(stack)

        for batch_size in (1, 7, rig.n):
            (
                _atlas,
                primary_count,
                secondary_count,
                primary_mad,
                secondary_mad,
            ) = render_consensus_atlas(
                rig, sources, batch_size=batch_size, dtype="float32"
            )
            assert np.array_equal(primary_count, out.tile_primary_count)
            assert np.array_equal(secondary_count, out.tile_secondary_count)
            assert np.array_equal(primary_mad, out.tile_primary_lum_mad, equal_nan=True)
            assert np.array_equal(
                secondary_mad, out.tile_secondary_lum_mad, equal_nan=True
            )


class TestRenderConsensusAtlasAPI:
    """Surface checks: shapes, return arity, error paths."""

    def test_return_shapes(self, seoul_bull_rig_and_sources):
        rig, sources = seoul_bull_rig_and_sources
        result = render_consensus_atlas(rig, sources, batch_size=8, dtype="float32")
        assert len(result) == 5
        atlas, primary_count, secondary_count, primary_mad, secondary_mad = result
        assert atlas.ndim == 3 and atlas.dtype == np.float32
        assert primary_count.shape == (rig.n,)
        assert primary_count.dtype == np.int32
        assert secondary_count.shape == (rig.n,)
        assert primary_mad.shape == (rig.n,)
        assert primary_mad.dtype == np.float32
        assert secondary_mad.shape == (rig.n,)

    def test_rejects_uint8_dtype(self, seoul_bull_rig_and_sources):
        """Validation plan #7: ``dtype="uint8"`` is rejected by the wrapper."""
        rig, sources = seoul_bull_rig_and_sources
        with pytest.raises(ValueError, match="float16.*float32|uint8"):
            render_consensus_atlas(rig, sources, batch_size=8, dtype="uint8")

    def test_rejects_zero_batch_size(self, seoul_bull_rig_and_sources):
        rig, sources = seoul_bull_rig_and_sources
        with pytest.raises(ValueError, match="batch_size"):
            render_consensus_atlas(rig, sources, batch_size=0, dtype="float32")
