// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tile-batched consensus atlas: bounded-memory panorama compositing.
//!
//! See `specs/core/tile-batched-consensus-atlas.md` for the design.
//!
//! The monolithic panorama path materialises one
//! [`PerSphericalTileSourceStack`] over **every** source image at once, so its
//! peak memory grows linearly with the source count. [`render_consensus_atlas`]
//! produces a byte-identical result while processing the rig's tiles in
//! **batches**: it builds, RANSAC-partitions, and collapses one batch's stack
//! at a time, so peak per-batch memory is governed by the heaviest single
//! batch rather than the whole-run total.
//!
//! The result is identical to building the monolithic stack, running
//! [`refine_photometric_ransac`], and calling
//! [`PerSphericalTileSourceStack::primary_consensus_atlas`] — for any
//! `batch_size` — because the per-tile RANSAC RNG is re-seeded from the
//! *global* tile index via [`RansacPhotometricParams::tile_index_base`], and
//! every other per-tile reduction in the chain is order-independent.

use crate::camera::intrinsics::CameraIntrinsics;
use crate::camera::remap::ImageU8;
use crate::geometry::rot_quaternion::RotQuaternion;
use crate::spherical::per_tile_source_stack::{
    BuildError, BuildParams, ConsensusAtlasError, PatchPixel, PerSphericalTileSourceStack,
};
use crate::spherical::photometric_ransac::{
    refine_photometric_ransac, RansacPhotometricError, RansacPhotometricParams,
};
use crate::spherical::tile_rig::SphericalTileRig;

/// Tunables for [`render_consensus_atlas`].
#[derive(Debug, Clone)]
pub struct ConsensusAtlasBatchParams {
    /// Tiles per batch. Smaller ⇒ lower peak memory, more per-batch fixed
    /// overhead. Must be `>= 1` (`0` is an error); a value above `n_tiles`
    /// acts as `n_tiles` (a single batch — equivalent to the monolithic path).
    pub batch_size: usize,
    /// Photometric-RANSAC tunables. The orchestrator overwrites
    /// `ransac.tile_index_base` per batch; whatever the caller sets there is
    /// ignored. All other fields are forwarded unchanged.
    pub ransac: RansacPhotometricParams,
    /// Forwarded verbatim to
    /// [`PerSphericalTileSourceStack::build_rotation_only`] each batch.
    pub build: BuildParams,
}

impl Default for ConsensusAtlasBatchParams {
    fn default() -> Self {
        Self {
            batch_size: 32,
            ransac: RansacPhotometricParams::default(),
            build: BuildParams::default(),
        }
    }
}

/// Atlas + per-tile validity signal, accumulated across batches.
///
/// The per-tile arrays are indexed by *global* tile index and are exactly what
/// a single monolithic [`refine_photometric_ransac`] would have produced — the
/// orchestrator scatters each batch's slice into the global arrays.
#[derive(Debug, Clone)]
pub struct ConsensusAtlasReport {
    /// Row-major `atlas_h × atlas_w × channels` `f32`, `NaN` where no
    /// consensus (empty primary cluster, all-invalid pixel, or trailing atlas
    /// slot). Identical layout to
    /// [`PerSphericalTileSourceStack::primary_consensus_atlas`].
    pub atlas: Vec<f32>,
    /// Per-tile primary cluster size, length `n_tiles`.
    pub tile_primary_count: Vec<i32>,
    /// Per-tile secondary cluster size, length `n_tiles`.
    pub tile_secondary_count: Vec<i32>,
    /// Per-tile primary luminance MAD (`NaN` where the cluster is below
    /// `min_inliers`), length `n_tiles`.
    pub tile_primary_lum_mad: Vec<f32>,
    /// Per-tile secondary luminance MAD (`NaN` where the cluster is below
    /// `min_inliers`), length `n_tiles`.
    pub tile_secondary_lum_mad: Vec<f32>,
}

/// Errors from [`render_consensus_atlas`].
#[derive(Debug)]
pub enum ConsensusAtlasBatchError {
    /// `params.batch_size == 0`.
    BatchSizeZero,
    /// `rig.patch_size()` is not a power of two.
    PatchSizeNotPowerOfTwo(u32),
    /// [`PerSphericalTileSourceStack::build_rotation_only`] failed for `batch`.
    Build { batch: usize, source: BuildError },
    /// [`refine_photometric_ransac`] failed for `batch`.
    Ransac {
        batch: usize,
        source: RansacPhotometricError,
    },
    /// [`PerSphericalTileSourceStack::consensus_patches_per_tile`] failed for
    /// `batch`.
    Consensus {
        batch: usize,
        source: ConsensusAtlasError,
    },
}

impl std::fmt::Display for ConsensusAtlasBatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BatchSizeZero => write!(f, "batch_size must be >= 1"),
            Self::PatchSizeNotPowerOfTwo(p) => write!(
                f,
                "rig.patch_size = {p} is not a power of two; \
                 call set_patch_size on the rig first"
            ),
            Self::Build { batch, source } => {
                write!(
                    f,
                    "building source stack for batch {batch} failed: {source}"
                )
            }
            Self::Ransac { batch, source } => {
                write!(f, "photometric RANSAC for batch {batch} failed: {source}")
            }
            Self::Consensus { batch, source } => {
                write!(f, "consensus atlas for batch {batch} failed: {source}")
            }
        }
    }
}

impl std::error::Error for ConsensusAtlasBatchError {}

/// Composite a consensus atlas for `rig` over `sources` in tile batches, never
/// holding more than one batch's stack live.
///
/// Equivalent to building the monolithic `PerSphericalTileSourceStack<T>`,
/// running [`refine_photometric_ransac`], and calling
/// [`PerSphericalTileSourceStack::primary_consensus_atlas`] — the `atlas` and
/// the per-tile arrays are byte-identical for any `batch_size` — but with peak
/// per-batch stack memory governed by the heaviest batch's row count.
///
/// `rig.patch_size()` must already be a power of two (as for the monolithic
/// build). `u8`-backed batches are accepted by the generic signature
/// (`as_f32` range-promotes); the Python wrapper rejects `dtype="uint8"` for
/// parity with the existing surface.
pub fn render_consensus_atlas<T: PatchPixel>(
    rig: &SphericalTileRig,
    sources: &[(CameraIntrinsics, RotQuaternion, ImageU8)],
    params: &ConsensusAtlasBatchParams,
) -> Result<ConsensusAtlasReport, ConsensusAtlasBatchError> {
    if params.batch_size == 0 {
        return Err(ConsensusAtlasBatchError::BatchSizeZero);
    }
    let patch_size = rig.patch_size();
    if !patch_size.is_power_of_two() {
        return Err(ConsensusAtlasBatchError::PatchSizeNotPowerOfTwo(patch_size));
    }

    let n_tiles = rig.len();
    // Same channel-fallback rule as `build_rotation_only`: an empty source set
    // yields a single-channel atlas.
    let channels = sources.first().map_or(1, |(_, _, img)| img.channels()) as usize;
    let ps = patch_size as usize;

    let (atlas_w, atlas_h) = rig.atlas_size();
    let atlas_w = atlas_w as usize;
    let atlas_h = atlas_h as usize;
    let row_stride = atlas_w * channels;
    let mut atlas = vec![f32::NAN; atlas_w * atlas_h * channels];

    let mut tile_primary_count = vec![0i32; n_tiles];
    let mut tile_secondary_count = vec![0i32; n_tiles];
    let mut tile_primary_lum_mad = vec![f32::NAN; n_tiles];
    let mut tile_secondary_lum_mad = vec![f32::NAN; n_tiles];

    let n_batches = n_tiles.div_ceil(params.batch_size);
    for b in 0..n_batches {
        let start = b * params.batch_size;
        let end = ((b + 1) * params.batch_size).min(n_tiles);

        // Only this batch's stack is live at a time — the whole point.
        let sub_rig = rig.tiles_subset(start..end);
        let sub_stack =
            PerSphericalTileSourceStack::<T>::build_rotation_only(&sub_rig, sources, &params.build)
                .map_err(|source| ConsensusAtlasBatchError::Build { batch: b, source })?;

        let mut ransac = params.ransac.clone();
        ransac.tile_index_base = start;
        let out = refine_photometric_ransac(&sub_stack, &ransac)
            .map_err(|source| ConsensusAtlasBatchError::Ransac { batch: b, source })?;

        let patches = sub_stack
            .consensus_patches_per_tile(&out.primary_mask)
            .map_err(|source| ConsensusAtlasBatchError::Consensus { batch: b, source })?;

        // Blit each tile's consensus patch into the full atlas at the parent
        // rig's slot — never materialising a per-batch atlas.
        for (local_t, patch) in patches.iter().enumerate() {
            debug_assert_eq!(patch.len(), ps * ps * channels);
            let (ox, oy) = rig.tile_atlas_origin(start + local_t);
            let ox = ox as usize;
            let oy = oy as usize;
            for dy in 0..ps {
                let src_off = dy * ps * channels;
                let dst_off = (oy + dy) * row_stride + ox * channels;
                atlas[dst_off..dst_off + ps * channels]
                    .copy_from_slice(&patch[src_off..src_off + ps * channels]);
            }
        }

        // Scatter the batch's per-tile signal into the global arrays.
        for local_t in 0..(end - start) {
            let g = start + local_t;
            tile_primary_count[g] = out.tile_primary_count[local_t];
            tile_secondary_count[g] = out.tile_secondary_count[local_t];
            tile_primary_lum_mad[g] = out.tile_primary_lum_mad[local_t];
            tile_secondary_lum_mad[g] = out.tile_secondary_lum_mad[local_t];
        }
        // `sub_stack` (and `sub_rig`, `patches`) drop here before the next batch.
    }

    Ok(ConsensusAtlasReport {
        atlas,
        tile_primary_count,
        tile_secondary_count,
        tile_primary_lum_mad,
        tile_secondary_lum_mad,
    })
}

#[cfg(test)]
mod tests;
