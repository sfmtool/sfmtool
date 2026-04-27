// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-spherical-tile source patch stack: a direction-space primitive built on
//! top of [`SphericalTileRig`].
//!
//! See `specs/core/per-spherical-tile-source-stack.md` for design details.
//!
//! For each tile in the rig and each contributing source image, this primitive
//! holds an image pyramid of warped patches: the source's view of the tile's
//! look direction, warped into the tile's local pinhole frame, then
//! 2× box-filter downsampled all the way to a 1×1 patch. Visibility is
//! computed once via centre-direction projection through each source camera;
//! sources that do not see the tile centre are dropped.
//!
//! The output is laid out struct-of-arrays per `(tile, level)`: every
//! contributor's patch at one pyramid level lives in a single contiguous
//! buffer, in source-index order. That matches the dominant access pattern
//! (NCC across all sources at one scale, GPU upload of a single buffer per
//! scale, etc.).

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use crate::camera_intrinsics::CameraIntrinsics;
use crate::remap::{remap_bilinear, ImageU8};
use crate::rot_quaternion::RotQuaternion;
use crate::spherical_tile_rig::SphericalTileRig;
use crate::warp_map::WarpMap;

/// All contributing sources' patches for one tile at one pyramid level, laid
/// out struct-of-arrays so per-`(tile, level)` consumers get a single
/// contiguous buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SphericalTilePatchLevel {
    /// Side length of this level in pixels (`base_patch_size >> level_idx`).
    pub size: u32,
    /// Number of contributing sources (matches the owning tile's
    /// `src_indices.len()`).
    pub n_contributors: u32,
    /// Channel count (matches the input image: 1, 3, or 4).
    pub channels: u32,
    /// All contributors' patches concatenated in `src_indices` order. Each
    /// per-source patch is row-major `size × size × channels` u8; total length
    /// is `n_contributors · size · size · channels`.
    pub patches: Vec<u8>,
    /// All contributors' valid masks, same SoA layout as `patches`. Level 0
    /// masks come from the warp's in-bounds test; level L > 0 uses the
    /// all-four rule (true iff every one of the four level-(L−1) source
    /// pixels was true). Length is `n_contributors · size · size`.
    pub valid: Vec<bool>,
}

/// All data for one spherical tile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SphericalTileSourceStack {
    /// Source-list indices of this tile's contributors, in the order their
    /// patches appear in every level's SoA buffers. Sorted ascending. Empty
    /// if no source sees the tile centre.
    pub src_indices: Vec<u32>,
    /// Pyramid levels in order: `levels[0]` is the base
    /// (`base_patch_size × base_patch_size`), `levels[L − 1]` is `1 × 1`.
    /// Every level holds `src_indices.len()` patches in matching order.
    pub levels: Vec<SphericalTilePatchLevel>,
}

/// Per-spherical-tile source patch stack.
///
/// See the module-level docs for layout and lifecycle. Construct via
/// [`PerSphericalTileSourceStack::build_rotation_only`].
#[derive(Debug, Clone)]
pub struct PerSphericalTileSourceStack {
    n_tiles: usize,
    base_patch_size: u32,
    pyramid_levels: u32,
    tiles: Vec<SphericalTileSourceStack>,
}

/// Build-time tuning knobs.
#[derive(Debug, Clone, Default)]
pub struct BuildParams {
    /// Reserved for future extension. Currently the build always processes
    /// sources sequentially in the outer loop and parallelises across each
    /// source's kept tiles via rayon (lock-free because each tile-task
    /// writes to its own tile's SoA buffer and a unique `pos` slot inside
    /// it). Setting this knob to `Some(1)` is therefore a no-op today, but
    /// the field exists so callers can opt into future parallel-source
    /// chunking without an API break.
    pub max_in_flight_sources: Option<usize>,
}

/// Construction-time validation errors for
/// [`PerSphericalTileSourceStack::build_rotation_only`].
#[derive(Debug, PartialEq, Eq)]
pub enum BuildError {
    /// `rig.patch_size()` is not a power of two. Call
    /// `rig.set_patch_size(rig.patch_size().next_power_of_two())` (or another
    /// power of two) before invoking the builder.
    PatchSizeNotPowerOfTwo(u32),
    /// Sources disagree on channel count. The whole slice must use the same
    /// value.
    MixedSourceChannels { first: u32, offending: u32 },
    /// A source has 0, 2, or > 4 channels (only 1, 3, 4 are accepted by the
    /// remap kernels).
    UnsupportedChannelCount(u32),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PatchSizeNotPowerOfTwo(p) => write!(
                f,
                "rig.patch_size = {p} is not a power of two; \
                 call set_patch_size on the rig first"
            ),
            Self::MixedSourceChannels { first, offending } => write!(
                f,
                "sources disagree on channel count: first source has {first} \
                 channels, another has {offending}"
            ),
            Self::UnsupportedChannelCount(c) => {
                write!(f, "channel count {c} not supported (must be 1, 3, or 4)")
            }
        }
    }
}

impl std::error::Error for BuildError {}

impl PerSphericalTileSourceStack {
    /// Build a rotation-only stack: every patch is the source's view of the
    /// tile assuming the scene is at infinite radial distance from the rig
    /// centre. Source positions are not used — only the relative orientation
    /// between each source and the rig.
    ///
    /// `sources[i]` is `(src_intrinsics_i, R_src_from_world_i, image_i)`. All
    /// images must share a channel count (1, 3, or 4).
    ///
    /// Returns [`BuildError::PatchSizeNotPowerOfTwo`] if `rig.patch_size()` is
    /// not a power of two. The caller is expected to round up via
    /// `rig.set_patch_size(rig.patch_size().next_power_of_two())` first.
    pub fn build_rotation_only(
        rig: &SphericalTileRig,
        sources: &[(CameraIntrinsics, RotQuaternion, ImageU8)],
        _params: &BuildParams,
    ) -> Result<Self, BuildError> {
        let b = rig.patch_size();
        if b == 0 || !b.is_power_of_two() {
            return Err(BuildError::PatchSizeNotPowerOfTwo(b));
        }
        // log2(b) + 1 levels: [B, B/2, ..., 1].
        let pyramid_levels = b.trailing_zeros() + 1;
        let n_tiles = rig.len();
        let n_sources = sources.len();

        // ── Channel-count validation ─────────────────────────────────────
        let channels = match sources.first() {
            Some((_, _, img)) => img.channels(),
            None => 1, // empty sources: pick any; nothing is allocated downstream
        };
        if !matches!(channels, 1 | 3 | 4) {
            return Err(BuildError::UnsupportedChannelCount(channels));
        }
        for (_, _, img) in sources.iter().skip(1) {
            let c = img.channels();
            if c != channels {
                return Err(BuildError::MixedSourceChannels {
                    first: channels,
                    offending: c,
                });
            }
        }

        // ── Pass 1: visibility matrix (sequential over sources for
        // deterministic src_indices ordering and trivial popcount/position) ─
        let mut visibility = vec![false; n_tiles * n_sources];
        for (i, (src_intrinsics, r_src_world, _)) in sources.iter().enumerate() {
            let r = r_src_world.to_rotation_matrix();
            let sw = src_intrinsics.width as f64;
            let sh = src_intrinsics.height as f64;
            for t in 0..n_tiles {
                let d_world = rig.direction(t);
                let d_src = r * Vector3::new(d_world[0], d_world[1], d_world[2]);
                if let Some((sx, sy)) = src_intrinsics.ray_to_pixel([d_src.x, d_src.y, d_src.z]) {
                    if sx >= 0.0 && sy >= 0.0 && sx < sw && sy < sh {
                        visibility[t * n_sources + i] = true;
                    }
                }
            }
        }

        // ── Pass 2: per-tile SoA allocation ──────────────────────────────
        let mut tiles: Vec<SphericalTileSourceStack> = (0..n_tiles)
            .map(|t| {
                let mut src_indices = Vec::new();
                for i in 0..n_sources {
                    if visibility[t * n_sources + i] {
                        src_indices.push(i as u32);
                    }
                }
                let k = src_indices.len();
                let levels = (0..pyramid_levels)
                    .map(|li| {
                        let s = b >> li;
                        let s_us = s as usize;
                        let c_us = channels as usize;
                        SphericalTilePatchLevel {
                            size: s,
                            n_contributors: k as u32,
                            channels,
                            patches: vec![0u8; k * s_us * s_us * c_us],
                            valid: vec![false; k * s_us * s_us],
                        }
                    })
                    .collect();
                SphericalTileSourceStack {
                    src_indices,
                    levels,
                }
            })
            .collect();

        // ── Pass 3: per-source warp + pyramid write ──────────────────────
        //
        // Outer loop is sequential over sources; inner work parallelises
        // across the source's kept tiles via rayon. Each tile-task writes to
        // a unique (tile, pos) slot in the SoA buffer it owns, so the writes
        // are race-free without atomics or locks.
        let tile_camera = rig.tile_camera();
        let tile_rotations: Vec<Matrix3<f64>> = (0..n_tiles)
            .map(|t| {
                let cols = rig.tile_rotation(t);
                Matrix3::from_columns(&[
                    Vector3::new(cols[0], cols[1], cols[2]),
                    Vector3::new(cols[3], cols[4], cols[5]),
                    Vector3::new(cols[6], cols[7], cols[8]),
                ])
            })
            .collect();

        for (src_idx, (src_intrinsics, r_src_world, image)) in sources.iter().enumerate() {
            let r_sw = r_src_world.to_rotation_matrix();
            let src_idx_u32 = src_idx as u32;

            tiles.par_iter_mut().enumerate().for_each(|(t, tile_data)| {
                let Some(pos) = tile_data.src_indices.iter().position(|&x| x == src_idx_u32) else {
                    return;
                };
                warp_and_downsample_into(
                    pos,
                    src_intrinsics,
                    &r_sw,
                    image,
                    &tile_camera,
                    &tile_rotations[t],
                    b,
                    pyramid_levels,
                    channels,
                    tile_data,
                );
            });
        }

        Ok(Self {
            n_tiles,
            base_patch_size: b,
            pyramid_levels,
            tiles,
        })
    }

    /// Number of tiles.
    pub fn n_tiles(&self) -> usize {
        self.n_tiles
    }

    /// Base patch size used to build this stack (= `rig.patch_size()` at
    /// build time). Always a power of two.
    pub fn base_patch_size(&self) -> u32 {
        self.base_patch_size
    }

    /// Number of pyramid levels per tile (= `log2(base_patch_size) + 1`).
    pub fn pyramid_levels(&self) -> u32 {
        self.pyramid_levels
    }

    /// All data for tile `idx`.
    ///
    /// # Panics
    /// Panics if `idx >= n_tiles()`.
    pub fn tile(&self, idx: usize) -> &SphericalTileSourceStack {
        &self.tiles[idx]
    }

    /// Number of contributing sources for tile `idx`.
    ///
    /// # Panics
    /// Panics if `idx >= n_tiles()`.
    pub fn n_contributors(&self, idx: usize) -> usize {
        self.tiles[idx].src_indices.len()
    }
}

/// Compute a single source's contribution to one tile: build the per-tile
/// rotation-only warp, remap to a `B × B` patch, then downsample through every
/// pyramid level. Writes directly into the SoA slot at `pos`.
#[allow(clippy::too_many_arguments)]
fn warp_and_downsample_into(
    pos: usize,
    src_intrinsics: &CameraIntrinsics,
    r_sw: &Matrix3<f64>,
    image: &ImageU8,
    tile_camera: &CameraIntrinsics,
    r_world_from_tile: &Matrix3<f64>,
    b: u32,
    pyramid_levels: u32,
    channels: u32,
    tile_data: &mut SphericalTileSourceStack,
) {
    // Build R_src_from_tile = R_src_from_world · R_world_from_tile.
    let r_st = r_sw * r_world_from_tile;
    let r_st_quat = RotQuaternion::from_rotation_matrix(r_st);
    let warp = WarpMap::from_cameras_with_rotation(src_intrinsics, tile_camera, &r_st_quat);
    let patch_img = remap_bilinear(image, &warp);
    debug_assert_eq!(patch_img.width(), b);
    debug_assert_eq!(patch_img.height(), b);
    debug_assert_eq!(patch_img.channels(), channels);

    let b_us = b as usize;
    let c_us = channels as usize;

    // Level-0 valid mask: warp's in-bounds bit.
    let mut prev_valid = Vec::with_capacity(b_us * b_us);
    for v in 0..b {
        for u in 0..b {
            prev_valid.push(warp.is_valid(u, v));
        }
    }
    let mut prev_patch = patch_img.data().to_vec();

    // Write level 0.
    {
        let level = &mut tile_data.levels[0];
        let pixel_count = b_us * b_us;
        let p_off = pos * pixel_count * c_us;
        level.patches[p_off..p_off + pixel_count * c_us].copy_from_slice(&prev_patch);
        let v_off = pos * pixel_count;
        level.valid[v_off..v_off + pixel_count].copy_from_slice(&prev_valid);
    }

    // Levels 1..L: 2× box-filter downsample with the all-four valid rule.
    let mut s_prev = b;
    for li in 1..pyramid_levels {
        let s = s_prev / 2;
        let s_us = s as usize;
        let s_prev_us = s_prev as usize;
        let pixel_count = s_us * s_us;
        let mut new_patch = vec![0u8; pixel_count * c_us];
        let mut new_valid = vec![false; pixel_count];
        for v in 0..s_us {
            for u in 0..s_us {
                let i00 = (2 * v) * s_prev_us + (2 * u);
                let i10 = (2 * v) * s_prev_us + (2 * u + 1);
                let i01 = (2 * v + 1) * s_prev_us + (2 * u);
                let i11 = (2 * v + 1) * s_prev_us + (2 * u + 1);
                let out_pix = v * s_us + u;
                new_valid[out_pix] =
                    prev_valid[i00] && prev_valid[i10] && prev_valid[i01] && prev_valid[i11];
                for ch in 0..c_us {
                    let v00 = prev_patch[i00 * c_us + ch] as u16;
                    let v10 = prev_patch[i10 * c_us + ch] as u16;
                    let v01 = prev_patch[i01 * c_us + ch] as u16;
                    let v11 = prev_patch[i11 * c_us + ch] as u16;
                    new_patch[out_pix * c_us + ch] = ((v00 + v10 + v01 + v11 + 2) / 4) as u8;
                }
            }
        }

        // Write into level li at slot pos.
        {
            let level = &mut tile_data.levels[li as usize];
            let p_off = pos * pixel_count * c_us;
            level.patches[p_off..p_off + pixel_count * c_us].copy_from_slice(&new_patch);
            let v_off = pos * pixel_count;
            level.valid[v_off..v_off + pixel_count].copy_from_slice(&new_valid);
        }

        prev_patch = new_patch;
        prev_valid = new_valid;
        s_prev = s;
    }
}

#[cfg(test)]
mod tests;
