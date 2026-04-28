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
//! Storage is CSR-flat across all tiles. For each pyramid level the
//! contributions of every kept `(tile, source)` pair live in a single
//! contiguous buffer of `total_contrib_rows · size² · channels` elements,
//! addressable in three ways:
//!
//! - **Per-tile slice** via [`patches_for_tile`] / [`valid_for_tile`] — what
//!   tile-major NCC / consensus consumers want.
//! - **Whole-level buffer** via [`level_patches`] / [`level_valid`] — a single
//!   `Tensor::from_data` source for autodiff / GPU pipelines that build one
//!   computational graph per pyramid level.
//! - **Per-row gather** via [`tile_id`] / [`src_id`] — segment-reduction by
//!   tile and gather-by-source, the access pattern joint-optimisation
//!   workloads (e.g. per-image colour parameters + per-tile cluster
//!   assignment) want.
//!
//! Pixel storage type is generic over [`PatchPixel`] — `u8` (compact, what
//! NCC / GPU-byte-texture consumers want) and `f32` (autodiff-ready) are
//! provided. The `u8` → `f32` conversion preserves range (`v as f32`,
//! 0–255), not scale, so `f32` storage is byte-equivalent to `u8` storage at
//! level 0 modulo type cost.
//!
//! [`patches_for_tile`]: PerSphericalTileSourceStack::patches_for_tile
//! [`valid_for_tile`]: PerSphericalTileSourceStack::valid_for_tile
//! [`level_patches`]: PerSphericalTileSourceStack::level_patches
//! [`level_valid`]: PerSphericalTileSourceStack::level_valid
//! [`tile_id`]: PerSphericalTileSourceStack::tile_id
//! [`src_id`]: PerSphericalTileSourceStack::src_id

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use crate::camera_intrinsics::CameraIntrinsics;
use crate::remap::{remap_bilinear, ImageU8};
use crate::rot_quaternion::RotQuaternion;
use crate::spherical_tile_rig::SphericalTileRig;
use crate::warp_map::WarpMap;

/// Pixel-element storage trait for [`PerSphericalTileSourceStack`].
///
/// Implementations define how a u8 source-image sample lands in storage and
/// how a 2× box-filter downsample is computed.
///
/// `from_u8` preserves **range** (0–255 maps to `0.0–255.0` for `f32`), not
/// scale: callers that want normalised inputs divide by 255 at their own
/// convenience.
pub trait PatchPixel: Copy + Default + Send + Sync + 'static {
    /// Convert a u8 source-image sample into storage type.
    fn from_u8(v: u8) -> Self;

    /// Mean of four storage-typed neighbours. For `u8` this is the
    /// round-to-nearest u16 mean `(a + b + c + d + 2) / 4`; for `f32` it
    /// is exact arithmetic `0.25 · (a + b + c + d)`.
    fn box_avg_4(a: Self, b: Self, c: Self, d: Self) -> Self;
}

impl PatchPixel for u8 {
    #[inline]
    fn from_u8(v: u8) -> Self {
        v
    }

    #[inline]
    fn box_avg_4(a: u8, b: u8, c: u8, d: u8) -> Self {
        ((a as u16 + b as u16 + c as u16 + d as u16 + 2) / 4) as u8
    }
}

impl PatchPixel for f32 {
    #[inline]
    fn from_u8(v: u8) -> Self {
        v as f32
    }

    #[inline]
    fn box_avg_4(a: f32, b: f32, c: f32, d: f32) -> Self {
        0.25 * (a + b + c + d)
    }
}

/// Per-pyramid-level CSR-packed storage for patches and valid masks.
///
/// Layout for `total_contrib_rows` rows, square patches of side `size`,
/// `C` channels:
///
/// - `patches`: row-major `total_contrib_rows × size² × C` — row `r`
///   occupies `[r · size² · C .. (r + 1) · size² · C)`.
/// - `valid`: row-major `total_contrib_rows × size²` — row `r` occupies
///   `[r · size² .. (r + 1) · size²)`. Values are strictly `{0, 1}`.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchLevel<T> {
    /// Side length of this level in pixels (`base_patch_size >> level_idx`).
    pub size: u32,
    /// All contributions concatenated in row-major form. Row indexing is
    /// global across tiles; within a row, pixels are row-major and channels
    /// are interleaved.
    pub patches: Vec<T>,
    /// Per-pixel valid mask, same row layout as `patches` minus the channel
    /// dimension. `0` = invalid, `1` = valid; no other values occur.
    pub valid: Vec<u8>,
}

/// Per-spherical-tile source patch stack with CSR-packed per-level storage.
///
/// Generic over the pixel storage type `T`; see [`PatchPixel`].
#[derive(Debug, Clone)]
pub struct PerSphericalTileSourceStack<T: PatchPixel> {
    n_tiles: usize,
    base_patch_size: u32,
    pyramid_levels: u32,
    channels: u32,
    /// Per-row source-list index, length `total_contrib_rows`.
    src_id: Vec<u32>,
    /// Per-row tile index, length `total_contrib_rows`. Materialised once at
    /// build time so segment-reduction consumers can use it directly as a
    /// gather index.
    tile_id: Vec<u32>,
    /// CSR offsets, length `n_tiles + 1`. Tile `t`'s rows are
    /// `tile_offsets[t]..tile_offsets[t + 1]`. `tile_offsets[n_tiles]
    /// == total_contrib_rows`.
    tile_offsets: Vec<u32>,
    /// Pyramid levels in order: `levels[0]` is the base
    /// (`base_patch_size × base_patch_size`), `levels[L − 1]` is `1 × 1`.
    levels: Vec<PatchLevel<T>>,
}

/// Build-time tuning knobs.
#[derive(Debug, Clone, Default)]
pub struct BuildParams {
    /// Reserved for future extension. Currently the build always processes
    /// sources sequentially in the outer loop and parallelises across each
    /// source's kept tiles via rayon. Setting this knob has no effect today;
    /// it is reserved so callers can opt into a future parallel-source
    /// chunking variant without an API break.
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

impl<T: PatchPixel> PerSphericalTileSourceStack<T> {
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
        let pyramid_levels = b.trailing_zeros() + 1;
        let n_tiles = rig.len();
        let n_sources = sources.len();

        // ── Channel-count validation ─────────────────────────────────────
        let channels = match sources.first() {
            Some((_, _, img)) => img.channels(),
            None => 1,
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

        // ── Pass 1: visibility matrix ────────────────────────────────────
        // visibility[t * n_sources + i] = does source `i` see tile `t`'s
        // centre direction inside its image?
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

        // ── Pass 1b: derive CSR offsets, src_id, tile_id, position ───────
        // `position[t * n_sources + i]` is the contributor's row offset
        // within tile `t`'s range, defined only for visible (t, i). Used at
        // Pass 3 to compute the global row index `tile_offsets[t] +
        // position[t][i]`.
        let mut tile_offsets = vec![0u32; n_tiles + 1];
        for t in 0..n_tiles {
            let count: u32 = (0..n_sources)
                .filter(|&i| visibility[t * n_sources + i])
                .count() as u32;
            tile_offsets[t + 1] = tile_offsets[t] + count;
        }
        let total_contrib_rows = tile_offsets[n_tiles] as usize;

        let mut src_id = vec![0u32; total_contrib_rows];
        let mut tile_id = vec![0u32; total_contrib_rows];
        let mut position = vec![0u32; n_tiles * n_sources];
        for t in 0..n_tiles {
            let mut p = 0u32;
            let base = tile_offsets[t] as usize;
            for i in 0..n_sources {
                if visibility[t * n_sources + i] {
                    src_id[base + p as usize] = i as u32;
                    tile_id[base + p as usize] = t as u32;
                    position[t * n_sources + i] = p;
                    p += 1;
                }
            }
        }

        // ── Pass 2: allocate per-level CSR buffers ───────────────────────
        let c_us = channels as usize;
        let mut levels: Vec<PatchLevel<T>> = (0..pyramid_levels)
            .map(|li| {
                let s = (b >> li) as usize;
                let pixel_count = s * s;
                PatchLevel {
                    size: b >> li,
                    patches: vec![T::default(); total_contrib_rows * pixel_count * c_us],
                    valid: vec![0u8; total_contrib_rows * pixel_count],
                }
            })
            .collect();

        // ── Pass 3: parallel per-source warp + pyramid write ─────────────
        // Sequential outer over sources; rayon-parallel inner over each
        // source's kept tiles. Each parallel task computes the global row
        // `r = tile_offsets[t] + position[t][i]` and writes into that row's
        // slot in every level's CSR buffer. The row is unique per kept
        // (t, i), so within a single source's parallel sweep the tasks
        // touch disjoint slices — see the SAFETY comment on
        // `SoaWriter::write_level_row`.
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

        let writer = SoaWriter::new(&mut levels, channels);

        for (src_idx, (src_intrinsics, r_src_world, image)) in sources.iter().enumerate() {
            let r_sw = r_src_world.to_rotation_matrix();
            let visibility_ref = &visibility;
            let position_ref = &position;
            let tile_offsets_ref = &tile_offsets;
            let writer_ref = &writer;
            let tile_rotations_ref = &tile_rotations;

            (0..n_tiles).into_par_iter().for_each(|t| {
                if !visibility_ref[t * n_sources + src_idx] {
                    return;
                }
                let p = position_ref[t * n_sources + src_idx];
                let row = (tile_offsets_ref[t] + p) as usize;
                // SAFETY: every kept (t, src_idx) pair maps to a distinct
                // global row (`tile_offsets[t]` is unique per t,
                // `position[t][src_idx]` is unique per src_idx within tile
                // t). Within this for_each over a fixed src_idx, no two
                // parallel tasks share a row; their writes hit disjoint
                // byte ranges in every level's CSR buffer.
                unsafe {
                    warp_and_downsample_into::<T>(
                        row,
                        src_intrinsics,
                        &r_sw,
                        image,
                        &tile_camera,
                        &tile_rotations_ref[t],
                        b,
                        pyramid_levels,
                        channels,
                        writer_ref,
                    );
                }
            });
        }
        // The writer's raw pointers alias `levels`; drop it before moving
        // `levels` into the returned struct.
        drop(writer);

        Ok(Self {
            n_tiles,
            base_patch_size: b,
            pyramid_levels,
            channels,
            src_id,
            tile_id,
            tile_offsets,
            levels,
        })
    }

    /// Number of tiles in the stack (mirrors `rig.len()`).
    pub fn n_tiles(&self) -> usize {
        self.n_tiles
    }

    /// Side length of level 0 across every tile (= `rig.patch_size` at
    /// build time). Always a power of two.
    pub fn base_patch_size(&self) -> u32 {
        self.base_patch_size
    }

    /// Number of pyramid levels (= `log2(base_patch_size) + 1`).
    pub fn pyramid_levels(&self) -> u32 {
        self.pyramid_levels
    }

    /// Channel count, uniform across all tiles, sources, and levels.
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Total CSR row count summed across all tiles
    /// (`tile_offsets[n_tiles]`).
    pub fn total_contrib_rows(&self) -> usize {
        self.src_id.len()
    }

    /// Source-list indices contributing to tile `t`, ascending. Slice of
    /// the flat `src_id` array.
    ///
    /// # Panics
    /// Panics if `t >= n_tiles()`.
    pub fn src_indices_for_tile(&self, t: usize) -> &[u32] {
        let start = self.tile_offsets[t] as usize;
        let end = self.tile_offsets[t + 1] as usize;
        &self.src_id[start..end]
    }

    /// Number of contributing sources for tile `t`.
    ///
    /// # Panics
    /// Panics if `t >= n_tiles()`.
    pub fn n_contributors(&self, t: usize) -> usize {
        (self.tile_offsets[t + 1] - self.tile_offsets[t]) as usize
    }

    /// Tile `t`'s patches at level `l`. Layout: row-major
    /// `n_contributors(t) × size² × channels`, contributors in
    /// `src_indices_for_tile(t)` order. Empty when `n_contributors(t) ==
    /// 0`.
    ///
    /// # Panics
    /// Panics if `t >= n_tiles()` or `l >= pyramid_levels()`.
    pub fn patches_for_tile(&self, t: usize, l: usize) -> &[T] {
        let level = &self.levels[l];
        let s = level.size as usize;
        let c = self.channels as usize;
        let start = self.tile_offsets[t] as usize * s * s * c;
        let end = self.tile_offsets[t + 1] as usize * s * s * c;
        &level.patches[start..end]
    }

    /// Tile `t`'s valid masks at level `l`. Layout: row-major
    /// `n_contributors(t) × size²`, contributors in the same order as
    /// `patches_for_tile`.
    ///
    /// # Panics
    /// Panics if `t >= n_tiles()` or `l >= pyramid_levels()`.
    pub fn valid_for_tile(&self, t: usize, l: usize) -> &[u8] {
        let level = &self.levels[l];
        let s = level.size as usize;
        let start = self.tile_offsets[t] as usize * s * s;
        let end = self.tile_offsets[t + 1] as usize * s * s;
        &level.valid[start..end]
    }

    /// Per-row source index. Length `total_contrib_rows`.
    pub fn src_id(&self) -> &[u32] {
        &self.src_id
    }

    /// Per-row tile index. Length `total_contrib_rows`.
    pub fn tile_id(&self) -> &[u32] {
        &self.tile_id
    }

    /// CSR offsets, length `n_tiles + 1`. `tile_offsets()[t]` is the row
    /// index where tile `t`'s contributions start.
    pub fn tile_offsets(&self) -> &[u32] {
        &self.tile_offsets
    }

    /// Pyramid level metadata + buffers as a struct.
    ///
    /// # Panics
    /// Panics if `l >= pyramid_levels()`.
    pub fn level(&self, l: usize) -> &PatchLevel<T> {
        &self.levels[l]
    }

    /// Whole-level patches buffer at level `l`. Length
    /// `total_contrib_rows · size_l² · channels`.
    pub fn level_patches(&self, l: usize) -> &[T] {
        &self.levels[l].patches
    }

    /// Whole-level valid buffer at level `l`. Length
    /// `total_contrib_rows · size_l²`.
    pub fn level_valid(&self, l: usize) -> &[u8] {
        &self.levels[l].valid
    }
}

// ── Unsafe SoA writer for parallel disjoint-row writes ──────────────────

/// Raw mutable view across all level buffers, suitable for parallel
/// disjoint-row writes during the build. Holds raw pointers into the
/// `levels` vector's per-level patches/valid buffers; safety relies on the
/// caller writing only to unique `(level, row)` pairs from any given
/// thread.
struct SoaWriter<T> {
    /// Per pyramid level: (patches_ptr, valid_ptr, size).
    levels: Vec<(*mut T, *mut u8, u32)>,
    channels: u32,
}

// SAFETY: the raw pointers are derived from `&mut [PatchLevel<T>]`, so the
// referent lives at least as long as the borrow that produced them. The
// build never reads through these pointers after writing — each row is
// written exactly once across the whole build, no aliasing reads.
unsafe impl<T: Send> Send for SoaWriter<T> {}
unsafe impl<T: Send> Sync for SoaWriter<T> {}

impl<T: PatchPixel> SoaWriter<T> {
    fn new(levels: &mut [PatchLevel<T>], channels: u32) -> Self {
        let v: Vec<_> = levels
            .iter_mut()
            .map(|lvl| (lvl.patches.as_mut_ptr(), lvl.valid.as_mut_ptr(), lvl.size))
            .collect();
        Self {
            levels: v,
            channels,
        }
    }

    /// Write one `(level, row)` slot.
    ///
    /// # Safety
    /// Caller must ensure no two threads write to the same
    /// `(level_idx, row)` concurrently, and that `row <
    /// total_contrib_rows` and `level_idx < pyramid_levels` for the
    /// underlying stack. `patch.len()` must equal `size² · channels`;
    /// `valid.len()` must equal `size²`.
    unsafe fn write_level_row(&self, level_idx: usize, row: usize, patch: &[T], valid: &[u8]) {
        let (p_ptr, v_ptr, size) = self.levels[level_idx];
        let s = size as usize;
        let c = self.channels as usize;
        debug_assert_eq!(patch.len(), s * s * c);
        debug_assert_eq!(valid.len(), s * s);
        let p_off = row * s * s * c;
        let v_off = row * s * s;
        // SAFETY: see struct-level docs and the call site's SAFETY note.
        unsafe {
            std::ptr::copy_nonoverlapping(patch.as_ptr(), p_ptr.add(p_off), s * s * c);
            std::ptr::copy_nonoverlapping(valid.as_ptr(), v_ptr.add(v_off), s * s);
        }
    }
}

/// Compute one `(source, tile)` contribution and write it through every
/// pyramid level. Allocates per-call scratch buffers for the level-by-level
/// downsample state.
///
/// # Safety
/// `writer` must satisfy the disjoint-row invariant for `(level_idx, row)`
/// across all concurrent callers — see [`SoaWriter::write_level_row`].
#[allow(clippy::too_many_arguments)]
unsafe fn warp_and_downsample_into<T: PatchPixel>(
    row: usize,
    src_intrinsics: &CameraIntrinsics,
    r_sw: &Matrix3<f64>,
    image: &ImageU8,
    tile_camera: &CameraIntrinsics,
    r_world_from_tile: &Matrix3<f64>,
    b: u32,
    pyramid_levels: u32,
    channels: u32,
    writer: &SoaWriter<T>,
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
    let pixel_count_0 = b_us * b_us;

    // Level-0 valid mask (0/1 u8) and level-0 patch in storage type T.
    let mut prev_valid: Vec<u8> = Vec::with_capacity(pixel_count_0);
    for v in 0..b {
        for u in 0..b {
            prev_valid.push(if warp.is_valid(u, v) { 1 } else { 0 });
        }
    }
    let mut prev_patch: Vec<T> = patch_img.data().iter().map(|&x| T::from_u8(x)).collect();

    // Write level 0.
    // SAFETY: forwarded from this function's caller — `row` is unique
    // among concurrently running calls.
    unsafe {
        writer.write_level_row(0, row, &prev_patch, &prev_valid);
    }

    // Levels 1..L: 2× box-filter downsample with the all-four valid rule.
    let mut s_prev = b;
    for li in 1..pyramid_levels {
        let s = s_prev / 2;
        let s_us = s as usize;
        let s_prev_us = s_prev as usize;
        let pixel_count = s_us * s_us;
        let mut new_patch: Vec<T> = vec![T::default(); pixel_count * c_us];
        let mut new_valid: Vec<u8> = vec![0u8; pixel_count];
        for v in 0..s_us {
            for u in 0..s_us {
                let i00 = (2 * v) * s_prev_us + (2 * u);
                let i10 = (2 * v) * s_prev_us + (2 * u + 1);
                let i01 = (2 * v + 1) * s_prev_us + (2 * u);
                let i11 = (2 * v + 1) * s_prev_us + (2 * u + 1);
                let out_pix = v * s_us + u;
                new_valid[out_pix] =
                    prev_valid[i00] & prev_valid[i10] & prev_valid[i01] & prev_valid[i11];
                for ch in 0..c_us {
                    let v00 = prev_patch[i00 * c_us + ch];
                    let v10 = prev_patch[i10 * c_us + ch];
                    let v01 = prev_patch[i01 * c_us + ch];
                    let v11 = prev_patch[i11 * c_us + ch];
                    new_patch[out_pix * c_us + ch] = T::box_avg_4(v00, v10, v01, v11);
                }
            }
        }
        // SAFETY: forwarded from this function's caller.
        unsafe {
            writer.write_level_row(li as usize, row, &new_patch, &new_valid);
        }
        prev_patch = new_patch;
        prev_valid = new_valid;
        s_prev = s;
    }
}

#[cfg(test)]
mod tests;
