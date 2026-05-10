// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Spherical tile rig: discretizing the sphere as a rig of pinhole tiles.
//!
//! See `specs/core/spherical-tiles-rig.md` for design details.
//!
//! A [`SphericalTileRig`] is a camera rig of `n` pinhole "tile" cameras that
//! share a single world-space optical centre. Every tile has identical
//! intrinsics (focal length, half-FOV, patch resolution); only the rotation
//! differs per tile so that the tiles look in different directions on the
//! unit sphere around the rig centre.
//!
//! Per-tile arrays (color / alpha / depth / NCC scratch) live in a single
//! "atlas" image — each tile occupies a `patch_size × patch_size` block at
//! `tile_atlas_origin(idx)`. The atlas-as-destination ([`warp_to_atlas_with_rotation`])
//! and atlas-as-source ([`warp_from_atlas_with_rotation`]) [`WarpMap`]
//! constructors do the per-pixel reprojection in a single rayon-parallel pass.
//!
//! [`warp_to_atlas_with_rotation`]: SphericalTileRig::warp_to_atlas_with_rotation
//! [`warp_from_atlas_with_rotation`]: SphericalTileRig::warp_from_atlas_with_rotation

use nalgebra::Vector3;
use rayon::prelude::*;

use crate::camera_intrinsics::{CameraIntrinsics, CameraModel};
use crate::rot_quaternion::RotQuaternion;
use crate::se3_transform::Se3Transform;
use crate::spatial::PointCloud3;
use crate::sphere_points::{evenly_distributed_sphere_points, random_sphere_points, RelaxConfig};
use crate::warp_map::WarpMap;

/// Lower bound on `patch_size`. Keeps NCC / gradient kernels meaningful at
/// small `n`.
const MIN_PATCH_SIZE: u32 = 5;

/// Threshold (in `|direction.y|`) above which a tile is treated as
/// "near a pole" — the world-up reference flips from ±Y to ±X to keep the
/// cross product well-conditioned.
const POLE_GUARD_Y: f64 = 0.95;

/// Number of probe points used to measure the worst-case Voronoi-cell
/// radius across the rig at construction time. With ~50K points, the probe
/// undersamples the sphere by at most ~0.018 rad; the `overlap_factor`
/// safety margin absorbs that.
const COVERAGE_PROBE_N: usize = 50_000;

/// Sizing parameters for [`SphericalTileRig::new`].
#[derive(Debug, Clone)]
pub struct SphericalTileRigParams {
    /// Rig optical centre in world space.
    pub centre: [f64; 3],
    /// Number of tiles. Must be ≥ 2.
    pub n: usize,
    /// Angular size of one tile pixel, in radians. For a target equirect
    /// of width `W`, pass `2π / W`. Must be `> 0`.
    pub arc_per_pixel: f64,
    /// Multiplicative safety margin on the **measured** worst-case
    /// nearest-centre gap. `1.15` = 15% overlap; default. Must be `≥ 1.0`.
    pub overlap_factor: f64,
    /// Optional override for the atlas column count. `None` ⇒ `ceil(√n)`.
    pub atlas_cols: Option<u32>,
    /// Forwarded to [`evenly_distributed_sphere_points`]. `None` ⇒ defaults
    /// (also unseeded — pass `Some(RelaxConfig { seed: Some(_), .. })` for
    /// reproducible runs).
    pub relax: Option<RelaxConfig>,
}

impl Default for SphericalTileRigParams {
    fn default() -> Self {
        Self {
            centre: [0.0, 0.0, 0.0],
            n: 320,
            arc_per_pixel: 2.0 * std::f64::consts::PI / 512.0,
            overlap_factor: 1.15,
            atlas_cols: None,
            relax: None,
        }
    }
}

/// Construction-time validation errors for [`SphericalTileRig::new`].
#[derive(Debug, PartialEq, Eq)]
pub enum SphericalTileRigError {
    /// `n < 2`.
    TooFewTiles,
    /// `arc_per_pixel <= 0` or non-finite.
    InvalidArcPerPixel,
    /// `overlap_factor < 1.0` or non-finite.
    InvalidOverlapFactor,
    /// `centre` contains a non-finite component.
    InvalidCentre,
}

impl std::fmt::Display for SphericalTileRigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewTiles => write!(f, "n must be >= 2"),
            Self::InvalidArcPerPixel => write!(f, "arc_per_pixel must be finite and > 0"),
            Self::InvalidOverlapFactor => write!(f, "overlap_factor must be finite and >= 1.0"),
            Self::InvalidCentre => write!(f, "centre must contain only finite components"),
        }
    }
}

impl std::error::Error for SphericalTileRigError {}

/// A rig of `n` shared-centre pinhole tile cameras tiling the unit sphere.
pub struct SphericalTileRig {
    /// World-space optical centre shared by every tile.
    centre: [f64; 3],
    /// Unit look directions in world frame. Length `n`.
    directions: Vec<[f64; 3]>,
    /// World-frame `(e_right, e_up)` tangent-plane bases per tile.
    /// Length `n`. `e_right × e_up = direction`.
    bases: Vec<[f64; 6]>,
    /// Per-tile half-FOV in radians (uniform).
    half_fov_rad: f64,
    /// Measured worst-case nearest-neighbour angular gap among tile
    /// directions, in radians. Diagnostic only.
    measured_max_nn_angle: f64,
    /// Measured worst-case Voronoi-cell radius across the sphere — the
    /// largest angle from any probe direction to its nearest tile,
    /// determined at construction by sampling the sphere with
    /// [`COVERAGE_PROBE_N`] points. `half_fov_rad` is derived from this.
    measured_max_coverage_angle: f64,
    /// Patch grid size (pixels per side). Uniform across tiles.
    patch_size: u32,
    /// Number of tile columns in the packed atlas.
    atlas_cols: u32,
    /// KD-tree over `directions` (as f32 unit vectors). Squared-Euclidean
    /// nearest-neighbour on unit vectors is monotone in angular distance,
    /// so this gives the angularly-closest tile.
    direction_tree: PointCloud3<f32>,
}

impl SphericalTileRig {
    /// Build from explicit sizing parameters.
    ///
    /// Tile look directions come from the sphere-point relaxer;
    /// `half_fov_rad` is derived from the **measured** worst-case nearest
    /// neighbour gap times `overlap_factor / 2`; `patch_size` is then
    /// `max(MIN_PATCH_SIZE, ceil(2 · half_fov_rad / arc_per_pixel))`.
    pub fn new(params: &SphericalTileRigParams) -> Result<Self, SphericalTileRigError> {
        // ── Validate ──────────────────────────────────────────────────────
        if params.n < 2 {
            return Err(SphericalTileRigError::TooFewTiles);
        }
        if !params.arc_per_pixel.is_finite() || params.arc_per_pixel <= 0.0 {
            return Err(SphericalTileRigError::InvalidArcPerPixel);
        }
        if !params.overlap_factor.is_finite() || params.overlap_factor < 1.0 {
            return Err(SphericalTileRigError::InvalidOverlapFactor);
        }
        if !params.centre.iter().all(|c| c.is_finite()) {
            return Err(SphericalTileRigError::InvalidCentre);
        }

        // ── Tile look directions ──────────────────────────────────────────
        let relax_cfg = params.relax.clone().unwrap_or_default();
        let flat = evenly_distributed_sphere_points(params.n, &relax_cfg);
        debug_assert_eq!(flat.len(), 3 * params.n);

        // Convert to Vec<[f64; 3]> (world frame, re-normalised to f64 precision
        // so the resulting tangent bases are orthonormal to f64 precision).
        let directions: Vec<[f64; 3]> = (0..params.n)
            .map(|i| {
                let x = flat[3 * i] as f64;
                let y = flat[3 * i + 1] as f64;
                let z = flat[3 * i + 2] as f64;
                let inv = 1.0 / (x * x + y * y + z * z).sqrt();
                [x * inv, y * inv, z * inv]
            })
            .collect();

        // ── Measure worst-case NN angle from the f32 cloud ────────────────
        let direction_tree = PointCloud3::<f32>::new(&flat, params.n);
        let nn_chords_f32 = direction_tree.nearest_neighbor_distances();
        // chord = 2 sin(angle/2)  ⇒  angle = 2 asin(chord / 2)
        let measured_max_nn_angle = nn_chords_f32
            .iter()
            .map(|&c| 2.0 * ((c as f64) * 0.5).clamp(-1.0, 1.0).asin())
            .fold(0.0_f64, f64::max);

        // ── Probe coverage: max Voronoi-cell radius across the sphere ─────
        // For well-relaxed point sets the Voronoi-cell radius is ≈ half the
        // NN angle, so this matches `0.5 · measured_max_nn_angle` to within
        // probe noise. For under-relaxed sets (small `n`) it is strictly
        // larger, which is exactly when half-NN is *not* a tight bound on
        // tile coverage. Using the probed radius makes coverage hold by
        // construction at all `n`.
        //
        // The probe seed is derived from the relaxer seed (or `0` if
        // unseeded) so the result is reproducible whenever the rest of the
        // rig is.
        let probe_seed = relax_cfg.seed.unwrap_or(0).wrapping_add(0xc0ffee);
        let probe_flat = random_sphere_points(COVERAGE_PROBE_N, Some(probe_seed));
        let probe_nn = direction_tree.nearest(&probe_flat, COVERAGE_PROBE_N);
        let mut max_coverage_chord_sq = 0.0_f64;
        for i in 0..COVERAGE_PROBE_N {
            let px = probe_flat[3 * i] as f64;
            let py = probe_flat[3 * i + 1] as f64;
            let pz = probe_flat[3 * i + 2] as f64;
            let d = directions[probe_nn[i] as usize];
            let dx = px - d[0];
            let dy = py - d[1];
            let dz = pz - d[2];
            let s = dx * dx + dy * dy + dz * dz;
            if s > max_coverage_chord_sq {
                max_coverage_chord_sq = s;
            }
        }
        let max_chord = max_coverage_chord_sq.sqrt();
        let measured_max_coverage_angle = 2.0 * (max_chord * 0.5).clamp(-1.0, 1.0).asin();

        // ── Sizing ────────────────────────────────────────────────────────
        let half_fov_rad = measured_max_coverage_angle * params.overlap_factor;
        let patch_size = {
            let raw = (2.0 * half_fov_rad / params.arc_per_pixel).ceil() as u32;
            raw.max(MIN_PATCH_SIZE)
        };
        let atlas_cols = params
            .atlas_cols
            .unwrap_or_else(|| (params.n as f64).sqrt().ceil() as u32)
            .max(1);

        // ── Build per-tile tangent basis ──────────────────────────────────
        let bases: Vec<[f64; 6]> = directions.iter().map(|d| build_basis(*d)).collect();

        Ok(Self {
            centre: params.centre,
            directions,
            bases,
            half_fov_rad,
            measured_max_nn_angle,
            measured_max_coverage_angle,
            patch_size,
            atlas_cols,
            direction_tree,
        })
    }

    /// Number of tiles in the rig.
    pub fn len(&self) -> usize {
        self.directions.len()
    }

    /// Whether the rig is empty (always `false` after a successful `new`).
    pub fn is_empty(&self) -> bool {
        self.directions.is_empty()
    }

    /// World-space optical centre shared by every tile.
    pub fn centre(&self) -> [f64; 3] {
        self.centre
    }

    /// Unit look direction of tile `idx` in world frame.
    pub fn direction(&self, idx: usize) -> [f64; 3] {
        self.directions[idx]
    }

    /// `(e_right, e_up)` tangent basis of tile `idx` in world frame.
    pub fn basis(&self, idx: usize) -> ([f64; 3], [f64; 3]) {
        let b = self.bases[idx];
        ([b[0], b[1], b[2]], [b[3], b[4], b[5]])
    }

    /// Per-tile half-FOV in radians.
    pub fn half_fov_rad(&self) -> f64 {
        self.half_fov_rad
    }

    /// Measured worst-case nearest-neighbour angular gap across all tile
    /// directions. Diagnostic.
    pub fn measured_max_nn_angle(&self) -> f64 {
        self.measured_max_nn_angle
    }

    /// Measured worst-case Voronoi-cell radius across the sphere — the
    /// largest angle from any probe direction to its nearest tile.
    /// `half_fov_rad = measured_max_coverage_angle · overlap_factor`.
    pub fn measured_max_coverage_angle(&self) -> f64 {
        self.measured_max_coverage_angle
    }

    /// Patch grid size (pixels per side). Uniform across tiles.
    pub fn patch_size(&self) -> u32 {
        self.patch_size
    }

    /// Override the per-tile patch size after construction.
    ///
    /// Use this when a downstream consumer needs a specific patch size the
    /// constructor's `arc_per_pixel`-driven formula doesn't produce — for
    /// example, rounding up to the next power of two so the per-tile patch
    /// can serve as an image-pyramid base.
    ///
    /// Tile directions, bases, half-FOV, and the KD-tree are unaffected; only
    /// [`tile_camera`](Self::tile_camera) and [`atlas_size`](Self::atlas_size)
    /// (and their derivatives) shift to the new resolution.
    ///
    /// # Panics
    /// Panics if `patch_size == 0`.
    pub fn set_patch_size(&mut self, patch_size: u32) {
        assert!(patch_size > 0, "patch_size must be > 0");
        self.patch_size = patch_size;
    }

    /// Pinhole `CameraIntrinsics` shared by every tile.
    pub fn tile_camera(&self) -> CameraIntrinsics {
        let half = self.patch_size as f64 / 2.0;
        let f = half / self.half_fov_rad.tan();
        CameraIntrinsics {
            model: CameraModel::Pinhole {
                focal_length_x: f,
                focal_length_y: f,
                principal_point_x: half,
                principal_point_y: half,
            },
            width: self.patch_size,
            height: self.patch_size,
        }
    }

    /// Build `R_world_from_tile` for tile `idx`, columns `[e_right | e_up | direction]`.
    pub fn tile_rotation(&self, idx: usize) -> [f64; 9] {
        let b = self.bases[idx];
        let d = self.directions[idx];
        // Column-major 3x3 (matches our convention: first three entries are the
        // first column, e_right; next three are e_up; last three are direction).
        [b[0], b[1], b[2], b[3], b[4], b[5], d[0], d[1], d[2]]
    }

    /// Apply an `Se3Transform` to the rig: rotates and translates the
    /// centre, rotates `directions` and `bases`, and rebuilds the
    /// direction KD-tree. Scale is **not** applied (the rig is unitless).
    pub fn apply_transform(&mut self, t: &Se3Transform) {
        let rot = t.rotation.to_rotation_matrix();

        // Centre: rotate then translate (scale ignored — directions are unit).
        let c = Vector3::new(self.centre[0], self.centre[1], self.centre[2]);
        let new_c = rot * c + t.translation;
        self.centre = [new_c.x, new_c.y, new_c.z];

        for d in &mut self.directions {
            let v = rot * Vector3::new(d[0], d[1], d[2]);
            *d = [v.x, v.y, v.z];
        }
        for b in &mut self.bases {
            let r = rot * Vector3::new(b[0], b[1], b[2]);
            let u = rot * Vector3::new(b[3], b[4], b[5]);
            *b = [r.x, r.y, r.z, u.x, u.y, u.z];
        }

        // Rebuild the KD-tree from the rotated directions (as f32).
        let flat: Vec<f32> = self
            .directions
            .iter()
            .flat_map(|d| [d[0] as f32, d[1] as f32, d[2] as f32])
            .collect();
        self.direction_tree = PointCloud3::<f32>::new(&flat, self.directions.len());
    }

    // ── Tile-to-image (atlas) mapping ─────────────────────────────────────

    /// Number of tile columns in the packed atlas.
    pub fn atlas_cols(&self) -> u32 {
        self.atlas_cols
    }

    /// Number of tile rows in the packed atlas (`ceil(n / atlas_cols)`).
    pub fn atlas_rows(&self) -> u32 {
        self.len().div_ceil(self.atlas_cols as usize) as u32
    }

    /// Atlas image `(width, height)` in pixels.
    pub fn atlas_size(&self) -> (u32, u32) {
        (
            self.atlas_cols * self.patch_size,
            self.atlas_rows() * self.patch_size,
        )
    }

    /// Top-left pixel `(x, y)` of tile `idx`'s sub-image inside the atlas.
    ///
    /// # Panics
    /// Panics if `idx >= len()`.
    pub fn tile_atlas_origin(&self, idx: usize) -> (u32, u32) {
        assert!(idx < self.len(), "tile index {idx} out of range");
        let idx_u32 = idx as u32;
        let col = idx_u32 % self.atlas_cols;
        let row = idx_u32 / self.atlas_cols;
        (col * self.patch_size, row * self.patch_size)
    }

    // ── Atlas warp maps ───────────────────────────────────────────────────

    /// Build a `WarpMap` with the atlas as the **destination** image.
    ///
    /// For each atlas pixel `(u, v)` the owning tile is read off the atlas
    /// row/column, the in-tile pixel is unprojected through `tile_camera()`
    /// to a tile-frame ray, the ray is rotated to the world frame via
    /// `R_world_from_tile = [e_right | e_up | direction]`, then to the src
    /// frame via `rot_src_from_world`, and finally projected through `src`.
    /// Atlas slots that do not belong to any tile (the trailing slots when
    /// `atlas_cols` does not divide `n` evenly) are filled with `NaN`.
    pub fn warp_to_atlas_with_rotation(
        &self,
        src: &CameraIntrinsics,
        rot_src_from_world: &RotQuaternion,
    ) -> WarpMap {
        let (atlas_w, atlas_h) = self.atlas_size();
        let src_w = src.width as f64;
        let src_h = src.height as f64;
        let r_sw = rot_src_from_world.to_rotation_matrix();
        let tile_cam = self.tile_camera();
        let patch_size = self.patch_size;
        let atlas_cols = self.atlas_cols;
        let n = self.len();

        let row_len = 2 * atlas_w as usize;
        let data: Vec<f32> = (0..atlas_h)
            .into_par_iter()
            .flat_map(|row| {
                let mut row_data = vec![f32::NAN; row_len];
                let tile_row = row / patch_size;
                let in_row = row - tile_row * patch_size;
                let v_tile_pix = in_row as f64 + 0.5;

                for col in 0..atlas_w {
                    let tile_col = col / patch_size;
                    let tile_idx = (tile_row * atlas_cols + tile_col) as usize;
                    if tile_idx >= n {
                        continue; // trailing atlas slot — leave as NaN
                    }
                    let in_col = col - tile_col * patch_size;
                    let u_tile_pix = in_col as f64 + 0.5;

                    // tile-frame ray
                    let t_ray = tile_cam.pixel_to_ray(u_tile_pix, v_tile_pix);

                    // tile → world via R_world_from_tile = [e_right | e_up | direction]
                    let basis = self.bases[tile_idx];
                    let dir = self.directions[tile_idx];
                    let world = Vector3::new(
                        basis[0] * t_ray[0] + basis[3] * t_ray[1] + dir[0] * t_ray[2],
                        basis[1] * t_ray[0] + basis[4] * t_ray[1] + dir[1] * t_ray[2],
                        basis[2] * t_ray[0] + basis[5] * t_ray[1] + dir[2] * t_ray[2],
                    );

                    // world → src via rot_src_from_world
                    let s = r_sw * world;
                    let (sx, sy) = match src.ray_to_pixel([s.x, s.y, s.z]) {
                        Some(pix) => pix,
                        None => continue, // leave as NaN
                    };
                    if sx < 0.0 || sy < 0.0 || sx >= src_w || sy >= src_h {
                        continue;
                    }

                    let idx = 2 * col as usize;
                    row_data[idx] = sx as f32;
                    row_data[idx + 1] = sy as f32;
                }
                row_data
            })
            .collect();

        WarpMap::new(atlas_w, atlas_h, data)
    }

    /// Resample an atlas image into a destination camera, blending the `k`
    /// angularly-nearest tiles per dst pixel by inverse-angular-distance
    /// weights.
    ///
    /// `atlas` is the per-pixel atlas data laid out row-major with channels
    /// interleaved: `atlas.len() == atlas_size().0 * atlas_size().1 * channels`.
    /// The result has length `dst.width * dst.height * channels`, same layout.
    /// `channels` may be any positive value (1 = grayscale, 3 = RGB, …).
    ///
    /// Each contributor's weight is a soft cell-membership ramp on the
    /// projected radius: inside the tile's Voronoi cell the weight is `1`;
    /// across a 2-pixel-wide ramp centred on the cell boundary the weight
    /// falls linearly to `0`; tiles whose projection escapes their own
    /// patch contribute nothing. This means `k = 1` already produces seam-
    /// smoothing inside the overlap rings between tiles, and `k > 1` only
    /// changes which farther neighbours are eligible to participate
    /// (typically a no-op away from cell boundaries). Setting `k > 1`
    /// guarantees both sides of every cell boundary are reachable.
    ///
    /// `k` is clamped to the number of tiles in the rig.
    ///
    /// **NaN handling:** atlas pixels carrying `NaN` are interpreted as
    /// "no data here". The bilinear sampler skips NaN corners and
    /// renormalises over the remaining ones (so a tile only goes NaN when
    /// every corner of its bilinear footprint is). Output is gated on the
    /// *closest* tile: if the nearest neighbour to the dst direction has
    /// no data, the direction is treated as unobserved and the output is
    /// `NaN`, regardless of what farther tiles hold — this keeps each
    /// tile's data within its Voronoi cell instead of letting the few
    /// valid tiles bleed into the empty hemisphere. When the closest tile
    /// is valid, farther neighbours still participate in the k-nearest
    /// blend, with weights renormalised over the non-NaN contributors.
    /// Callers can therefore pass an atlas with `NaN` holes (e.g. from
    /// [`PerSphericalTileSourceStack::primary_consensus_atlas`])
    /// and trust that valid neighbours dominate the blend without
    /// contaminating uncovered regions.
    ///
    /// Rows are computed in parallel via rayon.
    ///
    /// # Panics
    /// Panics if `k == 0`, `channels == 0`, or `atlas.len()` does not match
    /// `atlas_size().0 * atlas_size().1 * channels`.
    ///
    /// [`PerSphericalTileSourceStack::primary_consensus_atlas`]:
    ///     crate::per_spherical_tile_source_stack::PerSphericalTileSourceStack::primary_consensus_atlas
    pub fn resample_atlas(
        &self,
        atlas: &[f32],
        channels: usize,
        dst: &CameraIntrinsics,
        rot_world_from_dst: &RotQuaternion,
        k: usize,
    ) -> Vec<f32> {
        assert!(k >= 1, "k must be >= 1");
        assert!(channels >= 1, "channels must be >= 1");
        let (atlas_w, atlas_h) = self.atlas_size();
        let expected_len = atlas_w as usize * atlas_h as usize * channels;
        assert_eq!(
            atlas.len(),
            expected_len,
            "atlas length {} does not match atlas_size {:?} * channels {}",
            atlas.len(),
            (atlas_w, atlas_h),
            channels,
        );

        let dst_w = dst.width;
        let dst_h = dst.height;
        let r_wd = rot_world_from_dst.to_rotation_matrix();
        let half = self.patch_size as f64 / 2.0;
        let f = half / self.half_fov_rad.tan();
        let (fx_t, fy_t, cx_t, cy_t) = (f, f, half, half);
        let patch_size_f = self.patch_size as f64;
        // Voronoi-cell radius in tile-pixel units. Tiles tile the sphere
        // with `half_fov_rad`, but Voronoi cells span only the
        // `measured_max_coverage_angle` core of each FOV (the ratio is
        // `1 / overlap_factor`). Inside this radius a dst direction
        // belongs unambiguously to one tile; outside it (still inside
        // the patch) we are in the overlap zone where adjacent tiles
        // share coverage.
        let cell_radius_px = half * (self.measured_max_coverage_angle / self.half_fov_rad);
        // Half-width of the cross-cell blend, in tile pixels. A 2-pixel
        // total blend zone (`-1..+1` of signed cell-boundary distance)
        // gives a tight, geometrically meaningful seam smoothing.
        let blend_half_px = 1.0_f64;
        let n_tiles = self.len();
        let k_eff = k.min(n_tiles);

        let row_len = dst_w as usize * channels;
        let data: Vec<f32> = (0..dst_h)
            .into_par_iter()
            .flat_map(|row| {
                let mut row_data = vec![0.0_f32; row_len];
                let mut sample = vec![0.0_f32; channels];
                let mut accum = vec![0.0_f32; channels];
                let v_pix = row as f64 + 0.5;
                for col in 0..dst_w {
                    let u_pix = col as f64 + 0.5;
                    let d_dst = dst.pixel_to_ray(u_pix, v_pix);
                    let world = r_wd * Vector3::new(d_dst[0], d_dst[1], d_dst[2]);

                    // KD-tree k-nearest. The query is f32 unit vectors.
                    let q = [world.x as f32, world.y as f32, world.z as f32];
                    let nn = self.direction_tree.nearest_k(&q, 1, k_eff);

                    // The closest tile (nn[0]) gates the pixel: if its
                    // sample is NaN this direction is genuinely
                    // unobserved (no Voronoi-cell owner has data) and we
                    // emit NaN regardless of what farther tiles hold.
                    // Each contributor's weight is a soft cell-membership
                    // ramp on the projection radius — full inside the
                    // tile's Voronoi cell, falling linearly to zero
                    // across `2 * blend_half_px` centred on the cell
                    // boundary, and reject if the projection escapes
                    // the tile's own patch (so we can never bilinear-
                    // sample into a neighbour's atlas slot).
                    accum.iter_mut().for_each(|v| *v = 0.0);
                    let base = col as usize * channels;
                    let mut closest_valid = false;
                    let mut valid_weight_sum = 0.0_f64;
                    for (i, &idx_u32) in nn.iter().enumerate() {
                        let tile_idx = idx_u32 as usize;
                        let basis = self.bases[tile_idx];
                        let dir = self.directions[tile_idx];
                        // tile-frame ray = R_world_from_tile.transpose() · world
                        let tx = basis[0] * world.x + basis[1] * world.y + basis[2] * world.z;
                        let ty = basis[3] * world.x + basis[4] * world.y + basis[5] * world.z;
                        let tz = dir[0] * world.x + dir[1] * world.y + dir[2] * world.z;
                        // Reject tiles where the dst direction is behind
                        // (tz <= 0) or projects outside their patch — by
                        // rig construction the closest tile never falls
                        // here, but farther neighbours often do.
                        if tz <= 1e-9 {
                            if i == 0 {
                                break;
                            }
                            continue;
                        }
                        let in_x = fx_t * (tx / tz) + cx_t;
                        let in_y = fy_t * (ty / tz) + cy_t;
                        if in_x < 0.0 || in_x >= patch_size_f || in_y < 0.0 || in_y >= patch_size_f
                        {
                            if i == 0 {
                                break;
                            }
                            continue;
                        }
                        // Soft cell-membership weight from the projection's
                        // signed distance to the cell boundary (positive
                        // inside the cell, negative in the overlap ring).
                        let dx = in_x - cx_t;
                        let dy = in_y - cy_t;
                        let r_proj = (dx * dx + dy * dy).sqrt();
                        let signed_dist = cell_radius_px - r_proj;
                        let cell_w = ((signed_dist / blend_half_px) * 0.5 + 0.5).clamp(0.0, 1.0);
                        if cell_w <= 0.0 {
                            // Beyond the blend ramp on the outside — not
                            // worth sampling. Closest tile by definition
                            // has cell_w near 1, so this branch is only
                            // hit by farther neighbours.
                            continue;
                        }
                        let (ox, oy) = self.tile_atlas_origin(tile_idx);
                        let ax = (ox as f64 + in_x) as f32;
                        let ay = (oy as f64 + in_y) as f32;
                        let valid = sample_bilinear_f32_nan_aware(
                            atlas,
                            atlas_w,
                            atlas_h,
                            channels,
                            ax,
                            ay,
                            &mut sample,
                        );
                        if i == 0 {
                            if !valid {
                                break;
                            }
                            closest_valid = true;
                        }
                        if !valid {
                            continue;
                        }
                        valid_weight_sum += cell_w;
                        let w_i = cell_w as f32;
                        for ch in 0..channels {
                            accum[ch] += w_i * sample[ch];
                        }
                    }

                    if closest_valid && valid_weight_sum > 0.0 {
                        let inv_w = (1.0 / valid_weight_sum) as f32;
                        for ch in 0..channels {
                            row_data[base + ch] = accum[ch] * inv_w;
                        }
                    } else {
                        // Closest tile had no data → direction is in an
                        // unobserved region of the rig; propagate NaN.
                        for ch in 0..channels {
                            row_data[base + ch] = f32::NAN;
                        }
                    }
                }
                row_data
            })
            .collect();
        data
    }

    /// Build a `WarpMap` with the atlas as the **source** image.
    ///
    /// For each `dst` pixel `(u, v)` we ray-trace through `dst`, rotate to
    /// world frame via `rot_world_from_dst`, look up the angularly-closest
    /// tile via the rig's KD-tree, rotate the ray into that tile's local
    /// frame via `R_tile_from_world = R_world_from_tile.transpose()`,
    /// project through `tile_camera()` to an in-tile pixel, and offset by
    /// `tile_atlas_origin(tile_idx)` to get the atlas pixel.
    pub fn warp_from_atlas_with_rotation(
        &self,
        dst: &CameraIntrinsics,
        rot_world_from_dst: &RotQuaternion,
    ) -> WarpMap {
        let dst_w = dst.width;
        let dst_h = dst.height;
        let (atlas_w, atlas_h) = self.atlas_size();
        let r_wd = rot_world_from_dst.to_rotation_matrix();
        let tile_cam = self.tile_camera();
        let atlas_w_f = atlas_w as f64;
        let atlas_h_f = atlas_h as f64;

        let row_len = 2 * dst_w as usize;
        let data: Vec<f32> = (0..dst_h)
            .into_par_iter()
            .flat_map(|row| {
                let mut row_data = vec![f32::NAN; row_len];
                let v_pix = row as f64 + 0.5;
                for col in 0..dst_w {
                    let u_pix = col as f64 + 0.5;
                    let d_dst = dst.pixel_to_ray(u_pix, v_pix);
                    let world = r_wd * Vector3::new(d_dst[0], d_dst[1], d_dst[2]);

                    // KD-tree NN as f32 unit vectors.
                    let q = [world.x as f32, world.y as f32, world.z as f32];
                    let nn = self.direction_tree.nearest(&q, 1);
                    let tile_idx = nn[0] as usize;

                    // world → tile via R_world_from_tile.transpose() —
                    // tx = e_right · world, ty = e_up · world, tz = dir · world.
                    let basis = self.bases[tile_idx];
                    let dir = self.directions[tile_idx];
                    let tx = basis[0] * world.x + basis[1] * world.y + basis[2] * world.z;
                    let ty = basis[3] * world.x + basis[4] * world.y + basis[5] * world.z;
                    let tz = dir[0] * world.x + dir[1] * world.y + dir[2] * world.z;

                    let (in_x, in_y) = match tile_cam.ray_to_pixel([tx, ty, tz]) {
                        Some(p) => p,
                        None => continue,
                    };
                    let (ox, oy) = self.tile_atlas_origin(tile_idx);
                    let ax = ox as f64 + in_x;
                    let ay = oy as f64 + in_y;
                    if ax < 0.0 || ay < 0.0 || ax >= atlas_w_f || ay >= atlas_h_f {
                        continue;
                    }

                    let idx = 2 * col as usize;
                    row_data[idx] = ax as f32;
                    row_data[idx + 1] = ay as f32;
                }
                row_data
            })
            .collect();

        WarpMap::new(dst_w, dst_h, data)
    }
}

/// Bilinear sample from a packed row-major f32 atlas at fractional pixel
/// coordinates (pixel-center-at-0.5 convention). Out-of-bounds taps replicate
/// the edge pixel.
///
/// `NaN` corners are treated as "missing data": the bilinear weights of
/// non-NaN corners are renormalised over only the contributing corners.
/// All channels share the corner-validity decision (we test NaN on channel
/// 0; producers of multi-channel atlases store NaN per-pixel across all
/// channels together).
///
/// Returns `true` and fills `out` when at least one corner is valid.
/// Returns `false` and leaves `out` unspecified when every corner is NaN.
fn sample_bilinear_f32_nan_aware(
    img: &[f32],
    width: u32,
    height: u32,
    channels: usize,
    x: f32,
    y: f32,
    out: &mut [f32],
) -> bool {
    let gx = x - 0.5;
    let gy = y - 0.5;
    let w = width as i32;
    let h = height as i32;
    let x0 = gx.floor() as i32;
    let y0 = gy.floor() as i32;
    let x1 = x0.saturating_add(1);
    let y1 = y0.saturating_add(1);
    let fx = gx - x0 as f32;
    let fy = gy - y0 as f32;
    let cx0 = x0.clamp(0, w - 1) as usize;
    let cx1 = x1.clamp(0, w - 1) as usize;
    let cy0 = y0.clamp(0, h - 1) as usize;
    let cy1 = y1.clamp(0, h - 1) as usize;
    let stride = width as usize * channels;

    let i00 = cy0 * stride + cx0 * channels;
    let i10 = cy0 * stride + cx1 * channels;
    let i01 = cy1 * stride + cx0 * channels;
    let i11 = cy1 * stride + cx1 * channels;
    let valid00 = !img[i00].is_nan();
    let valid10 = !img[i10].is_nan();
    let valid01 = !img[i01].is_nan();
    let valid11 = !img[i11].is_nan();

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let mut wsum = 0.0_f32;
    if valid00 {
        wsum += w00;
    }
    if valid10 {
        wsum += w10;
    }
    if valid01 {
        wsum += w01;
    }
    if valid11 {
        wsum += w11;
    }
    if wsum == 0.0 {
        return false;
    }
    let inv_w = 1.0 / wsum;

    for ch in 0..channels {
        let mut s = 0.0_f32;
        if valid00 {
            s += w00 * img[i00 + ch];
        }
        if valid10 {
            s += w10 * img[i10 + ch];
        }
        if valid01 {
            s += w01 * img[i01 + ch];
        }
        if valid11 {
            s += w11 * img[i11 + ch];
        }
        out[ch] = s * inv_w;
    }
    true
}

/// Compute `(e_right, e_up)` for a unit world direction.
///
/// Picks `world_up = ±Y` unless `direction` is near a Y pole
/// (`|direction.y| > POLE_GUARD_Y`), in which case it falls back to `±X` to
/// keep `cross(world_up, direction)` well-conditioned. The basis is
/// right-handed: `e_right × e_up = direction`.
fn build_basis(d: [f64; 3]) -> [f64; 6] {
    let dir = Vector3::new(d[0], d[1], d[2]);
    // Branchless guard: pick world-up = Y unless |y| is near 1.
    let near_pole = d[1].abs() > POLE_GUARD_Y;
    let world_up = if near_pole {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };

    let e_right = world_up.cross(&dir).normalize();
    let e_up = dir.cross(&e_right).normalize();

    [e_right.x, e_right.y, e_right.z, e_up.x, e_up.y, e_up.z]
}

#[cfg(test)]
mod tests;
