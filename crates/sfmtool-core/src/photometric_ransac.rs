// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Photometric refinement via per-tile RANSAC subset partition.
//!
//! See `specs/drafts/photometric-subsets-ransac.md` for the spec.
//!
//! Operates on a [`PerSphericalTileSourceStack<f32>`]. Recovers per-source
//! log-gain (mean-zero by construction) plus a primary / secondary cluster
//! partition over the per-`(tile, source)` rows. Inputs:
//!
//! - The pyramid level matching `params.target_patch_size` is read from the
//!   stack (the algorithm errors if no level matches exactly).
//! - The central `params.scoring_patch_size × params.scoring_patch_size`
//!   sub-patch is linearised by `gamma`, has saturated pixels masked out, and
//!   is the unit the RANSAC kernel scores.
//!
//! The algorithm alternates per-tile RANSAC (primary cluster) with a
//! per-image least-squares log-gain solve, until either the primary mask
//! stops changing or `max_outer_iters` is hit. After convergence one extra
//! RANSAC pass over the primary-rejected rows yields the secondary cluster.

use std::collections::HashSet;

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::per_spherical_tile_source_stack::PerSphericalTileSourceStack;

/// Tuning knobs; defaults match the spec's recommendations.
#[derive(Debug, Clone)]
pub struct RansacPhotometricParams {
    /// Residual threshold (u8 luminance units) separating inliers from
    /// outliers in the patch-L1 score.
    pub inlier_threshold: f32,
    /// Decoding gamma applied to each pixel (`linear = 255 * (p/255)^gamma`)
    /// before mean-luminance and patch-L1 reductions. Held fixed.
    pub gamma: f32,
    /// Side length of patches read from the pyramid; the algorithm picks the
    /// level whose patch side equals this exactly, otherwise errors.
    pub target_patch_size: u32,
    /// Side length of the centred sub-patch the scorer operates on.
    pub scoring_patch_size: u32,
    /// RANSAC minimal subset size `m`. Recommended default 2.
    pub subset_size: u32,
    /// Cap on candidate subsets per tile when `C(K, m) > max_subsets`.
    pub max_subsets_per_tile: u32,
    /// Tiles with fewer contributors than this are skipped.
    pub min_inliers: u32,
    /// Cap on outer alternation iterations.
    pub max_outer_iters: u32,
    /// Convergence tolerance: stop when fewer than
    /// `mask_change_tolerance * R` rows in the primary mask flip.
    pub mask_change_tolerance: f32,
    /// Per-channel u8 cutoff above which a pixel is treated as saturated;
    /// pixels at or above this value have their working validity zeroed.
    pub saturation_threshold: u8,
    /// RNG seed for reproducible subset sampling.
    pub seed: u64,
}

impl Default for RansacPhotometricParams {
    fn default() -> Self {
        Self {
            inlier_threshold: 8.0,
            gamma: 1.0,
            target_patch_size: 4,
            scoring_patch_size: 2,
            subset_size: 2,
            max_subsets_per_tile: 64,
            min_inliers: 2,
            max_outer_iters: 8,
            mask_change_tolerance: 0.05,
            saturation_threshold: 254,
            seed: 0,
        }
    }
}

/// Output of [`refine_photometric_ransac`]; arrays match spec shapes.
#[derive(Debug, Clone)]
pub struct RansacPhotometricOutput {
    /// Per-source log-gain, mean-zero by construction. Length `n_sources`.
    pub log_gain: Vec<f32>,
    /// Per-row primary-cluster flag. Length `total_contrib_rows`.
    pub primary_mask: Vec<bool>,
    /// Per-row secondary-cluster flag (largest agreeing subset within the
    /// primary's complement). Length `total_contrib_rows`.
    pub secondary_mask: Vec<bool>,
    /// Per-tile primary cluster size (0 for skipped tiles).
    pub tile_primary_count: Vec<i32>,
    /// Per-tile secondary cluster size (0 where there's no runner-up).
    pub tile_secondary_count: Vec<i32>,
    /// Median Absolute Deviation of `row_lum_corrected` over each tile's
    /// primary cluster, NaN where `tile_primary_count < min_inliers`.
    pub tile_primary_lum_mad: Vec<f32>,
    /// Same MAD over each tile's secondary cluster; NaN where
    /// `tile_secondary_count < min_inliers`.
    pub tile_secondary_lum_mad: Vec<f32>,
    /// Number of outer iterations executed.
    pub outer_iters: u32,
    /// Per-iteration `primary_mask` Hamming distance from the previous
    /// iteration. Length `outer_iters`.
    pub mask_change_history: Vec<u32>,
}

/// Errors from [`refine_photometric_ransac`].
#[derive(Debug, Clone, PartialEq)]
pub enum RansacPhotometricError {
    /// No pyramid level has patch side equal to `target_patch_size`. Caller
    /// must build a stack whose pyramid contains the requested level.
    NoMatchingLevel {
        target_patch_size: u32,
        available: Vec<u32>,
    },
    /// `scoring_patch_size` violates one of:
    /// `2 <= scoring <= target`, `scoring` even, or `(target - scoring)` even.
    InvalidScoringPatchSize { target: u32, scoring: u32 },
    /// `subset_size < 2`. Subset sampling only has meaning for `m >= 2`.
    SubsetSizeOutOfRange { subset_size: u32 },
    /// Catch-all for invalid params.
    InvalidParam(String),
}

impl std::fmt::Display for RansacPhotometricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMatchingLevel {
                target_patch_size,
                available,
            } => write!(
                f,
                "no pyramid level has patch side equal to {target_patch_size}; \
                 available level sides: {available:?}"
            ),
            Self::InvalidScoringPatchSize { target, scoring } => write!(
                f,
                "scoring_patch_size {scoring} invalid for target_patch_size {target}: \
                 must be even, 2 <= scoring <= target, and (target - scoring) even"
            ),
            Self::SubsetSizeOutOfRange { subset_size } => {
                write!(f, "subset_size {subset_size} must be at least 2")
            }
            Self::InvalidParam(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for RansacPhotometricError {}

/// Refine photometric agreement on `stack` per the spec.
///
/// Picks the pyramid level whose patch side equals
/// `params.target_patch_size` exactly; reads patches and valid masks from
/// that level; runs the alternation; returns per-source log-gain plus the
/// primary / secondary cluster partition.
///
/// `n_sources` is taken to be `max(stack.src_id()) + 1` (or 0 for an empty
/// stack). Sources that contribute no rows correspond to all-zero columns of
/// the LSQ matrix and end up with `log_gain[i] = 0` — the minimum-norm
/// answer.
pub fn refine_photometric_ransac(
    stack: &PerSphericalTileSourceStack<f32>,
    params: &RansacPhotometricParams,
) -> Result<RansacPhotometricOutput, RansacPhotometricError> {
    validate_params(params)?;

    // ── Pick the pyramid level matching target_patch_size ────────────────
    let pyramid_levels = stack.pyramid_levels() as usize;
    let mut chosen_level: Option<usize> = None;
    let mut available = Vec::with_capacity(pyramid_levels);
    for l in 0..pyramid_levels {
        let s = stack.level(l).size;
        available.push(s);
        if s == params.target_patch_size {
            chosen_level = Some(l);
        }
    }
    let level = chosen_level.ok_or(RansacPhotometricError::NoMatchingLevel {
        target_patch_size: params.target_patch_size,
        available,
    })?;

    // ── Pull flat layout out of the stack ────────────────────────────────
    let s = stack.level(level).size as usize;
    let c = stack.channels() as usize;
    let r_total = stack.total_contrib_rows();
    let n_tiles = stack.n_tiles();
    let patches = stack.level_patches(level); // [R * s * s * C]
    let valid = stack.level_valid(level); // [R * s * s], strictly {0, 1}
    let src_index = stack.src_id();
    let tile_offsets = stack.tile_offsets();
    let n_sources = src_index
        .iter()
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(0);

    refine_flat(
        patches,
        valid,
        src_index,
        tile_offsets,
        s as u32,
        c as u32,
        n_sources,
        n_tiles,
        r_total,
        params,
    )
}

/// Lower-level entry point that runs the algorithm against pre-flattened
/// row-major buffers; used by the unit tests to construct exact inputs.
///
/// `patches` is `[R * s * s * C]` and `valid` is `[R * s * s]` where
/// `R = tile_offsets[n_tiles]`.
#[allow(clippy::too_many_arguments)]
pub fn refine_photometric_ransac_flat(
    patches: &[f32],
    valid: &[u8],
    src_index: &[u32],
    tile_offsets: &[u32],
    target_patch_size: u32,
    channels: u32,
    n_sources: usize,
    n_tiles: usize,
    params: &RansacPhotometricParams,
) -> Result<RansacPhotometricOutput, RansacPhotometricError> {
    validate_params(params)?;
    if params.target_patch_size != target_patch_size {
        return Err(RansacPhotometricError::NoMatchingLevel {
            target_patch_size: params.target_patch_size,
            available: vec![target_patch_size],
        });
    }
    let r_total = if tile_offsets.is_empty() {
        0
    } else {
        tile_offsets[n_tiles] as usize
    };
    refine_flat(
        patches,
        valid,
        src_index,
        tile_offsets,
        target_patch_size,
        channels,
        n_sources,
        n_tiles,
        r_total,
        params,
    )
}

fn validate_params(params: &RansacPhotometricParams) -> Result<(), RansacPhotometricError> {
    if params.subset_size < 2 {
        return Err(RansacPhotometricError::SubsetSizeOutOfRange {
            subset_size: params.subset_size,
        });
    }
    let target = params.target_patch_size;
    let scoring = params.scoring_patch_size;
    if !target.is_power_of_two() || target < 2 {
        return Err(RansacPhotometricError::InvalidParam(format!(
            "target_patch_size {target} must be a power of two >= 2"
        )));
    }
    if scoring < 2
        || scoring > target
        || !scoring.is_multiple_of(2)
        || !(target - scoring).is_multiple_of(2)
    {
        return Err(RansacPhotometricError::InvalidScoringPatchSize { target, scoring });
    }
    if params.min_inliers < 1 {
        return Err(RansacPhotometricError::InvalidParam(
            "min_inliers must be >= 1".into(),
        ));
    }
    if params.gamma <= 0.0 || !params.gamma.is_finite() {
        return Err(RansacPhotometricError::InvalidParam(format!(
            "gamma must be a positive finite number; got {}",
            params.gamma
        )));
    }
    if params.inlier_threshold < 0.0 {
        return Err(RansacPhotometricError::InvalidParam(format!(
            "inlier_threshold must be >= 0; got {}",
            params.inlier_threshold
        )));
    }
    if !params.mask_change_tolerance.is_finite() {
        return Err(RansacPhotometricError::InvalidParam(format!(
            "mask_change_tolerance must be finite; got {}",
            params.mask_change_tolerance
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn refine_flat(
    patches: &[f32],
    valid: &[u8],
    src_index: &[u32],
    tile_offsets: &[u32],
    target_patch_size: u32,
    channels: u32,
    n_sources: usize,
    n_tiles: usize,
    r_total: usize,
    params: &RansacPhotometricParams,
) -> Result<RansacPhotometricOutput, RansacPhotometricError> {
    let s = target_patch_size as usize;
    let ss = params.scoring_patch_size as usize;
    let c = channels as usize;
    let half_diff = (s - ss) / 2;
    let pixel_count_ss = ss * ss;
    let row_patch_stride = pixel_count_ss * c;
    let row_valid_stride = pixel_count_ss;

    // Sanity-check buffer lengths.
    let expected_patches_len = r_total * s * s * c;
    let expected_valid_len = r_total * s * s;
    if patches.len() != expected_patches_len {
        return Err(RansacPhotometricError::InvalidParam(format!(
            "patches buffer length {} != expected R*s*s*C = {}",
            patches.len(),
            expected_patches_len
        )));
    }
    if valid.len() != expected_valid_len {
        return Err(RansacPhotometricError::InvalidParam(format!(
            "valid buffer length {} != expected R*s*s = {}",
            valid.len(),
            expected_valid_len
        )));
    }
    if src_index.len() != r_total {
        return Err(RansacPhotometricError::InvalidParam(format!(
            "src_index length {} != R = {}",
            src_index.len(),
            r_total
        )));
    }
    if tile_offsets.len() != n_tiles + 1 {
        return Err(RansacPhotometricError::InvalidParam(format!(
            "tile_offsets length {} != n_tiles + 1 = {}",
            tile_offsets.len(),
            n_tiles + 1
        )));
    }

    // ── Step 1: Per-row scoring patch + mean luminance (once) ────────────
    let mut row_patch_uncorrected = vec![0.0f32; r_total * row_patch_stride];
    let mut centre_v = vec![0.0f32; r_total * row_valid_stride];
    let mut row_lum_uncorrected = vec![0.0f32; r_total];
    let sat_thresh = params.saturation_threshold as f32;
    let gamma = params.gamma;
    let gamma_is_one = (gamma - 1.0).abs() < 1e-6;

    #[allow(clippy::needless_range_loop)]
    for r in 0..r_total {
        let p_in_base = r * s * s * c;
        let v_in_base = r * s * s;
        let p_out_base = r * row_patch_stride;
        let v_out_base = r * row_valid_stride;
        let mut sum = 0.0f64;
        let mut denom = 0.0f64;
        for vi in 0..ss {
            for ui in 0..ss {
                let v_in = vi + half_diff;
                let u_in = ui + half_diff;
                let v_value = valid[v_in_base + v_in * s + u_in] as f32;
                // Saturation: zero working validity iff any channel is at or
                // above the threshold.
                let mut max_chan = 0.0f32;
                for ch in 0..c {
                    let pix = patches[p_in_base + (v_in * s + u_in) * c + ch];
                    if pix > max_chan {
                        max_chan = pix;
                    }
                }
                let v_after = if max_chan >= sat_thresh { 0.0 } else { v_value };
                let v_out_idx = v_out_base + vi * ss + ui;
                centre_v[v_out_idx] = v_after;
                let v_after_f64 = v_after as f64;
                for ch in 0..c {
                    let raw = patches[p_in_base + (v_in * s + u_in) * c + ch];
                    let linearised = if gamma_is_one {
                        raw
                    } else {
                        255.0 * (raw / 255.0).powf(gamma)
                    };
                    row_patch_uncorrected[p_out_base + (vi * ss + ui) * c + ch] = linearised;
                    sum += linearised as f64 * v_after_f64;
                }
                denom += v_after_f64 * c as f64;
            }
        }
        row_lum_uncorrected[r] = (sum / denom.max(1.0)) as f32;
    }

    // Precompute row → tile lookup (avoids a binary search inside the LSQ
    // build).
    let mut row_to_tile = vec![0u32; r_total];
    for t in 0..n_tiles {
        let a = tile_offsets[t] as usize;
        let b = tile_offsets[t + 1] as usize;
        row_to_tile[a..b].fill(t as u32);
    }

    // ── Outer loop ───────────────────────────────────────────────────────
    let mut log_gain = vec![0.0f32; n_sources];
    let mut primary_mask = vec![false; r_total];
    let mut mask_change_history: Vec<u32> = Vec::with_capacity(params.max_outer_iters as usize);

    let max_outer = params.max_outer_iters.max(1) as usize;
    let mask_change_cap = (params.mask_change_tolerance * r_total as f32).round() as i64;
    let min_inliers = params.min_inliers as usize;

    let mut row_patch_corrected = vec![0.0f32; r_total * row_patch_stride];
    let mut row_lum_corrected = vec![0.0f32; r_total];

    let mut outer_iters: u32 = 0;
    for _iter_idx in 0..max_outer {
        // Step 2: apply current gains to row patches and luminances.
        apply_gains(
            &row_patch_uncorrected,
            &row_lum_uncorrected,
            src_index,
            &log_gain,
            &mut row_patch_corrected,
            &mut row_lum_corrected,
            row_patch_stride,
        );

        // Step 3: per-tile RANSAC → new primary mask.
        let new_primary_mask = run_per_tile_ransac(
            &row_patch_corrected,
            &centre_v,
            tile_offsets,
            n_tiles,
            ss,
            c,
            params.inlier_threshold,
            params.subset_size as usize,
            params.max_subsets_per_tile as usize,
            min_inliers,
            params.seed,
            r_total,
        );

        // Step 4: LSQ for log-gain.
        log_gain = solve_log_gain_lsq(
            &row_lum_uncorrected,
            src_index,
            &row_to_tile,
            &new_primary_mask,
            n_sources,
            &log_gain,
        );

        // Step 5: convergence.
        let mask_change = hamming(&new_primary_mask, &primary_mask) as i64;
        mask_change_history.push(mask_change as u32);
        outer_iters += 1;
        primary_mask = new_primary_mask;
        if mask_change == 0 {
            break;
        }
        if mask_change < mask_change_cap {
            break;
        }
    }

    // Step 6: secondary cluster — re-apply final gains, then RANSAC over
    // each tile's primary-rejected rows.
    apply_gains(
        &row_patch_uncorrected,
        &row_lum_uncorrected,
        src_index,
        &log_gain,
        &mut row_patch_corrected,
        &mut row_lum_corrected,
        row_patch_stride,
    );
    let secondary_mask = run_secondary_ransac(
        &row_patch_corrected,
        &centre_v,
        tile_offsets,
        &primary_mask,
        n_tiles,
        ss,
        c,
        params.inlier_threshold,
        params.subset_size as usize,
        params.max_subsets_per_tile as usize,
        min_inliers,
        params.seed,
        r_total,
    );

    // Per-tile counts and lum-MADs.
    let mut tile_primary_count = vec![0i32; n_tiles];
    let mut tile_secondary_count = vec![0i32; n_tiles];
    for t in 0..n_tiles {
        let a = tile_offsets[t] as usize;
        let b = tile_offsets[t + 1] as usize;
        for r in a..b {
            if primary_mask[r] {
                tile_primary_count[t] += 1;
            }
            if secondary_mask[r] {
                tile_secondary_count[t] += 1;
            }
        }
    }
    let tile_primary_lum_mad = compute_tile_lum_mad(
        &row_lum_corrected,
        tile_offsets,
        &primary_mask,
        n_tiles,
        min_inliers,
    );
    let tile_secondary_lum_mad = compute_tile_lum_mad(
        &row_lum_corrected,
        tile_offsets,
        &secondary_mask,
        n_tiles,
        min_inliers,
    );

    Ok(RansacPhotometricOutput {
        log_gain,
        primary_mask,
        secondary_mask,
        tile_primary_count,
        tile_secondary_count,
        tile_primary_lum_mad,
        tile_secondary_lum_mad,
        outer_iters,
        mask_change_history,
    })
}

fn apply_gains(
    row_patch_uncorrected: &[f32],
    row_lum_uncorrected: &[f32],
    src_index: &[u32],
    log_gain: &[f32],
    row_patch_corrected: &mut [f32],
    row_lum_corrected: &mut [f32],
    row_patch_stride: usize,
) {
    let r_total = row_lum_uncorrected.len();
    for r in 0..r_total {
        let g = (log_gain[src_index[r] as usize] as f64).exp() as f32;
        row_lum_corrected[r] = row_lum_uncorrected[r] * g;
        let base = r * row_patch_stride;
        for k in 0..row_patch_stride {
            row_patch_corrected[base + k] = row_patch_uncorrected[base + k] * g;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_per_tile_ransac(
    row_patch_corrected: &[f32],
    centre_v: &[f32],
    tile_offsets: &[u32],
    n_tiles: usize,
    ss: usize,
    c: usize,
    threshold: f32,
    subset_size: usize,
    max_subsets: usize,
    min_inliers: usize,
    seed: u64,
    r_total: usize,
) -> Vec<bool> {
    let row_patch_stride = ss * ss * c;
    let row_valid_stride = ss * ss;
    let per_tile: Vec<Vec<bool>> = (0..n_tiles)
        .into_par_iter()
        .map(|t| {
            let a = tile_offsets[t] as usize;
            let b = tile_offsets[t + 1] as usize;
            let k = b - a;
            if k < min_inliers {
                return vec![false; k];
            }
            let row_patch = &row_patch_corrected[a * row_patch_stride..b * row_patch_stride];
            let valid_pp = &centre_v[a * row_valid_stride..b * row_valid_stride];
            let mut rng = StdRng::seed_from_u64(seed.rotate_left(32) ^ (t as u64));
            ransac_cluster_for_tile(
                row_patch,
                valid_pp,
                k,
                ss,
                c,
                threshold,
                subset_size,
                max_subsets,
                &mut rng,
            )
        })
        .collect();
    let mut out = Vec::with_capacity(r_total);
    for v in per_tile {
        out.extend(v);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_secondary_ransac(
    row_patch_corrected: &[f32],
    centre_v: &[f32],
    tile_offsets: &[u32],
    primary_mask: &[bool],
    n_tiles: usize,
    ss: usize,
    c: usize,
    threshold: f32,
    subset_size: usize,
    max_subsets: usize,
    min_inliers: usize,
    seed: u64,
    r_total: usize,
) -> Vec<bool> {
    let row_patch_stride = ss * ss * c;
    let row_valid_stride = ss * ss;
    let per_tile: Vec<Vec<(usize, bool)>> = (0..n_tiles)
        .into_par_iter()
        .map(|t| {
            let a = tile_offsets[t] as usize;
            let b = tile_offsets[t + 1] as usize;
            let mut rejected_local: Vec<usize> = Vec::with_capacity(b - a);
            for (i, r) in (a..b).enumerate() {
                if !primary_mask[r] {
                    rejected_local.push(i);
                }
            }
            if rejected_local.len() < min_inliers {
                return Vec::new();
            }
            // Slice out the primary-rejected sub-stack.
            let mut sub_patches: Vec<f32> =
                Vec::with_capacity(rejected_local.len() * row_patch_stride);
            let mut sub_valid: Vec<f32> =
                Vec::with_capacity(rejected_local.len() * row_valid_stride);
            for &local_i in &rejected_local {
                let p_base = (a + local_i) * row_patch_stride;
                let v_base = (a + local_i) * row_valid_stride;
                sub_patches
                    .extend_from_slice(&row_patch_corrected[p_base..p_base + row_patch_stride]);
                sub_valid.extend_from_slice(&centre_v[v_base..v_base + row_valid_stride]);
            }
            let mut rng = StdRng::seed_from_u64(
                seed.rotate_left(32) ^ ((t as u64).wrapping_add(0xA5A5_A5A5_A5A5_A5A5)),
            );
            let sub_mask = ransac_cluster_for_tile(
                &sub_patches,
                &sub_valid,
                rejected_local.len(),
                ss,
                c,
                threshold,
                subset_size,
                max_subsets,
                &mut rng,
            );
            let mut out = Vec::with_capacity(rejected_local.len());
            for (i, &m) in rejected_local.iter().zip(&sub_mask) {
                out.push((a + i, m));
            }
            out
        })
        .collect();
    let mut secondary_mask = vec![false; r_total];
    for v in per_tile {
        for (r, m) in v {
            if m {
                secondary_mask[r] = true;
            }
        }
    }
    secondary_mask
}

#[allow(clippy::too_many_arguments)]
fn ransac_cluster_for_tile(
    row_patch: &[f32], // [k * ss * ss * c]
    valid_pp: &[f32],  // [k * ss * ss]
    k: usize,
    ss: usize,
    c: usize,
    threshold: f32,
    subset_size: usize,
    max_subsets: usize,
    rng: &mut StdRng,
) -> Vec<bool> {
    if k == 0 {
        return Vec::new();
    }
    let pixel_count = ss * ss;
    let row_patch_stride = pixel_count * c;

    // Below-sampling fallback: per-pixel-channel median + threshold.
    if k <= subset_size {
        let med = per_pixel_median(row_patch, k, ss, c);
        let mut out = vec![false; k];
        for i in 0..k {
            let score = patch_l1_score(
                &row_patch[i * row_patch_stride..(i + 1) * row_patch_stride],
                &valid_pp[i * pixel_count..(i + 1) * pixel_count],
                &med,
                ss,
                c,
            );
            out[i] = score <= threshold;
        }
        return out;
    }

    // Build the candidate subset list — exhaustive when feasible.
    let n_combos = comb(k, subset_size);
    let candidates: Vec<Vec<usize>> = if n_combos <= max_subsets {
        enumerate_combinations(k, subset_size)
    } else {
        sample_combinations(rng, k, subset_size, max_subsets)
    };

    let mut best_count: i64 = -1;
    let mut best_score = f32::INFINITY;
    let mut best_inliers: Option<Vec<bool>> = None;

    let mut hyp = vec![0.0f32; row_patch_stride];
    let mut scores = vec![0.0f32; k];
    let mut inliers = vec![false; k];

    for idx_set in candidates {
        consensus_patch(row_patch, valid_pp, &idx_set, ss, c, &mut hyp);
        for i in 0..k {
            scores[i] = patch_l1_score(
                &row_patch[i * row_patch_stride..(i + 1) * row_patch_stride],
                &valid_pp[i * pixel_count..(i + 1) * pixel_count],
                &hyp,
                ss,
                c,
            );
            inliers[i] = scores[i] <= threshold;
        }
        let n_in = inliers.iter().filter(|&&x| x).count() as i64;
        let mean_in_score = if n_in > 0 {
            let mut s_sum = 0.0f64;
            for i in 0..k {
                if inliers[i] {
                    s_sum += scores[i] as f64;
                }
            }
            (s_sum / n_in as f64) as f32
        } else {
            f32::INFINITY
        };
        if n_in > best_count {
            best_count = n_in;
            best_score = mean_in_score;
            best_inliers = Some(inliers.clone());
        } else if n_in == best_count && n_in > 0 && mean_in_score < best_score {
            best_score = mean_in_score;
            best_inliers = Some(inliers.clone());
        }
    }
    best_inliers.unwrap_or_else(|| vec![false; k])
}

/// Per-pixel-channel median across `k` rows (no validity weighting; matches
/// the spec's median-fallback definition).
fn per_pixel_median(row_patch: &[f32], k: usize, ss: usize, c: usize) -> Vec<f32> {
    let pixel_count = ss * ss * c;
    let mut out = vec![0.0f32; pixel_count];
    let mut buf: Vec<f32> = vec![0.0; k];
    for p in 0..pixel_count {
        for i in 0..k {
            buf[i] = row_patch[i * pixel_count + p];
        }
        // Total order on f32 (no NaN: input is finite from gain * bounded
        // pixel value).
        buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out[p] = if k % 2 == 1 {
            buf[k / 2]
        } else {
            0.5 * (buf[k / 2 - 1] + buf[k / 2])
        };
    }
    out
}

/// Validity-weighted per-pixel-per-channel mean over the indexed subset of
/// rows, written into `out`. Per-pixel `Σ_i row[i,p,c] * v[i,p] / max(Σ_i
/// v[i,p], 1e-3)`.
fn consensus_patch(
    row_patch: &[f32],
    valid_pp: &[f32],
    idx_set: &[usize],
    ss: usize,
    c: usize,
    out: &mut [f32],
) {
    let pixel_count = ss * ss;
    for p in 0..pixel_count {
        let mut denom = 0.0f64;
        for &i in idx_set {
            denom += valid_pp[i * pixel_count + p] as f64;
        }
        let denom = denom.max(1e-3);
        for ch in 0..c {
            let mut num = 0.0f64;
            for &i in idx_set {
                num += row_patch[i * pixel_count * c + p * c + ch] as f64
                    * valid_pp[i * pixel_count + p] as f64;
            }
            out[p * c + ch] = (num / denom) as f32;
        }
    }
}

/// L1-mean residual of one row against `hyp`, validity-weighted.
///
/// Returns `f32::INFINITY` when the row's working validity is identically
/// zero — a fully-masked row carries no information, so it cannot
/// "agree" with any consensus hypothesis. Without this guard the spec's
/// `num / max(denom, 1.0)` formula returns 0 for such rows (because
/// `num` is also 0), which would make every fully-masked row a
/// gain-collapsing inlier in every tile.
fn patch_l1_score(row: &[f32], valid_row: &[f32], hyp: &[f32], ss: usize, c: usize) -> f32 {
    let pixel_count = ss * ss;
    let mut num = 0.0f64;
    let mut denom = 0.0f64;
    for p in 0..pixel_count {
        let v = valid_row[p] as f64;
        denom += v;
        let mut chan_sum = 0.0f64;
        for ch in 0..c {
            chan_sum += (row[p * c + ch] - hyp[p * c + ch]).abs() as f64;
        }
        num += chan_sum * v;
    }
    if denom <= 0.0 {
        return f32::INFINITY;
    }
    (num / (denom * c as f64)) as f32
}

fn enumerate_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut buf = Vec::with_capacity(k);
    fn rec(n: usize, k: usize, start: usize, buf: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if buf.len() == k {
            out.push(buf.clone());
            return;
        }
        let remaining = k - buf.len();
        let upper = n.saturating_sub(remaining);
        for i in start..=upper {
            buf.push(i);
            rec(n, k, i + 1, buf, out);
            buf.pop();
        }
    }
    rec(n, k, 0, &mut buf, &mut out);
    out
}

fn sample_combinations(rng: &mut StdRng, n: usize, k: usize, count: usize) -> Vec<Vec<usize>> {
    let mut seen: HashSet<Vec<usize>> = HashSet::with_capacity(count);
    let mut out: Vec<Vec<usize>> = Vec::with_capacity(count);
    let max_attempts = count.saturating_mul(8).max(64);
    let mut attempts = 0usize;
    while out.len() < count && attempts < max_attempts {
        attempts += 1;
        let mut idx: Vec<usize> = rand::seq::index::sample(rng, n, k).into_vec();
        idx.sort_unstable();
        if seen.insert(idx.clone()) {
            out.push(idx);
        }
    }
    out
}

fn comb(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: u128 = 1;
    for i in 0..k {
        result = result * (n - i) as u128 / (i as u128 + 1);
    }
    if result > usize::MAX as u128 {
        usize::MAX
    } else {
        result as usize
    }
}

fn solve_log_gain_lsq(
    row_lum_uncorrected: &[f32],
    src_index: &[u32],
    row_to_tile: &[u32],
    primary_mask: &[bool],
    n_sources: usize,
    current_log_gain: &[f32],
) -> Vec<f32> {
    let r_total = row_lum_uncorrected.len();
    if n_sources == 0 {
        return Vec::new();
    }
    let log_lum: Vec<f64> = row_lum_uncorrected
        .iter()
        .map(|&v| (v as f64).max(1e-3).ln())
        .collect();

    let primary_rows: Vec<usize> = (0..r_total).filter(|&r| primary_mask[r]).collect();
    if primary_rows.is_empty() {
        return current_log_gain.to_vec();
    }

    // Per-tile primary aggregates.
    let n_tiles = row_to_tile
        .iter()
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(0);
    let mut tile_primary_srcs: Vec<Vec<u32>> = vec![Vec::new(); n_tiles];
    let mut tile_primary_log_lum_sum: Vec<f64> = vec![0.0; n_tiles];
    let mut tile_primary_count: Vec<usize> = vec![0; n_tiles];
    for &r in &primary_rows {
        let t = row_to_tile[r] as usize;
        tile_primary_srcs[t].push(src_index[r]);
        tile_primary_log_lum_sum[t] += log_lum[r];
        tile_primary_count[t] += 1;
    }
    let tile_primary_log_lum_mean: Vec<f64> = (0..n_tiles)
        .map(|t| {
            if tile_primary_count[t] > 0 {
                tile_primary_log_lum_sum[t] / tile_primary_count[t] as f64
            } else {
                0.0
            }
        })
        .collect();

    let k_rows = primary_rows.len();
    let mut a = DMatrix::<f64>::zeros(k_rows, n_sources);
    let mut b = DVector::<f64>::zeros(k_rows);
    for (k, &r) in primary_rows.iter().enumerate() {
        let t = row_to_tile[r] as usize;
        let n_in_t = tile_primary_count[t];
        if n_in_t < 2 {
            continue;
        }
        let s = src_index[r] as usize;
        a[(k, s)] += 1.0;
        let inv_n = 1.0 / n_in_t as f64;
        for &s2 in &tile_primary_srcs[t] {
            a[(k, s2 as usize)] -= inv_n;
        }
        b[k] = tile_primary_log_lum_mean[t] - log_lum[r];
    }

    let svd = nalgebra::SVD::new(a, true, true);
    let x: DVector<f64> = match svd.solve(&b, 1e-9) {
        Ok(v) => v,
        Err(_) => DVector::zeros(n_sources),
    };
    let mean = x.iter().sum::<f64>() / n_sources as f64;
    let centred: Vec<f32> = x.iter().map(|&v| (v - mean) as f32).collect();
    centred
}

fn hamming(a: &[bool], b: &[bool]) -> u32 {
    a.iter().zip(b).filter(|(x, y)| x != y).count() as u32
}

fn compute_tile_lum_mad(
    row_lum_corrected: &[f32],
    tile_offsets: &[u32],
    mask: &[bool],
    n_tiles: usize,
    min_inliers: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; n_tiles];
    for t in 0..n_tiles {
        let a = tile_offsets[t] as usize;
        let b = tile_offsets[t + 1] as usize;
        let mut vals: Vec<f32> = Vec::new();
        for r in a..b {
            if mask[r] {
                vals.push(row_lum_corrected[r]);
            }
        }
        if vals.len() >= min_inliers {
            out[t] = mad(&vals);
        }
    }
    out
}

fn mad(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        0.5 * (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2])
    };
    let mut deviations: Vec<f32> = sorted.iter().map(|&v| (v - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if deviations.len() % 2 == 1 {
        deviations[deviations.len() / 2]
    } else {
        0.5 * (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2])
    }
}

#[cfg(test)]
mod tests;
