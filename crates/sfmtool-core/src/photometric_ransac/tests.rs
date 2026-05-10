// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`refine_photometric_ransac`] / `_flat`.
//!
//! The flat-array entry point lets these tests construct exact patch
//! tensors and tile/source assignments without going through a
//! `PerSphericalTileSourceStack`. Tests track the validation plan in
//! `specs/drafts/photometric-subsets-ransac.md` (items 1–9 and 14–18).

// Indexed loops mirror the indexed math in the synthetic fixtures
// (`target_l[t]`, `truth[i]`, …); converting them to iterator form would
// obscure the structure of the tests.
#![allow(clippy::needless_range_loop)]

use super::*;

// ── Synthesis helpers ────────────────────────────────────────────────────

/// Builder for a flat synthetic stack: every tile has the same set of
/// contributors (one row per source). Patch pixels and validity start at
/// the test's chosen defaults and are mutated through `set_*` methods.
struct SyntheticStack {
    patches: Vec<f32>,
    valid: Vec<u8>,
    src_index: Vec<u32>,
    tile_offsets: Vec<u32>,
    n_tiles: usize,
    n_sources: usize,
    target_size: u32,
    channels: u32,
}

impl SyntheticStack {
    fn full(n_tiles: usize, n_sources: usize, target_size: u32, channels: u32) -> Self {
        let r = n_tiles * n_sources;
        let s = target_size as usize;
        let c = channels as usize;
        let tile_offsets: Vec<u32> = (0..=n_tiles).map(|t| (t * n_sources) as u32).collect();
        let src_index: Vec<u32> = (0..r).map(|r| (r % n_sources) as u32).collect();
        Self {
            patches: vec![0.0; r * s * s * c],
            valid: vec![1; r * s * s],
            src_index,
            tile_offsets,
            n_tiles,
            n_sources,
            target_size,
            channels,
        }
    }

    fn row_of(&self, tile: usize, src: usize) -> usize {
        let a = self.tile_offsets[tile] as usize;
        let b = self.tile_offsets[tile + 1] as usize;
        for r in a..b {
            if self.src_index[r] as usize == src {
                return r;
            }
        }
        panic!("source {src} not in tile {tile}");
    }

    /// Set every pixel/channel of the (tile, src) row's level patch to
    /// `value`.
    fn set_uniform(&mut self, tile: usize, src: usize, value: f32) {
        let r = self.row_of(tile, src);
        let s = self.target_size as usize;
        let c = self.channels as usize;
        let base = r * s * s * c;
        for k in 0..(s * s * c) {
            self.patches[base + k] = value;
        }
    }

    /// Write a `2 × 2` central pattern (in scoring coordinates) into the
    /// (tile, src) row's level patch. Surrounding pixels are filled with
    /// `0.0` because they fall outside the central crop.
    fn set_central_2x2(&mut self, tile: usize, src: usize, pattern: [[f32; 2]; 2]) {
        assert_eq!(self.target_size, 4);
        let r = self.row_of(tile, src);
        let s = self.target_size as usize;
        let c = self.channels as usize;
        let half_diff = (s - 2) / 2;
        let base = r * s * s * c;
        for k in 0..(s * s * c) {
            self.patches[base + k] = 0.0;
        }
        for i in 0..2 {
            for j in 0..2 {
                let v = pattern[i][j];
                let row = i + half_diff;
                let col = j + half_diff;
                for ch in 0..c {
                    self.patches[base + (row * s + col) * c + ch] = v;
                }
            }
        }
    }

    fn run(&self, params: &RansacPhotometricParams) -> RansacPhotometricOutput {
        super::refine_photometric_ransac_flat(
            &self.patches,
            &self.valid,
            &self.src_index,
            &self.tile_offsets,
            self.target_size,
            self.channels,
            self.n_sources,
            self.n_tiles,
            params,
        )
        .expect("flat run succeeded")
    }
}

fn loose_threshold_params() -> RansacPhotometricParams {
    RansacPhotometricParams {
        // Loose enough to admit gain-only disagreements among synthetic
        // sources differing by ~10 % luminance on the first iteration.
        inlier_threshold: 16.0,
        ..Default::default()
    }
}

// ── 1. Single-cluster recovery ───────────────────────────────────────────

#[test]
fn single_cluster_recovery_with_known_gains() {
    let truth = [0.0_f32, 0.1, -0.05, 0.02];
    // After mean-zero recentering, what the algorithm should recover:
    let mean = truth.iter().sum::<f32>() / truth.len() as f32;
    let recovered_truth: [f32; 4] = std::array::from_fn(|i| truth[i] - mean);

    let n_tiles = 5;
    let n_sources = 4;
    let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 3);
    let target_l = [50.0_f32, 80.0, 120.0, 150.0, 180.0];
    for t in 0..n_tiles {
        for i in 0..n_sources {
            // Source i observed L_target * exp(-truth[i]); the algorithm's
            // gain * uncorrected reproduces L_target.
            let l_obs = target_l[t] * (-truth[i]).exp();
            stack.set_uniform(t, i, l_obs);
        }
    }

    let out = stack.run(&loose_threshold_params());

    // Mean-zero check: |mean(log_gain)| < 1e-6 (also test 5 below).
    let mean_lg = out.log_gain.iter().sum::<f32>() / out.log_gain.len() as f32;
    assert!(mean_lg.abs() < 1e-6, "mean log_gain = {mean_lg}");

    // Per-source within 0.01.
    for i in 0..n_sources {
        let err = (out.log_gain[i] - recovered_truth[i]).abs();
        assert!(
            err < 0.01,
            "source {i}: recovered {} vs truth {} (err {err})",
            out.log_gain[i],
            recovered_truth[i]
        );
    }

    // All rows in primary cluster, none in secondary.
    assert!(
        out.primary_mask.iter().all(|&b| b),
        "primary cluster should contain every row"
    );
    assert!(
        out.secondary_mask.iter().all(|&b| !b),
        "secondary cluster should be empty"
    );

    // Spec: "one outer iteration suffices" — but note that the strict
    // mask-stable convergence is checked between consecutive iterations,
    // so even if iteration 1 produces the perfect mask, iteration 2 sees
    // mask_change == 0 and breaks. Allow up to 2.
    assert!(
        out.outer_iters <= 2,
        "should converge in <=2 iterations, got {}",
        out.outer_iters
    );
}

// ── 2. Two-cluster partition ─────────────────────────────────────────────

#[test]
fn two_cluster_partition_recovers_both_groups() {
    let n_tiles = 3;
    let n_sources = 6;
    let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 3);
    let target_l = [60.0_f32, 100.0, 140.0];
    // Sources 0..4 see L_t; sources 4..6 see L_t + 50.
    for t in 0..n_tiles {
        for i in 0..4 {
            stack.set_uniform(t, i, target_l[t]);
        }
        for i in 4..6 {
            stack.set_uniform(t, i, target_l[t] + 50.0);
        }
    }

    let params = RansacPhotometricParams {
        inlier_threshold: 8.0,
        ..Default::default()
    };
    let out = stack.run(&params);

    for t in 0..n_tiles {
        let a = t * n_sources;
        for i in 0..4 {
            assert!(out.primary_mask[a + i], "tile {t}, src {i}: primary");
            assert!(
                !out.secondary_mask[a + i],
                "tile {t}, src {i}: not secondary"
            );
        }
        for i in 4..6 {
            assert!(!out.primary_mask[a + i], "tile {t}, src {i}: not primary");
            assert!(out.secondary_mask[a + i], "tile {t}, src {i}: secondary");
        }
    }

    // Recovered gains should be ~zero on the primary subset (sources 0..4
    // all see the same L_t, so no per-source gain to fit). Sources 4..5,
    // which never enter the primary, are unidentifiable — pinned to the
    // mean by the SVD min-norm + recentering, which leaves them at 0.
    for i in 0..n_sources {
        assert!(
            out.log_gain[i].abs() < 0.05,
            "log_gain[{i}] = {} unexpectedly large",
            out.log_gain[i]
        );
    }
}

// ── 3. Below-threshold tile (K ≤ m) — symmetric rejection ───────────────

#[test]
fn k_le_m_uses_median_fallback_with_symmetric_rejection() {
    // Two contributors per tile (m=2) — exactly at the K==m boundary.
    let n_tiles = 1;
    let n_sources = 2;
    let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 1);
    // Difference > 2 * threshold → both rows equidistant from median →
    // both rejected.
    stack.set_uniform(0, 0, 100.0);
    stack.set_uniform(0, 1, 130.0); // diff 30 > 2*8 = 16
    let params = RansacPhotometricParams {
        inlier_threshold: 8.0,
        ..Default::default()
    };
    let out = stack.run(&params);
    assert!(
        !out.primary_mask[0] && !out.primary_mask[1],
        "both rows should be rejected (diff > 2*threshold)"
    );
    assert_eq!(out.tile_primary_count[0], 0);
    // Secondary needs >= min_inliers post-rejection rows; both got
    // rejected, so secondary should be empty too.
    assert!(!out.secondary_mask[0] && !out.secondary_mask[1]);
}

#[test]
fn k_le_m_keeps_close_pair() {
    let n_tiles = 1;
    let n_sources = 2;
    let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 1);
    // Difference < 2 * threshold → both rows within `threshold` of median →
    // both inliers.
    stack.set_uniform(0, 0, 100.0);
    stack.set_uniform(0, 1, 105.0); // diff 5 < 2*8 = 16
    let params = RansacPhotometricParams {
        inlier_threshold: 8.0,
        ..Default::default()
    };
    let out = stack.run(&params);
    assert!(out.primary_mask[0] && out.primary_mask[1]);
    assert_eq!(out.tile_primary_count[0], 2);
}

// ── 4. Empty / single-contributor tiles ─────────────────────────────────

#[test]
fn empty_and_single_contributor_tiles_are_skipped() {
    // Two tiles: tile 0 has 0 contributors; tile 1 has 1 contributor;
    // tile 2 has 3 contributors so the LSQ has at least one row to act on.
    let n_sources = 3;
    let n_tiles = 3;
    let s = 4_u32;
    let c = 1_u32;
    let s_us = s as usize;
    let tile_offsets = vec![0u32, 0, 1, 4];
    let src_index = vec![0u32, 0, 1, 2];
    let r_total = 4;
    let mut patches = vec![0.0_f32; r_total * s_us * s_us * c as usize];
    let valid = vec![1u8; r_total * s_us * s_us];
    // Tile 1's single row.
    for k in 0..(s_us * s_us) {
        patches[k] = 100.0;
    }
    // Tile 2's three rows — agreeing.
    for r in 1..4 {
        let base = r * s_us * s_us;
        for k in 0..(s_us * s_us) {
            patches[base + k] = 120.0;
        }
    }
    let params = RansacPhotometricParams::default();
    let out = super::refine_photometric_ransac_flat(
        &patches,
        &valid,
        &src_index,
        &tile_offsets,
        s,
        c,
        n_sources,
        n_tiles,
        &params,
    )
    .expect("ok");

    // Tile 0 (empty): nothing to assert beyond no crash.
    assert_eq!(out.tile_primary_count[0], 0);
    assert_eq!(out.tile_secondary_count[0], 0);

    // Tile 1 (single): skipped — primary/secondary masks false.
    assert!(!out.primary_mask[0]);
    assert!(!out.secondary_mask[0]);
    assert_eq!(out.tile_primary_count[1], 0);

    // Tile 2 (three contributors): not skipped.
    assert_eq!(out.tile_primary_count[2], 3);
}

// ── 5. Mean-zero output ─────────────────────────────────────────────────

#[test]
fn log_gain_is_mean_zero() {
    // Re-uses the test 1 setup; assert via the explicit invariant.
    let truth = [0.0_f32, 0.1, -0.05, 0.02];
    let n_tiles = 5;
    let n_sources = 4;
    let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 3);
    let target_l = [50.0_f32, 80.0, 120.0, 150.0, 180.0];
    for t in 0..n_tiles {
        for i in 0..n_sources {
            let l_obs = target_l[t] * (-truth[i]).exp();
            stack.set_uniform(t, i, l_obs);
        }
    }
    let out = stack.run(&loose_threshold_params());
    let m = out.log_gain.iter().sum::<f32>() / out.log_gain.len() as f32;
    assert!(m.abs() < 1e-6, "mean = {m}");
}

// ── 6. Shift invariance ─────────────────────────────────────────────────

#[test]
fn constant_shift_in_truth_gains_yields_identical_recovery() {
    let truth = [0.0_f32, 0.1, -0.05, 0.02];
    let n_tiles = 4;
    let n_sources = 4;
    let target_l = [60.0_f32, 90.0, 120.0, 150.0];
    let build = |shift: f32| {
        let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 3);
        for t in 0..n_tiles {
            for i in 0..n_sources {
                let g_i = truth[i] + shift;
                let l_obs = target_l[t] * (-g_i).exp();
                stack.set_uniform(t, i, l_obs);
            }
        }
        stack.run(&loose_threshold_params())
    };
    let a = build(0.0);
    let b = build(0.5);
    for i in 0..n_sources {
        let err = (a.log_gain[i] - b.log_gain[i]).abs();
        assert!(
            err < 1e-5,
            "shift-invariance violated at source {i}: {} vs {} (err {err})",
            a.log_gain[i],
            b.log_gain[i]
        );
    }
}

// ── 7. Subset-enumeration boundary ──────────────────────────────────────

#[test]
fn subset_enumeration_boundary_seed_independence() {
    // Build a single tile with K contributors. Vary K across {11, 12, 13},
    // and across {seed_0..seed_7}, comparing primary masks.
    fn build_and_run(k: usize, seed: u64) -> Vec<bool> {
        let mut stack = SyntheticStack::full(1, k, 4, 1);
        // Mostly-agreeing rows with small jitter; one row well outside the
        // threshold so the partition has a real choice to make.
        for i in 0..k {
            let v = if i < k - 1 {
                100.0 + 0.3 * (i as f32 - (k as f32) / 2.0)
            } else {
                160.0
            };
            stack.set_uniform(0, i, v);
        }
        let params = RansacPhotometricParams {
            seed,
            ..Default::default()
        };
        stack.run(&params).primary_mask
    }

    // K=11 with m=2 → C(11,2)=55 ≤ 64, exhaustive: byte-identical across
    // all seeds.
    let baseline_11 = build_and_run(11, 0);
    for seed in 1..8 {
        let m = build_and_run(11, seed);
        assert_eq!(
            baseline_11, m,
            "K=11 should be seed-independent (exhaustive)"
        );
    }

    // K=12 and K=13 sample (C(K,2) > 64). Hamming distance across seeds
    // should be small (< 1 % of K). Note the spec quotes a 1 % bound that
    // is generous; on these clean synthetic stacks the actual flutter is
    // typically zero.
    for k in [12_usize, 13] {
        let baseline = build_and_run(k, 0);
        for seed in 1..8 {
            let m = build_and_run(k, seed);
            let h: usize = baseline.iter().zip(&m).filter(|(a, b)| a != b).count();
            let bound = ((k as f32) * 0.01).ceil() as usize;
            assert!(
                h <= bound.max(1),
                "K={k}, seed {seed}: hamming {h} exceeds bound {bound}",
            );
        }
    }
}

// ── 8. Secondary cluster recovery ───────────────────────────────────────

#[test]
fn secondary_cluster_recovers_runner_up_group() {
    let n_tiles = 1;
    let n_sources = 8;
    let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 1);
    // 5 rows at L=100, 3 rows at L=160 (well above threshold).
    for i in 0..5 {
        stack.set_uniform(0, i, 100.0);
    }
    for i in 5..8 {
        stack.set_uniform(0, i, 160.0);
    }

    let params = RansacPhotometricParams {
        inlier_threshold: 8.0,
        ..Default::default()
    };
    let out = stack.run(&params);
    for i in 0..5 {
        assert!(
            out.primary_mask[i] && !out.secondary_mask[i],
            "row {i}: primary"
        );
    }
    for i in 5..8 {
        assert!(
            !out.primary_mask[i] && out.secondary_mask[i],
            "row {i}: secondary"
        );
    }
    assert_eq!(out.tile_primary_count[0], 5);
    assert_eq!(out.tile_secondary_count[0], 3);
}

#[test]
fn secondary_skipped_when_runner_up_too_small() {
    // 5 rows at L=100, 1 row at L=160. Secondary should skip (1 < min_inliers=2).
    let n_tiles = 1;
    let n_sources = 6;
    let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 1);
    for i in 0..5 {
        stack.set_uniform(0, i, 100.0);
    }
    stack.set_uniform(0, 5, 160.0);
    let params = RansacPhotometricParams::default();
    let out = stack.run(&params);
    assert_eq!(out.tile_secondary_count[0], 0);
    assert!(out.secondary_mask.iter().all(|&b| !b));
}

// ── 9. Patch-aware scoring distinguishes spatial pattern ────────────────

#[test]
fn patch_aware_scoring_separates_checker_from_constant() {
    let n_tiles = 1;
    let n_sources = 4;
    let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 1);
    // Two checker rows (identical pattern).
    stack.set_central_2x2(0, 0, [[100.0, 200.0], [200.0, 100.0]]);
    stack.set_central_2x2(0, 1, [[100.0, 200.0], [200.0, 100.0]]);
    // Two constant patches at the same scalar mean (~150) but differing
    // by 4 — within threshold for scalar-mean scoring, but the patch-L1
    // residual against a checker hypothesis is well above 8.0.
    stack.set_central_2x2(0, 2, [[150.0, 150.0], [150.0, 150.0]]);
    stack.set_central_2x2(0, 3, [[154.0, 154.0], [154.0, 154.0]]);
    let params = RansacPhotometricParams {
        inlier_threshold: 8.0,
        ..Default::default()
    };
    let out = stack.run(&params);

    // Primary should be the two checker rows (mean-inlier-residual 0
    // beats the constant pair's mean-inlier-residual 2 on tie-break).
    assert!(out.primary_mask[0] && out.primary_mask[1]);
    assert!(!out.primary_mask[2] && !out.primary_mask[3]);
    // Secondary should be the two constant rows.
    assert!(!out.secondary_mask[0] && !out.secondary_mask[1]);
    assert!(out.secondary_mask[2] && out.secondary_mask[3]);
}

// ── 14. Permutation invariance of source order ──────────────────────────

#[test]
fn permuting_source_order_permutes_log_gain() {
    // Build two stacks: identical except sources 0 and 2 are swapped.
    let truth = [0.0_f32, 0.1, -0.05, 0.02];
    let n_tiles = 4;
    let n_sources = 4;
    let target_l = [60.0_f32, 100.0, 130.0, 150.0];
    let build = |perm: &[usize; 4]| {
        let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 3);
        for t in 0..n_tiles {
            for slot in 0..n_sources {
                // After perm, slot `slot` corresponds to original source perm[slot].
                let orig = perm[slot];
                let l_obs = target_l[t] * (-truth[orig]).exp();
                stack.set_uniform(t, slot, l_obs);
            }
        }
        stack.run(&loose_threshold_params())
    };
    let identity: [usize; 4] = [0, 1, 2, 3];
    let permuted: [usize; 4] = [2, 1, 0, 3];
    let a = build(&identity);
    let b = build(&permuted);
    for slot in 0..n_sources {
        let orig = permuted[slot];
        let err = (b.log_gain[slot] - a.log_gain[orig]).abs();
        assert!(
            err < 1e-4,
            "permutation invariance: slot {slot} (orig {orig}) {err}",
        );
    }
}

// ── 15. Permutation invariance of row order within a tile ───────────────

#[test]
fn permuting_rows_within_tile_permutes_masks() {
    // For a single tile, shuffle the row order and verify the cluster
    // memberships rename consistently.
    let n_tiles = 1;
    let n_sources = 6;
    let mut stack_a = SyntheticStack::full(n_tiles, n_sources, 4, 1);
    for i in 0..4 {
        stack_a.set_uniform(0, i, 100.0);
    }
    for i in 4..6 {
        stack_a.set_uniform(0, i, 160.0);
    }
    let out_a = stack_a.run(&RansacPhotometricParams::default());

    // Permute: reverse source order within tile 0.
    let perm: Vec<usize> = (0..n_sources).rev().collect();
    let mut stack_b = SyntheticStack::full(n_tiles, n_sources, 4, 1);
    for slot in 0..n_sources {
        let orig = perm[slot];
        stack_b.set_uniform(0, slot, if orig < 4 { 100.0 } else { 160.0 });
    }
    let out_b = stack_b.run(&RansacPhotometricParams::default());

    for slot in 0..n_sources {
        let orig = perm[slot];
        assert_eq!(out_b.primary_mask[slot], out_a.primary_mask[orig]);
        assert_eq!(out_b.secondary_mask[slot], out_a.secondary_mask[orig]);
    }
}

// ── 16. Idempotence at the fixed point ──────────────────────────────────

#[test]
fn one_more_iter_at_fixed_point_does_not_change_outputs() {
    let truth = [0.0_f32, 0.05, -0.02, 0.03];
    let n_tiles = 4;
    let n_sources = 4;
    let target_l = [80.0_f32, 100.0, 120.0, 150.0];
    let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 3);
    for t in 0..n_tiles {
        for i in 0..n_sources {
            let l_obs = target_l[t] * (-truth[i]).exp();
            stack.set_uniform(t, i, l_obs);
        }
    }
    // Run with cap = 8.
    let out_8 = stack.run(&loose_threshold_params());
    // Run with cap = out_8.outer_iters + 1.
    let extended_params = RansacPhotometricParams {
        max_outer_iters: out_8.outer_iters + 1,
        ..loose_threshold_params()
    };
    let out_n1 = super::refine_photometric_ransac_flat(
        &stack.patches,
        &stack.valid,
        &stack.src_index,
        &stack.tile_offsets,
        stack.target_size,
        stack.channels,
        stack.n_sources,
        stack.n_tiles,
        &extended_params,
    )
    .expect("ok");

    assert_eq!(out_n1.primary_mask, out_8.primary_mask);
    for i in 0..n_sources {
        let err = (out_n1.log_gain[i] - out_8.log_gain[i]).abs();
        assert!(err < 1e-6, "log_gain[{i}] drifted by {err}");
    }
}

// ── 17. All rows fully masked ───────────────────────────────────────────

#[test]
fn fully_masked_input_returns_zero_gain_and_no_crash() {
    let n_tiles = 3;
    let n_sources = 4;
    let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 3);
    // Set every validity entry to 0.
    for v in stack.valid.iter_mut() {
        *v = 0;
    }
    let out = stack.run(&RansacPhotometricParams::default());
    // Every row's mean luminance is the floor (sum=0 / max(denom,1) = 0),
    // and all-zero gains is the defensible fallback. The algorithm must
    // not crash and must produce mean-zero log_gain.
    assert!(out.log_gain.iter().all(|&g| g.is_finite()));
    let mean = out.log_gain.iter().sum::<f32>() / out.log_gain.len() as f32;
    assert!(mean.abs() < 1e-5);
    // Per-tile MAD on fully-zeroed rows is 0 (when classified) or NaN
    // (when below min_inliers). Either is acceptable here; just ensure
    // we don't panic.
    let _ = (out.tile_primary_lum_mad, out.tile_secondary_lum_mad);
}

// ── 18. Single-tile underdetermined ────────────────────────────────────

#[test]
fn single_tile_underdetermined_returns_sensible_gains() {
    let n_sources = 10;
    let n_tiles = 1;
    let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 1);
    // Truth gains for all 10 sources.
    let truth: Vec<f32> = (0..n_sources)
        .map(|i| 0.05 * (i as f32 - (n_sources as f32 - 1.0) / 2.0))
        .collect();
    let target_l = 100.0_f32;
    for i in 0..n_sources {
        let l_obs = target_l * (-truth[i]).exp();
        stack.set_uniform(0, i, l_obs);
    }
    let out = stack.run(&loose_threshold_params());
    // Mean-zero.
    let m = out.log_gain.iter().sum::<f32>() / out.log_gain.len() as f32;
    assert!(m.abs() < 1e-5, "mean = {m}");
    // No source's gain is wildly out of plausible range.
    for &g in &out.log_gain {
        assert!(g.abs() < 1.0);
    }
}
