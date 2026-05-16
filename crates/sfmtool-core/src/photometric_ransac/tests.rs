// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`refine_photometric_ransac`] / `_flat`.
//!
//! The flat-array entry point lets these tests construct exact patch
//! tensors and tile/source assignments without going through a
//! `PerSphericalTileSourceStack`.

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
            &self.tile_offsets,
            self.target_size,
            self.channels,
            self.n_tiles,
            params,
        )
        .expect("flat run succeeded")
    }
}

// ── Two-cluster partition ────────────────────────────────────────────────

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
}

// ── Below-threshold tile (K ≤ m) — symmetric rejection ──────────────────

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

// ── Empty / single-contributor tiles ────────────────────────────────────

#[test]
fn empty_and_single_contributor_tiles_are_skipped() {
    // Two tiles: tile 0 has 0 contributors; tile 1 has 1 contributor;
    // tile 2 has 3 contributors so the RANSAC has at least one row to act on.
    let n_tiles = 3;
    let s = 4_u32;
    let c = 1_u32;
    let s_us = s as usize;
    let tile_offsets = vec![0u32, 0, 1, 4];
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
        &tile_offsets,
        s,
        c,
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

// ── Subset-enumeration boundary ─────────────────────────────────────────

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

// ── Secondary cluster recovery ──────────────────────────────────────────

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

// ── Patch-aware scoring distinguishes spatial pattern ───────────────────

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

// ── Permutation invariance of row order within a tile ───────────────────

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

// ── Fully-masked input ──────────────────────────────────────────────────

#[test]
fn fully_masked_input_does_not_crash() {
    let n_tiles = 3;
    let n_sources = 4;
    let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 3);
    for v in stack.valid.iter_mut() {
        *v = 0;
    }
    let out = stack.run(&RansacPhotometricParams::default());
    let _ = (out.tile_primary_lum_mad, out.tile_secondary_lum_mad);
}

// ── tile_index_base plumbing ────────────────────────────────────────────

/// Per-tile rows `[t * n_sources, (t + 1) * n_sources)`.
fn tile_rows(out: &[bool], t: usize, n_sources: usize) -> &[bool] {
    &out[t * n_sources..(t + 1) * n_sources]
}

/// `tile_index_base` shifts every tile's RNG stream by a constant offset.
///
/// On an over-cap stack (`C(16, 2) = 120 > max_subsets_per_tile = 64`, so each
/// tile uses sampled — RNG-seeded — candidate subsets) where every tile holds
/// *identical* data, the only thing that can vary tile-to-tile is the seed.
/// Two equal-size, well-separated clusters make the surviving cluster a
/// function purely of which candidate pairs the RNG samples first, so the
/// per-tile partition genuinely depends on the seed.
///
/// Running with `tile_index_base = k` must therefore reproduce, for tile `t`,
/// exactly what `tile_index_base = 0` produces for global tile `t + k`.
#[test]
fn tile_index_base_shifts_per_tile_rng_streams() {
    let n_tiles = 24;
    let n_sources = 16; // C(16, 2) = 120 > 64 ⇒ sampled candidates, RNG in play.
    let k = 7;

    // Every tile identical: 8 sources at luminance 100, 8 at 200. The two
    // clusters are equal-size and well-separated (gap 100 ≫ inlier_threshold).
    let mut stack = SyntheticStack::full(n_tiles, n_sources, 4, 3);
    for t in 0..n_tiles {
        for i in 0..8 {
            stack.set_uniform(t, i, 100.0);
        }
        for i in 8..16 {
            stack.set_uniform(t, i, 200.0);
        }
    }

    let base = RansacPhotometricParams::default();
    let out_0 = stack.run(&base);
    let out_k = stack.run(&RansacPhotometricParams {
        tile_index_base: k,
        ..base.clone()
    });

    // Non-vacuity: the partition really is seed-dependent — not every tile
    // resolves to the same cluster.
    let first = tile_rows(&out_0.primary_mask, 0, n_sources).to_vec();
    assert!(
        (0..n_tiles).any(|t| tile_rows(&out_0.primary_mask, t, n_sources) != first.as_slice()),
        "fixture is seed-insensitive — the test would pass vacuously"
    );

    // The shift property: tile `t` at base `k` == tile `t + k` at base 0.
    for t in 0..(n_tiles - k) {
        assert_eq!(
            tile_rows(&out_k.primary_mask, t, n_sources),
            tile_rows(&out_0.primary_mask, t + k, n_sources),
            "primary_mask: tile {t} (base {k}) != tile {} (base 0)",
            t + k
        );
        assert_eq!(
            tile_rows(&out_k.secondary_mask, t, n_sources),
            tile_rows(&out_0.secondary_mask, t + k, n_sources),
            "secondary_mask: tile {t} (base {k}) != tile {} (base 0)",
            t + k
        );
    }
}
