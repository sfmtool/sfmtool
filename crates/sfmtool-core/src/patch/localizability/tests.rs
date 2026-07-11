// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Synthetic-patch validation of the structure-tensor scorer (the spec's
//! [Validation](../../../../specs/core/patch-localizability.md) table) plus a
//! numeric cross-check against the throwaway Python prototype (not in-tree; see
//! the spec's Evidence section) whose math this scorer ports.

use super::scorer::{patch_localizability, score_localizability_stack, Localizability};
use crate::patch::normal_refine::{build_support, PatchWindow};

const R: usize = 24;
const C: usize = 4;

fn window() -> crate::patch::normal_refine::Support {
    build_support(PatchWindow::GaussianDisk { sigma: 0.6 }, R as u32)
}

/// Wrap an `R×R` luminance grid into an RGBA (`R×R×4`) f32 patch: `R=G=B=gray`,
/// alpha fully opaque. Luminance weights sum to 1, so the scorer recovers `gray`.
fn rgba_from_gray(gray: &[f64]) -> Vec<f32> {
    let mut out = vec![0.0f32; R * R * C];
    for (px, &g) in gray.iter().enumerate() {
        out[px * C] = g as f32;
        out[px * C + 1] = g as f32;
        out[px * C + 2] = g as f32;
        out[px * C + 3] = 255.0;
    }
    out
}

fn flat() -> Vec<f32> {
    rgba_from_gray(&vec![128.0; R * R])
}

fn edge_vertical() -> Vec<f32> {
    // A step in x (bright right half): gradient purely horizontal, so the surface
    // slides vertically (along the edge).
    let mut g = vec![0.0f64; R * R];
    for row in 0..R {
        for col in 0..R {
            g[row * R + col] = if col < R / 2 { 50.0 } else { 200.0 };
        }
    }
    rgba_from_gray(&g)
}

fn edge_horizontal() -> Vec<f32> {
    let mut g = vec![0.0f64; R * R];
    for row in 0..R {
        for col in 0..R {
            g[row * R + col] = if row < R / 2 { 50.0 } else { 200.0 };
        }
    }
    rgba_from_gray(&g)
}

fn corner() -> Vec<f32> {
    // An L-corner: one quadrant bright. Two orthogonal edges → both eigenvalues
    // large → pinned in 2D.
    let mut g = vec![0.0f64; R * R];
    for row in 0..R {
        for col in 0..R {
            g[row * R + col] = if row < R / 2 && col < R / 2 {
                200.0
            } else {
                50.0
            };
        }
    }
    rgba_from_gray(&g)
}

fn blob() -> Vec<f32> {
    // A Gaussian bump: radial gradients in every direction → both eigenvalues large.
    let c = (R as f64 - 1.0) / 2.0;
    let mut g = vec![0.0f64; R * R];
    for row in 0..R {
        for col in 0..R {
            let dx = col as f64 - c;
            let dy = row as f64 - c;
            g[row * R + col] = 50.0 + 200.0 * (-(dx * dx + dy * dy) / (2.0 * 16.0)).exp();
        }
    }
    rgba_from_gray(&g)
}

fn score(patch: &[f32]) -> Localizability {
    patch_localizability(patch, R, C, &window(), 3.0)
}

#[test]
fn lam2_ranks_2d_features_above_edges_and_flats() {
    let corner = score(&corner());
    let blob = score(&blob());
    let edge = score(&edge_vertical());
    let flat = score(&flat());

    // Corner and blob localize in 2D: the weak eigenvalue is large.
    assert!(corner.lam2 > 10.0, "corner lam2 = {}", corner.lam2);
    assert!(blob.lam2 > 10.0, "blob lam2 = {}", blob.lam2);
    // A straight edge cannot pin the along-edge axis: lam2 collapses to zero.
    assert!(edge.lam2 < 1e-6, "edge lam2 = {}", edge.lam2);
    // A flat patch has no structure at all: both eigenvalues vanish.
    assert!(flat.lam2 < 1e-9, "flat lam2 = {}", flat.lam2);
    assert!(flat.lam1 < 1e-9, "flat lam1 = {}", flat.lam1);
    // The edge still has one strong axis (across the edge) — that is what makes it
    // an *edge* rather than a *flat*.
    assert!(edge.lam1 > 100.0, "edge lam1 = {}", edge.lam1);
    // Every 2D feature is far better localized than an edge or flat, and that is
    // exactly what σ_pos reports (smaller = better).
    assert!(corner.sigma_pos_grid < edge.sigma_pos_grid);
    assert!(corner.sigma_pos_grid < flat.sigma_pos_grid);
    assert!(blob.sigma_pos_grid < edge.sigma_pos_grid);
}

#[test]
fn weak_eigenvector_aligns_with_the_edge() {
    // Vertical edge → slide axis is vertical (y). theta is the angle of the weak
    // eigenvector in the (x=col, y=row) frame; vertical means cos(theta) ≈ 0.
    let v = score(&edge_vertical());
    assert!(
        v.theta.cos().abs() < 1e-9,
        "vertical-edge slide theta = {} (cos {})",
        v.theta,
        v.theta.cos()
    );
    // Horizontal edge → slide axis is horizontal (x): sin(theta) ≈ 0.
    let h = score(&edge_horizontal());
    assert!(
        h.theta.sin().abs() < 1e-9,
        "horizontal-edge slide theta = {} (sin {})",
        h.theta,
        h.theta.sin()
    );
}

#[test]
fn linear_ramp_matches_the_analytic_tensor() {
    // A pure linear ramp in x has constant gradient (k, 0) everywhere — including
    // the one-sided borders — so the tensor is exactly diag(k²·W, 0): a
    // closed-form check of the window-weighted accumulation. lam2 = 0, so σ_pos
    // saturates on the flat-axis floor.
    let k = 1.5;
    let mut g = vec![0.0f64; R * R];
    for row in 0..R {
        for col in 0..R {
            g[row * R + col] = k * col as f64;
        }
    }
    let win = window();
    let s = patch_localizability(&rgba_from_gray(&g), R, C, &win, 3.0);
    let expected_lam1 = k * k * win.total_weight;
    assert!(
        (s.lam1 - expected_lam1).abs() < 1e-6 * expected_lam1,
        "lam1 {} vs analytic {}",
        s.lam1,
        expected_lam1
    );
    assert!(s.lam2.abs() < 1e-9, "lam2 = {}", s.lam2);
    let expected_sigma = 3.0 / (1e-12f64).sqrt();
    assert!((s.sigma_pos_grid - expected_sigma).abs() < 1e-3 * expected_sigma);
}

#[test]
fn matches_python_prototype_numerically() {
    // Reference values captured from the throwaway Python prototype (not in-tree)
    // on the exact `REF_VAL` patch below (RGBA with R=G=B=REF_VAL, alpha 255),
    // sigma_noise=3.0. The prototype computes luminance in f32; the tolerance covers that.
    let gray: Vec<f64> = REF_VAL.iter().map(|&v| v as f64).collect();
    let s = patch_localizability(&rgba_from_gray(&gray), R, C, &window(), 3.0);

    let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
    assert!(rel(s.lam1, 177487.98729078675) < 2e-3, "lam1 = {}", s.lam1);
    assert!(rel(s.lam2, 17114.369000570616) < 2e-3, "lam2 = {}", s.lam2);
    assert!(
        (s.theta - (-1.5746049728304614)).abs() < 2e-3,
        "theta = {}",
        s.theta
    );
    assert!(
        rel(s.sigma_pos_grid, 0.022931940642136706) < 2e-3,
        "sigma_grid = {}",
        s.sigma_pos_grid
    );
}

#[test]
fn grayscale_scores_like_the_replicated_rgb() {
    // A 1-channel patch IS its own luminance: identical content must score
    // identically whether supplied as grayscale or replicated to R=G=B (the
    // Rec.601 partial red weight must not shrink single-channel gradients).
    let gray: Vec<f64> = REF_VAL.iter().map(|&v| v as f64).collect();
    let gray_f32: Vec<f32> = gray.iter().map(|&v| v as f32).collect();
    let rgb = rgba_from_gray(&gray);
    let win = window();
    let s1 = patch_localizability(&gray_f32, R, 1, &win, 3.0);
    let s3 = patch_localizability(&rgb, R, C, &win, 3.0);
    let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
    assert!(
        rel(s1.lam1, s3.lam1) < 1e-6,
        "lam1 {} vs {}",
        s1.lam1,
        s3.lam1
    );
    assert!(
        rel(s1.lam2, s3.lam2) < 1e-6,
        "lam2 {} vs {}",
        s1.lam2,
        s3.lam2
    );
    assert!(
        rel(s1.sigma_pos_grid, s3.sigma_pos_grid) < 1e-6,
        "sigma {} vs {}",
        s1.sigma_pos_grid,
        s3.sigma_pos_grid
    );
}

#[test]
fn empty_patch_scores_nan() {
    let empty = vec![0.0f32; R * R * C];
    let s = patch_localizability(&empty, R, C, &window(), 3.0);
    assert!(s.lam1.is_nan() && s.lam2.is_nan());
    assert!(s.theta.is_nan() && s.sigma_pos_grid.is_nan());
}

#[test]
fn alpha_only_patch_is_flat_not_empty() {
    // A patch with zero RGB but non-zero alpha is *covered* (not culled), so it is
    // scored (flat → huge σ_pos), not treated as empty — matching the prototype's
    // "any non-zero over the whole RGBA stack" validity test.
    let mut p = vec![0.0f32; R * R * C];
    for px in 0..R * R {
        p[px * C + 3] = 255.0;
    }
    let s = patch_localizability(&p, R, C, &window(), 3.0);
    assert!(!s.sigma_pos_grid.is_nan());
    assert!(s.lam2 < 1e-9, "alpha-only lam2 = {}", s.lam2);
}

#[test]
fn batch_matches_per_patch() {
    let patches = [corner(), edge_vertical(), flat(), blob()];
    let mut stack = Vec::new();
    for p in &patches {
        stack.extend_from_slice(p);
    }
    let batch = score_localizability_stack(
        &stack,
        patches.len(),
        R,
        C,
        PatchWindow::GaussianDisk { sigma: 0.6 },
        3.0,
    );
    assert_eq!(batch.len(), patches.len());
    for (b, p) in batch.iter().zip(&patches) {
        let single = score(p);
        assert_eq!(b.lam1, single.lam1);
        assert_eq!(b.lam2, single.lam2);
        assert_eq!(b.sigma_pos_grid, single.sigma_pos_grid);
    }
}

/// The exact `R×R` (row-major) luminance grid the prototype was run on to produce
/// the [`matches_python_prototype_numerically`] reference values.
#[rustfmt::skip]
const REF_VAL: [u8; R * R] = [
    168,206,235,247,240,215,179,139,107,89,91,111,145,185,220,243,247,231,200,161,124,97,88,97,
    166,204,233,246,238,214,177,138,105,88,89,109,143,183,218,241,245,230,199,160,122,95,86,96,
    161,199,228,240,233,208,172,132,100,82,84,104,138,178,213,236,240,224,193,155,117,90,81,90,
    152,191,220,232,225,200,164,124,92,74,76,96,130,170,205,227,232,216,185,146,109,82,72,82,
    142,180,209,222,215,190,153,114,81,64,65,86,120,159,195,217,221,206,175,136,98,72,62,72,
    130,169,198,210,203,178,142,102,70,52,54,74,108,148,183,205,209,194,163,124,87,60,50,60,
    118,157,186,198,191,166,130,90,58,40,42,62,96,136,171,193,198,182,151,112,75,48,38,48,
    107,146,175,187,180,155,119,79,47,29,31,51,85,125,160,182,186,171,140,101,64,37,27,37,
    98,136,165,178,171,146,109,70,37,20,21,42,76,115,151,173,177,162,131,92,54,28,18,28,
    91,130,159,171,164,139,103,63,31,13,15,35,69,109,144,166,170,155,124,85,48,21,11,21,
    88,126,155,168,161,136,99,60,27,10,11,31,66,105,140,163,167,152,121,82,44,18,8,18,
    88,126,155,168,161,136,99,60,27,10,11,32,66,105,141,163,167,152,121,82,44,18,8,18,
    92,130,159,171,164,140,103,64,31,13,15,35,69,109,144,167,171,156,125,86,48,21,12,22,
    98,137,166,178,171,146,110,70,38,20,22,42,76,116,151,174,178,162,131,92,55,28,18,28,
    108,146,175,188,181,156,119,80,47,30,31,51,86,125,160,183,187,172,141,102,64,38,28,38,
    119,157,186,199,192,167,130,91,59,41,42,63,97,136,172,194,198,183,152,113,76,49,39,49,
    131,169,198,211,204,179,142,103,70,53,54,75,109,148,184,206,210,195,164,125,87,61,51,61,
    143,181,210,222,215,190,154,115,82,64,66,86,120,160,195,218,222,206,176,137,99,72,63,73,
    153,191,220,233,226,201,164,125,92,75,76,96,131,170,205,228,232,217,186,147,109,83,73,83,
    161,199,228,241,234,209,172,133,100,83,84,104,139,178,213,236,240,225,194,155,117,91,81,91,
    166,204,233,246,239,214,177,138,105,88,89,109,144,183,218,241,245,230,199,160,122,96,86,96,
    167,206,235,247,240,215,179,139,107,89,91,111,145,185,220,243,247,231,200,161,124,97,87,97,
    166,204,233,245,238,213,177,137,105,87,89,109,143,183,218,241,245,229,198,159,122,95,86,95,
    160,198,227,240,233,208,171,132,100,82,83,104,138,177,213,235,239,224,193,154,117,90,80,90,
];
