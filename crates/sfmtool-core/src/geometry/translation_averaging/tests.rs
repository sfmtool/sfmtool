// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for translation averaging, relative lengths and the orientation bit.

use super::*;

/// Eight frames in a general (non-colinear) constellation.
const TRUE_CENTRES: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.1, -0.2],
    [2.1, 0.0, 0.3],
    [3.0, -0.4, 0.1],
    [3.8, 0.5, -0.5],
    [4.9, 0.2, 0.6],
    [5.7, -0.3, 0.0],
    [6.6, 0.1, 0.4],
];

/// Six frames on one straight line, evenly spaced: every baseline of it
/// carries the same direction, which is what a camera walking a line produces.
fn colinear() -> Vec<[f64; 3]> {
    (0..6).map(|f| [0.4 * f as f64, 0.0, 0.0]).collect()
}

fn all_pairs(n: usize) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            out.push((i as u32, j as u32));
        }
    }
    out
}

/// The exact unit directions of `pairs` over `centres`, flattened.
fn edges(centres: &[[f64; 3]], pairs: &[(u32, u32)]) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>) {
    let mut ii = Vec::new();
    let mut jj = Vec::new();
    let mut dirs = Vec::new();
    let mut w = Vec::new();
    for &(i, j) in pairs {
        let a = centres[i as usize];
        let b = centres[j as usize];
        let v = Vector3::new(b[0] - a[0], b[1] - a[1], b[2] - a[2]).normalize();
        ii.push(i);
        jj.push(j);
        dirs.extend_from_slice(&[v.x, v.y, v.z]);
        w.push(10.0);
    }
    (ii, jj, dirs, w)
}

/// The true relative lengths of `pairs`, gauged to a median of one.
fn lengths_of(centres: &[[f64; 3]], pairs: &[(u32, u32)]) -> Vec<f64> {
    let mut out: Vec<f64> = pairs
        .iter()
        .map(|&(i, j)| {
            let a = centres[i as usize];
            let b = centres[j as usize];
            ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
        })
        .collect();
    let med = median(&out);
    for v in out.iter_mut() {
        *v /= med;
    }
    out
}

/// The constellation modulo scale and shift: unit pairwise vectors.
fn shape(centres: &[[f64; 3]]) -> Vec<Vector3<f64>> {
    let n = centres.len();
    let mut out = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (centres[i], centres[j]);
            out.push(Vector3::new(b[0] - a[0], b[1] - a[1], b[2] - a[2]).normalize());
        }
    }
    out
}

fn shape_error(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    shape(a)
        .iter()
        .zip(shape(b))
        .map(|(p, q)| (p - q).abs().max())
        .fold(0.0f64, f64::max)
}

/// The consecutive gaps of a chain of centres, over their own largest.
fn spacing(centres: &[[f64; 3]]) -> Vec<f64> {
    let gaps: Vec<f64> = (1..centres.len())
        .map(|k| {
            let (a, b) = (centres[k - 1], centres[k]);
            ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
        })
        .collect();
    let biggest = gaps.iter().fold(0.0f64, |m, &v| m.max(v));
    gaps.iter().map(|g| g / biggest).collect()
}

#[test]
fn the_constellation_is_the_forms_own_null_space() {
    let pairs = all_pairs(TRUE_CENTRES.len());
    let (ii, jj, d, w) = edges(&TRUE_CENTRES, &pairs);
    let out = average_translations(
        TranslationGraph {
            edge_i: &ii,
            edge_j: &jj,
            directions: &d,
            weights: &w,
            lengths: None,
            length_weights: None,
            n_frames: TRUE_CENTRES.len(),
        },
        IRLS_ROUNDS,
    );
    assert!(out.census.solved);
    assert!(out.census.read_off_null);
    assert_eq!(out.census.n_null, 1);
    assert_eq!(out.census.n_free, 0);
    assert!(shape_error(&out.centres, &TRUE_CENTRES) < 1e-9);
    // Every edge points forward along its own measured direction.
    assert!(out.lambda.iter().all(|&v| v > 0.0));
    assert!(out.residual.iter().all(|&v| v < 1e-9));
}

#[test]
fn a_colinear_path_leaves_one_null_direction_per_free_spacing() {
    let centres = colinear();
    let pairs = all_pairs(centres.len());
    let (ii, jj, d, w) = edges(&centres, &pairs);
    let graph = TranslationGraph {
        edge_i: &ii,
        edge_j: &jj,
        directions: &d,
        weights: &w,
        lengths: None,
        length_weights: None,
        n_frames: centres.len(),
    };
    let bare = average_translations(graph, IRLS_ROUNDS);
    // Six frames on a line have five gaps, one of which the scale gauge fixes.
    assert_eq!(bare.census.n_free, centres.len() - 2);
    assert_eq!(bare.census.n_lengths, 0);
    assert!(!bare.census.read_off_null);
    // And what comes back is not the spacing the frames have.
    assert!(spacing(&bare.centres)
        .iter()
        .any(|g| (g - 1.0).abs() > 1e-3));

    // The pairs' own lengths close every one of those freedoms.
    let ell = lengths_of(&centres, &pairs);
    let with = average_translations(
        TranslationGraph {
            lengths: Some(&ell),
            length_weights: Some(&w),
            ..graph
        },
        IRLS_ROUNDS,
    );
    assert_eq!(with.census.n_free, 0);
    assert_eq!(with.census.n_lengths, pairs.len());
    assert!(spacing(&with.centres)
        .iter()
        .all(|g| (g - 1.0).abs() < 1e-6));
}

#[test]
fn the_directions_own_reading_is_what_the_capture_determines() {
    let centres = colinear();
    let pairs = all_pairs(centres.len());
    let (ii, jj, d, w) = edges(&centres, &pairs);
    let ell = lengths_of(&centres, &pairs);
    let graph = TranslationGraph {
        edge_i: &ii,
        edge_j: &jj,
        directions: &d,
        weights: &w,
        lengths: Some(&ell),
        length_weights: Some(&w),
        n_frames: centres.len(),
    };
    // The reading ignores the lengths the graph carries, which is what makes
    // it a property of the capture rather than of a solve.
    let alone = direction_reading(graph);
    let with = average_translations(graph, IRLS_ROUNDS).census;
    assert_eq!(alone.n_free, centres.len() - 2);
    assert_eq!(with.n_free, 0);
    assert!(alone.lam2_rel < with.lam2_rel);
    assert_eq!(alone.n_lengths, 0);
}

#[test]
fn a_frame_on_one_lengthless_edge_is_loose() {
    // Frame 8 hangs off frame 0 by a single edge and states no length for it,
    // so nothing says how far along that edge it sits.
    let mut centres: Vec<[f64; 3]> = TRUE_CENTRES.to_vec();
    centres.push([
        TRUE_CENTRES[0][0],
        TRUE_CENTRES[0][1] + 2.0,
        TRUE_CENTRES[0][2],
    ]);
    let mut pairs = all_pairs(TRUE_CENTRES.len());
    pairs.push((0, 8));
    let (ii, jj, d, w) = edges(&centres, &pairs);
    let mut ell = lengths_of(&centres, &pairs);
    let last = ell.len() - 1;
    ell[last] = f64::NAN;
    let out = average_translations(
        TranslationGraph {
            edge_i: &ii,
            edge_j: &jj,
            directions: &d,
            weights: &w,
            lengths: Some(&ell),
            length_weights: Some(&w),
            n_frames: centres.len(),
        },
        IRLS_ROUNDS,
    );
    assert_eq!(out.census.n_loose, 1);
    assert!(!out.census.read_off_null);
    assert_eq!(out.census.n_lengths, pairs.len() - 1);
    // The eight frames the graph does determine are still where they belong.
    let head: Vec<[f64; 3]> = out.centres[..8].to_vec();
    assert!(shape_error(&head, &TRUE_CENTRES) < 1e-3);
}

#[test]
fn a_contradicted_edge_is_reweighted_out_of_the_solve() {
    let pairs = all_pairs(TRUE_CENTRES.len());
    let (ii, jj, mut d, w) = edges(&TRUE_CENTRES, &pairs);
    // One edge reversed and tilted perpendicular to the truth, at the same
    // base weight as every other edge.
    let bad = pairs
        .iter()
        .position(|&p| p == (2, 5))
        .expect("edge (2, 5)");
    let good = Vector3::new(d[3 * bad], d[3 * bad + 1], d[3 * bad + 2]);
    let tilted = (-good + Vector3::new(0.0, 0.9, 0.0)).normalize();
    d[3 * bad] = tilted.x;
    d[3 * bad + 1] = tilted.y;
    d[3 * bad + 2] = tilted.z;
    let out = average_translations(
        TranslationGraph {
            edge_i: &ii,
            edge_j: &jj,
            directions: &d,
            weights: &w,
            lengths: None,
            length_weights: None,
            n_frames: TRUE_CENTRES.len(),
        },
        IRLS_ROUNDS,
    );
    // The contradicted edge is the one the graph disagrees with most, and the
    // rest of the constellation survives it.
    let worst = out
        .residual
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("a non-empty graph")
        .0;
    assert_eq!(worst, bad);
    assert!(shape_error(&out.centres, &TRUE_CENTRES) < 5e-2);
}

#[test]
fn a_graph_stating_no_baseline_solves_nothing() {
    let ii = [0u32, 1];
    let jj = [1u32, 2];
    let d = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let w = [0.0, 0.0];
    let out = average_translations(
        TranslationGraph {
            edge_i: &ii,
            edge_j: &jj,
            directions: &d,
            weights: &w,
            lengths: None,
            length_weights: None,
            n_frames: 3,
        },
        IRLS_ROUNDS,
    );
    assert!(!out.census.solved);
    assert!(out.centres.is_empty());
}

#[test]
fn two_runs_of_the_averaging_are_bit_identical() {
    let pairs = all_pairs(TRUE_CENTRES.len());
    let (ii, jj, d, w) = edges(&TRUE_CENTRES, &pairs);
    let graph = TranslationGraph {
        edge_i: &ii,
        edge_j: &jj,
        directions: &d,
        weights: &w,
        lengths: None,
        length_weights: None,
        n_frames: TRUE_CENTRES.len(),
    };
    let a = average_translations(graph, IRLS_ROUNDS);
    let b = average_translations(graph, IRLS_ROUNDS);
    assert_eq!(a, b);
}

// ── Relative lengths ──────────────────────────────────────────────────────

/// Four frames on a line at deliberately UNEVEN spacing, so the lengths carry
/// something the directions cannot: every baseline of them points the same way.
const SPACED: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.5, 0.0, 0.0],
    [4.0, 0.0, 0.0],
];

fn cloud() -> Vec<[f64; 3]> {
    (0..40)
        .map(|k| {
            let t = k as f64 / 39.0;
            [-6.0 + 12.0 * t, -4.0 + 8.0 * t, -(9.0 + 21.0 * t)]
        })
        .collect()
}

/// The depth rows of every pair, read the way an edge reads them: with `c_i`
/// at the origin and `c_j` at `+d`, so both depths are in units of the pair's
/// own baseline. `point_shift` moves edge zero's points out of everyone else's
/// numbering, which is how a lone edge is built.
fn depth_rows(centres: &[[f64; 3]], point_shift: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f64>) {
    let pts = cloud();
    let pairs = all_pairs(centres.len());
    let (mut ee, mut ff, mut pp, mut zz) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (e, &(i, j)) in pairs.iter().enumerate() {
        let ci = Vector3::from_row_slice(&centres[i as usize]);
        let cj = Vector3::from_row_slice(&centres[j as usize]);
        let base = (cj - ci).norm();
        let shift = if e == 0 { point_shift } else { 0 };
        for (c, p) in pts.iter().enumerate() {
            let p = Vector3::from_row_slice(p);
            // Both rays meet exactly, so each depth is the true range over the
            // pair's own baseline.
            ee.push(e as u32);
            ff.push(i);
            pp.push(c as u32 + shift);
            zz.push((p - ci).norm() / base);
            ee.push(e as u32);
            ff.push(j);
            pp.push(c as u32 + shift);
            zz.push((p - cj).norm() / base);
        }
    }
    (ee, ff, pp, zz)
}

#[test]
fn the_depths_state_the_ratio_of_the_baselines() {
    let (ee, ff, pp, zz) = depth_rows(&SPACED, 0);
    let pairs = all_pairs(SPACED.len());
    let out = relative_lengths(
        DepthRows {
            edge_of_row: &ee,
            frame_of_row: &ff,
            point_of_row: &pp,
            depth_of_row: &zz,
            n_edges: pairs.len(),
        },
        LENGTH_IRLS_ROUNDS,
        MIN_TIED_ROWS,
    );
    let truth = lengths_of(&SPACED, &pairs);
    assert!(out.lengths.iter().all(|v| v.is_finite()));
    let med = median(&out.lengths);
    for (got, want) in out.lengths.iter().zip(&truth) {
        assert!((got / med - want).abs() < 1e-9, "{got} vs {want}");
    }
    // The fit explains its own rows, so the scatter it leaves is nothing.
    assert!(out.scatter.iter().all(|&v| v < 1e-9));
    assert!(out.n_tied.iter().all(|&v| v > 0));
}

#[test]
fn an_edge_that_shares_no_point_states_no_length() {
    // One edge is given points nobody else saw, so nothing ties its depths to
    // another baseline: a length for it would be invented rather than read.
    let (ee, ff, pp, zz) = depth_rows(&SPACED, 10_000);
    let pairs = all_pairs(SPACED.len());
    let out = relative_lengths(
        DepthRows {
            edge_of_row: &ee,
            frame_of_row: &ff,
            point_of_row: &pp,
            depth_of_row: &zz,
            n_edges: pairs.len(),
        },
        LENGTH_IRLS_ROUNDS,
        MIN_TIED_ROWS,
    );
    assert!(out.lengths[0].is_nan());
    assert_eq!(out.n_tied[0], 0);
    assert!(out.lengths[1..].iter().all(|v| v.is_finite()));
}

#[test]
fn a_graph_with_no_tied_row_states_nothing() {
    let ee = [0u32, 0, 0, 1, 1, 1];
    let ff = [0u32, 0, 0, 1, 1, 1];
    let pp = [0u32, 1, 2, 3, 4, 5];
    let zz = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out = relative_lengths(
        DepthRows {
            edge_of_row: &ee,
            frame_of_row: &ff,
            point_of_row: &pp,
            depth_of_row: &zz,
            n_edges: 2,
        },
        LENGTH_IRLS_ROUNDS,
        MIN_TIED_ROWS,
    );
    assert!(out.lengths.iter().all(|v| v.is_nan()));
    assert!(out.scatter.iter().all(|v| v.is_nan()));
    assert_eq!(out.n_tied, vec![0, 0]);
}

#[test]
fn a_wild_depth_moves_nothing() {
    let (ee, ff, pp, mut zz) = depth_rows(&SPACED, 0);
    let pairs = all_pairs(SPACED.len());
    let clean = relative_lengths(
        DepthRows {
            edge_of_row: &ee,
            frame_of_row: &ff,
            point_of_row: &pp,
            depth_of_row: &zz,
            n_edges: pairs.len(),
        },
        LENGTH_IRLS_ROUNDS,
        MIN_TIED_ROWS,
    );
    let of_two: Vec<usize> = (0..ee.len()).filter(|&r| ee[r] == 2).collect();
    for &r in &of_two[..of_two.len() / 6] {
        zz[r] *= 1000.0;
    }
    let dirty = relative_lengths(
        DepthRows {
            edge_of_row: &ee,
            frame_of_row: &ff,
            point_of_row: &pp,
            depth_of_row: &zz,
            n_edges: pairs.len(),
        },
        LENGTH_IRLS_ROUNDS,
        MIN_TIED_ROWS,
    );
    for (a, b) in dirty.lengths.iter().zip(&clean.lengths) {
        assert!((a - b).abs() < 1e-3 * b.abs(), "{a} vs {b}");
    }
    // The edge that carries the wild rows is the one whose scatter shows it.
    let worst = dirty
        .scatter
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("a non-empty graph")
        .0;
    assert_eq!(worst, 2);
}

// ── Orientation ───────────────────────────────────────────────────────────

/// Frames along a short arc, points in front of every one of them.
fn constellation() -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
    let n_frames = 5u32;
    let centres: Vec<f64> = (0..n_frames)
        .flat_map(|f| [0.6 * f as f64, 0.0, 0.0])
        .collect();
    let pts: Vec<[f64; 3]> = (0..9)
        .map(|k| {
            let t = k as f64 / 8.0;
            [-2.0 + 4.0 * t, -1.0 + 2.0 * t, -6.0]
        })
        .collect();
    let (mut rays, mut pof, mut fof) = (Vec::new(), Vec::new(), Vec::new());
    for f in 0..n_frames {
        let c = Vector3::new(0.6 * f as f64, 0.0, 0.0);
        for (k, p) in pts.iter().enumerate() {
            let u = (Vector3::from_row_slice(p) - c).normalize();
            rays.extend_from_slice(&[u.x, u.y, u.z]);
            pof.push(k as u32);
            fof.push(f);
        }
    }
    (centres, rays, pof, fof)
}

#[test]
fn the_right_way_up_reads_positive() {
    let (centres, rays, pof, fof) = constellation();
    let got = orientation_reading(
        OrientationRays {
            centres: &centres,
            rays_world: &rays,
            point_of_ray: &pof,
            frame_of_ray: &fof,
        },
        0.5f64.to_radians(),
    );
    assert!(got.angw > 0.0);
    assert_eq!(got.obs_frac, 1.0);
    assert_eq!(got.margin_frac, 1.0);
    assert_eq!(got.points, 9);
    assert_eq!(got.behind, 0);
    assert_eq!(got.thin, 0);
}

#[test]
fn reflecting_the_centres_negates_the_reading_exactly() {
    let (centres, rays, pof, fof) = constellation();
    let flipped: Vec<f64> = centres.iter().map(|v| -v).collect();
    let bound = 0.5f64.to_radians();
    let up = orientation_reading(
        OrientationRays {
            centres: &centres,
            rays_world: &rays,
            point_of_ray: &pof,
            frame_of_ray: &fof,
        },
        bound,
    );
    let down = orientation_reading(
        OrientationRays {
            centres: &flipped,
            rays_world: &rays,
            point_of_ray: &pof,
            frame_of_ray: &fof,
        },
        bound,
    );
    assert_eq!(down.angw, -up.angw);
    assert_eq!(down.obs_front, up.obs_total - up.obs_front);
    assert_eq!(down.margin_frac, up.margin_frac);
}
