// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The direction between two camera centres, read off ray coplanarity with the
//! rotations held.
//!
//! With both rotations known, the baseline `b = c_j - c_i` is coplanar with
//! every point's two world rays: `b . (u_i x u_j) = 0`. So `b` is the null
//! space of the matrix whose rows are those normals. The normal
//! `u_i x u_j` has norm `sin(parallax angle)`, which is literally how much
//! baseline that point saw, so a row inside an angular bound carries no
//! information at all: inside the bound the two rays are the same ray and the
//! cross product is noise. Those rows are DROPPED rather than down-weighted,
//! and the rest are normalized to unit length so no single wide pair carries
//! the fit.
//!
//! The null space is refit on its own best fraction for a fixed number of
//! rounds, and its sign is fixed by cheirality: with one centre at the origin
//! and the other at `+d`, the direction that puts more points in front of both
//! cameras is the one kept.
//!
//! Edges are flattened CSR-style, so a whole covisibility graph is one call.
//!
//! See `specs/core/geometry/baseline-direction.md` for the design.

use nalgebra::{MatrixXx3, Vector3};
use rayon::prelude::*;

use crate::numeric::median_in_place;

/// One edge's baseline, and what its own rows say about how well it is
/// determined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaselineDirection {
    /// Unit direction from the first centre to the second, sign fixed by
    /// cheirality.
    pub direction: [f64; 3],
    /// Rows the edge was given.
    pub n_rows: usize,
    /// Rows past the angular bound, which are the rows the fit read.
    pub n_used: usize,
    /// Conditioning of the null space: the second-smallest singular value over
    /// the smallest, on the rows the final fit kept. Infinite where the
    /// smallest is zero.
    pub condition: f64,
    /// Median parallax of the rows used, in degrees.
    pub parallax_median_deg: f64,
    /// Widest parallax over ALL the edge's rows, in degrees, including the ones
    /// the bound dropped.
    pub parallax_max_deg: f64,
    /// Fraction of the used rows that triangulate in front of both cameras at
    /// the sign kept.
    pub cheiral_fraction: f64,
    /// Median absolute coplanarity residual of the used rows, in radians.
    pub residual_median_rad: f64,
}

/// How the null space is trimmed.
#[derive(Debug, Clone, Copy)]
pub struct BaselineTrim {
    /// Rows below which an edge states no direction at all.
    pub tol_rad: f64,
    /// Refit rounds, each keeping the best fraction of the rows past the bound.
    pub rounds: usize,
    /// The fraction kept, of the rows past the bound, never fewer than three.
    pub keep_fraction: f64,
}

/// The baseline direction of every edge of a graph.
///
/// `rays_i` and `rays_j` are the two frames' unit WORLD rays, three components
/// per row, one row per shared point, with all edges concatenated; `offsets`
/// (length `n_edge + 1`) delimits the edges CSR-style. An edge with fewer than
/// three rows past the bound states no direction and comes back `None`.
pub fn baseline_directions(
    rays_i: &[f64],
    rays_j: &[f64],
    offsets: &[usize],
    trim: BaselineTrim,
) -> Vec<Option<BaselineDirection>> {
    assert_eq!(
        rays_i.len(),
        rays_j.len(),
        "rays_i and rays_j length mismatch"
    );
    assert!(
        rays_i.len().is_multiple_of(3),
        "rays must be three components per row"
    );
    let n_edge = offsets.len().saturating_sub(1);
    (0..n_edge)
        .into_par_iter()
        .map(|e| one_edge(rays_i, rays_j, offsets[e], offsets[e + 1], trim))
        .collect()
}

/// One edge, from its own rows.
fn one_edge(
    rays_i: &[f64],
    rays_j: &[f64],
    lo: usize,
    hi: usize,
    trim: BaselineTrim,
) -> Option<BaselineDirection> {
    let n_rows = hi - lo;
    let mut normals: Vec<Vector3<f64>> = Vec::new();
    let mut ui: Vec<Vector3<f64>> = Vec::new();
    let mut uj: Vec<Vector3<f64>> = Vec::new();
    let mut parallax: Vec<f64> = Vec::with_capacity(n_rows);
    let mut used_parallax: Vec<f64> = Vec::new();
    for r in lo..hi {
        let a = Vector3::new(rays_i[3 * r], rays_i[3 * r + 1], rays_i[3 * r + 2]);
        let b = Vector3::new(rays_j[3 * r], rays_j[3 * r + 1], rays_j[3 * r + 2]);
        let cross = a.cross(&b);
        let norm = cross.norm();
        let par = norm.clamp(0.0, 1.0).asin();
        parallax.push(par);
        if par > trim.tol_rad {
            normals.push(cross / norm);
            ui.push(a);
            uj.push(b);
            used_parallax.push(par);
        }
    }
    let n_used = normals.len();
    if n_used < 3 {
        return None;
    }

    let take = round_half_even(trim.keep_fraction * n_used as f64).max(3.0) as usize;
    let take = take.min(n_used);
    let mut kept: Vec<usize> = (0..n_used).collect();
    for _ in 0..trim.rounds {
        let (d, _) = null_space(&normals, &kept)?;
        kept = best_rows(&normals, &d, take);
    }
    let (direction, condition) = null_space(&normals, &kept)?;

    // Cheirality fixes the sign: with c_i at the origin and c_j at +d, a point
    // in front of both cameras has positive depth along both rays.
    let pos = cheiral_votes(&ui, &uj, &direction);
    let neg = cheiral_votes(&ui, &uj, &(-direction));
    let (direction, pos, neg) = if neg > pos {
        (-direction, neg, pos)
    } else {
        (direction, pos, neg)
    };

    let mut residuals: Vec<f64> = normals.iter().map(|n| n.dot(&direction).abs()).collect();
    let mut used = used_parallax;
    Some(BaselineDirection {
        direction: [direction.x, direction.y, direction.z],
        n_rows,
        n_used,
        condition,
        parallax_median_deg: median_in_place(&mut used).to_degrees(),
        parallax_max_deg: parallax
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .to_degrees(),
        cheiral_fraction: pos as f64 / (pos + neg).max(1) as f64,
        residual_median_rad: median_in_place(&mut residuals),
    })
}

/// The null direction of the retained rows and the conditioning of that
/// reading, from one decomposition. The direction is the right singular
/// vector of the smallest singular value; the conditioning is the ratio of
/// the second-smallest singular value to the smallest, `inf` when the
/// smallest is zero or there are fewer than two. The direction's raw sign is
/// whatever the decomposition produced, and cheirality overrides it
/// downstream.
fn null_space(normals: &[Vector3<f64>], kept: &[usize]) -> Option<(Vector3<f64>, f64)> {
    let m = MatrixXx3::from_rows(
        &kept
            .iter()
            .map(|&k| normals[k].transpose())
            .collect::<Vec<_>>(),
    );
    let svd = nalgebra::SVD::new(m, false, true);
    let v_t = svd.v_t?;
    let sv = svd.singular_values.as_slice();
    let k = smallest_index(sv);
    let direction = Vector3::new(v_t[(k, 0)], v_t[(k, 1)], v_t[(k, 2)]).normalize();
    let mut sorted: Vec<f64> = sv.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let condition = if sorted.len() < 2 || sorted[0] <= 0.0 {
        f64::INFINITY
    } else {
        sorted[1] / sorted[0]
    };
    Some((direction, condition))
}

fn smallest_index(values: &[f64]) -> usize {
    let mut best = 0usize;
    for (k, v) in values.iter().enumerate() {
        if *v < values[best] {
            best = k;
        }
    }
    best
}

/// The `take` rows whose coplanarity residual is smallest, ties in row order.
fn best_rows(normals: &[Vector3<f64>], direction: &Vector3<f64>, take: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..normals.len()).collect();
    let residual: Vec<f64> = normals.iter().map(|n| n.dot(direction).abs()).collect();
    order.sort_by(|&a, &b| {
        residual[a]
            .partial_cmp(&residual[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.truncate(take);
    order.sort_unstable();
    order
}

/// How many rows triangulate in front of both cameras at this baseline.
fn cheiral_votes(ui: &[Vector3<f64>], uj: &[Vector3<f64>], d: &Vector3<f64>) -> usize {
    ui.iter()
        .zip(uj)
        .filter(|(a, b)| {
            // c_i = 0, c_j = d. Solve s u_i - t u_j = d in least squares.
            let a11 = a.dot(a);
            let a12 = -a.dot(b);
            let a22 = b.dot(b);
            let b1 = a.dot(d);
            let b2 = -b.dot(d);
            let det = a11 * a22 - a12 * a12;
            let s = (a22 * b1 - a12 * b2) / det;
            let t = (a11 * b2 - a12 * b1) / det;
            s > 0.0 && t > 0.0
        })
        .count()
}

/// Round half to even, so a trim fraction landing exactly on a half row does
/// not always round up.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

#[cfg(test)]
mod tests;
