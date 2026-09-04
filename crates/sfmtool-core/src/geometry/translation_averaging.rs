// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Camera centres from pairwise baselines, with every rotation held.
//!
//! With the rotations known and the direction of each connected pair's baseline
//! read off correspondences ([`super::baseline_direction`]), the centres of the
//! whole graph are one linear problem: [`average_translations`] fits them to the
//! stated directions, and to the relative lengths wherever a caller supplies
//! them.
//!
//! Two further operations sit beside the solve, because they are consumed
//! separately: [`relative_lengths`] reads the lengths off the two-view depths
//! the pairs imply, and [`orientation_reading`] reads the one bit the
//! directions cannot, which of the two mirror-image constellations has the
//! structure in front of the cameras.
//!
//! See `specs/core/geometry/translation-averaging.md` for the design — the
//! objective and its gauges, why the constellation is the fitted form's own
//! null space rather than the solution of a linear system posed against it, and
//! what the census reports when the graph does not determine it.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector, Matrix3, Point3, SymmetricEigen, Vector3};

use crate::numeric::{median, median_in_place};
use crate::reconstruction::point_estimation::smallest_pairwise_cosine;
use crate::reconstruction::triangulation::triangulate_batch;

/// Rounds of the centre solve's reweighting.
///
/// The reweighting has no convergence test: each round is one reading of the
/// graph's own residual distribution, and the count is what bounds the work.
pub const IRLS_ROUNDS: usize = 5;

/// Rounds of the length fit's row reweighting.
///
/// The rows here are individual depths rather than whole edges, so a wild
/// depth needs more rounds to be squeezed out than a wild edge does.
pub const LENGTH_IRLS_ROUNDS: usize = 8;

/// The most conjugate-gradient steps one length round takes.
///
/// It bounds the work and decides no outcome, because the fit is a
/// least-squares solve with one answer.
pub const CG_STEPS: usize = 200;

/// The relative residual the conjugate gradient stops at, measured against the
/// largest entry of the right-hand side. Like [`CG_STEPS`] it bounds the work
/// and decides no outcome.
pub const CG_TOL: f64 = 1e-12;

/// The fewest tied rows an edge needs before it states a length.
///
/// It is the baseline-direction operation's own floor: below three rows a
/// reading is the rows themselves rather than a fit of them.
pub const MIN_TIED_ROWS: usize = 3;

/// What the form read of the graph, whichever reading was taken.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AveragingCensus {
    /// Largest eigenvalue of the form, which every other reading is relative
    /// to. Zero for an empty graph.
    pub lam_max: f64,
    /// Smallest eigenvalue over [`Self::lam_max`]; `NaN` where the form is
    /// empty.
    pub lam1_rel: f64,
    /// Second-smallest eigenvalue over [`Self::lam_max`]; `NaN` where the form
    /// has fewer than two.
    pub lam2_rel: f64,
    /// Conditioning of the constellation: the smallest eigenvalue over the
    /// second-smallest. `NaN` where the second is not positive.
    pub gap: f64,
    /// Eigenvalues at or below the numerical rank tolerance.
    pub n_null: usize,
    /// Frames owning more than half of a null dimension.
    pub n_loose: usize,
    /// Null dimensions beyond the constellation's own, one per spacing the
    /// directions leave free.
    pub n_free: usize,
    /// Whether the centres were read off the null space rather than from the
    /// range solution.
    pub read_off_null: bool,
    /// Edges that stated a length.
    pub n_lengths: usize,
    /// Whether the solve produced centres at all. False when the gauge vector
    /// has no component along the answer, which is the graph stating no
    /// baseline.
    pub solved: bool,
}

/// The centres, what each edge did at them, and the census.
#[derive(Debug, Clone, PartialEq)]
pub struct TranslationAveraging {
    /// One centre per frame, mean-centred and at the gauge scale. Empty when
    /// the solve returned nothing.
    pub centres: Vec<[f64; 3]>,
    /// Projected baseline length `d_ij . (c_j - c_i)` per edge. A negative
    /// entry is an edge the constellation placed backwards. Empty when the
    /// solve returned nothing.
    pub lambda: Vec<f64>,
    /// Direction residual per edge: the length of the part of its baseline the
    /// direction says should not be there. Empty when the solve returned
    /// nothing.
    pub residual: Vec<f64>,
    /// What the form read of the graph.
    pub census: AveragingCensus,
}

/// The pairwise readings one averaging is posed on.
#[derive(Debug, Clone, Copy)]
pub struct TranslationGraph<'a> {
    /// First frame index of each edge, `n_edge` entries.
    pub edge_i: &'a [u32],
    /// Second frame index of each edge, `n_edge` entries.
    pub edge_j: &'a [u32],
    /// Unit direction from the first centre to the second, three components
    /// per edge.
    pub directions: &'a [f64],
    /// How far each direction is trusted, `n_edge` entries.
    pub weights: &'a [f64],
    /// Relative baseline length per edge on one common scale, `NaN` where the
    /// edge states none. `None` is no length anywhere.
    pub lengths: Option<&'a [f64]>,
    /// How far each length is trusted; an entry at or below zero states no
    /// length. `None` is no length anywhere.
    pub length_weights: Option<&'a [f64]>,
    /// How many frames the edge indices address.
    pub n_frames: usize,
}

impl TranslationGraph<'_> {
    /// How many edges the graph carries.
    fn n_edges(&self) -> usize {
        self.edge_i.len()
    }

    /// The per-edge directions as vectors, and the checks every caller shares.
    fn unpack(&self) -> Vec<Vector3<f64>> {
        let m = self.n_edges();
        assert_eq!(self.edge_j.len(), m, "edge_i and edge_j length mismatch");
        assert_eq!(
            self.directions.len(),
            3 * m,
            "directions must be n_edge * 3"
        );
        assert_eq!(self.weights.len(), m, "weights must be one per edge");
        (0..m)
            .map(|e| {
                Vector3::new(
                    self.directions[3 * e],
                    self.directions[3 * e + 1],
                    self.directions[3 * e + 2],
                )
            })
            .collect()
    }

    /// The length statements, already zeroed where an edge states none, and
    /// the flag saying which edges did state one.
    fn stated_lengths(&self) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
        let m = self.n_edges();
        let mut ell = vec![0.0; m];
        let mut a = vec![0.0; m];
        let mut stated = vec![false; m];
        for e in 0..m {
            let l = self.lengths.map_or(f64::NAN, |v| v[e]);
            let w = self.length_weights.map_or(0.0, |v| v[e]);
            if l.is_finite() && w > 0.0 {
                ell[e] = l;
                a[e] = w;
                stated[e] = true;
            }
        }
        (ell, a, stated)
    }
}

/// Camera centres from pairwise baselines, by weighted linear averaging.
///
/// The scale gauge is `sum_ij w_ij d_ij . (c_j - c_i) = sum_ij w_ij` (the
/// centres are scaled so the weighted mean baseline projects to one) and the
/// shift gauge is `sum_j c_j = 0`. Both are exactly the freedoms the pairwise
/// readings cannot see, so fixing them adds nothing.
///
/// `rounds` rounds of reweighting follow the solve, each charging every edge's
/// direction residual and length slip against the median of that half over the
/// graph, so an edge stops carrying the solve when it is worse than the
/// graph's own typical edge. Reweighting never raises a weight above the one
/// the caller passed.
pub fn average_translations(graph: TranslationGraph<'_>, rounds: usize) -> TranslationAveraging {
    let n = graph.n_frames;
    let m = graph.n_edges();
    let d = graph.unpack();
    let ii: Vec<usize> = graph.edge_i.iter().map(|&v| v as usize).collect();
    let jj: Vec<usize> = graph.edge_j.iter().map(|&v| v as usize).collect();
    let (ell, base_a, stated) = graph.stated_lengths();
    let base_w: Vec<f64> = graph.weights.to_vec();
    let any_stated = stated.iter().any(|&s| s);
    let n_lengths = stated.iter().filter(|&&s| s).count();
    let base_w_sum: f64 = base_w.iter().sum();

    let mut w = base_w.clone();
    let mut a = base_a.clone();
    let mut centres = vec![[0.0f64; 3]; n];
    let mut lambda = vec![0.0f64; m];
    let mut residual = vec![0.0f64; m];
    let mut census = AveragingCensus {
        lam_max: 0.0,
        lam1_rel: f64::NAN,
        lam2_rel: f64::NAN,
        gap: f64::NAN,
        n_null: 0,
        n_loose: 0,
        n_free: 0,
        read_off_null: false,
        n_lengths,
        solved: false,
    };

    for _round in 0..rounds {
        let (big, vec) = assemble(n, &ii, &jj, &d, &w, &a, &ell);
        let (x, read) = spectrum(&big, &vec, n);
        census = AveragingCensus {
            n_lengths,
            solved: false,
            ..read
        };
        let along = vec.dot(&x);
        if !along.is_finite() || along == 0.0 {
            return TranslationAveraging {
                centres: Vec::new(),
                lambda: Vec::new(),
                residual: Vec::new(),
                census,
            };
        }
        census.solved = true;

        // The gauge scale, then the shift gauge: the centres are mean-centred.
        let k = base_w_sum / along;
        let mut mean = [0.0f64; 3];
        for f in 0..n {
            for c in 0..3 {
                centres[f][c] = k * x[3 * f + c];
                mean[c] += centres[f][c];
            }
        }
        if n > 0 {
            for v in mean.iter_mut() {
                *v /= n as f64;
            }
            for centre in centres.iter_mut() {
                for c in 0..3 {
                    centre[c] -= mean[c];
                }
            }
        }

        for e in 0..m {
            let b = Vector3::new(
                centres[jj[e]][0] - centres[ii[e]][0],
                centres[jj[e]][1] - centres[ii[e]][1],
                centres[jj[e]][2] - centres[ii[e]][2],
            );
            let lam = b.dot(&d[e]);
            lambda[e] = lam;
            residual[e] = (b - d[e] * lam).norm();
        }

        let floor = median_floor(&residual);
        for e in 0..m {
            w[e] = base_w[e] / (1.0 + residual[e] / floor);
        }
        if any_stated {
            let quad: f64 = (0..m).map(|e| base_a[e] * (ell[e] * ell[e])).sum();
            let scale = if quad > 0.0 {
                (0..m)
                    .map(|e| base_a[e] * (ell[e] * lambda[e]))
                    .sum::<f64>()
                    / quad
            } else {
                0.0
            };
            let slip: Vec<f64> = (0..m)
                .map(|e| {
                    if stated[e] {
                        (lambda[e] - scale * ell[e]).abs()
                    } else {
                        0.0
                    }
                })
                .collect();
            let of_stated: Vec<f64> = (0..m).filter(|&e| stated[e]).map(|e| slip[e]).collect();
            let floor = median_floor(&of_stated);
            for e in 0..m {
                a[e] = base_a[e] / (1.0 + slip[e] / floor);
            }
        }
    }

    if !census.solved {
        // No round ran, so nothing was read and nothing is stated.
        return TranslationAveraging {
            centres: Vec::new(),
            lambda: Vec::new(),
            residual: Vec::new(),
            census,
        };
    }
    TranslationAveraging {
        centres,
        lambda,
        residual,
        census,
    }
}

/// What the DIRECTIONS alone determine, before any length is read in.
///
/// The same form [`average_translations`] builds, at the weights the edges
/// came with and with the length half empty, decomposed once and not
/// reweighted. It states what the graph's geometry determines on its own,
/// which is a property of the capture rather than of a solve, and it is what
/// tells a colinear path (`n_free > 0`) from a general one. Any length the
/// graph carries is ignored.
pub fn direction_reading(graph: TranslationGraph<'_>) -> AveragingCensus {
    let n = graph.n_frames;
    let m = graph.n_edges();
    let d = graph.unpack();
    let ii: Vec<usize> = graph.edge_i.iter().map(|&v| v as usize).collect();
    let jj: Vec<usize> = graph.edge_j.iter().map(|&v| v as usize).collect();
    let zero = vec![0.0; m];
    let (big, vec) = assemble(n, &ii, &jj, &d, graph.weights, &zero, &zero);
    let (_x, read) = spectrum(&big, &vec, n);
    read
}

/// The graph's own scale for a reweighting: the median of the residuals, or
/// one where they have no positive median.
fn median_floor(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    let med = median(values);
    if med > 0.0 {
        med
    } else {
        1.0
    }
}

/// `(form, gauge gradient)` of the objective at these weights.
///
/// The block for an edge is `w P + a d d^T`: the part of the baseline
/// perpendicular to its direction, weighted by how much the direction is
/// trusted, plus the part along it, weighted by how much the length is. The
/// length statements are RELATIVE, so the scale that turns them into distances
/// is eliminated here, which leaves the form homogeneous and the overall scale
/// still a gauge. The shift gauge is applied by lifting the three translation
/// directions out of the spectrum, so the null space read below is never the
/// trivial one.
fn assemble(
    n: usize,
    ii: &[usize],
    jj: &[usize],
    d: &[Vector3<f64>],
    w: &[f64],
    a: &[f64],
    lengths: &[f64],
) -> (DMatrix<f64>, DVector<f64>) {
    let dim = 3 * n;
    let mut big = DMatrix::<f64>::zeros(dim, dim);
    let mut vec = DVector::<f64>::zeros(dim);
    let mut row = DVector::<f64>::zeros(dim);
    let identity = Matrix3::<f64>::identity();
    for e in 0..ii.len() {
        let de = d[e];
        let dd = de * de.transpose();
        let blk = (identity - dd) * w[e] + dd * a[e];
        let (si, sj) = (3 * ii[e], 3 * jj[e]);
        for r in 0..3 {
            for c in 0..3 {
                big[(si + r, si + c)] += blk[(r, c)];
                big[(sj + r, sj + c)] += blk[(r, c)];
                big[(si + r, sj + c)] -= blk[(r, c)];
                big[(sj + r, si + c)] -= blk[(r, c)];
            }
        }
        let al = a[e] * lengths[e];
        for c in 0..3 {
            vec[si + c] -= w[e] * de[c];
            vec[sj + c] += w[e] * de[c];
            row[si + c] -= al * de[c];
            row[sj + c] += al * de[c];
        }
    }
    let quad: f64 = (0..ii.len())
        .map(|e| a[e] * (lengths[e] * lengths[e]))
        .sum();
    if quad > 0.0 {
        for p in 0..dim {
            for q in 0..dim {
                big[(p, q)] -= row[p] * row[q] / quad;
            }
        }
    }
    // The shift gauge: `gauge` has one row per translation direction, so
    // `gauge^T gauge` is one where two coordinates share an axis. Lifting it by
    // the form's own mean diagonal keeps it out of the null space read below.
    let lift = big.trace() / dim.max(1) as f64;
    for p in 0..dim {
        for q in 0..dim {
            if p % 3 == q % 3 {
                big[(p, q)] += lift;
            }
        }
    }
    (big, vec)
}

/// The form's own reading of what it determines, and the centres it states.
///
/// The solve is `argmin x^T B x` under the scale gauge `vec . x = 1`. Where
/// the form is positive definite that is `B^-1 vec`, which is what a linear
/// solve returns; where it is singular the answer is its null space, and the
/// linear solve returns the one part of the answer the measurement does not
/// carry. The null space is only the constellation when the WHOLE of it is
/// shared: one dimension no frame owns half of. Anything else is a frame free
/// to slide, and there the answer is the range solution.
fn spectrum(
    big: &DMatrix<f64>,
    vec: &DVector<f64>,
    n_frames: usize,
) -> (DVector<f64>, AveragingCensus) {
    let dim = big.nrows();
    let eig = SymmetricEigen::new(big.clone());
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&p, &q| {
        eig.eigenvalues[p]
            .partial_cmp(&eig.eigenvalues[q])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(p.cmp(&q))
    });
    let lam: Vec<f64> = order.iter().map(|&k| eig.eigenvalues[k]).collect();
    let proj: Vec<f64> = order
        .iter()
        .map(|&k| eig.eigenvectors.column(k).dot(vec))
        .collect();

    let lam_max = lam.last().copied().unwrap_or(0.0);
    let tol = f64::EPSILON * dim as f64 * lam_max.max(0.0);
    let null: Vec<bool> = lam.iter().map(|&v| v <= tol).collect();
    let n_null = null.iter().filter(|&&v| v).count();

    // Ownership of the null space, with no basis chosen: the trace of a
    // frame's own three-by-three block of the projector is how many null
    // dimensions are that frame moving.
    let mut n_loose = 0usize;
    if n_null > 0 {
        for f in 0..n_frames {
            let mut own = 0.0;
            for (k, &is_null) in null.iter().enumerate() {
                if !is_null {
                    continue;
                }
                let col = eig.eigenvectors.column(order[k]);
                for c in 0..3 {
                    own += col[3 * f + c] * col[3 * f + c];
                }
            }
            if own > 0.5 {
                n_loose += 1;
            }
        }
    }

    let readable = n_null == 1 && n_loose == 0;
    let null_reach = (0..dim)
        .filter(|&k| null[k])
        .map(|k| proj[k].abs())
        .fold(0.0f64, f64::max);
    let mut x = DVector::<f64>::zeros(dim);
    if readable && null_reach > 0.0 {
        for k in 0..dim {
            if null[k] {
                x += eig.eigenvectors.column(order[k]) * proj[k];
            }
        }
    } else {
        for k in 0..dim {
            if !null[k] {
                x += eig.eigenvectors.column(order[k]) * (proj[k] / lam[k]);
            }
        }
    }

    let census = AveragingCensus {
        lam_max,
        lam1_rel: if lam_max > 0.0 {
            lam[0] / lam_max
        } else {
            f64::NAN
        },
        lam2_rel: if lam_max > 0.0 && dim > 1 {
            lam[1] / lam_max
        } else {
            f64::NAN
        },
        gap: if dim > 1 && lam[1] > 0.0 {
            lam[0] / lam[1]
        } else {
            f64::NAN
        },
        n_null,
        n_loose,
        n_free: n_null.saturating_sub(1),
        read_off_null: readable,
        n_lengths: 0,
        solved: false,
    };
    (x, census)
}

// ── Relative lengths ──────────────────────────────────────────────────────

/// One depth per `(edge, frame, point)`, flattened.
///
/// A pair's direction fixes its geometry up to how long its baseline is, so
/// the two-view depths that come out of the pair are in units of THAT
/// baseline. A point two edges both see from the same frame therefore has two
/// depths for one world distance, and their ratio is the ratio of the two
/// baselines. Only rows with positive depth on both rays belong here; the
/// caller supplies them already filtered.
#[derive(Debug, Clone, Copy)]
pub struct DepthRows<'a> {
    /// Which edge each row came from.
    pub edge_of_row: &'a [u32],
    /// Which frame each row's depth was read from.
    pub frame_of_row: &'a [u32],
    /// Which point each row saw.
    pub point_of_row: &'a [u32],
    /// The depth itself, in units of its own edge's baseline.
    pub depth_of_row: &'a [f64],
    /// How many edges the rows index.
    pub n_edges: usize,
}

/// One relative length per edge, with the scatter and the tie count behind it.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativeLengths {
    /// The relative baseline length of each edge, gauged to a median log
    /// length of zero. `NaN` for an edge that states none.
    pub lengths: Vec<f64>,
    /// Median absolute log residual of the edge's own rows. `NaN` for an edge
    /// with no row in the fit.
    pub scatter: Vec<f64>,
    /// How many of the edge's rows another edge also saw.
    pub n_tied: Vec<i64>,
}

/// The eliminated system in the edges alone, at one set of row weights.
///
/// Each point's world depth is at its own weighted mean over the rows that saw
/// it, so it never enters the solve; what is left is the residual of every row
/// against that mean, summed onto the edge it came from. The operator is
/// therefore a pass over the rows and never a matrix.
struct Fit<'a> {
    ee: &'a [usize],
    gid: &'a [usize],
    n_edge: usize,
    n_group: usize,
    ww: &'a [f64],
    /// Weight sum per group, floored at one so an empty group never divides.
    gsum: Vec<f64>,
    /// The diagonal of the operator, which preconditions the solve.
    diag: Vec<f64>,
}

impl<'a> Fit<'a> {
    fn new(
        ee: &'a [usize],
        gid: &'a [usize],
        n_edge: usize,
        n_group: usize,
        ww: &'a [f64],
    ) -> Self {
        let mut gsum = vec![0.0; n_group];
        for (r, &g) in gid.iter().enumerate() {
            gsum[g] += ww[r];
        }
        for v in gsum.iter_mut() {
            if *v <= 0.0 {
                *v = 1.0;
            }
        }
        let mut diag = vec![0.0; n_edge];
        for (r, &e) in ee.iter().enumerate() {
            let share = ww[r] / gsum[gid[r]];
            diag[e] += ww[r] * (1.0 - share);
        }
        for v in diag.iter_mut() {
            if *v <= 0.0 {
                *v = 1.0;
            }
        }
        Fit {
            ee,
            gid,
            n_edge,
            n_group,
            ww,
            gsum,
            diag,
        }
    }

    /// Each row less its own point's weighted mean over the rows.
    fn centred(&self, row: &[f64], out: &mut [f64]) {
        let mut mean = vec![0.0; self.n_group];
        for (r, &g) in self.gid.iter().enumerate() {
            mean[g] += self.ww[r] * row[r];
        }
        for (g, v) in mean.iter_mut().enumerate() {
            *v /= self.gsum[g];
        }
        for (r, &g) in self.gid.iter().enumerate() {
            out[r] = row[r] - mean[g];
        }
    }

    /// The operator: spread onto rows, centre, weight, gather back.
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        let spread: Vec<f64> = self.ee.iter().map(|&e| x[e]).collect();
        let mut centred = vec![0.0; spread.len()];
        self.centred(&spread, &mut centred);
        let mut out = vec![0.0; self.n_edge];
        for (r, &e) in self.ee.iter().enumerate() {
            out[e] += self.ww[r] * centred[r];
        }
        out
    }

    /// What the depths themselves put on the right-hand side.
    fn rhs(&self, vv: &[f64]) -> Vec<f64> {
        let mut centred = vec![0.0; vv.len()];
        self.centred(vv, &mut centred);
        let mut out = vec![0.0; self.n_edge];
        for (r, &e) in self.ee.iter().enumerate() {
            out[e] -= self.ww[r] * centred[r];
        }
        out
    }

    /// Per row, what the fit leaves of `x_edge + log z - log depth`.
    fn residual(&self, x: &[f64], vv: &[f64]) -> Vec<f64> {
        let sum: Vec<f64> = self
            .ee
            .iter()
            .enumerate()
            .map(|(r, &e)| x[e] + vv[r])
            .collect();
        let mut out = vec![0.0; sum.len()];
        self.centred(&sum, &mut out);
        out
    }
}

/// The least-squares lengths, by preconditioned conjugate gradient.
///
/// The operator is positive semi-definite with the constant vector in its null
/// space, which is the one freedom a set of RELATIVE lengths does not have;
/// the right-hand side is orthogonal to it, so the iteration never leaves the
/// complement. The result is gauged to a median log length of zero.
fn conjugate_gradient(
    fit: &Fit<'_>,
    rhs: &[f64],
    steps: usize,
    tol: f64,
    start: &[f64],
) -> Vec<f64> {
    let mut x = start.to_vec();
    let ax = fit.apply(&x);
    let mut r: Vec<f64> = (0..fit.n_edge).map(|e| rhs[e] - ax[e]).collect();
    let mut z: Vec<f64> = (0..fit.n_edge).map(|e| r[e] / fit.diag[e]).collect();
    let mut p = z.clone();
    let mut rz: f64 = (0..fit.n_edge).map(|e| r[e] * z[e]).sum();
    let rhs_max = rhs.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let bar = tol * if rhs_max == 0.0 { 1.0 } else { rhs_max };
    for _step in 0..steps {
        let r_max = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if r_max <= bar || rz <= 0.0 {
            break;
        }
        let ap = fit.apply(&p);
        let denom: f64 = (0..fit.n_edge).map(|e| p[e] * ap[e]).sum();
        if denom <= 0.0 {
            break;
        }
        let alpha = rz / denom;
        for e in 0..fit.n_edge {
            x[e] += alpha * p[e];
            r[e] -= alpha * ap[e];
            z[e] = r[e] / fit.diag[e];
        }
        let rz_next: f64 = (0..fit.n_edge).map(|e| r[e] * z[e]).sum();
        let beta = rz_next / rz;
        for e in 0..fit.n_edge {
            p[e] = z[e] + beta * p[e];
        }
        rz = rz_next;
    }
    let med = median(&x);
    for v in x.iter_mut() {
        *v -= med;
    }
    x
}

/// Relative baseline lengths, from the depths each pair's own solve implies.
///
/// The whole graph is one fit of
/// `log z(edge, frame, point) = D(frame, point) - x(edge)`, with `x` the log
/// baseline length and `D` the log world depth.
/// The `D` are eliminated rather than solved for, so the operator is a pass
/// over the rows and never a matrix.
///
/// A row is TIED when another edge saw the same point from the same frame;
/// only tied rows relate one baseline to another. An edge needs `min_tied`
/// tied rows before it states a length, and an edge without them comes back
/// `NaN` so that a centre solve constrains its direction only. Rows are
/// reweighted by their own absolute residual against the graph's median for
/// `rounds` rounds, so a wild depth stops carrying the fit.
pub fn relative_lengths(rows: DepthRows<'_>, rounds: usize, min_tied: usize) -> RelativeLengths {
    let n_row = rows.edge_of_row.len();
    assert_eq!(
        rows.frame_of_row.len(),
        n_row,
        "frame_of_row length mismatch"
    );
    assert_eq!(
        rows.point_of_row.len(),
        n_row,
        "point_of_row length mismatch"
    );
    assert_eq!(
        rows.depth_of_row.len(),
        n_row,
        "depth_of_row length mismatch"
    );
    let n_edge = rows.n_edges;
    let mut out = RelativeLengths {
        lengths: vec![f64::NAN; n_edge],
        scatter: vec![f64::NAN; n_edge],
        n_tied: vec![0i64; n_edge],
    };
    if n_row == 0 {
        return out;
    }

    let ee_all: Vec<usize> = rows.edge_of_row.iter().map(|&v| v as usize).collect();
    let vv_all: Vec<f64> = rows.depth_of_row.iter().map(|&z| z.ln()).collect();

    // Groups are `(frame, point)` pairs. Their labels never enter the
    // arithmetic (every reduction runs in row order), so they are handed out
    // in first-appearance order.
    let mut label: HashMap<(u32, u32), usize> = HashMap::new();
    let mut gid_all: Vec<usize> = Vec::with_capacity(n_row);
    for r in 0..n_row {
        let key = (rows.frame_of_row[r], rows.point_of_row[r]);
        let next = label.len();
        gid_all.push(*label.entry(key).or_insert(next));
    }
    let mut group_rows = vec![0usize; label.len()];
    for &g in &gid_all {
        group_rows[g] += 1;
    }

    // A row is tied when another edge saw the same point from the same frame.
    for r in 0..n_row {
        if group_rows[gid_all[r]] > 1 {
            out.n_tied[ee_all[r]] += 1;
        }
    }
    let has: Vec<bool> = out.n_tied.iter().map(|&t| t >= min_tied as i64).collect();
    if !ee_all.iter().any(|&e| has[e]) {
        return out;
    }

    let mut ee: Vec<usize> = Vec::new();
    let mut vv: Vec<f64> = Vec::new();
    let mut gid: Vec<usize> = Vec::new();
    let mut relabel: HashMap<usize, usize> = HashMap::new();
    for r in 0..n_row {
        if !has[ee_all[r]] {
            continue;
        }
        ee.push(ee_all[r]);
        vv.push(vv_all[r]);
        let next = relabel.len();
        gid.push(*relabel.entry(gid_all[r]).or_insert(next));
    }
    let n_group = relabel.len();

    let mut ww = vec![1.0; ee.len()];
    let mut x = vec![0.0; n_edge];
    let mut resid = vec![0.0; ee.len()];
    for _round in 0..rounds.max(1) {
        let fit = Fit::new(&ee, &gid, n_edge, n_group, &ww);
        let rhs = fit.rhs(&vv);
        x = conjugate_gradient(&fit, &rhs, CG_STEPS, CG_TOL, &x);
        resid = fit.residual(&x, &vv).iter().map(|v| v.abs()).collect();
        let med = median_floor(&resid);
        for (r, w) in ww.iter_mut().enumerate() {
            *w = 1.0 / (1.0 + resid[r] / med);
        }
    }

    // The scatter is each edge's own median absolute residual.
    let mut by_edge: Vec<Vec<f64>> = vec![Vec::new(); n_edge];
    for (r, &e) in ee.iter().enumerate() {
        by_edge[e].push(resid[r]);
    }
    for (e, rs) in by_edge.iter_mut().enumerate() {
        if !rs.is_empty() {
            out.scatter[e] = median_in_place(rs);
        }
    }
    for e in 0..n_edge {
        if has[e] {
            out.lengths[e] = x[e].exp();
        }
    }
    out
}

// ── Orientation ───────────────────────────────────────────────────────────

/// One world ray per observation, with the frame it was seen from and the
/// point it saw.
#[derive(Debug, Clone, Copy)]
pub struct OrientationRays<'a> {
    /// Camera centres, three components per frame.
    pub centres: &'a [f64],
    /// Unit world rays, three components per observation.
    pub rays_world: &'a [f64],
    /// Which point each ray saw.
    pub point_of_ray: &'a [u32],
    /// Which frame each ray was seen from.
    pub frame_of_ray: &'a [u32],
}

/// The orientation reading of one constellation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrientationReading {
    /// The parallax-weighted front-minus-behind vote, in radians. Negative
    /// says the constellation should be reflected.
    pub angw: f64,
    /// Observations seeing their point in front.
    pub obs_front: usize,
    /// Observations of every point the reading solved.
    pub obs_total: usize,
    /// [`Self::obs_front`] over [`Self::obs_total`].
    pub obs_frac: f64,
    /// [`Self::angw`] over [`Self::obs_total`].
    pub angw_per_obs: f64,
    /// How far [`Self::obs_frac`] sits from one half, doubled.
    pub margin_frac: f64,
    /// Points in front of every camera that saw them.
    pub points: usize,
    /// Points whose widest ray pair is inside the caller's angular bound.
    pub thin: usize,
    /// Points behind at least one camera that saw them.
    pub behind: usize,
}

/// The one bit the directions cannot state: which of the two mirror-image
/// constellations has the structure in front of the cameras.
///
/// Pairwise directions determine the constellation only up to the point
/// reflection `c -> -c`, because the form the averaging builds is quadratic in
/// the centres and does not contain them. For every point with two or more
/// observations the point is solved at the given centres by the least-squares
/// midpoint over its rays and each observation's depth along its own ray is
/// read; the reading is the parallax-weighted vote
/// `sum_points theta_widest * (n_front - n_behind)`. A point's cheirality
/// statement is worth exactly the parallax it was measured with: a point
/// inside `angular_bound` is a bearing whose depth sign is a coin toss, and it
/// contributes nothing beyond its own small angle.
///
/// The reading is exactly antisymmetric under `c -> -c`, so one pass describes
/// both orientations.
pub fn orientation_reading(rays: OrientationRays<'_>, angular_bound: f64) -> OrientationReading {
    let n_ray = rays.point_of_ray.len();
    assert_eq!(
        rays.frame_of_ray.len(),
        n_ray,
        "frame_of_ray length mismatch"
    );
    assert_eq!(
        rays.rays_world.len(),
        3 * n_ray,
        "rays_world must be n_ray * 3"
    );

    // Points are grouped in first-appearance order, and each point's rays keep
    // the caller's own row order, so every reduction is sequential in it.
    let mut label: HashMap<u32, usize> = HashMap::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    for r in 0..n_ray {
        let next = label.len();
        let slot = *label.entry(rays.point_of_ray[r]).or_insert(next);
        if slot == members.len() {
            members.push(Vec::new());
        }
        members[slot].push(r);
    }

    let mut dirs: Vec<Vector3<f64>> = Vec::new();
    let mut cs: Vec<Point3<f64>> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    for rowset in &members {
        if rowset.len() < 2 {
            continue;
        }
        offsets.push(dirs.len());
        for &r in rowset {
            dirs.push(Vector3::new(
                rays.rays_world[3 * r],
                rays.rays_world[3 * r + 1],
                rays.rays_world[3 * r + 2],
            ));
            let f = rays.frame_of_ray[r] as usize;
            cs.push(Point3::new(
                rays.centres[3 * f],
                rays.centres[3 * f + 1],
                rays.centres[3 * f + 2],
            ));
        }
    }
    offsets.push(dirs.len());
    let tris = triangulate_batch(&dirs, &cs, &offsets);

    let mut angw = 0.0f64;
    let mut obs_front = 0usize;
    let mut obs_total = 0usize;
    let (mut points, mut thin, mut behind) = (0usize, 0usize, 0usize);
    for (k, tri) in tris.iter().enumerate() {
        let (lo, hi) = (offsets[k], offsets[k + 1]);
        let widest = smallest_pairwise_cosine(&dirs[lo..hi])
            .clamp(-1.0, 1.0)
            .acos();
        let p = tri.point.coords;
        let mut front = 0usize;
        for r in lo..hi {
            if (p - cs[r].coords).dot(&dirs[r]) > 0.0 {
                front += 1;
            }
        }
        let total = hi - lo;
        obs_front += front;
        obs_total += total;
        angw += widest * (front as f64 - (total - front) as f64);
        if widest <= angular_bound {
            thin += 1;
        } else if front < total {
            behind += 1;
        } else {
            points += 1;
        }
    }
    let obs_frac = obs_front as f64 / obs_total.max(1) as f64;
    OrientationReading {
        angw,
        obs_front,
        obs_total,
        obs_frac,
        angw_per_obs: angw / obs_total.max(1) as f64,
        margin_frac: (2.0 * obs_frac - 1.0).abs(),
        points,
        thin,
        behind,
    }
}

#[cfg(test)]
mod tests;
