// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Levenberg–Marquardt fit [`grown_plane_normal`](super::grown_plane_normal)
//! runs at every stage, the objectives it fits, and the information a fit
//! carries about the tilt.
//!
//! Every objective is a score to maximise, a ZNCC or a weighted mean of
//! ZNCCs, and a residual frozen at the current parameters whose sum of squares
//! is `2 · (1 − score)` there ([`Block`]): the samples, the channels and the
//! cells' robust weights are fixed at the linearisation point, and only the
//! view's samples move with the parameters (both views' in the pair fit, which
//! normalises both again at every evaluation). The Jacobian is analytic, read
//! at the render the residual was frozen at: a sample's value moves with an
//! image shift by its image gradient, and with the tilt by that gradient
//! times how its pixel moves as the plane turns, and both go through the
//! block's normalisation by the chain rule ([`Block::add_derivative`]), so a
//! Jacobian costs no render. The fit's steps read the gradient from the
//! level's Sobel gradient images ([`GradientRead::Sobel`]), which is
//! continuous across pixel edges; the information about the tilt reads the
//! derivative of the bilinear read itself ([`GradientRead::Bilinear`]), the
//! local slope of the score the fit maximises. A step is kept only when the true score, read
//! again from a fresh render, rises; otherwise the damping grows and the step
//! is tried again, and a stage that cannot rise keeps the parameters it was
//! given. The render that scores a kept step also gives the residual the next
//! iteration linearises at, so a kept point is rendered once. A fit stops
//! once a kept step raises the score by less than `min_improvement`, or by
//! less than [`MIN_RELATIVE_GAIN`] of what its first kept step raised it.

use nalgebra::{DMatrix, DVector, Matrix2};

use super::render::{
    robust_score, textured, zncc, Block, GradientRead, Gradients, Samples, ViewRender,
};

/// The largest change of the tilt one step may make, tilt units: a step
/// past it is scaled back, so that the fit moves the plane in steps no larger
/// than about six degrees and stays near the basin it started in.
const MAX_TILT_STEP: f64 = 0.1;

/// The largest change of a shift one step may make, image px.
const MAX_SHIFT_STEP_PX: f64 = 2.0;

/// How many damped steps one iteration tries before the fit gives up. More
/// than half of all fits end on an iteration that cannot rise, and each try
/// costs a render; three tries kept the quality that eight gave on the
/// ground-truth sweep.
pub(super) const DAMPING_TRIES: u32 = 3;

/// A fit stops once a kept step raises the score by less than this share of
/// what the fit's first kept step raised it. A step is kept only when the
/// score rises, so the first step's gain is positive, and the rule first
/// applies to the second step: a fit whose first step does nearly all the
/// work stops after its second. On the ground-truth sweep, run alone, it took
/// 9% less time than three damping tries without it and still had 55 of the
/// 63 judged points right, as they did.
pub(super) const MIN_RELATIVE_GAIN: f64 = 0.01;

/// How a fit iterates.
#[derive(Debug, Clone, Copy)]
pub(super) struct Iteration {
    pub(super) max_iterations: u32,
    pub(super) damping: f64,
    pub(super) min_improvement: f64,
}

/// A residual frozen at a point, and its value there.
pub(super) type Linearised<F> = (F, Vec<f64>);

/// An objective over a parameter vector whose first two entries are a tilt
/// and whose others are shifts in image px.
pub(super) trait Objective {
    type Frozen;

    /// The score at `p`, higher better, the residual frozen there and that
    /// residual's value at `p`, read from the same render as the score; no
    /// frozen residual where nothing can be read. The value is the one
    /// `residuals`, a test-only method, gives at `p`, bit for bit.
    fn freeze(&self, p: &[f64]) -> (f64, Option<Linearised<Self::Frozen>>);

    /// The frozen residual read at `p`, from a render of its own. The fit
    /// itself never calls it; it is what [`forward_difference`] differences.
    #[cfg(test)]
    fn residuals(&self, frozen: &Self::Frozen, p: &[f64], out: &mut Vec<f64>);

    /// The Jacobian of the frozen residual at `p`, the point it was frozen
    /// at, whose residual is `r0`, with each sample's image gradient read as
    /// `read` says: one row per entry of `r0`, one column per parameter.
    fn jacobian(
        &self,
        frozen: &Self::Frozen,
        p: &[f64],
        r0: &[f64],
        read: GradientRead,
    ) -> DMatrix<f64>;
}

/// The forward-difference Jacobian of `objective`'s frozen residual at `p`,
/// whose residual is `r0`, with step `steps[i]` for parameter `i`: one render
/// per parameter. The check the analytic Jacobians are tested against.
#[cfg(test)]
pub(super) fn forward_difference<O: Objective>(
    objective: &O,
    frozen: &O::Frozen,
    p: &[f64],
    r0: &[f64],
    steps: &[f64],
) -> DMatrix<f64> {
    let mut j = DMatrix::zeros(r0.len(), p.len());
    let mut r = Vec::with_capacity(r0.len());
    for (i, &h) in steps.iter().enumerate() {
        let mut q = p.to_vec();
        q[i] += h;
        r.clear();
        objective.residuals(frozen, &q, &mut r);
        for (row, (a, b)) in r.iter().zip(r0).enumerate() {
            j[(row, i)] = (a - b) / h;
        }
    }
    j
}

/// How sample `k`'s value in channel `c` of a view moves with the fit's
/// parameters, written to `out`: through `pixel`, how its pixel moves with
/// the tilt, into the first two entries when the tilt is free, then through
/// its gradient alone into the next two when the view's shift is free, since
/// a shift moves the pixel by itself.
fn value_derivative(
    gradients: &Gradients,
    pixel: Option<&[[[f64; 2]; 2]]>,
    shift_from: Option<usize>,
    k: usize,
    c: usize,
    out: &mut [f64],
) {
    let g = gradients.at(k, c);
    if let Some(pixel) = pixel {
        let m = pixel[k];
        out[0] = g[0] * m[0][0] + g[1] * m[1][0];
        out[1] = g[0] * m[0][1] + g[1] * m[1][1];
    }
    if let Some(i) = shift_from {
        out[i] = g[0];
        out[i + 1] = g[1];
    }
}

/// Fit `objective` from `p0`; the parameters reached, their score and how
/// many steps were taken. The score never falls below the start's. The fit
/// stops after a kept step that gains less than `it.min_improvement`, or less
/// than [`MIN_RELATIVE_GAIN`] of the first kept step's gain.
pub(super) fn maximise<O: Objective>(
    objective: &O,
    p0: &[f64],
    it: &Iteration,
) -> (Vec<f64>, f64, u32) {
    let mut p = p0.to_vec();
    let (mut score, mut frozen) = objective.freeze(&p);
    let mut damping = it.damping;
    let mut taken = 0;
    let mut first_gain = None;
    for _ in 0..it.max_iterations {
        let Some((fz, r0)) = frozen.as_ref() else {
            break;
        };
        if r0.is_empty() {
            break;
        }
        let j = objective.jacobian(fz, &p, r0, GradientRead::Sobel);
        let a = j.transpose() * &j;
        let g = j.transpose() * DVector::from_column_slice(r0);
        let mut accepted = None;
        for _ in 0..DAMPING_TRIES {
            let Some(delta) = damped_step(&a, &g, damping) else {
                damping *= 10.0;
                continue;
            };
            let q: Vec<f64> = p
                .iter()
                .zip(capped(delta).iter())
                .map(|(x, d)| x + d)
                .collect();
            let (s, f) = objective.freeze(&q);
            if s > score + 1e-12 {
                damping = (damping / 10.0).max(1e-9);
                accepted = Some((q, s, f));
                break;
            }
            damping *= 10.0;
        }
        let Some((q, s, f)) = accepted else {
            break;
        };
        let gain = s - score;
        (p, score, frozen) = (q, s, f);
        taken += 1;
        let first = *first_gain.get_or_insert(gain);
        if gain < it.min_improvement || gain < MIN_RELATIVE_GAIN * first {
            break;
        }
    }
    (p, score, taken)
}

/// The Levenberg–Marquardt step `−(A + λ·diag(A))⁻¹ g`, with a small floor on
/// the diagonal so a parameter the residual does not see stays put.
fn damped_step(a: &DMatrix<f64>, g: &DVector<f64>, damping: f64) -> Option<DVector<f64>> {
    let mut m = a.clone();
    let scale = (0..a.nrows())
        .map(|i| a[(i, i)])
        .fold(0.0, f64::max)
        .max(1e-12);
    for i in 0..a.nrows() {
        m[(i, i)] += damping * a[(i, i)] + 1e-9 * scale;
    }
    let step = m.cholesky()?.solve(&(-g));
    step.iter().all(|v| v.is_finite()).then_some(step)
}

/// `delta` scaled back so that no tilt entry moves more than
/// [`MAX_TILT_STEP`] and no shift entry more than [`MAX_SHIFT_STEP_PX`].
fn capped(delta: DVector<f64>) -> DVector<f64> {
    let mut factor: f64 = 1.0;
    for (i, d) in delta.iter().enumerate() {
        let cap = if i < 2 {
            MAX_TILT_STEP
        } else {
            MAX_SHIFT_STEP_PX
        };
        if d.abs() > cap {
            factor = factor.min(cap / d.abs());
        }
    }
    delta * factor
}

/// The information the objective carries about the tilt at `p`: the normal
/// matrix `JᵀJ` of its frozen residual with every shift eliminated (the
/// Schur complement), which is minus the Hessian of the score in the tilt
/// with the shifts fitted again; zero where nothing can be read.
pub(super) fn tilt_information<O: Objective>(objective: &O, p: &[f64]) -> Matrix2<f64> {
    normal_matrix(objective, p).map_or_else(Matrix2::zeros, |n| schur_tilt(&n))
}

/// The tilt block of `n` with the other parameters eliminated:
/// `N_tt − N_ts N_ss⁻¹ N_st`, symmetrised, its negative eigenvalues set to 0.
pub(super) fn schur_tilt(n: &DMatrix<f64>) -> Matrix2<f64> {
    let k = n.nrows();
    let ntt = Matrix2::new(n[(0, 0)], n[(0, 1)], n[(1, 0)], n[(1, 1)]);
    let info = if k > 2 {
        let nss = n.view((2, 2), (k - 2, k - 2)).into_owned();
        let nst = n.view((2, 0), (k - 2, 2)).into_owned();
        let scale = (0..k - 2)
            .map(|i| nss[(i, i)])
            .fold(0.0, f64::max)
            .max(1e-12);
        let mut reg = nss.clone();
        for i in 0..k - 2 {
            reg[(i, i)] += 1e-12 * scale;
        }
        match reg.cholesky() {
            Some(chol) => {
                let x = chol.solve(&nst);
                let t = nst.transpose() * x;
                ntt - Matrix2::new(t[(0, 0)], t[(0, 1)], t[(1, 0)], t[(1, 1)])
            }
            None => ntt,
        }
    } else {
        ntt
    };
    let sym = (info + info.transpose()) * 0.5;
    let eigen = sym.symmetric_eigen();
    let clipped = eigen.eigenvalues.map(|v| v.max(0.0));
    eigen.eigenvectors * Matrix2::from_diagonal(&clipped) * eigen.eigenvectors.transpose()
}

/// What one view's fit reads: a window's plain ZNCC, or the whole grid's
/// robust cell score.
#[derive(Clone)]
pub(super) enum Reading {
    /// The mean-channel ZNCC over these samples.
    Window(Vec<usize>),
    /// The robust score over the grid's live cells.
    Grid,
}

/// One view against the reference: parameters `[a, b, sx, sy]`, or
/// `[sx, sy]` with the tilt held at `held`.
pub(super) struct ViewFit<'r, 'g, 'a> {
    pub(super) render: &'r ViewRender<'g, 'a>,
    pub(super) reference: &'r Samples,
    /// Which reference cells are textured, for [`Reading::Grid`].
    pub(super) live: &'r [bool; 9],
    pub(super) reading: Reading,
    pub(super) held: Option<[f64; 2]>,
    pub(super) min_coverage: f64,
}

impl ViewFit<'_, '_, '_> {
    /// The tilt and shift `p` stands for.
    fn split(&self, p: &[f64]) -> ([f64; 2], [f64; 2]) {
        match self.held {
            Some(tilt) => (tilt, [p[0], p[1]]),
            None => ([p[0], p[1]], [p[2], p[3]]),
        }
    }

    /// The samples this fit renders.
    fn rows(&self) -> Vec<usize> {
        match &self.reading {
            Reading::Window(rows) => rows.clone(),
            Reading::Grid => self.render.grid.all(),
        }
    }

    /// The score at `tilt` and `shift`.
    pub(super) fn score_at(&self, tilt: [f64; 2], shift: [f64; 2]) -> f64 {
        let samples = self.render.render(tilt, shift, &self.rows());
        self.score_of(&samples).0
    }

    /// The score of `samples`, with each cell's robust weight for
    /// [`Reading::Grid`].
    fn score_of(&self, samples: &Samples) -> (f64, [f64; 9]) {
        match &self.reading {
            Reading::Window(rows) => (
                zncc(self.reference, samples, rows, self.min_coverage),
                [0.0; 9],
            ),
            Reading::Grid => {
                let cells = cell_znccs(
                    self.render,
                    self.reference,
                    self.live,
                    samples,
                    self.min_coverage,
                );
                robust_score(&cells)
            }
        }
    }
}

/// Each cell's ZNCC in `samples`, `None` for a cell the reference is flat in.
pub(super) fn cell_znccs(
    render: &ViewRender<'_, '_>,
    reference: &Samples,
    live: &[bool; 9],
    samples: &Samples,
    min_coverage: f64,
) -> [Option<f64>; 9] {
    std::array::from_fn(|k| {
        live[k].then(|| zncc(reference, samples, &render.grid.cells[k], min_coverage))
    })
}

/// Which of the grid's cells the reference is textured in.
pub(super) fn live_cells(render_cells: &[Vec<usize>; 9], reference: &Samples) -> [bool; 9] {
    std::array::from_fn(|k| textured(reference, &render_cells[k]))
}

/// A view's residual frozen at one render: its blocks, and the render's
/// samples, which the Jacobian there normalises through.
pub(super) struct ViewFrozen {
    blocks: Vec<Block>,
    samples: Samples,
}

impl Objective for ViewFit<'_, '_, '_> {
    type Frozen = ViewFrozen;

    fn freeze(&self, p: &[f64]) -> (f64, Option<Linearised<ViewFrozen>>) {
        let (tilt, shift) = self.split(p);
        let samples = self.render.render(tilt, shift, &self.rows());
        let (score, weights) = self.score_of(&samples);
        let blocks: Vec<Block> = match &self.reading {
            Reading::Window(rows) => {
                Block::freeze(self.reference, &samples, rows, 1.0, self.min_coverage)
                    .into_iter()
                    .collect()
            }
            Reading::Grid => {
                let total: f64 = weights.iter().sum();
                if total <= 0.0 {
                    Vec::new()
                } else {
                    (0..9)
                        .filter(|&k| weights[k] > 0.0)
                        .filter_map(|k| {
                            Block::freeze(
                                self.reference,
                                &samples,
                                &self.render.grid.cells[k],
                                weights[k] / total,
                                self.min_coverage,
                            )
                        })
                        .collect()
                }
            }
        };
        if blocks.is_empty() {
            return (score, None);
        }
        let mut r0 = Vec::new();
        for block in &blocks {
            block.residuals(&samples, &mut r0);
        }
        let frozen = ViewFrozen { blocks, samples };
        (score, Some((frozen, r0)))
    }

    #[cfg(test)]
    fn residuals(&self, frozen: &ViewFrozen, p: &[f64], out: &mut Vec<f64>) {
        let (tilt, shift) = self.split(p);
        let samples = self.render.render(tilt, shift, &self.rows());
        for block in &frozen.blocks {
            block.residuals(&samples, out);
        }
    }

    /// The view's samples move with the shift by their gradient and, when
    /// the tilt is free, with the tilt by their gradient through how their
    /// pixels move; the reference is fixed, so the residual moves as minus
    /// the view's normalised samples.
    fn jacobian(
        &self,
        frozen: &ViewFrozen,
        p: &[f64],
        r0: &[f64],
        read: GradientRead,
    ) -> DMatrix<f64> {
        let (tilt, shift) = self.split(p);
        let gradients = self.render.gradients(tilt, shift, &self.rows(), read);
        let pixel = self
            .held
            .is_none()
            .then(|| self.render.pixel_tilt_derivatives(tilt, &self.rows()));
        let shift_from = Some(if self.held.is_some() { 0 } else { 2 });
        let mut j = DMatrix::zeros(r0.len(), p.len());
        let mut row = 0;
        for block in &frozen.blocks {
            row += block.add_derivative(
                &frozen.samples,
                -1.0,
                |k, c, out| value_derivative(&gradients, pixel.as_deref(), shift_from, k, c, out),
                &mut j,
                row,
            );
        }
        j
    }
}

/// Two views against each other through one plane: parameters `[a, b, rx,
/// ry]`, the first view at its held shift and the second at its held shift
/// plus the relative shift `(rx, ry)`. The score is the robust cell score of
/// the ZNCC between the two renders over the cells the reference is textured
/// in. Both renders move with the tilt, so the residual normalises both sides
/// afresh at every evaluation; only the samples, channels and cell weights
/// are frozen.
pub(super) struct PairFit<'r, 'g, 'a> {
    pub(super) first: &'r ViewRender<'g, 'a>,
    pub(super) second: &'r ViewRender<'g, 'a>,
    pub(super) live: &'r [bool; 9],
    /// Each view's shift against the reference, held, image px.
    pub(super) shifts: [[f64; 2]; 2],
    pub(super) min_coverage: f64,
}

impl PairFit<'_, '_, '_> {
    /// The second view's shift at `p`: its held shift plus the relative
    /// shift.
    fn second_shift(&self, p: &[f64]) -> [f64; 2] {
        [self.shifts[1][0] + p[2], self.shifts[1][1] + p[3]]
    }

    /// Both views' renders over the whole grid at `p`.
    fn renders(&self, p: &[f64]) -> (Samples, Samples) {
        let tilt = [p[0], p[1]];
        let rows = self.first.grid.all();
        let a = self.first.render(tilt, self.shifts[0], &rows);
        let b = self.second.render(tilt, self.second_shift(p), &rows);
        (a, b)
    }

    /// The robust score of `a` against `b` and each cell's weight.
    fn score_of(&self, a: &Samples, b: &Samples) -> (f64, [f64; 9]) {
        let cells: [Option<f64>; 9] = std::array::from_fn(|k| {
            self.live[k].then(|| zncc(a, b, &self.first.grid.cells[k], self.min_coverage))
        });
        robust_score(&cells)
    }
}

/// A pair's residual frozen at one render of each view: its blocks, and each
/// render's samples.
pub(super) struct PairFrozen {
    blocks: Vec<Block>,
    first: Samples,
    second: Samples,
}

impl Objective for PairFit<'_, '_, '_> {
    type Frozen = PairFrozen;

    fn freeze(&self, p: &[f64]) -> (f64, Option<Linearised<PairFrozen>>) {
        let (first, second) = self.renders(p);
        let (a, b) = (&first, &second);
        let (score, weights) = self.score_of(a, b);
        let total: f64 = weights.iter().sum();
        let blocks: Vec<Block> = if total <= 0.0 {
            Vec::new()
        } else {
            (0..9)
                .filter(|&k| weights[k] > 0.0)
                .filter_map(|k| {
                    Block::freeze(
                        a,
                        b,
                        &self.first.grid.cells[k],
                        weights[k] / total,
                        self.min_coverage,
                    )
                })
                .collect()
        };
        if blocks.is_empty() {
            return (score, None);
        }
        let mut r0 = Vec::new();
        for block in &blocks {
            block.pair_residuals(a, b, &mut r0);
        }
        let frozen = PairFrozen {
            blocks,
            first,
            second,
        };
        (score, Some((frozen, r0)))
    }

    #[cfg(test)]
    fn residuals(&self, frozen: &PairFrozen, p: &[f64], out: &mut Vec<f64>) {
        let (a, b) = self.renders(p);
        for block in &frozen.blocks {
            block.pair_residuals(&a, &b, out);
        }
    }

    /// Both renders move with the tilt and only the second with the
    /// relative shift; the residual is the first's normalised samples less
    /// the second's, each normalised over the block afresh.
    fn jacobian(
        &self,
        frozen: &PairFrozen,
        p: &[f64],
        r0: &[f64],
        read: GradientRead,
    ) -> DMatrix<f64> {
        let tilt = [p[0], p[1]];
        let rows = self.first.grid.all();
        let first_gradients = self.first.gradients(tilt, self.shifts[0], &rows, read);
        let second_gradients = self
            .second
            .gradients(tilt, self.second_shift(p), &rows, read);
        let first_pixel = self.first.pixel_tilt_derivatives(tilt, &rows);
        let second_pixel = self.second.pixel_tilt_derivatives(tilt, &rows);
        let mut j = DMatrix::zeros(r0.len(), p.len());
        let mut row = 0;
        for block in &frozen.blocks {
            block.add_derivative(
                &frozen.first,
                1.0,
                |k, c, out| value_derivative(&first_gradients, Some(&first_pixel), None, k, c, out),
                &mut j,
                row,
            );
            row += block.add_derivative(
                &frozen.second,
                -1.0,
                |k, c, out| {
                    value_derivative(&second_gradients, Some(&second_pixel), Some(2), k, c, out)
                },
                &mut j,
                row,
            );
        }
        j
    }
}

/// The normal matrix `JᵀJ` of `objective`'s frozen residual at `p`, over every
/// parameter, with the derivative of the bilinear read as each sample's
/// gradient; `None` where nothing can be read.
pub(super) fn normal_matrix<O: Objective>(objective: &O, p: &[f64]) -> Option<DMatrix<f64>> {
    let (_, frozen) = objective.freeze(p);
    let (fz, r0) = frozen?;
    if r0.is_empty() {
        return None;
    }
    let j = objective.jacobian(&fz, p, &r0, GradientRead::Bilinear);
    Some(j.transpose() * &j)
}

/// How the shifts of a fit whose normal matrix is `n`, over `[a, b, s…]`,
/// move when the tilt moves by `dt` and the shifts are fitted again:
/// `−N_ss⁻¹ N_st · dt`, the Gauss–Newton minimiser of the shifts at the new
/// tilt. Zero where `N_ss` cannot be inverted.
pub(super) fn shift_response(n: &DMatrix<f64>, dt: [f64; 2]) -> Vec<f64> {
    let k = n.nrows();
    if k <= 2 {
        return Vec::new();
    }
    let nss = n.view((2, 2), (k - 2, k - 2)).into_owned();
    let nst = n.view((2, 0), (k - 2, 2)).into_owned();
    let scale = (0..k - 2)
        .map(|i| nss[(i, i)])
        .fold(0.0, f64::max)
        .max(1e-12);
    let mut reg = nss;
    for i in 0..k - 2 {
        reg[(i, i)] += 1e-12 * scale;
    }
    match reg.cholesky() {
        Some(chol) => {
            let rhs = nst * DVector::from_column_slice(&dt);
            let x = chol.solve(&rhs);
            x.iter().map(|v| -v).collect()
        }
        None => vec![0.0; k - 2],
    }
}
