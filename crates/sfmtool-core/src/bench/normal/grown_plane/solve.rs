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
//! view's samples move with the parameters. The Jacobian is a forward
//! difference of that residual. A step is kept only when the true score, read
//! again from a fresh render, rises; otherwise the damping grows and the step
//! is tried again, and a stage that cannot rise keeps the parameters it was
//! given.

use nalgebra::{DMatrix, DVector, Matrix2};

use super::render::{robust_score, textured, zncc, Block, Samples, ViewRender};

/// The largest change of the tilt one step may make, tilt units: a step
/// past it is scaled back, so that the fit moves the plane in steps no larger
/// than about six degrees and stays near the basin it started in.
const MAX_TILT_STEP: f64 = 0.1;

/// The largest change of a shift one step may make, image px.
const MAX_SHIFT_STEP_PX: f64 = 2.0;

/// How many times one iteration raises the damping before it gives up.
const DAMPING_TRIES: u32 = 8;

/// How a fit iterates.
#[derive(Debug, Clone, Copy)]
pub(super) struct Iteration {
    pub(super) max_iterations: u32,
    pub(super) damping: f64,
    pub(super) min_improvement: f64,
    pub(super) tilt_step: f64,
    pub(super) shift_step: f64,
}

/// An objective over a parameter vector whose first two entries are a tilt
/// and whose others are shifts in image px.
pub(super) trait Objective {
    type Frozen;

    /// The score at `p`, higher better, and the residual frozen there; no
    /// frozen residual where nothing can be read.
    fn freeze(&self, p: &[f64]) -> (f64, Option<Self::Frozen>);

    /// The frozen residual read at `p`.
    fn residuals(&self, frozen: &Self::Frozen, p: &[f64], out: &mut Vec<f64>);

    /// The forward-difference Jacobian of the frozen residual at `p`, whose
    /// residual is `r0`, with step `steps[i]` for parameter `i`.
    fn jacobian(
        &self,
        frozen: &Self::Frozen,
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
            self.residuals(frozen, &q, &mut r);
            for (row, (a, b)) in r.iter().zip(r0).enumerate() {
                j[(row, i)] = (a - b) / h;
            }
        }
        j
    }
}

/// The Jacobian steps for `n` parameters, tilt first.
fn steps_for(n: usize, it: &Iteration) -> Vec<f64> {
    (0..n)
        .map(|i| if i < 2 { it.tilt_step } else { it.shift_step })
        .collect()
}

/// Fit `objective` from `p0`; the parameters reached, their score and how
/// many steps were taken. The score never falls below the start's.
pub(super) fn maximise<O: Objective>(
    objective: &O,
    p0: &[f64],
    it: &Iteration,
) -> (Vec<f64>, f64, u32) {
    let mut p = p0.to_vec();
    let (mut score, mut frozen) = objective.freeze(&p);
    let steps = steps_for(p.len(), it);
    let mut damping = it.damping;
    let mut r0 = Vec::new();
    let mut taken = 0;
    for _ in 0..it.max_iterations {
        let Some(fz) = frozen.as_ref() else {
            break;
        };
        r0.clear();
        objective.residuals(fz, &p, &mut r0);
        if r0.is_empty() {
            break;
        }
        let j = objective.jacobian(fz, &p, &r0, &steps);
        let a = j.transpose() * &j;
        let g = j.transpose() * DVector::from_column_slice(&r0);
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
        if gain < it.min_improvement {
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
pub(super) fn tilt_information<O: Objective>(
    objective: &O,
    p: &[f64],
    it: &Iteration,
) -> Matrix2<f64> {
    let (_, frozen) = objective.freeze(p);
    let Some(fz) = frozen else {
        return Matrix2::zeros();
    };
    let mut r0 = Vec::new();
    objective.residuals(&fz, p, &mut r0);
    if r0.is_empty() {
        return Matrix2::zeros();
    }
    let j = objective.jacobian(&fz, p, &r0, &steps_for(p.len(), it));
    let n = j.transpose() * &j;
    schur_tilt(&n)
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

impl Objective for ViewFit<'_, '_, '_> {
    type Frozen = Vec<Block>;

    fn freeze(&self, p: &[f64]) -> (f64, Option<Vec<Block>>) {
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
        (score, (!blocks.is_empty()).then_some(blocks))
    }

    fn residuals(&self, frozen: &Vec<Block>, p: &[f64], out: &mut Vec<f64>) {
        let (tilt, shift) = self.split(p);
        let samples = self.render.render(tilt, shift, &self.rows());
        for block in frozen {
            block.residuals(&samples, out);
        }
    }
}

/// Every joined view against the reference with one shared tilt: parameters
/// `[a, b, sx₁, sy₁, sx₂, sy₂, …]`, the score the mean of the views' robust
/// grid scores. Each view's residual is scaled by `1/√m` for `m` views, so
/// the sum of squares is `2 · (1 − mean score)`.
pub(super) struct JointFit<'r, 'g, 'a> {
    pub(super) views: Vec<ViewFit<'r, 'g, 'a>>,
}

impl JointFit<'_, '_, '_> {
    /// View `i`'s own parameters `[a, b, sx, sy]` from the joint `p`.
    fn own(p: &[f64], i: usize) -> [f64; 4] {
        [p[0], p[1], p[2 + 2 * i], p[3 + 2 * i]]
    }
}

impl Objective for JointFit<'_, '_, '_> {
    type Frozen = Vec<Option<Vec<Block>>>;

    fn freeze(&self, p: &[f64]) -> (f64, Option<Self::Frozen>) {
        let m = self.views.len() as f64;
        let mut total = 0.0;
        let mut frozen = Vec::with_capacity(self.views.len());
        for (i, view) in self.views.iter().enumerate() {
            let (s, f) = view.freeze(&Self::own(p, i));
            total += s;
            frozen.push(f.map(|blocks| scaled(blocks, 1.0 / m)));
        }
        let any = frozen.iter().any(Option::is_some);
        (total / m, any.then_some(frozen))
    }

    fn residuals(&self, frozen: &Self::Frozen, p: &[f64], out: &mut Vec<f64>) {
        for (i, (view, blocks)) in self.views.iter().zip(frozen).enumerate() {
            if let Some(blocks) = blocks {
                view.residuals(blocks, &Self::own(p, i), out);
            }
        }
    }

    /// Each view's own four columns, placed: the two tilt columns are shared
    /// and a view's shift columns touch only its own rows.
    fn jacobian(
        &self,
        frozen: &Self::Frozen,
        p: &[f64],
        r0: &[f64],
        steps: &[f64],
    ) -> DMatrix<f64> {
        let mut j = DMatrix::zeros(r0.len(), p.len());
        let mut row = 0;
        for (i, (view, blocks)) in self.views.iter().zip(frozen).enumerate() {
            let Some(blocks) = blocks else {
                continue;
            };
            let own = Self::own(p, i);
            let mut base = Vec::new();
            view.residuals(blocks, &own, &mut base);
            let own_steps = [steps[0], steps[1], steps[2 + 2 * i], steps[3 + 2 * i]];
            let local = view.jacobian(blocks, &own, &base, &own_steps);
            let cols = [0, 1, 2 + 2 * i, 3 + 2 * i];
            for r in 0..base.len() {
                for (c, &col) in cols.iter().enumerate() {
                    j[(row + r, col)] = local[(r, c)];
                }
            }
            row += base.len();
        }
        j
    }
}

/// `blocks` with each weight multiplied by `factor`.
fn scaled(blocks: Vec<Block>, factor: f64) -> Vec<Block> {
    blocks.into_iter().map(|b| b.weighted(factor)).collect()
}
