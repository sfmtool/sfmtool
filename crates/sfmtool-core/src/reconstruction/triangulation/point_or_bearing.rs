// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Whether a track's rays ask for a depth: a bearing fit, a point fit, and the
//! likelihood-ratio test between them.
//!
//! A track is stored either as a finite point or as a bearing, a direction with
//! no depth. Both are fitted to the same residuals. The residual of ray `i` is
//! `Wᵢ (I − dᵢdᵢᵀ) m`: the component of the model's unit direction `m` from
//! that camera perpendicular to the ray `dᵢ`, which is the sine of the angle
//! between them, through the ray's 2×3 noise weight `Wᵢ`. The weight maps a
//! world-frame change of direction to pixels over the pixel noise, so the
//! residual is in units of the noise, and costs (sums of squared residuals)
//! are χ²-scaled. [`observed_ray`] builds a ray and its weight from a camera
//! model, a pose and a pixel; [`isotropic_ray_weight`] builds a weight from a
//! scalar angular noise.
//!
//! The point model is an anchor `a` (by default the centroid of the observing
//! camera centres), a unit direction `u` and an inverse depth `ρ ≥ 0`, with
//! the point at `a + u / ρ`. The direction from camera `i` to the point is
//! proportional to `u + ρ (a − cᵢ)`, which is smooth through `ρ = 0`, where it
//! is `u` from every camera: the bearing model. The bearing is the point model
//! with `ρ` held at its boundary, so the point fit can only lower the cost, and
//! the cost reduction `Λ` is a likelihood-ratio statistic. For a track truly at
//! infinity with Gaussian noise of the stated weight,
//! `P(Λ > t) = ½ P(χ²₁ > t)`.
//!
//! Two tiers, because deciding and placing cost different amounts:
//!
//! - [`bearing_score`] decides. The bearing is closed form (the eigenvector of
//!   the smallest eigenvalue of `M = Σ Aᵢᵀ Aᵢ`, `Aᵢ = Wᵢ (I − dᵢdᵢᵀ)`).
//!   `depth_score` is the score statistic `gᵀ H⁻¹ g` of the point model at
//!   `(bearing, ρ = 0)`, which approximates `Λ` closely where the rays are
//!   close to parallel, and `midpoint_bound` is the cost reduction at the
//!   weighted linear midpoint, a lower bound on `Λ` that decides the
//!   wide-angle tracks the score cannot. [`is_finite`] reads the verdict.
//! - [`fit_point_and_bearing`] places. It runs the iterative point fit and
//!   returns `Λ` exactly, with an optional soft-L1 loss.
//!
//! Tracks are flattened CSR-style, as for [`super::triangulate_batch`].
//!
//! See `specs/core/reconstruction/batch-triangulation-api.md` § "Point or
//! bearing" for the design.

mod ray_weight;

pub use ray_weight::{isotropic_ray_weight, isotropic_ray_weights, observed_ray, ObservedRay};

use nalgebra::{Matrix2, Matrix2x3, Matrix3, Point3, SymmetricEigen, Vector2, Vector3};
use rayon::prelude::*;

/// The value of `depth_score` (and of `Λ`) above which [`is_finite`] calls a
/// track a finite point. Under the half-χ²₁ law of a true bearing, a 1 in 3.5
/// million chance of calling it finite.
pub const DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD: f64 = 25.0;

/// The default [`PointBearingFitOptions::soft_l1_scale`], in noise units.
pub const DEFAULT_SOFT_L1_SCALE: f64 = 3.0;

/// The default [`PointBearingFitOptions::max_iterations`].
pub const DEFAULT_POINT_FIT_MAX_ITERATIONS: usize = 20;

/// The bearing that best explains one track's rays, and how strongly the rays
/// ask for a depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BearingScore {
    /// The unit direction minimising `Σ ‖Aᵢ u‖²`, `Aᵢ = Wᵢ (I − dᵢdᵢᵀ)`: the
    /// eigenvector of the smallest eigenvalue of `M = Σ Aᵢᵀ Aᵢ`, signed to
    /// point along the weighted mean ray.
    pub bearing: Vector3<f64>,
    /// The bearing's cost in units of the noise, which is the smallest
    /// eigenvalue of `M`. It is summed from the rays' residuals rather than read
    /// off the eigensolve, which is the more precise of the two. `Λ` can never
    /// exceed it, so `bearing_cost < threshold` decides a bearing exactly.
    pub bearing_cost: f64,
    /// The score statistic for adding an inverse depth at the bearing,
    /// `gᵀ H⁻¹ g` with `g` and `H` the Gauss-Newton gradient and normal matrix
    /// of the point model at `(bearing, ρ = 0)`. 0 when the Gauss-Newton step
    /// in `ρ` is not positive (the rays converge behind the cameras, or the
    /// bearing is no description of rays spread over a wide angle) and when
    /// the camera centres give no leverage on `ρ`.
    pub depth_score: f64,
    /// `bearing_cost` less the cost at the weighted linear midpoint
    /// `(Σ Aᵢᵀ Aᵢ)⁻¹ Σ Aᵢᵀ Aᵢ cᵢ`, when that point lies in front of every
    /// camera, and 0 otherwise. A lower bound on `Λ`, so it can only call a
    /// track finite that is finite. It is what decides tracks whose rays
    /// spread over a wide angle (an arc of cameras around an object), where
    /// the bearing is a poor model, `ρ = 0` is far from the point, and the
    /// score linearised there says nothing.
    pub midpoint_bound: f64,
    /// [`Self::bearing`] lies in front of every observing camera:
    /// `dᵢ · bearing > 0` for every ray. The sine residual cannot tell a ray
    /// from its opposite, so rays along `u` and `−u` fit the bearing `u`
    /// exactly and give it no cost. When this is false the bearing does not
    /// describe every sighting, and a classifier does not store it on the
    /// strength of a bearing verdict: it treats the track as it treats a finite
    /// point behind a camera, pruning the sightings the bearing is behind or
    /// dropping the track.
    pub bearing_in_front_of_all_cameras: bool,
    /// The number of rays used: those with a finite non-zero direction, a
    /// finite centre, and a finite non-zero weight whose squared entries do
    /// not overflow.
    pub num_views: usize,
}

/// Both fits of one track and the exact statistic that compares them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointBearingFit {
    /// The bearing minimising the bearing model's cost. With
    /// [`PointBearingFitOptions::soft_l1_scale`] unset this is
    /// [`BearingScore::bearing`]; with it set, that bearing refined under the
    /// robust loss.
    pub bearing: Vector3<f64>,
    /// The bearing model's cost at [`Self::bearing`], in noise units (under the
    /// robust loss when one is set).
    pub bearing_cost: f64,
    /// The point model's anchor: the caller's, or the centroid of the
    /// observing camera centres.
    pub anchor: Point3<f64>,
    /// The point model's unit direction `u` from the anchor.
    pub direction: Vector3<f64>,
    /// The point model's inverse depth `ρ ≥ 0`. 0 means the best point is the
    /// bearing `direction`.
    pub inverse_depth: f64,
    /// `anchor + direction / inverse_depth`, or `None` when `inverse_depth` is
    /// 0 and the point is at infinity.
    pub point: Option<Point3<f64>>,
    /// The point model's cost at its fit, `≤ bearing_cost` (under the robust
    /// loss when one is set).
    pub point_cost: f64,
    /// `bearing_cost − point_cost`, `≥ 0`: `Λ` under the loss the fit used,
    /// taken at the optimum the fit reaches. It can fall slightly short of the
    /// best point in front of every camera (232 of 30,000 adversarial fits,
    /// at worst by 0.08%, none changing the verdict); decisions come from
    /// `BearingScore`'s score and bound. Only plain least squares
    /// (`soft_l1_scale: None`) gives the statistic that `depth_score`
    /// approximates and the half-χ²₁ law describes; the robust value is
    /// smaller wherever a residual is past the loss's scale. Meaningless when
    /// [`Self::in_front_of_all_cameras`] is false.
    pub depth_likelihood_ratio: f64,
    /// The fitted point lies in front of every observing camera: the direction
    /// from each camera to it agrees in sign with that camera's ray. For
    /// `inverse_depth = 0` this reads the direction itself. The sine residual
    /// cannot tell an angle from its opposite, so a point behind the cameras
    /// can fit the rays; the fit prefers a result in front, and this flag says
    /// when none was found.
    pub in_front_of_all_cameras: bool,
    /// The number of rays used, as for [`BearingScore::num_views`].
    pub num_views: usize,
}

/// How [`fit_point_and_bearing`] fits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointBearingFitOptions {
    /// Per-component robust loss. `None` is plain least squares; `Some(c)` is
    /// soft-L1 with scale `c` noise units, `c² ρ(r²/c²)` with
    /// `ρ(z) = 2(√(1 + z) − 1)`, the loss bundle adjustment uses.
    pub soft_l1_scale: Option<f64>,
    /// The most Levenberg-Marquardt iterations per fit.
    pub max_iterations: usize,
}

impl Default for PointBearingFitOptions {
    fn default() -> Self {
        Self {
            soft_l1_scale: Some(DEFAULT_SOFT_L1_SCALE),
            max_iterations: DEFAULT_POINT_FIT_MAX_ITERATIONS,
        }
    }
}

/// The decision. Finite when `bearing_cost ≥ threshold` and either
/// `depth_score` or `midpoint_bound` reaches the threshold; otherwise a
/// bearing. `bearing_cost < threshold` decides a bearing exactly, and
/// `midpoint_bound ≥ threshold` decides a finite point exactly.
pub fn is_finite(score: &BearingScore, threshold: f64) -> bool {
    score.bearing_cost >= threshold
        && (score.depth_score >= threshold || score.midpoint_bound >= threshold)
}

/// Score one track. `dirs` are unit world rays, `centers` the camera centres,
/// `weights` the per-ray 2×3 noise weights (see [`observed_ray`] and
/// [`isotropic_ray_weight`]). `None` when fewer than two rays are usable, or
/// when the weights are so large that the bearing's cost overflows.
pub fn bearing_score(
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    weights: &[Matrix2x3<f64>],
) -> Option<BearingScore> {
    check_lengths(dirs, centers, weights);
    let track = Track::new(dirs, centers, weights, None)?;
    let (bearing, bearing_cost) = track.closed_form_bearing();
    if !bearing_cost.is_finite() {
        return None;
    }
    let depth_score = track.depth_score(bearing);
    let midpoint_bound = track
        .midpoint_in_front()
        .map(|p| (bearing_cost - track.cost_at_point(&p)).max(0.0))
        .unwrap_or(0.0);
    Some(BearingScore {
        bearing,
        bearing_cost,
        depth_score,
        midpoint_bound,
        bearing_in_front_of_all_cameras: track.rays.iter().all(|r| r.d.dot(&bearing) > 0.0),
        num_views: track.rays.len(),
    })
}

/// [`bearing_score`] over a batch of tracks, CSR-style: track `t` owns
/// `dirs[offsets[t]..offsets[t+1]]` and the matching `centers` and `weights`.
/// Each entry is bit-identical to the single-track call.
pub fn bearing_score_batch(
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    offsets: &[usize],
    weights: &[Matrix2x3<f64>],
) -> Vec<Option<BearingScore>> {
    check_lengths(dirs, centers, weights);
    let m = offsets.len().saturating_sub(1);
    (0..m)
        .into_par_iter()
        .map(|t| {
            let r = offsets[t]..offsets[t + 1];
            bearing_score(&dirs[r.clone()], &centers[r.clone()], &weights[r])
        })
        .collect()
}

/// Fit both models to one track and compare them. Inputs as for
/// [`bearing_score`].
///
/// `start`, when given, is a point to start the point fit from (a stored
/// position, or bundle adjustment's current estimate). When there is none, or
/// its fit ends above the bearing's cost or behind a camera, the fit also runs
/// from the bearing at `ρ = 0` and from the weighted linear midpoint (when
/// that lies in front of every camera). When the warm fit is kept, the
/// midpoint is still refined if it costs less than that fit. A fit started in
/// front of every camera stays in front. When none of those results is in
/// front of every camera, the fit runs once more from the cheapest point along
/// one of the rays that is. Of the results that cost no more than the bearing,
/// it keeps the cheapest in front of every camera, or the cheapest when none
/// is; the bearing itself is always one of them, so
/// `point_cost ≤ bearing_cost`. With plain least squares, the midpoint's
/// refinement is an in-front result costing no more than the midpoint, so
/// `Λ ≥ BearingScore::midpoint_bound` at any `max_iterations`, up to the
/// round-off of evaluating the midpoint's cost through `u + ρ (a − cᵢ)`
/// rather than directly, which grows with the anchor's distance from the
/// cameras (a few parts per million with an anchor 10⁶ away).
///
/// With `soft_l1_scale` set, the robust bearing is refined from the
/// closed-form one, and held in front of every camera when that one is, so
/// it can stop short of the robust minimum on the far side of a camera.
///
/// `anchor`, when given and finite, is the point model's anchor; otherwise it
/// is the centroid of the observing camera centres. The fitted point and `Λ`
/// do not depend on it, but the parametrisation does, and bundle adjustment
/// keeps a point's anchor fixed across rounds. The anchor itself is the one
/// point the model cannot represent (`ρ` would be infinite): a start or a
/// midpoint exactly at the anchor is skipped, and in that case alone `Λ` can
/// fall below `midpoint_bound`.
///
/// In near-camera geometry the in-front point the fit finds can sit almost
/// on a near camera's centre: that is the sine cost's genuine minimum in
/// front of every camera, but a point at almost zero depth from a camera is
/// not one to store, and a consumer applies a minimum depth of its own.
pub fn fit_point_and_bearing(
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    weights: &[Matrix2x3<f64>],
    start: Option<Point3<f64>>,
    anchor: Option<Point3<f64>>,
    options: &PointBearingFitOptions,
) -> Option<PointBearingFit> {
    check_lengths(dirs, centers, weights);
    let track = Track::new(dirs, centers, weights, anchor)?;
    let scale = options.soft_l1_scale;
    let max_iter = options.max_iterations;

    let (closed_form, plain_cost) = track.closed_form_bearing();
    let bearing_state = match scale {
        None => State {
            u: closed_form,
            rho: 0.0,
            cost: plain_cost,
        },
        Some(_) => track.refine(closed_form, 0.0, true, scale, max_iter),
    };
    let bearing = bearing_state.u;
    let bearing_cost = bearing_state.cost;
    if !bearing_cost.is_finite() {
        return None;
    }

    let mut runs = vec![bearing_state];
    let warm = start
        .and_then(|p| track.to_inverse_depth(p))
        .map(|(u0, rho0)| {
            let s = track.refine(u0, rho0, false, scale, max_iter);
            runs.push(s);
            s
        });
    let warm_accepted = warm.filter(|s| s.cost <= bearing_cost && track.in_front(s));
    let midpoint = track
        .midpoint_in_front()
        .and_then(|p| track.to_inverse_depth(p));
    match warm_accepted {
        None => {
            runs.push(track.refine(bearing, 0.0, false, scale, max_iter));
            if let Some((u0, rho0)) = midpoint {
                runs.push(track.refine(u0, rho0, false, scale, max_iter));
            }
        }
        // An accepted warm fit still has to beat the midpoint, so that the fit
        // never reports a `Λ` below `BearingScore::midpoint_bound`; the
        // midpoint is refined only when it is the cheaper start. Its
        // refinement stays in front and costs no more than it.
        Some(w) => {
            if let Some((u0, rho0)) = midpoint {
                if track.cost(&u0, rho0, scale) < w.cost {
                    runs.push(track.refine(u0, rho0, false, scale, max_iter));
                }
            }
        }
    }

    let (mut best, mut best_front) = track.select(&runs, bearing_cost);
    // No start so far led to a point in front of every camera: the bearing is
    // behind a camera, and the midpoint is behind one too or missing. That
    // happens when one camera is very close to the point. Start once more from
    // the cheapest point along one of the rays that is in front of every
    // camera.
    if !best_front {
        if let Some((u0, rho0)) = track
            .ray_point_in_front()
            .and_then(|p| track.to_inverse_depth(p))
        {
            runs.push(track.refine(u0, rho0, false, scale, max_iter));
            (best, best_front) = track.select(&runs, bearing_cost);
        }
    }

    let point = (best.rho > 0.0).then(|| track.anchor + best.u / best.rho);
    Some(PointBearingFit {
        bearing,
        bearing_cost,
        anchor: track.anchor,
        direction: best.u,
        inverse_depth: best.rho,
        point,
        point_cost: best.cost,
        depth_likelihood_ratio: (bearing_cost - best.cost).max(0.0),
        in_front_of_all_cameras: best_front,
        num_views: track.rays.len(),
    })
}

/// [`fit_point_and_bearing`] over a batch of tracks, CSR-style as for
/// [`bearing_score_batch`]. `starts` and `anchors`, when given, have one entry
/// per track; a non-finite entry means none for that track. Each entry is
/// bit-identical to the single-track call.
pub fn fit_point_and_bearing_batch(
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    offsets: &[usize],
    weights: &[Matrix2x3<f64>],
    starts: Option<&[Point3<f64>]>,
    anchors: Option<&[Point3<f64>]>,
    options: &PointBearingFitOptions,
) -> Vec<Option<PointBearingFit>> {
    check_lengths(dirs, centers, weights);
    let m = offsets.len().saturating_sub(1);
    if let Some(s) = starts {
        assert_eq!(s.len(), m, "starts must have one entry per track");
    }
    if let Some(a) = anchors {
        assert_eq!(a.len(), m, "anchors must have one entry per track");
    }
    let finite = |p: &Point3<f64>| p.coords.iter().all(|c| c.is_finite());
    (0..m)
        .into_par_iter()
        .map(|t| {
            let r = offsets[t]..offsets[t + 1];
            let start = starts.map(|s| s[t]).filter(finite);
            let anchor = anchors.map(|a| a[t]).filter(finite);
            fit_point_and_bearing(
                &dirs[r.clone()],
                &centers[r.clone()],
                &weights[r],
                start,
                anchor,
                options,
            )
        })
        .collect()
}

fn check_lengths(dirs: &[Vector3<f64>], centers: &[Point3<f64>], weights: &[Matrix2x3<f64>]) {
    assert_eq!(
        dirs.len(),
        centers.len(),
        "dirs and centers must have equal length"
    );
    assert_eq!(
        weights.len(),
        dirs.len(),
        "weights must have one entry per ray"
    );
}

/// A fit has converged when a step would lower the cost by no more than this
/// fraction of it plus [`CONVERGED_ABS`].
const CONVERGED_REL: f64 = 1e-12;

/// The absolute part of the convergence test, in noise units squared: far
/// below any cost difference a threshold on `Λ` could read, and it stops the
/// fit of an exact bearing from creeping off `ρ = 0` on round-off.
const CONVERGED_ABS: f64 = 1e-12;

/// One usable ray, prepared for the residual.
#[derive(Debug, Clone, Copy)]
struct Ray {
    /// The unit ray.
    d: Vector3<f64>,
    /// `W (I − d dᵀ)`: the residual is `a m`.
    a: Matrix2x3<f64>,
    /// `a − c`: the anchor relative to this ray's camera centre.
    offset: Vector3<f64>,
    center: Point3<f64>,
}

/// A track's usable rays and the anchor of its point model.
struct Track {
    rays: Vec<Ray>,
    anchor: Point3<f64>,
}

/// A point-model state and its cost.
#[derive(Debug, Clone, Copy)]
struct State {
    u: Vector3<f64>,
    rho: f64,
    cost: f64,
}

/// An orthonormal pair perpendicular to the unit vector `d`.
fn tangent_basis(d: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    // Cross with the axis least aligned with d, for a well-conditioned basis.
    let a = d.abs();
    let helper = if a.x <= a.y && a.x <= a.z {
        Vector3::x()
    } else if a.y <= a.z {
        Vector3::y()
    } else {
        Vector3::z()
    };
    let b1 = d.cross(&helper).normalize();
    let b2 = d.cross(&b1);
    (b1, b2)
}

/// The loss of one residual component, in noise units squared.
fn component_loss(r: f64, scale: Option<f64>) -> f64 {
    match scale {
        None => r * r,
        Some(s) => {
            let z = (r / s) * (r / s);
            s * s * 2.0 * ((1.0 + z).sqrt() - 1.0)
        }
    }
}

/// The Triggs scaling of one residual component under the soft-L1 loss, as in
/// bundle adjustment: the Jacobian row is multiplied by `(1 + z)^(−¾)` and the
/// residual by `(1 + z)^(¼)`, `z = (r/s)²`, so that `Jᵀr` is the robust
/// gradient and `JᵀJ` its Gauss-Newton curvature.
fn robust_scales(r: f64, scale: Option<f64>) -> (f64, f64) {
    match scale {
        None => (1.0, 1.0),
        Some(s) => {
            let z = (r / s) * (r / s);
            ((1.0 + z).powf(-0.75), (1.0 + z).powf(0.25))
        }
    }
}

impl Ray {
    /// The residual at `(u, ρ)`, or `None` where the model's direction from
    /// this camera is undefined (the point is at the camera centre).
    fn residual(&self, u: &Vector3<f64>, rho: f64) -> Option<Vector2<f64>> {
        let v = u + rho * self.offset;
        let n = v.norm();
        if n <= 0.0 || !n.is_finite() {
            return None;
        }
        Some(self.a * (v / n))
    }

    /// The residual and its 2×3 Jacobian with respect to `(δ₁, δ₂, ρ)`, where
    /// the direction moves as `u ← normalize(u + δ₁ t1 + δ₂ t2)` with `(t1, t2)`
    /// an orthonormal basis perpendicular to `u`.
    ///
    /// With `v = u + ρ e` (`e` the anchor offset) and `m = v / ‖v‖`,
    /// `∂m/∂v = (I − m mᵀ) / ‖v‖`, `∂v/∂δ = [t1 t2]` and `∂v/∂ρ = e`.
    fn residual_and_jacobian(
        &self,
        u: &Vector3<f64>,
        t1: &Vector3<f64>,
        t2: &Vector3<f64>,
        rho: f64,
    ) -> Option<(Vector2<f64>, Matrix2x3<f64>)> {
        let v = u + rho * self.offset;
        let n = v.norm();
        if n <= 0.0 || !n.is_finite() {
            return None;
        }
        let m = v / n;
        let r = self.a * m;
        let cols = [t1, t2, &self.offset];
        let mut j = Matrix2x3::zeros();
        for k in 0..2 {
            // Row aₖᵀ (I − m mᵀ) / ‖v‖.
            let a_k: Vector3<f64> = self.a.row(k).transpose();
            let row = (a_k - a_k.dot(&m) * m) / n;
            for (c, col) in cols.iter().enumerate() {
                j[(k, c)] = row.dot(col);
            }
        }
        Some((r, j))
    }
}

impl Track {
    /// The usable rays of one track, or `None` when fewer than two.
    fn new(
        dirs: &[Vector3<f64>],
        centers: &[Point3<f64>],
        weights: &[Matrix2x3<f64>],
        anchor: Option<Point3<f64>>,
    ) -> Option<Self> {
        let identity = Matrix3::<f64>::identity();
        let mut usable: Vec<(Vector3<f64>, Point3<f64>, Matrix2x3<f64>)> =
            Vec::with_capacity(dirs.len());
        for ((d, c), w) in dirs.iter().zip(centers).zip(weights) {
            let n = d.norm();
            let ok = d.iter().all(|x| x.is_finite())
                && c.coords.iter().all(|x| x.is_finite())
                && w.iter().all(|x| x.is_finite())
                && n > 0.0
                && w.norm_squared() > 0.0
                && w.norm_squared().is_finite();
            if ok {
                let d = d / n;
                usable.push((d, *c, w * (identity - d * d.transpose())));
            }
        }
        if usable.len() < 2 {
            return None;
        }
        let anchor = anchor
            .filter(|p| p.coords.iter().all(|x| x.is_finite()))
            .unwrap_or_else(|| {
                let mut sum = Vector3::zeros();
                for (_, c, _) in &usable {
                    sum += c.coords;
                }
                Point3::from(sum / usable.len() as f64)
            });
        let rays = usable
            .into_iter()
            .map(|(d, center, a)| Ray {
                d,
                a,
                offset: anchor - center,
                center,
            })
            .collect();
        Some(Self { rays, anchor })
    }

    /// The bearing minimising the plain least-squares cost, and that cost.
    fn closed_form_bearing(&self) -> (Vector3<f64>, f64) {
        let mut m = Matrix3::<f64>::zeros();
        let mut mean = Vector3::<f64>::zeros();
        for r in &self.rays {
            let q = r.a.transpose() * r.a;
            m += q;
            mean += q.trace() * r.d;
        }
        let eig = SymmetricEigen::new(m);
        let mut k_min = 0;
        for k in 1..3 {
            if eig.eigenvalues[k] < eig.eigenvalues[k_min] {
                k_min = k;
            }
        }
        let mut u: Vector3<f64> = eig.eigenvectors.column(k_min).into_owned().normalize();
        if u.dot(&mean) < 0.0 {
            u = -u;
        }
        // Σ ‖Aᵢ u‖² is the eigenvalue, summed from non-negative terms.
        let cost = self.rays.iter().map(|r| (r.a * u).norm_squared()).sum();
        (u, cost)
    }

    /// The score statistic for `ρ` at `(bearing, ρ = 0)`, one-sided.
    fn depth_score(&self, bearing: Vector3<f64>) -> f64 {
        let (t1, t2) = tangent_basis(&bearing);
        let (h, g) = match self.normal_equations(&bearing, &t1, &t2, 0.0, None) {
            Some(x) => x,
            None => return 0.0,
        };
        // Block form: the tangent block, its coupling to ρ, and ρ's Schur
        // complement. gᵀH⁻¹g = g_tᵀH_tt⁻¹g_t + g̃²/S, with g̃ the ρ gradient
        // after re-optimising the direction.
        let h_tt = Matrix2::new(h[(0, 0)], h[(0, 1)], h[(1, 0)], h[(1, 1)]);
        let h_tr = Vector2::new(h[(0, 2)], h[(1, 2)]);
        let h_rr = h[(2, 2)];
        let g_t = Vector2::new(g[0], g[1]);
        let Some(h_tt_inv) = h_tt.try_inverse() else {
            return 0.0;
        };
        let y = h_tt_inv * g_t;
        let z = h_tt_inv * h_tr;
        let schur = h_rr - h_tr.dot(&z);
        if !(h_rr.is_finite() && schur.is_finite()) || h_rr <= 0.0 || schur <= 1e-12 * h_rr {
            return 0.0;
        }
        let g_rho = g[2] - h_tr.dot(&y);
        // The Gauss-Newton step in ρ is −g̃/S; a step to ρ ≤ 0 earns nothing.
        if g_rho.is_nan() || g_rho >= 0.0 {
            return 0.0;
        }
        (g_t.dot(&y) + g_rho * g_rho / schur).max(0.0)
    }

    /// `H = Σ J̃ᵀJ̃` and `g = Σ J̃ᵀr̃` at `(u, ρ)`, with the robust scaling when
    /// a scale is set. `None` where a residual is undefined.
    fn normal_equations(
        &self,
        u: &Vector3<f64>,
        t1: &Vector3<f64>,
        t2: &Vector3<f64>,
        rho: f64,
        scale: Option<f64>,
    ) -> Option<(Matrix3<f64>, Vector3<f64>)> {
        let mut h = Matrix3::<f64>::zeros();
        let mut g = Vector3::<f64>::zeros();
        for ray in &self.rays {
            let (r, j) = ray.residual_and_jacobian(u, t1, t2, rho)?;
            for c in 0..2 {
                let (js, rs) = robust_scales(r[c], scale);
                let row = j.row(c).transpose() * js;
                h += row * row.transpose();
                g += row * (r[c] * rs);
            }
        }
        Some((h, g))
    }

    /// The model's cost at `(u, ρ)`; `∞` where a residual is undefined.
    fn cost(&self, u: &Vector3<f64>, rho: f64, scale: Option<f64>) -> f64 {
        let mut total = 0.0;
        for ray in &self.rays {
            match ray.residual(u, rho) {
                Some(r) => total += component_loss(r[0], scale) + component_loss(r[1], scale),
                None => return f64::INFINITY,
            }
        }
        total
    }

    /// Among `runs` that cost no more than `bearing_cost`, the one in front of
    /// every camera if any is, then the cheapest; earlier runs win ties.
    /// `runs[0]` is the bearing state, which always qualifies. Returns the
    /// state and whether it is in front.
    fn select(&self, runs: &[State], bearing_cost: f64) -> (State, bool) {
        let mut best = runs[0];
        let mut best_front = self.in_front(&best);
        for s in &runs[1..] {
            if s.cost > bearing_cost {
                continue;
            }
            let front = self.in_front(s);
            if (front && !best_front) || (front == best_front && s.cost < best.cost) {
                best = *s;
                best_front = front;
            }
        }
        (best, best_front)
    }

    /// The cheapest point in front of every camera found along the rays
    /// themselves: on each ray, the distance that best fits the other rays (a
    /// one-dimensional linear solve) and a ladder of distances from 10⁻⁶ to
    /// 10³ times the camera spread, four to a decade. `O(K²)` per ladder
    /// step, so it is a last resort, run only when no other start leads in
    /// front. A point on a ray is in front of that ray's camera by
    /// construction, and one close to a camera near the point can be in front
    /// of the others where the midpoint is not.
    fn ray_point_in_front(&self) -> Option<Point3<f64>> {
        let mut spread: f64 = 0.0;
        for (i, ri) in self.rays.iter().enumerate() {
            for rj in &self.rays[i + 1..] {
                spread = spread.max((ri.center - rj.center).norm());
            }
        }
        if !(spread > 0.0 && spread.is_finite()) {
            spread = 1.0;
        }
        let mut best: Option<(Point3<f64>, f64)> = None;
        let mut consider = |p: Point3<f64>| {
            if !self.rays.iter().all(|r| (p - r.center).dot(&r.d) > 0.0) {
                return;
            }
            let cost = self.cost_at_point(&p);
            if cost.is_finite() && best.is_none_or(|(_, c)| cost < c) {
                best = Some((p, cost));
            }
        };
        for ri in &self.rays {
            // min over s of Σⱼ ‖Aⱼ (cᵢ + s dᵢ − cⱼ)‖².
            let (mut num, mut den) = (0.0, 0.0);
            for rj in &self.rays {
                let q = rj.a.transpose() * rj.a;
                let qd = q * ri.d;
                num += qd.dot(&(rj.center - ri.center));
                den += qd.dot(&ri.d);
            }
            if den > 0.0 && num / den > 0.0 {
                consider(ri.center + (num / den) * ri.d);
            }
            for step in -24..=12 {
                let s = spread * 10f64.powf(step as f64 / 4.0);
                consider(ri.center + s * ri.d);
            }
        }
        best.map(|(p, _)| p)
    }

    /// The plain least-squares cost of the finite point `p`, independent of the
    /// anchor; `∞` where `p` is at a camera centre.
    fn cost_at_point(&self, p: &Point3<f64>) -> f64 {
        let mut total = 0.0;
        for ray in &self.rays {
            let v = p - ray.center;
            let n = v.norm();
            if n <= 0.0 || !n.is_finite() {
                return f64::INFINITY;
            }
            total += (ray.a * (v / n)).norm_squared();
        }
        total
    }

    /// The state lies in front of every camera.
    fn in_front(&self, s: &State) -> bool {
        self.rays
            .iter()
            .all(|r| (s.u + s.rho * r.offset).dot(&r.d) > 0.0)
    }

    /// The weighted linear midpoint `(Σ Aᵢᵀ Aᵢ)⁻¹ Σ Aᵢᵀ Aᵢ cᵢ`, the point
    /// minimising `Σ ‖Aᵢ (X − cᵢ)‖²`, when it is finite and lies in front of
    /// every camera.
    fn midpoint_in_front(&self) -> Option<Point3<f64>> {
        let mut m = Matrix3::<f64>::zeros();
        let mut b = Vector3::<f64>::zeros();
        for r in &self.rays {
            let q = r.a.transpose() * r.a;
            m += q;
            b += q * r.center.coords;
        }
        let p = Point3::from(m.cholesky()?.solve(&b));
        (p.coords.iter().all(|c| c.is_finite())
            && self.rays.iter().all(|r| (p - r.center).dot(&r.d) > 0.0))
        .then_some(p)
    }

    /// A point as `(u, ρ)` about the anchor; `None` at the anchor itself.
    fn to_inverse_depth(&self, p: Point3<f64>) -> Option<(Vector3<f64>, f64)> {
        let w = p - self.anchor;
        let n = w.norm();
        (n > 0.0 && n.is_finite()).then(|| (w / n, 1.0 / n))
    }

    /// Levenberg-Marquardt over `(u, ρ)` from `(u, ρ)`, with `ρ` clamped at 0,
    /// or over `u` alone with `ρ` held when `fix_rho`. Accepts only steps that
    /// lower the cost, so the result costs no more than the start, and, from a
    /// start in front of every camera, only steps that stay in front: the sine
    /// residual would otherwise let the fit slide to a cheap point behind a
    /// camera, which the caller then has to discard.
    fn refine(
        &self,
        mut u: Vector3<f64>,
        mut rho: f64,
        fix_rho: bool,
        scale: Option<f64>,
        max_iterations: usize,
    ) -> State {
        let mut cost = self.cost(&u, rho, scale);
        if !cost.is_finite() {
            return State { u, rho, cost };
        }
        let keep_in_front = self.in_front(&State { u, rho, cost });
        let mut lambda = 1e-6;
        for _ in 0..max_iterations {
            if cost == 0.0 {
                break;
            }
            let (t1, t2) = tangent_basis(&u);
            let Some((h, g)) = self.normal_equations(&u, &t1, &t2, rho, scale) else {
                break;
            };
            let diag_max = h[(0, 0)].max(h[(1, 1)]).max(h[(2, 2)]);
            if diag_max.is_nan() || diag_max <= 0.0 {
                break;
            }
            let floor = 1e-12 * diag_max;
            let mut accepted = false;
            let mut converged = false;
            while lambda < 1e12 {
                let Some(step) = damped_step(&h, &g, lambda, floor, rho, fix_rho) else {
                    lambda *= 10.0;
                    continue;
                };
                // The quadratic model's predicted reduction; a negligible one
                // means the fit is at its optimum.
                let predicted = -(2.0 * g.dot(&step) + step.dot(&(h * step)));
                if predicted.is_nan() || predicted <= CONVERGED_REL * cost + CONVERGED_ABS {
                    converged = true;
                    break;
                }
                let u_new = (u + step[0] * t1 + step[1] * t2).normalize();
                let rho_new = if fix_rho {
                    rho
                } else {
                    (rho + step[2]).max(0.0)
                };
                let cost_new = self.cost(&u_new, rho_new, scale);
                let leaves_front = keep_in_front
                    && !self.in_front(&State {
                        u: u_new,
                        rho: rho_new,
                        cost: cost_new,
                    });
                if cost_new < cost && !leaves_front {
                    let decrease = cost - cost_new;
                    u = u_new;
                    rho = rho_new;
                    cost = cost_new;
                    lambda = (lambda / 10.0).max(1e-12);
                    accepted = true;
                    converged = decrease <= CONVERGED_REL * cost + CONVERGED_ABS;
                    break;
                }
                lambda *= 10.0;
            }
            if converged || !accepted {
                break;
            }
        }
        State { u, rho, cost }
    }
}

/// One damped Gauss-Newton step `(δ₁, δ₂, δρ)` solving
/// `(H + λ D) Δ = −g`, `D` the diagonal of `H` floored at `floor`. With `ρ`
/// held, or at its bound with the step pointing below it, `δρ = 0` and the
/// direction is solved alone.
fn damped_step(
    h: &Matrix3<f64>,
    g: &Vector3<f64>,
    lambda: f64,
    floor: f64,
    rho: f64,
    fix_rho: bool,
) -> Option<Vector3<f64>> {
    let mut a = *h;
    for k in 0..3 {
        a[(k, k)] += lambda * h[(k, k)].max(floor);
    }
    if !fix_rho {
        let step = -(a.cholesky()?.solve(g));
        if rho > 0.0 || step[2] >= 0.0 {
            return Some(step);
        }
    }
    let a2 = Matrix2::new(a[(0, 0)], a[(0, 1)], a[(1, 0)], a[(1, 1)]);
    let s2 = -(a2.cholesky()?.solve(&Vector2::new(g[0], g[1])));
    Some(Vector3::new(s2[0], s2[1], 0.0))
}

#[cfg(test)]
mod tests;
