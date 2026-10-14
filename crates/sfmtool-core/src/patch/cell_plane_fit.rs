// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Cell triangulation and plane fit: the rays of each of a patch's nine cells
//! in, a world position per cell and a plane through them out, with which
//! axes of the plane's normal the cells fix.
//!
//! The caller decides where each cell lies in each view and how precisely: it
//! turns a pixel of a posed view into a [`CellRay`] with
//! [`PosedView::ray_through`], giving the ray its pixel-noise floor, and hands
//! the nine cells' rays to [`fit_cell_plane`]. The cell plane normals of a
//! cluster-patches file
//! ([`cell_plane_normals`](crate::patch::cell_plane_normals::cell_plane_normals))
//! build the rays from the file's member positions, shapes and cell
//! displacements, and the floor from the member's shape.
//!
//! Each cell's rays are intersected in the least-squares sense, each ray
//! weighted by `(f / ρ)²` for its focal length `f` and depth `ρ`, so it pulls
//! in pixels of its own image; the position's covariance is `σ² Λ⁻¹`, with `σ`
//! the ray-intersection residual floored at the root mean square of the rays'
//! floors and at [`CellPlaneParams::ray_noise_floor_px`]. The positions are
//! fitted with a plane by Tukey IRLS, each cell weighted by the inverse of its
//! position's variance along the normal times its Tukey weight, and the cells
//! the fit gives any weight decide the [`NormalDeterminacy`] from their spread
//! within the plane.
//!
//! See `specs/core/patch/cell-plane-normals.md`.

use nalgebra::{Matrix3, SymmetricEigen, Vector3};

use crate::camera::CameraIntrinsics;
use crate::geometry::RigidTransform;
use crate::numeric::median_in_place;
use crate::patch::cell_plane_normals::{
    CellPlaneNormal, CellPlaneParams, CellPlaneStatus, NormalDeterminacy,
};

/// Median-absolute-deviation to standard-deviation conversion for a Gaussian.
const MAD_TO_SIGMA: f64 = 1.4826;
/// Weight sums at or below this leave the IRLS loop where it was.
const EPS_STALL: f64 = 1e-300;
/// Eigenvalue floor below which an in-plane spread counts as none.
const EPS_SPREAD: f64 = 1e-300;

/// One posed image's camera and pose, unpacked.
pub(crate) struct PosedView<'a> {
    camera: &'a CameraIntrinsics,
    /// `world_from_cam` rotation.
    r_wc: Matrix3<f64>,
    centre: Vector3<f64>,
    /// Focal length in pixels, the scale angular residuals are read at.
    focal_px: f64,
}

impl<'a> PosedView<'a> {
    /// The view of an image taken with `camera` at the `cam_from_world` pose
    /// `pose`.
    pub(crate) fn new(camera: &'a CameraIntrinsics, pose: &RigidTransform) -> Self {
        let r_cw = pose.to_rotation_matrix();
        let r_wc = r_cw.transpose();
        let centre = -(r_wc * pose.translation);
        let (fx, fy) = camera.focal_lengths();
        PosedView {
            camera,
            r_wc,
            centre,
            focal_px: 0.5 * (fx + fy),
        }
    }

    /// The world ray through `pixel` of this view, with pixel-noise floor
    /// `floor_px`; `None` when the camera model returns no finite direction.
    pub(crate) fn ray_through(&self, pixel: [f64; 2], floor_px: f64) -> Option<CellRay> {
        let d = self.camera.pixel_to_ray(pixel[0], pixel[1]);
        let d = self.r_wc * Vector3::new(d[0], d[1], d[2]);
        let len = d.norm();
        if !(len.is_finite() && len > 0.0) {
            return None;
        }
        Some(CellRay {
            origin: self.centre,
            dir: d / len,
            focal_px: self.focal_px,
            floor_px,
        })
    }
}

/// One cell's sighting in one view.
pub(crate) struct CellRay {
    origin: Vector3<f64>,
    dir: Vector3<f64>,
    focal_px: f64,
    /// The ray's pixel-noise floor: how precisely the caller placed the cell
    /// in this view, in its pixels.
    floor_px: f64,
}

/// A triangulated cell.
struct CellPoint {
    position: Vector3<f64>,
    /// Position covariance, world units squared.
    cov: Matrix3<f64>,
    /// Sum of unit directions from the point to the rays' cameras.
    toward_cameras: Vector3<f64>,
}

/// Triangulate each of the nine cells from its rays, `[row * 3 + col]` from
/// the top-left cell, and fit the plane through the cells that triangulate.
///
/// The answer's `cell_rays` counts each cell's rays; a cell that does not
/// triangulate reads `TooFewRays`, `NarrowBaseline` or `BehindCamera`. With
/// fewer than [`CellPlaneParams::min_cells`] triangulated cells no plane is
/// fitted and they read `Positioned`; otherwise each reads `InPlane` or
/// `PlaneOutlier` by its final weight, and the normal and its
/// [`NormalDeterminacy`] are set as [`CellPlaneNormal`] describes.
pub(crate) fn fit_cell_plane(
    cell_rays: &[Vec<CellRay>; 9],
    params: &CellPlaneParams,
) -> CellPlaneNormal {
    let mut out = CellPlaneNormal::empty();

    // ── Triangulation per cell ───────────────────────────────────────────
    let mut points: [Option<CellPoint>; 9] = Default::default();
    for (j, rays) in cell_rays.iter().enumerate() {
        out.cell_rays[j] = rays.len() as u32;
        match triangulate(rays, params) {
            Ok((point, residual_px)) => {
                out.cell_positions[j] = point.position.into();
                out.cell_residual_px[j] = residual_px;
                out.cell_status[j] = CellPlaneStatus::Positioned;
                points[j] = Some(point);
            }
            Err(status) => out.cell_status[j] = status,
        }
    }

    let positioned: Vec<usize> = (0..9).filter(|&j| points[j].is_some()).collect();
    if positioned.is_empty() {
        return out;
    }
    let mut toward = Vector3::zeros();
    for &j in &positioned {
        toward += points[j].as_ref().unwrap().toward_cameras;
    }
    let view_dir = if toward.norm() > 0.0 {
        toward / toward.norm()
    } else {
        Vector3::z()
    };
    out.view_dir = view_dir.into();
    if positioned.len() < params.min_cells {
        return out;
    }

    // ── Plane fit, Tukey IRLS ────────────────────────────────────────────
    let pts: Vec<&CellPoint> = positioned
        .iter()
        .map(|&j| points[j].as_ref().unwrap())
        .collect();
    let m = pts.len();
    let mut robust = vec![1.0f64; m];
    let mut normal = view_dir;
    let mut z = vec![0.0f64; m];
    let mut scratch = vec![0.0f64; m];
    for _ in 0..params.irls_iters {
        let weights = combined_weights(&pts, &normal, &robust);
        let (centroid, n) = weighted_plane(&pts, &weights);
        normal = n;
        for (zi, p) in z.iter_mut().zip(&pts) {
            let var = normal.dot(&(p.cov * normal)).max(f64::MIN_POSITIVE);
            *zi = (p.position - centroid).dot(&normal).abs() / var.sqrt();
        }
        scratch.copy_from_slice(&z);
        let mut scale = MAD_TO_SIGMA * median_in_place(&mut scratch);
        if !scale.is_finite() {
            scale = 1.0;
        }
        // The residuals are in units of each cell's own modelled noise, so a
        // scale below one would trust the cells more than their rays allow.
        let scale = scale.max(1.0);
        let next: Vec<f64> = z
            .iter()
            .map(|&zi| tukey(zi / (params.tukey_c * scale)))
            .collect();
        if next.iter().sum::<f64>() <= EPS_STALL {
            break;
        }
        robust = next;
    }
    let weights = combined_weights(&pts, &normal, &robust);
    let (centroid, mut normal) = weighted_plane(&pts, &weights);
    if normal.dot(&view_dir) < 0.0 {
        normal = -normal;
    }

    let sum_w: f64 = weights.iter().sum();
    let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
    let sum_wr2: f64 = weights
        .iter()
        .zip(&pts)
        .map(|(w, p)| w * (p.position - centroid).dot(&normal).powi(2))
        .sum();
    out.n_eff = if sum_w2 > 0.0 {
        sum_w * sum_w / sum_w2
    } else {
        0.0
    };
    out.plane_rms = if sum_w > 0.0 {
        (sum_wr2 / sum_w).sqrt()
    } else {
        f64::NAN
    };
    for (i, &j) in positioned.iter().enumerate() {
        out.cell_weight[j] = weights[i];
        out.cell_status[j] = if weights[i] > 0.0 {
            CellPlaneStatus::InPlane
        } else {
            CellPlaneStatus::PlaneOutlier
        };
    }

    let (anisotropy, verdict) = determinacy(&pts, &weights, &normal, &view_dir, params);
    out.anisotropy = anisotropy;
    if let Some((normal, determinacy)) = verdict {
        out.normal = normal;
        out.determinacy = determinacy;
    }
    out
}

/// Which axes of the fitted `normal` the cells fix, from the spread within
/// the plane of the live cells, those with a positive `weights` entry.
///
/// Returns the anisotropy `λ_mid / λ_max` of that spread (`0` without any),
/// and the normal with its verdict: `None` with fewer than
/// [`CellPlaneParams::min_cells`] live cells or no spread, both axes when the
/// anisotropy reaches [`CellPlaneParams::det_aniso`], and otherwise one axis,
/// the line the cells lie along, with the normal perpendicular to it and
/// closest to `view_dir`.
fn determinacy(
    pts: &[&CellPoint],
    weights: &[f64],
    normal: &Vector3<f64>,
    view_dir: &Vector3<f64>,
    params: &CellPlaneParams,
) -> (f64, Option<([f64; 3], NormalDeterminacy)>) {
    // A cell is live when the robust fit gives it any weight. Its precision
    // does not enter: a two-ray cell is as much a sample of where the surface
    // is as a five-ray one, and the precision already sets how hard each
    // pulls on the plane.
    let live: Vec<usize> = (0..pts.len()).filter(|&i| weights[i] > 0.0).collect();
    // The live cells' spread within the fitted plane, with equal weights: the
    // verdict asks where the cells are, not how sure the fit is of each, and
    // a cell's offset from the plane is noise, not spread.
    let mut mean = Vector3::zeros();
    for &i in &live {
        mean += pts[i].position;
    }
    mean /= live.len().max(1) as f64;
    let mut scatter = Matrix3::zeros();
    for &i in &live {
        let d = pts[i].position - mean;
        let d = d - d.dot(normal) * normal;
        scatter += d * d.transpose();
    }
    let (evals, evecs) = sorted_eigen(&scatter);
    let anisotropy = if evals[2] > EPS_SPREAD {
        evals[1] / evals[2]
    } else {
        0.0
    };
    if live.len() < params.min_cells || evals[2] <= EPS_SPREAD {
        return (anisotropy, None);
    }
    if anisotropy >= params.det_aniso {
        return (
            anisotropy,
            Some(((*normal).into(), NormalDeterminacy::BothAxes)),
        );
    }
    let axis = evecs[2];
    // The cells fix only the normal's tilt along the line; about the line
    // it stays at the prior, the mean viewing direction.
    let mut n = view_dir - view_dir.dot(&axis) * axis;
    if n.norm() <= 1e-12 {
        n = normal - normal.dot(&axis) * axis;
    }
    let n = n / n.norm();
    (
        anisotropy,
        Some((
            n.into(),
            NormalDeterminacy::OneAxis {
                free_axis: axis.into(),
            },
        )),
    )
}

/// Nearest point of `rays`, with its covariance and its residual in pixels;
/// the failing status otherwise.
///
/// The plain least-squares point gives each ray's depth `ρᵢ`; the point is
/// then solved again with each ray's projector weighted by `(fᵢ/ρᵢ)²`, so
/// each ray pulls in pixels of its own image, and the covariance `σ² Λ⁻¹`
/// inverts that same weighted normal matrix `Λ`. One pass, from the plain
/// point's depths.
fn triangulate(
    rays: &[CellRay],
    params: &CellPlaneParams,
) -> Result<(CellPoint, f64), CellPlaneStatus> {
    if rays.len() < params.min_rays.max(2) {
        return Err(CellPlaneStatus::TooFewRays);
    }
    let min_cos = params.min_triangulation_angle_deg.to_radians().cos();
    let wide = rays.iter().enumerate().any(|(a, ra)| {
        rays[a + 1..]
            .iter()
            .any(|rb| ra.dir.dot(&rb.dir) <= min_cos)
    });
    if !wide {
        return Err(CellPlaneStatus::NarrowBaseline);
    }

    let x0 = nearest_point(rays, &vec![1.0; rays.len()])?.0;
    let weights = rays
        .iter()
        .map(|r| {
            let depth = (x0 - r.origin).dot(&r.dir);
            if depth > 0.0 {
                Ok((r.focal_px / depth).powi(2))
            } else {
                Err(CellPlaneStatus::BehindCamera)
            }
        })
        .collect::<Result<Vec<f64>, _>>()?;
    let (x, info) = nearest_point(rays, &weights)?;

    let mut sum_e2 = 0.0;
    let mut sum_floor2 = 0.0;
    let mut toward = Vector3::zeros();
    for (r, &w) in rays.iter().zip(&weights) {
        let v = x - r.origin;
        if v.dot(&r.dir) <= 0.0 {
            return Err(CellPlaneStatus::BehindCamera);
        }
        sum_e2 += w * (v - v.dot(&r.dir) * r.dir).norm_squared();
        sum_floor2 += r.floor_px * r.floor_px;
        toward -= v / v.norm();
    }
    // Two coordinates per ray, three for the point.
    let n = rays.len() as f64;
    let residual_px = (sum_e2 / (2.0 * n - 3.0)).sqrt();
    let floor_px = (sum_floor2 / n).sqrt().max(params.ray_noise_floor_px);
    let sigma = residual_px.max(floor_px);
    let Some(info_inv) = info.try_inverse() else {
        return Err(CellPlaneStatus::NarrowBaseline);
    };
    Ok((
        CellPoint {
            position: x,
            cov: (sigma * sigma) * info_inv,
            toward_cameras: toward,
        },
        residual_px,
    ))
}

/// The point minimising `Σ wᵢ ‖Pᵢ (x − oᵢ)‖²` with `Pᵢ = I − dᵢdᵢᵀ`, and the
/// normal matrix `Σ wᵢ Pᵢ`.
fn nearest_point(
    rays: &[CellRay],
    weights: &[f64],
) -> Result<(Vector3<f64>, Matrix3<f64>), CellPlaneStatus> {
    let mut a = Matrix3::zeros();
    let mut b = Vector3::zeros();
    for (r, &w) in rays.iter().zip(weights) {
        let p = w * (Matrix3::identity() - r.dir * r.dir.transpose());
        a += p;
        b += p * r.origin;
    }
    let x = a
        .try_inverse()
        .map(|inv| inv * b)
        .ok_or(CellPlaneStatus::NarrowBaseline)?;
    if x.iter().all(|v| v.is_finite()) {
        Ok((x, a))
    } else {
        Err(CellPlaneStatus::NarrowBaseline)
    }
}

/// Each cell's robust weight times the inverse of its variance along `normal`.
fn combined_weights(pts: &[&CellPoint], normal: &Vector3<f64>, robust: &[f64]) -> Vec<f64> {
    pts.iter()
        .zip(robust)
        .map(|(p, &rob)| {
            let var = normal.dot(&(p.cov * normal));
            if var > 0.0 && var.is_finite() {
                rob / var
            } else {
                0.0
            }
        })
        .collect()
}

/// The weighted centroid and least-variance direction of the points.
fn weighted_plane(pts: &[&CellPoint], weights: &[f64]) -> (Vector3<f64>, Vector3<f64>) {
    let sum_w: f64 = weights.iter().sum();
    let mut centroid = Vector3::zeros();
    for (p, &w) in pts.iter().zip(weights) {
        centroid += w * p.position;
    }
    centroid /= sum_w.max(f64::MIN_POSITIVE);
    let mut scatter = Matrix3::zeros();
    for (p, &w) in pts.iter().zip(weights) {
        let d = p.position - centroid;
        scatter += w * (d * d.transpose());
    }
    let (_, evecs) = sorted_eigen(&scatter);
    (centroid, evecs[0])
}

/// Eigenvalues ascending with their eigenvectors, ordered by explicit
/// comparison.
fn sorted_eigen(m: &Matrix3<f64>) -> ([f64; 3], [Vector3<f64>; 3]) {
    let eig = SymmetricEigen::new(*m);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    (
        order.map(|i| eig.eigenvalues[i]),
        order.map(|i| eig.eigenvectors.column(i).into_owned()),
    )
}

/// Tukey biweight of a residual already divided by its cut-off.
fn tukey(u: f64) -> f64 {
    if u < 1.0 {
        let t = 1.0 - u * u;
        t * t
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests;
