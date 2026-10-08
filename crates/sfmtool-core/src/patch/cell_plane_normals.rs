// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Cell plane normals: a cluster's patch normal from the per-cell
//! displacements the piecewise refinement stored, once camera poses exist.
//!
//! The piecewise refinement cuts each cluster's `R × R` patch grid into a
//! three-by-three split of cells and stores, for every kept member, where each
//! cell's content lies relative to where the member's affine shape places it.
//! With poses, a cell's location in a member is a pixel and the pixel is a ray.
//! The rays of one cell across the cluster's members meet near one world
//! point; the nine points (fewer where cells were refused) are fitted with a
//! plane, and the plane's normal is the patch normal. No image is read.
//!
//! The plane fit is a weighted least-variance fit with Tukey IRLS. Each cell's
//! weight is the inverse of its position's variance along the plane normal,
//! from the rays' geometry (how many, how far, how wide the baseline) and the
//! ray-intersection residual. Alongside the normal the kernel reports which
//! axes of it the cells fix: both, one (the cells lie on a line, and the normal
//! may still turn about that line), or none. A one-axis normal is the one
//! perpendicular to the line closest to the mean viewing direction, and a
//! normal with no fixed axis is `NaN`.
//!
//! See `specs/drafts/cell-plane-normals.md` for the design.

use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use rayon::prelude::*;
use sfmtool_matches_format::{ClusterCellStatus, ClusterMemberStatus};

use crate::camera::CameraIntrinsics;
use crate::geometry::RigidTransform;
use crate::numeric::median_in_place;
use crate::patch::normal_refine::grid_bounds;

/// `reference_members` value of a cluster that has no reference member.
pub const NO_REFERENCE: u32 = u32::MAX;

/// Median-absolute-deviation to standard-deviation conversion for a Gaussian.
const MAD_TO_SIGMA: f64 = 1.4826;
/// A cell's weight counts toward the determinacy verdict when it is at least
/// this fraction of the cluster's largest weight.
const LIVE_WEIGHT_FRACTION: f64 = 0.25;
/// Weight sums at or below this leave the IRLS loop where it was.
const EPS_STALL: f64 = 1e-300;
/// Eigenvalue floor below which an in-plane spread counts as none.
const EPS_SPREAD: f64 = 1e-300;

/// Tuning for [`cell_plane_normals`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellPlaneParams {
    /// Also triangulate cells the refinement stored as `refused_outlier`: their
    /// displacement was measured and passed the ZNCC and curvature bars, but the
    /// member's affine fit gave it no weight.
    pub include_refused_outlier: bool,
    /// Fewest rays a cell is triangulated from. The reference member's ray
    /// counts, since its cell lies at its own centre by construction.
    pub min_rays: usize,
    /// A cell whose widest pair of rays subtends less than this, in degrees, is
    /// not triangulated: its depth is not pinned.
    pub min_triangulation_angle_deg: f64,
    /// Floor on a cell's ray-intersection residual, in pixels: below this the
    /// residual is treated as noise of this size.
    pub ray_noise_floor_px: f64,
    /// Number of IRLS passes before the final plane solve.
    pub irls_iters: u32,
    /// Tukey biweight tuning constant, in robust scales.
    pub tukey_c: f64,
    /// Fewest live cells a normal is fitted from.
    pub min_cells: usize,
    /// Both axes are fixed when the live cells' in-plane anisotropy
    /// `λ_mid / λ_max` reaches this; below it they lie on a line.
    pub det_aniso: f64,
}

impl Default for CellPlaneParams {
    fn default() -> Self {
        Self {
            include_refused_outlier: false,
            min_rays: 2,
            min_triangulation_angle_deg: 2.0,
            ray_noise_floor_px: 0.05,
            irls_iters: 3,
            tukey_c: 4.685,
            min_cells: 3,
            det_aniso: 0.10,
        }
    }
}

/// The per-cluster data of a cluster-patches file the kernel reads, borrowed
/// from its member-parallel arrays.
///
/// `K` is the member count and `C` the cluster count. The cell arrays are
/// indexed `[member][row * 3 + col]`, the top-left cell first.
#[derive(Clone, Copy, Debug)]
pub struct CellPlaneClusters<'a> {
    /// `C + 1` member-range boundaries.
    pub cluster_starts: &'a [u32],
    /// `C` global member index of each cluster's reference, or [`NO_REFERENCE`].
    pub reference_members: &'a [u32],
    /// `K` image index of each member, into the camera arrays.
    pub member_images: &'a [u32],
    /// `K` member statuses.
    pub member_status: &'a [ClusterMemberStatus],
    /// `K` member keypoint positions, source-image pixels.
    pub member_positions: &'a [[f32; 2]],
    /// `K` member affine shapes `S`, `S[row][col]`.
    pub member_shapes: &'a [[[f32; 2]; 2]],
    /// `K` per-cell displacements `[x, y]` in grid px.
    pub cell_shift_px: &'a [[[f32; 2]; 9]],
    /// `K` per-cell statuses.
    pub cell_status: &'a [[ClusterCellStatus; 9]],
    /// The patch edge in the detector's canonical frame
    /// (`refine_options.patch_size`).
    pub patch_size: f64,
    /// `R`, the patch grid's samples per side (`refine_options.resolution`).
    pub resolution: u32,
}

/// Per-image cameras and poses.
#[derive(Clone, Copy, Debug)]
pub struct CellPlaneCameras<'a> {
    /// One entry per camera.
    pub cameras: &'a [CameraIntrinsics],
    /// One entry per image: the index into `cameras`.
    pub image_camera: &'a [u32],
    /// One entry per image: its `cam_from_world` pose, `None` when unposed.
    pub cam_from_world: &'a [Option<RigidTransform>],
}

/// Which axes of a normal the cells fix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormalDeterminacy {
    /// The live cells spread in two directions: the normal is fixed.
    BothAxes,
    /// The live cells lie on a line along `free_axis` (unit, world frame). The
    /// normal is perpendicular to it, and its rotation about it is not fixed.
    OneAxis {
        /// The line's direction, about which the normal is free.
        free_axis: [f64; 3],
    },
    /// Fewer than [`CellPlaneParams::min_cells`] live cells, or no spread.
    None,
}

impl NormalDeterminacy {
    /// `0` none, `1` one axis, `2` both axes.
    pub fn code(&self) -> u8 {
        match self {
            NormalDeterminacy::None => 0,
            NormalDeterminacy::OneAxis { .. } => 1,
            NormalDeterminacy::BothAxes => 2,
        }
    }
}

/// What became of one cell.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellPlaneStatus {
    /// Triangulated and given weight in the plane fit.
    InPlane = 0,
    /// Triangulated, but the robust plane fit gave it no weight.
    PlaneOutlier = 1,
    /// Triangulated, but too few cells were for a plane to be fitted.
    Positioned = 2,
    /// Fewer than [`CellPlaneParams::min_rays`] posed rays.
    TooFewRays = 3,
    /// The widest pair of rays is narrower than
    /// [`CellPlaneParams::min_triangulation_angle_deg`].
    NarrowBaseline = 4,
    /// The rays' nearest point lies behind one of their cameras.
    BehindCamera = 5,
}

impl CellPlaneStatus {
    /// The names, in discriminant order.
    pub const NAMES: [&'static str; 6] = [
        "in_plane",
        "plane_outlier",
        "positioned",
        "too_few_rays",
        "narrow_baseline",
        "behind_camera",
    ];
}

/// One cluster's answer.
#[derive(Clone, Debug, PartialEq)]
pub struct CellPlaneNormal {
    /// Unit normal, sign-aligned to [`Self::view_dir`]; `NaN` when
    /// [`Self::determinacy`] is [`NormalDeterminacy::None`].
    pub normal: [f64; 3],
    /// Which axes of the normal the cells fix.
    pub determinacy: NormalDeterminacy,
    /// Mean unit direction from the cells toward the cameras of their rays;
    /// `NaN` when no cell was triangulated.
    pub view_dir: [f64; 3],
    /// Each cell's world position; `NaN` where it was not triangulated.
    pub cell_positions: [[f64; 3]; 9],
    /// Posed rays per cell.
    pub cell_rays: [u32; 9],
    /// What became of each cell.
    pub cell_status: [CellPlaneStatus; 9],
    /// Final plane-fit weight per cell; `0` where it has none.
    pub cell_weight: [f64; 9],
    /// Ray-intersection residual per cell, pixels (RMS over the rays,
    /// degrees-of-freedom corrected); `NaN` where not triangulated.
    pub cell_residual_px: [f64; 9],
    /// Weighted RMS of the cells' distances from the plane, world units;
    /// `NaN` when no plane was fitted.
    pub plane_rms: f64,
    /// `λ_mid / λ_max` of the live cells' weighted scatter; `NaN` when no
    /// plane was fitted.
    pub anisotropy: f64,
    /// `(Σw)² / Σw²` over the weighted cells; `NaN` when no plane was fitted.
    pub n_eff: f64,
}

impl CellPlaneNormal {
    fn empty() -> Self {
        let nan3 = [f64::NAN; 3];
        Self {
            normal: nan3,
            determinacy: NormalDeterminacy::None,
            view_dir: nan3,
            cell_positions: [nan3; 9],
            cell_rays: [0; 9],
            cell_status: [CellPlaneStatus::TooFewRays; 9],
            cell_weight: [0.0; 9],
            cell_residual_px: [f64::NAN; 9],
            plane_rms: f64::NAN,
            anisotropy: f64::NAN,
            n_eff: f64::NAN,
        }
    }
}

/// One posed image's camera and pose, unpacked.
struct View<'a> {
    camera: &'a CameraIntrinsics,
    /// `world_from_cam` rotation.
    r_wc: Matrix3<f64>,
    centre: Vector3<f64>,
    /// Focal length in pixels, the scale angular residuals are read at.
    focal_px: f64,
}

/// One cell's sighting in one member.
struct Ray {
    origin: Vector3<f64>,
    dir: Vector3<f64>,
    focal_px: f64,
}

/// A triangulated cell.
struct CellPoint {
    position: Vector3<f64>,
    /// Position covariance, world units squared.
    cov: Matrix3<f64>,
    /// Sum of unit directions from the point to the rays' cameras.
    toward_cameras: Vector3<f64>,
}

/// Estimate a patch normal for every cluster from its stored cell
/// displacements and the images' poses.
///
/// For each cell, the reference member's ray through the cell's centre and,
/// for every `kept` member in a posed image whose cell status is `fitted` (and
/// `refused_outlier` with [`CellPlaneParams::include_refused_outlier`]), the
/// ray through `position + (patch_size / R) · S · (c + d)` are intersected in
/// the least-squares sense. The positions are fitted with a plane (Tukey
/// IRLS, each cell weighted by the inverse of its position's variance along the
/// normal), and the result says which of the normal's axes the cells fix.
///
/// The per-cluster work is independent and runs in parallel, with no
/// randomness and a fixed pass count, so the result does not depend on the
/// thread count.
///
/// # Panics
/// If the member-parallel slices disagree on length, `cluster_starts` is not
/// `C + 1` long or not a valid range partition of the members, a member image
/// or camera index is out of range, or `resolution < 3`.
pub fn cell_plane_normals(
    clusters: &CellPlaneClusters<'_>,
    cameras: &CellPlaneCameras<'_>,
    params: &CellPlaneParams,
) -> Vec<CellPlaneNormal> {
    let n_members = clusters.member_images.len();
    assert_eq!(clusters.member_status.len(), n_members);
    assert_eq!(clusters.member_positions.len(), n_members);
    assert_eq!(clusters.member_shapes.len(), n_members);
    assert_eq!(clusters.cell_shift_px.len(), n_members);
    assert_eq!(clusters.cell_status.len(), n_members);
    let n_clusters = clusters.reference_members.len();
    assert_eq!(
        clusters.cluster_starts.len(),
        n_clusters + 1,
        "cluster_starts must have C + 1 entries"
    );
    assert!(
        clusters
            .cluster_starts
            .windows(2)
            .all(|w| w[0] <= w[1] && w[1] as usize <= n_members),
        "cluster_starts must be non-decreasing and within the member count"
    );
    assert!(clusters.resolution >= 3, "resolution must be at least 3");
    let n_images = cameras.image_camera.len();
    assert_eq!(cameras.cam_from_world.len(), n_images);
    assert!(
        clusters
            .member_images
            .iter()
            .all(|&i| (i as usize) < n_images),
        "member image index out of range"
    );
    assert!(
        cameras
            .image_camera
            .iter()
            .all(|&c| (c as usize) < cameras.cameras.len()),
        "image camera index out of range"
    );

    let views: Vec<Option<View<'_>>> = (0..n_images)
        .map(|i| {
            let pose = cameras.cam_from_world[i].as_ref()?;
            let camera = &cameras.cameras[cameras.image_camera[i] as usize];
            let r_cw = pose.to_rotation_matrix();
            let r_wc = r_cw.transpose();
            let centre = -(r_wc * pose.translation);
            let (fx, fy) = camera.focal_lengths();
            Some(View {
                camera,
                r_wc,
                centre,
                focal_px: 0.5 * (fx + fy),
            })
        })
        .collect();

    let centres = cell_centres(clusters.resolution);
    let step = clusters.patch_size / clusters.resolution as f64;

    (0..n_clusters)
        .into_par_iter()
        .map(|c| fit_cluster(c, clusters, &views, &centres, step, params))
        .collect()
}

/// Cell centres in grid coordinates centred on the grid, `[x, y]`, row-major.
/// The same split the piecewise refinement measures:
/// `[0, R/3, R − R/3, R]` on each axis.
fn cell_centres(resolution: u32) -> [[f64; 2]; 9] {
    let bounds = grid_bounds(resolution);
    let mid = (resolution as f64 - 1.0) / 2.0;
    let span = |t: usize| (bounds[t] + bounds[t + 1] - 1) as f64 / 2.0 - mid;
    std::array::from_fn(|i| [span(i % 3), span(i / 3)])
}

/// The world ray through `pixel` of `view`; `None` when the camera model
/// returns no finite direction.
fn ray_through(view: &View<'_>, pixel: [f64; 2]) -> Option<Ray> {
    let d = view.camera.pixel_to_ray(pixel[0], pixel[1]);
    let d = view.r_wc * Vector3::new(d[0], d[1], d[2]);
    let len = d.norm();
    if !(len.is_finite() && len > 0.0) {
        return None;
    }
    Some(Ray {
        origin: view.centre,
        dir: d / len,
        focal_px: view.focal_px,
    })
}

/// `position + step · S · u`.
fn grid_pixel(position: [f32; 2], shape: &[[f32; 2]; 2], step: f64, u: [f64; 2]) -> [f64; 2] {
    let s = |r: usize, c: usize| f64::from(shape[r][c]);
    [
        f64::from(position[0]) + step * (s(0, 0) * u[0] + s(0, 1) * u[1]),
        f64::from(position[1]) + step * (s(1, 0) * u[0] + s(1, 1) * u[1]),
    ]
}

fn fit_cluster(
    c: usize,
    clusters: &CellPlaneClusters<'_>,
    views: &[Option<View<'_>>],
    centres: &[[f64; 2]; 9],
    step: f64,
    params: &CellPlaneParams,
) -> CellPlaneNormal {
    let mut out = CellPlaneNormal::empty();
    let reference = clusters.reference_members[c];
    if reference == NO_REFERENCE {
        return out;
    }
    let range = clusters.cluster_starts[c] as usize..clusters.cluster_starts[c + 1] as usize;
    let accepted = |s: ClusterCellStatus| {
        s == ClusterCellStatus::Fitted
            || (params.include_refused_outlier && s == ClusterCellStatus::RefusedOutlier)
    };

    // ── Rays and triangulation per cell ──────────────────────────────────
    let mut points: [Option<CellPoint>; 9] = Default::default();
    for (j, centre) in centres.iter().enumerate() {
        let mut rays: Vec<Ray> = Vec::new();
        let r = reference as usize;
        if let Some(view) = &views[clusters.member_images[r] as usize] {
            let px = grid_pixel(
                clusters.member_positions[r],
                &clusters.member_shapes[r],
                step,
                *centre,
            );
            rays.extend(ray_through(view, px));
        }
        for k in range.clone() {
            if clusters.member_status[k] != ClusterMemberStatus::Kept
                || !accepted(clusters.cell_status[k][j])
            {
                continue;
            }
            let Some(view) = &views[clusters.member_images[k] as usize] else {
                continue;
            };
            let d = clusters.cell_shift_px[k][j];
            if !(d[0].is_finite() && d[1].is_finite()) {
                continue;
            }
            let u = [centre[0] + f64::from(d[0]), centre[1] + f64::from(d[1])];
            let px = grid_pixel(
                clusters.member_positions[k],
                &clusters.member_shapes[k],
                step,
                u,
            );
            rays.extend(ray_through(view, px));
        }
        out.cell_rays[j] = rays.len() as u32;
        match triangulate(&rays, params) {
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

    // ── Determinacy ──────────────────────────────────────────────────────
    let max_w = weights.iter().copied().fold(0.0f64, f64::max);
    let live: Vec<usize> = (0..m)
        .filter(|&i| weights[i] > 0.0 && weights[i] >= LIVE_WEIGHT_FRACTION * max_w)
        .collect();
    // The live cells' spread, with equal weights: the verdict asks where the
    // cells are, not how sure the fit is of each.
    let mut mean = Vector3::zeros();
    for &i in &live {
        mean += pts[i].position;
    }
    mean /= live.len().max(1) as f64;
    let mut scatter = Matrix3::zeros();
    for &i in &live {
        let d = pts[i].position - mean;
        scatter += d * d.transpose();
    }
    let (evals, evecs) = sorted_eigen(&scatter);
    out.anisotropy = if evals[2] > EPS_SPREAD {
        evals[1] / evals[2]
    } else {
        0.0
    };
    if live.len() < params.min_cells || evals[2] <= EPS_SPREAD {
        return out;
    }
    if out.anisotropy >= params.det_aniso {
        out.normal = normal.into();
        out.determinacy = NormalDeterminacy::BothAxes;
    } else {
        let axis = evecs[2];
        // The cells fix only the normal's tilt along the line; about the line
        // it stays at the prior, the mean viewing direction.
        let mut n = view_dir - view_dir.dot(&axis) * axis;
        if n.norm() <= 1e-12 {
            n = normal - normal.dot(&axis) * axis;
        }
        let n = n / n.norm();
        out.normal = n.into();
        out.determinacy = NormalDeterminacy::OneAxis {
            free_axis: axis.into(),
        };
    }
    out
}

/// Least-squares nearest point of `rays`, with its covariance and its
/// residual in pixels; the failing status otherwise.
fn triangulate(
    rays: &[Ray],
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
    let mut a = Matrix3::zeros();
    let mut b = Vector3::zeros();
    for r in rays {
        let p = Matrix3::identity() - r.dir * r.dir.transpose();
        a += p;
        b += p * r.origin;
    }
    let Some(x) = a.try_inverse().map(|inv| inv * b) else {
        return Err(CellPlaneStatus::NarrowBaseline);
    };
    if !(x[0].is_finite() && x[1].is_finite() && x[2].is_finite()) {
        return Err(CellPlaneStatus::NarrowBaseline);
    }

    let mut info = Matrix3::zeros();
    let mut sum_e2 = 0.0;
    let mut toward = Vector3::zeros();
    for r in rays {
        let v = x - r.origin;
        let depth = v.dot(&r.dir);
        if depth <= 0.0 {
            return Err(CellPlaneStatus::BehindCamera);
        }
        let p = Matrix3::identity() - r.dir * r.dir.transpose();
        let perp = (p * v).norm();
        let k = r.focal_px / depth;
        sum_e2 += (perp * k).powi(2);
        info += (k * k) * p;
        toward -= v / v.norm();
    }
    // Two coordinates per ray, three for the point.
    let dof = (2 * rays.len() - 3) as f64;
    let residual_px = (sum_e2 / dof).sqrt();
    let sigma = residual_px.max(params.ray_noise_floor_px);
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
