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
//! ray-intersection residual, floored at the precision of a measured cell
//! shift. Alongside the normal the kernel reports which axes of it the cells
//! fix: both, one (the cells lie on a line, and the normal may still turn
//! about that line), or none. A cell counts toward that verdict when the
//! robust fit gives it any weight. A one-axis normal is the one perpendicular
//! to the line closest to the mean viewing direction, and a normal with no
//! fixed axis is `NaN`.
//!
//! This module builds each cell's rays from the cluster file's columns and
//! the ray-noise floor from each member's shape; the triangulation, the plane
//! fit and the determinacy verdict are the shared cell plane kernel,
//! `crate::patch::cell_plane_fit`.
//!
//! See `specs/core/patch/cell-plane-normals.md` for the design.

use nalgebra::Matrix2;
use rayon::prelude::*;
use sfmtool_matches_format::{
    ClusterCellStatus, ClusterMemberStatus, CLUSTER_REFERENCE_UNREFINABLE,
};

use crate::camera::CameraIntrinsics;
use crate::geometry::RigidTransform;
use crate::patch::cell_plane_fit::{fit_cell_plane, CellRay, PosedView};
use crate::patch::normal_refine::grid_cell_centres;

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
    /// Precision of a measured cell shift, in grid px. A ray's pixel noise is
    /// at least this shift mapped into its member's image,
    /// `cell_shift_precision_grid_px · (patch_size / R) · ‖S‖` with `‖S‖`
    /// the largest singular value of the member's stored shape, and a cell's
    /// residual is floored at the root mean square of its rays' floors.
    pub cell_shift_precision_grid_px: f64,
    /// Absolute floor on a cell's ray-intersection residual, in pixels, under
    /// the floor that [`Self::cell_shift_precision_grid_px`] sets.
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
            cell_shift_precision_grid_px: 0.1,
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
    /// `C` global member index of each cluster's reference, or
    /// [`CLUSTER_REFERENCE_UNREFINABLE`] for none.
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
    /// Fewer than [`CellPlaneParams::min_cells`] live cells (cells the robust
    /// plane fit gives any weight), or no spread.
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
    /// degrees-of-freedom corrected), before any floor; `NaN` where not
    /// triangulated.
    pub cell_residual_px: [f64; 9],
    /// Weighted RMS of the cells' distances from the plane, world units;
    /// `NaN` when no plane was fitted.
    pub plane_rms: f64,
    /// `λ_mid / λ_max` of the live cells' scatter, projected onto the fitted
    /// plane and with equal weights; `NaN` when no plane was fitted.
    pub anisotropy: f64,
    /// `(Σw)² / Σw²` over the weighted cells; `NaN` when no plane was fitted.
    pub n_eff: f64,
}

impl CellPlaneNormal {
    /// Nothing triangulated and no plane: every cell `TooFewRays` with no
    /// rays, every reading `NaN`.
    pub(in crate::patch) fn empty() -> Self {
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

/// Estimate a patch normal for every cluster from its stored cell
/// displacements and the images' poses.
///
/// For each cell, the reference member's ray through the cell's centre and,
/// for every `kept` member in a posed image whose cell status is `fitted` (and
/// `refused_outlier` with [`CellPlaneParams::include_refused_outlier`]), the
/// ray through `position + (patch_size / R) · S · (c + d)` are intersected in
/// the least-squares sense, each ray weighted by `(f / ρ)²` for its focal
/// length `f` and depth `ρ`. The positions are fitted with a plane (Tukey
/// IRLS, each cell weighted by the inverse of its position's variance along the
/// normal), and the result says which of the normal's axes the cells with a
/// nonzero Tukey weight fix, from their spread within the plane.
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

    let views: Vec<Option<PosedView<'_>>> = (0..n_images)
        .map(|i| {
            let pose = cameras.cam_from_world[i].as_ref()?;
            let camera = &cameras.cameras[cameras.image_camera[i] as usize];
            Some(PosedView::new(camera, pose))
        })
        .collect();

    let centres = cell_centres(clusters.resolution);
    let step = clusters.patch_size / clusters.resolution as f64;

    (0..n_clusters)
        .into_par_iter()
        .map(|c| fit_cluster(c, clusters, &views, &centres, step, params))
        .collect()
}

/// Cell centres in grid coordinates centred on the grid, `[x, y]`, row-major:
/// the centres the piecewise refinement measures its cells about.
fn cell_centres(resolution: u32) -> [[f64; 2]; 9] {
    let grid = grid_cell_centres(resolution);
    std::array::from_fn(|i| grid[i / 3][i % 3])
}

/// Largest singular value of a stored shape `S`: the most a unit step on the
/// patch grid can stretch to in the member's image.
fn shape_gain(shape: &[[f32; 2]; 2]) -> f64 {
    let m = Matrix2::new(
        f64::from(shape[0][0]),
        f64::from(shape[0][1]),
        f64::from(shape[1][0]),
        f64::from(shape[1][1]),
    );
    m.singular_values().max()
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
    views: &[Option<PosedView<'_>>],
    centres: &[[f64; 2]; 9],
    step: f64,
    params: &CellPlaneParams,
) -> CellPlaneNormal {
    let reference = clusters.reference_members[c];
    if reference == CLUSTER_REFERENCE_UNREFINABLE {
        return CellPlaneNormal::empty();
    }
    let range = clusters.cluster_starts[c] as usize..clusters.cluster_starts[c + 1] as usize;
    // A ray's pixel-noise floor: the cell-shift precision mapped through the
    // member's grid-to-pixel map.
    let ray_floor = |k: usize| {
        params.cell_shift_precision_grid_px * step * shape_gain(&clusters.member_shapes[k])
    };
    let accepted = |s: ClusterCellStatus| {
        s == ClusterCellStatus::Fitted
            || (params.include_refused_outlier && s == ClusterCellStatus::RefusedOutlier)
    };

    // The reference member's ray through each cell's centre, and each
    // accepted member's ray through where the cell's content lies in it.
    let cell_rays: [Vec<CellRay>; 9] = std::array::from_fn(|j| {
        let centre = centres[j];
        let mut rays: Vec<CellRay> = Vec::new();
        let r = reference as usize;
        if let Some(view) = &views[clusters.member_images[r] as usize] {
            let px = grid_pixel(
                clusters.member_positions[r],
                &clusters.member_shapes[r],
                step,
                centre,
            );
            rays.extend(view.ray_through(px, ray_floor(r)));
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
            rays.extend(view.ray_through(px, ray_floor(k)));
        }
        rays
    });
    fit_cell_plane(&cell_rays, params)
}

#[cfg(test)]
mod tests;
