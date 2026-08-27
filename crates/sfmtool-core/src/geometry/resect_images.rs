// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Re-estimate a set of images' poses against structure that did not depend on
//! any of them.
//!
//! [`resect_images`] is the whole mechanism of `specs/gui/gui-resect-image.md`:
//! the target set's contribution to structure is removed (every finite point
//! any target observes is re-triangulated from the *non-target* observations
//! alone, and every direction a target observes is re-derived from the
//! non-target rotations), the targets' poses are re-estimated against what is
//! left, and the points the set observes are re-triangulated once more at the
//! resected poses. The source reconstruction is never modified — the answer
//! comes back as a new reconstruction plus one [`ResectImageReport`] per
//! target and a [`ResectTotals`] over the set.
//!
//! Why the hold-out matters: a stored pose was fit jointly with the points it
//! observes, so it always agrees with them and its own residuals can never
//! falsify it. Structure re-triangulated without it can. Holding a *set* out
//! together extends that to a group of images that corroborate each other: a
//! point two targets share is re-triangulated from neither.
//!
//! Two correspondence sources, both feeding the same estimate (see
//! [`ResectSource`]): each target's own stored observations, or the match graph
//! of a `.matches` file, which admits points the reconstruction never assigned
//! to the target.
//!
//! Deterministic: the finite path is [`resect_images_batch`] at the caller's
//! seed (which seeds each image as a pure function of `(seed, image index)`),
//! the rotation-only path is a closed-form fit with a fixed trimming schedule,
//! and every gather below walks the reconstruction in storage order.

use std::collections::{HashMap, HashSet};

use nalgebra::{Matrix3, Point3, Quaternion, UnitQuaternion, Vector3};

use matches_format::MatchesData;

use crate::camera::CameraIntrinsics;
use crate::geometry::batch_resection::{resect_images_batch, ResectOptions};
use crate::geometry::focal_vote::column_scan::kabsch;
use crate::reconstruction::triangulation::triangulate_batch;
use crate::reconstruction::{
    ObservationSource, ReconstructionError, SfmrImage, SfmrReconstruction,
};

mod matches_join;

#[cfg(test)]
mod tests;

/// How many *non-target* posed images the source must carry before a target set
/// can be resected against the rest.
///
/// Three is the floor at which "the rest of the reconstruction" is a
/// reconstruction: two cameras fix structure only up to the pair's own
/// degenerate freedoms, and re-estimating a further pose against it measures the
/// pair rather than the scene.
pub const MIN_OTHER_POSED_IMAGES: usize = 3;

/// Bearings the rotation-only path needs before it will fit anything. Three
/// unit vectors fix a rotation; fewer leave it underdetermined.
pub const MIN_BEARINGS: usize = 3;

/// Trimming schedule of the rotation-only fit: rounds, and the fraction of the
/// bearings kept by residual angle in each round after the first.
///
/// Fixed rather than an option because the fit is closed-form — each round is
/// one 3×3 SVD — so there is no cost to spending all of them, and a caller
/// choosing between them would be choosing between answers rather than between
/// budgets.
const ROTATION_TRIM_ROUNDS: usize = 5;
/// Fraction of the bearings the rotation-only fit keeps per trimming round.
const ROTATION_KEEP_FRACTION: f64 = 0.6;

/// Inlier bound on a reprojection, in pixels — the batch-registration
/// primitive's own (`reconstruction_growth::INLIER_PX`), so the reported inlier
/// count and the fraction the gate is applied to are the same measurement.
const INLIER_PX: f64 = 3.0;

/// One 2D–3D pair the estimate is fit to: the point it stands for, the pixel
/// the target observed it at, and the world position it is scored against.
type Correspondence = (usize, [f64; 2], [f64; 3]);

/// One bearing correspondence of the rotation-only path: the point's held-out
/// world direction, and the unit ray the target sees it along.
type BearingPair = (Vector3<f64>, Vector3<f64>);

/// Where a target image's 2D–3D pairs come from.
#[derive(Clone, Copy)]
pub enum ResectSource<'a> {
    /// Each target's own observations, joined to the held-out positions of the
    /// points the set observes. The default.
    StoredObservations,
    /// The match graph of a parsed `.matches` file: each target's keypoints, to
    /// matched keypoints in the non-target posed images, to the points those
    /// images' observations stand for. Admits points the reconstruction never
    /// assigned to the target, and requires a `sift_files` reconstruction
    /// (match rows are joined through feature indexes).
    Matches(&'a MatchesData),
}

impl ResectSource<'_> {
    /// The provenance string recorded in the derived reconstruction's metadata,
    /// and the word the status line uses.
    pub fn name(&self) -> &'static str {
        match self {
            ResectSource::StoredObservations => "observations",
            ResectSource::Matches(_) => "matches",
        }
    }
}

/// Settings of [`resect_images`].
#[derive(Clone, Debug, Default)]
pub struct ResectImageOptions {
    /// The batch-registration primitive's own options: the observation floor
    /// below which the finite path is unavailable, the acceptance gate on the
    /// all-observation inlier fraction, and the RANSAC seed.
    pub resect: ResectOptions,
}

/// What one target's resection did, in the quantities the caller reports.
///
/// Every count is *this target's* share: the points it observes, the pairs its
/// own estimate saw. Points two targets share are counted in both reports, and
/// once each in [`ResectTotals`].
#[derive(Clone, Debug, PartialEq)]
pub struct ResectImageReport {
    /// Index of the resected image in the source reconstruction.
    pub image_index: usize,
    /// The resected image's workspace-relative name.
    pub image_name: String,
    /// Which correspondence source produced the 2D–3D pairs
    /// ([`ResectSource::name`]).
    pub source: &'static str,
    /// Whether the rotation-only path ran (the finite support was below
    /// `ResectOptions::min_obs`, or the reconstruction is rotation-only).
    pub rotation_only: bool,
    /// 2D–3D pairs the estimate saw.
    pub correspondences: usize,
    /// How many of them the resected pose puts within the batch-registration
    /// primitive's 3 px inlier bound (`INLIER_PX`); the rotation-only path
    /// counts bearings within that bound's angular equivalent on this camera.
    pub inliers: usize,
    /// `inliers / correspondences` — the fraction the acceptance gate was
    /// applied to.
    pub inlier_fraction: f64,
    /// Whether the estimate cleared `ResectOptions::accept_gate`. When false,
    /// this image keeps its stored pose while the rest of the set proceeds.
    pub accepted: bool,
    /// Why the estimate was refused, when it was.
    pub refusal: Option<String>,
    /// Angle between the stored and resected world-to-camera rotations,
    /// degrees.
    pub rotation_deg: f64,
    /// Distance between the stored and resected camera centres, in the
    /// reconstruction's own units.
    pub translation: f64,
    /// [`ResectImageReport::translation`] in units of
    /// [`ResectImageReport::scene_scale`]; `None` when the scale is
    /// undefined (a rotation-only reconstruction has no camera-to-structure
    /// distance).
    pub translation_scene: Option<f64>,
    /// The source's median over images of that image's median
    /// camera-to-structure distance — the unit the evaluation channels report
    /// displacements in.
    pub scene_scale: Option<f64>,
    /// Points this target observes that the non-target images could
    /// re-triangulate without the set (step 2).
    pub held_out_points: usize,
    /// Points this target observes that were re-triangulated at the resected
    /// poses (step 4). Zero where no accepted target observes them, which
    /// leaves the held-out positions standing.
    pub retriangulated: usize,
    /// Points this target observes that were dropped: neither the hold-out nor
    /// the re-triangulation at the new poses could place them.
    pub removed_points: usize,
}

/// The set's totals, counting each point once however many targets observe it.
#[derive(Clone, Debug, PartialEq)]
pub struct ResectTotals {
    /// Targets the call was asked for.
    pub targets: usize,
    /// Targets whose estimate cleared the gate.
    pub accepted: usize,
    /// Targets whose estimate was refused; their stored poses were kept.
    pub refused: usize,
    /// 2D–3D pairs summed over the targets' estimates.
    pub correspondences: usize,
    /// Inliers summed over the targets' estimates.
    pub inliers: usize,
    /// `inliers / correspondences` over the whole set; `0.0` when the set saw
    /// no correspondences at all.
    pub inlier_fraction: f64,
    /// Distinct points the non-target images could re-triangulate without the
    /// set.
    pub held_out_points: usize,
    /// Distinct points re-triangulated at the resected poses.
    pub retriangulated: usize,
    /// Distinct points dropped.
    pub removed_points: usize,
    /// The source's scene scale ([`ResectImageReport::scene_scale`]).
    pub scene_scale: Option<f64>,
}

/// A resected image set: the derived reconstruction, one report per target in
/// the order the targets were asked for, and the set's totals.
pub struct ResectedImages {
    /// The derived reconstruction — the source, with the targets' poses and the
    /// points they observe replaced. Everything else is copied unchanged.
    pub reconstruction: SfmrReconstruction,
    /// One report per target, in the order the caller listed them.
    pub reports: Vec<ResectImageReport>,
    /// The set's totals.
    pub totals: ResectTotals,
}

/// The reports alone — an [`SfmrReconstruction`] is not `Debug`, and what a
/// caller (or a failing assertion) wants to see is what the estimates did.
impl std::fmt::Debug for ResectedImages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResectedImages")
            .field("reports", &self.reports)
            .field("totals", &self.totals)
            .finish_non_exhaustive()
    }
}

/// Why a resection could not be attempted at all.
///
/// Distinct from a *refused estimate*, which is one target's outcome: it leaves
/// that target's stored pose standing, still produces the derived
/// reconstruction, and reports itself through [`ResectImageReport::refusal`].
/// Everything here is a property of the call rather than of one estimate.
#[derive(Debug)]
pub enum ResectImageError {
    /// The target set is empty.
    NoTargets,
    /// An index is not an image of this reconstruction.
    ImageOutOfRange {
        /// The index asked for.
        index: usize,
        /// How many images the reconstruction has.
        count: usize,
    },
    /// The same image was named twice in the target set.
    DuplicateTarget(usize),
    /// A target image carries no usable pose (a non-finite quaternion or
    /// translation).
    NotPosed(usize),
    /// Fewer than [`MIN_OTHER_POSED_IMAGES`] non-target images are posed.
    TooFewPosedImages(usize),
    /// The 2D observations could not be read (a missing or short `.sift` file).
    Observations(ReconstructionError),
    /// The `.matches` source could not be joined to this reconstruction.
    Matches(String),
}

impl std::fmt::Display for ResectImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResectImageError::NoTargets => write!(f, "no target images were named"),
            ResectImageError::ImageOutOfRange { index, count } => {
                write!(f, "image {index} is out of range ({count} images)")
            }
            ResectImageError::DuplicateTarget(index) => {
                write!(f, "image {index} is named twice in the target set")
            }
            ResectImageError::NotPosed(index) => write!(f, "image {index} is not posed"),
            ResectImageError::TooFewPosedImages(n) => write!(
                f,
                "only {n} non-target posed image{} ({MIN_OTHER_POSED_IMAGES} needed)",
                if *n == 1 { "" } else { "s" }
            ),
            ResectImageError::Observations(e) => write!(f, "{e}"),
            ResectImageError::Matches(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ResectImageError {}

impl From<ReconstructionError> for ResectImageError {
    fn from(value: ReconstructionError) -> Self {
        ResectImageError::Observations(value)
    }
}

/// Re-estimate the poses of `image_indexes` against structure held out from all
/// of them, and return the result as a new reconstruction.
///
/// The whole mechanism of `specs/gui/gui-resect-image.md`:
///
/// 1. **Held-out structure.** Every finite point any target observes that keeps
///    at least two *non-target* observations is re-triangulated from those
///    alone, at the non-target images' stored poses; the stored position is
///    discarded. A point with fewer than two non-target observations has no
///    held-out position and is excluded from the estimates. A point at infinity
///    is a direction, which one rotation already fixes, so its held-out bearing
///    is the mean of the non-target observations' world rays.
/// 2. **Pose estimates.** A target with at least `ResectOptions::min_obs`
///    held-out finite correspondences takes the finite path:
///    [`resect_images_batch`] resects every such target of a shared camera
///    together, against the held-out structure. A target below that floor takes
///    the rotation-only path — its rotation is fit in closed form to the
///    bearings it observes (trimmed, iterated) and its translation is left at
///    its stored value. Each estimate is accepted or refused on
///    `ResectOptions::accept_gate`, independently of the others; a refusal
///    keeps that image's stored pose and reports itself rather than failing the
///    call.
/// 3. **Re-triangulation.** With the accepted targets at their resected poses,
///    the finite points they observe are re-triangulated from *all* their
///    observations. A point that fails keeps its held-out position when it has
///    one, and is otherwise dropped with its observations. No bundle adjustment
///    runs.
///
/// The derived reconstruction's metadata records the operation, the targets,
/// the correspondence source and the estimates' inlier fractions, so a later
/// save carries provenance.
///
/// Errors only on a property of the call itself — see [`ResectImageError`].
pub fn resect_images(
    recon: &SfmrReconstruction,
    image_indexes: &[usize],
    source: ResectSource<'_>,
    options: &ResectImageOptions,
) -> Result<ResectedImages, ResectImageError> {
    let count = recon.images.len();
    if image_indexes.is_empty() {
        return Err(ResectImageError::NoTargets);
    }
    let posed: Vec<bool> = recon.images.iter().map(is_posed).collect();
    let mut is_target = vec![false; count];
    for &t in image_indexes {
        if t >= count {
            return Err(ResectImageError::ImageOutOfRange { index: t, count });
        }
        if !posed[t] {
            return Err(ResectImageError::NotPosed(t));
        }
        if std::mem::replace(&mut is_target[t], true) {
            return Err(ResectImageError::DuplicateTarget(t));
        }
    }
    let others = (0..count).filter(|&i| posed[i] && !is_target[i]).count();
    if others < MIN_OTHER_POSED_IMAGES {
        return Err(ResectImageError::TooFewPosedImages(others));
    }
    // The non-target posed images, as the mask every hold-out and every join
    // reads: "posed, and not one of the images being questioned".
    let posed_others: Vec<bool> = (0..count).map(|i| posed[i] && !is_target[i]).collect();

    // ── The targets' own observations, and the points behind them ──────────
    // `tracks` is sorted by point then image, so each target's rows are already
    // ascending in point index — which is what the batch primitive's cluster
    // ordering wants.
    let mut target_rows: HashMap<usize, Vec<usize>> =
        image_indexes.iter().map(|&t| (t, Vec::new())).collect();
    for (row, obs) in recon.tracks.iter().enumerate() {
        let image = obs.image_index as usize;
        if is_target[image] {
            target_rows.get_mut(&image).expect("target row").push(row);
        }
    }
    let observed: HashMap<usize, Vec<usize>> = target_rows
        .iter()
        .map(|(&t, rows)| {
            (
                t,
                rows.iter()
                    .map(|&row| recon.tracks[row].point_index as usize)
                    .collect(),
            )
        })
        .collect();

    // Every point the set observes, once, in storage order.
    let mut all_observed: Vec<usize> = observed.values().flatten().copied().collect();
    all_observed.sort_unstable();
    all_observed.dedup();

    // Every observation row of every one of those points: the hold-out reads
    // them, the re-triangulation at the new poses reads them again, and both
    // want the pixels exactly once.
    let mut gathered: Vec<usize> = Vec::new();
    for &p in &all_observed {
        gathered.extend(recon.observation_offsets[p]..recon.observation_offsets[p + 1]);
    }
    gathered.sort_unstable();
    gathered.dedup();
    let pixels = observation_pixels(recon, &gathered)?;
    let pixel_of: HashMap<usize, [f64; 2]> = gathered.iter().copied().zip(pixels).collect();

    // ── Step 1: held-out structure ─────────────────────────────────────────
    let finite_observed: Vec<usize> = all_observed
        .iter()
        .copied()
        .filter(|&p| !recon.points[p].is_at_infinity())
        .collect();
    let no_replacement: Vec<Option<Pose>> = vec![None; count];
    let held_out_list = triangulate_points(
        recon,
        &finite_observed,
        &pixel_of,
        &posed_others,
        &no_replacement,
    );
    let held_out: HashMap<usize, [f64; 3]> = finite_observed
        .iter()
        .zip(&held_out_list)
        .filter_map(|(&p, position)| position.map(|w| (p, w)))
        .collect();
    let bearing_of = held_out_bearings(recon, &all_observed, &pixel_of, &posed_others);
    let observed_set: HashSet<usize> = all_observed.iter().copied().collect();

    // ── Step 2: the pose estimates ─────────────────────────────────────────
    let mut pairs_of: HashMap<usize, Vec<Correspondence>> = HashMap::new();
    for &t in image_indexes {
        let pairs = match source {
            ResectSource::StoredObservations => {
                let rows = &target_rows[&t];
                let points = &observed[&t];
                let mut out = Vec::with_capacity(rows.len());
                for (k, &row) in rows.iter().enumerate() {
                    let p = points[k];
                    if let Some(&world) = held_out.get(&p) {
                        out.push((p, pixel_of[&row], world));
                    }
                }
                out
            }
            ResectSource::Matches(matches) => {
                let mut out = matches_join::correspondences(recon, t, matches, &posed_others)?;
                // A matched point the set also observes must be scored against
                // the *held-out* position, not the stored one it helped fit.
                out.retain_mut(|(p, _, world)| {
                    if !observed_set.contains(p) {
                        // Not a point the set observes at all: its stored
                        // position owes the targets nothing.
                        return true;
                    }
                    match held_out.get(p) {
                        Some(&h) => {
                            *world = h;
                            true
                        }
                        // Nothing held out could place it; a pair scored against
                        // the set's own contribution is no evidence.
                        None => false,
                    }
                });
                out
            }
        };
        pairs_of.insert(t, pairs);
    }

    let bearings_of: HashMap<usize, Vec<BearingPair>> = image_indexes
        .iter()
        .map(|&t| {
            let camera = &recon.cameras[recon.images[t].camera_index as usize];
            (
                t,
                gather_bearings(
                    &target_rows[&t],
                    &observed[&t],
                    &pixel_of,
                    &bearing_of,
                    camera,
                ),
            )
        })
        .collect();

    // The finite path runs as one batch per camera model, so every target that
    // shares a camera is resected against the held-out structure together.
    let finite_targets: Vec<usize> = image_indexes
        .iter()
        .copied()
        .filter(|t| pairs_of[t].len() >= options.resect.min_obs)
        .collect();
    let mut estimates = finite_estimates(
        recon,
        &finite_targets,
        &pairs_of,
        &posed_others,
        &gathered,
        &pixel_of,
        options,
    );
    for &t in image_indexes {
        if estimates.contains_key(&t) {
            continue;
        }
        let bearings = &bearings_of[&t];
        let camera = &recon.cameras[recon.images[t].camera_index as usize];
        let estimate = if bearings.len() >= MIN_BEARINGS {
            rotation_estimate(recon, t, bearings, camera, options)
        } else {
            Estimate::no_support(recon, t, pairs_of[&t].len(), bearings.len())
        };
        estimates.insert(t, estimate);
    }

    // ── Step 3: re-triangulation at the resected poses ─────────────────────
    let mut replacement: Vec<Option<Pose>> = vec![None; count];
    let mut accepted_targets: Vec<usize> = Vec::new();
    for &t in image_indexes {
        let estimate = &estimates[&t];
        if estimate.accepted {
            replacement[t] = Some((estimate.rotation, estimate.translation));
            accepted_targets.push(t);
        }
    }
    // Only the points an accepted target observes have a new pose to be
    // re-triangulated at; the rest keep their held-out positions.
    let mut refit_points: Vec<usize> = accepted_targets
        .iter()
        .flat_map(|t| observed[t].iter().copied())
        .filter(|&p| !recon.points[p].is_at_infinity())
        .collect();
    refit_points.sort_unstable();
    refit_points.dedup();
    // Every posed image contributes, the accepted targets at their new poses.
    let refit_list = triangulate_points(recon, &refit_points, &pixel_of, &posed, &replacement);
    let refit: HashMap<usize, [f64; 3]> = refit_points
        .iter()
        .zip(&refit_list)
        .filter_map(|(&p, position)| position.map(|w| (p, w)))
        .collect();

    // ── The derived reconstruction ─────────────────────────────────────────
    let mut out = recon.clone();
    for &t in image_indexes {
        let estimate = &estimates[&t];
        if estimate.accepted {
            out.images[t].quaternion_wxyz = estimate.rotation;
            out.images[t].translation_xyz = estimate.translation;
        }
    }
    let mut retriangulated: HashSet<usize> = HashSet::new();
    let mut drop_mask = vec![false; recon.points.len()];
    for &p in &finite_observed {
        match (refit.get(&p), held_out.get(&p)) {
            (Some(&position), _) => {
                out.points[p].position = Point3::new(position[0], position[1], position[2]);
                retriangulated.insert(p);
            }
            (None, Some(&position)) => {
                out.points[p].position = Point3::new(position[0], position[1], position[2]);
            }
            (None, None) => drop_mask[p] = true,
        }
    }
    let removed: HashSet<usize> = drop_mask
        .iter()
        .enumerate()
        .filter_map(|(p, &d)| d.then_some(p))
        .collect();
    if !removed.is_empty() {
        let keep: Vec<bool> = drop_mask.iter().map(|&d| !d).collect();
        out = out.filter_points_by_mask(&keep);
    }

    // ── The reports ────────────────────────────────────────────────────────
    let scale = scene_scale(recon);
    let reports: Vec<ResectImageReport> = image_indexes
        .iter()
        .map(|&t| {
            let estimate = &estimates[&t];
            let stored = &recon.images[t];
            let (rotation, translation) = if estimate.accepted {
                (estimate.rotation, estimate.translation)
            } else {
                (stored.quaternion_wxyz, stored.translation_xyz)
            };
            let centre_delta =
                (world_centre(&rotation, &translation) - stored.camera_center()).norm();
            let mine = &observed[&t];
            ResectImageReport {
                image_index: t,
                image_name: stored.name.clone(),
                source: source.name(),
                rotation_only: estimate.rotation_only,
                correspondences: estimate.correspondences,
                inliers: estimate.inliers,
                inlier_fraction: estimate.inlier_fraction,
                accepted: estimate.accepted,
                refusal: estimate.refusal.clone(),
                rotation_deg: rotation
                    .rotation_to(&stored.quaternion_wxyz)
                    .angle()
                    .to_degrees(),
                translation: centre_delta,
                translation_scene: scale.map(|s| centre_delta / s),
                scene_scale: scale,
                held_out_points: distinct(mine, |p| held_out.contains_key(&p)),
                retriangulated: distinct(mine, |p| retriangulated.contains(&p)),
                removed_points: distinct(mine, |p| removed.contains(&p)),
            }
        })
        .collect();

    let correspondences: usize = reports.iter().map(|r| r.correspondences).sum();
    let inliers: usize = reports.iter().map(|r| r.inliers).sum();
    let accepted = reports.iter().filter(|r| r.accepted).count();
    let totals = ResectTotals {
        targets: reports.len(),
        accepted,
        refused: reports.len() - accepted,
        correspondences,
        inliers,
        inlier_fraction: if correspondences == 0 {
            0.0
        } else {
            inliers as f64 / correspondences as f64
        },
        held_out_points: held_out.len(),
        retriangulated: retriangulated.len(),
        removed_points: removed.len(),
        scene_scale: scale,
    };
    write_provenance(&mut out, &reports, &totals);
    Ok(ResectedImages {
        reconstruction: out,
        reports,
        totals,
    })
}

/// How many distinct members of `points` satisfy `predicate`.
fn distinct(points: &[usize], predicate: impl Fn(usize) -> bool) -> usize {
    let mut seen: HashSet<usize> = HashSet::new();
    points
        .iter()
        .filter(|&&p| predicate(p) && seen.insert(p))
        .count()
}

/// A world-to-camera pose: rotation and translation.
type Pose = (UnitQuaternion<f64>, Vector3<f64>);

/// Whether an image row carries a pose at all. Every `.sfmr` image has the
/// fields; a non-finite one is a placeholder rather than a registration.
fn is_posed(image: &SfmrImage) -> bool {
    image.quaternion_wxyz.coords.iter().all(|c| c.is_finite())
        && image.translation_xyz.iter().all(|c| c.is_finite())
}

/// The camera centre of a world-to-camera pose: `C = -Rᵀ t`.
fn world_centre(rotation: &UnitQuaternion<f64>, translation: &Vector3<f64>) -> Point3<f64> {
    Point3::from(-(rotation.inverse() * translation))
}

/// The 2D pixel of each requested observation row, in the same order.
///
/// An `embedded_patches` reconstruction carries them inline; a `sift_files` one
/// keeps them in the images' `.sift` companions, which are read once per image
/// touched rather than once per row.
fn observation_pixels(
    recon: &SfmrReconstruction,
    rows: &[usize],
) -> Result<Vec<[f64; 2]>, ReconstructionError> {
    match &recon.observations {
        ObservationSource::EmbeddedPatches { keypoints_xy, .. } => Ok(rows
            .iter()
            .map(|&row| [keypoints_xy[[row, 0]] as f64, keypoints_xy[[row, 1]] as f64])
            .collect()),
        ObservationSource::SiftFiles {
            feature_indexes, ..
        } => {
            let mut images: Vec<usize> = rows
                .iter()
                .map(|&row| recon.tracks[row].image_index as usize)
                .collect();
            images.sort_unstable();
            images.dedup();
            let mut positions: HashMap<usize, Vec<[f32; 2]>> = HashMap::new();
            for image in images {
                positions.insert(image, read_sift_positions(recon, image)?);
            }
            rows.iter()
                .map(|&row| {
                    let image = recon.tracks[row].image_index as usize;
                    let feature = feature_indexes[row] as usize;
                    positions[&image]
                        .get(feature)
                        .map(|p| [p[0] as f64, p[1] as f64])
                        .ok_or_else(|| ReconstructionError::SiftRead {
                            path: recon.sift_path_for_image(image),
                            source: format!(
                                "observation {row} references feature {feature}, beyond the file"
                            ),
                        })
                })
                .collect()
        }
    }
}

/// One image's `.sift` feature positions, read up to the highest feature index
/// any track of that image references.
fn read_sift_positions(
    recon: &SfmrReconstruction,
    image: usize,
) -> Result<Vec<[f32; 2]>, ReconstructionError> {
    let path = recon.sift_path_for_image(image);
    let count = recon.max_track_feature_index[image] as usize + 1;
    sift_format::read_sift_positions(&path, count).map_err(|e| ReconstructionError::SiftRead {
        path,
        source: e.to_string(),
    })
}

/// Ray-midpoint triangulation of `points`, one entry per input point.
///
/// `contributes` is the per-image mask of whose observations are read — the
/// hold-out passes the non-target posed images, the re-triangulation at the new
/// poses passes every posed image. `replace` substitutes a pose per image, so
/// the same gather serves both. A point is placed only when at least two
/// observations survive, the solve puts it in front of every one of them, and
/// its depth is observable at all (parallel rays leave the normal matrix
/// rank-deficient, which the triangulation reports as an infinite condition
/// number).
fn triangulate_points(
    recon: &SfmrReconstruction,
    points: &[usize],
    pixel_of: &HashMap<usize, [f64; 2]>,
    contributes: &[bool],
    replace: &[Option<Pose>],
) -> Vec<Option<[f64; 3]>> {
    let mut dirs: Vec<Vector3<f64>> = Vec::new();
    let mut centers: Vec<Point3<f64>> = Vec::new();
    let mut offsets: Vec<usize> = vec![0];
    for &p in points {
        for row in recon.observation_offsets[p]..recon.observation_offsets[p + 1] {
            let image = recon.tracks[row].image_index as usize;
            if !contributes[image] {
                continue;
            }
            let Some(uv) = pixel_of.get(&row) else {
                continue;
            };
            let (rotation, translation) = match replace[image] {
                Some(pose) => pose,
                None => (
                    recon.images[image].quaternion_wxyz,
                    recon.images[image].translation_xyz,
                ),
            };
            let camera = &recon.cameras[recon.images[image].camera_index as usize];
            let ray = camera.pixel_to_ray(uv[0], uv[1]);
            let world = rotation.inverse() * Vector3::new(ray[0], ray[1], ray[2]);
            let norm = world.norm();
            if norm <= 0.0 || norm.is_nan() {
                continue;
            }
            dirs.push(world / norm);
            centers.push(world_centre(&rotation, &translation));
        }
        offsets.push(dirs.len());
    }

    triangulate_batch(&dirs, &centers, &offsets)
        .into_iter()
        .enumerate()
        .map(|(t, tri)| {
            let usable = offsets[t + 1] - offsets[t] >= 2
                && tri.in_front_of_all_cameras
                && tri.condition_number.is_finite()
                && tri.point.coords.iter().all(|c| c.is_finite());
            usable.then_some([tri.point.x, tri.point.y, tri.point.z])
        })
        .collect()
}

/// The held-out direction of every point at infinity the target set observes:
/// the mean of the world rays the *non-target* images see it along.
///
/// A direction is fixed by one rotation, so a single non-target observation is
/// already a held-out bearing; a point at infinity none of them observes has
/// none, and the rotation-only path does not see it.
fn held_out_bearings(
    recon: &SfmrReconstruction,
    all_observed: &[usize],
    pixel_of: &HashMap<usize, [f64; 2]>,
    posed_others: &[bool],
) -> HashMap<usize, Vector3<f64>> {
    let mut out = HashMap::new();
    for &p in all_observed {
        if !recon.points[p].is_at_infinity() {
            continue;
        }
        let mut mean = Vector3::zeros();
        for row in recon.observation_offsets[p]..recon.observation_offsets[p + 1] {
            let image = recon.tracks[row].image_index as usize;
            if !posed_others[image] {
                continue;
            }
            let Some(uv) = pixel_of.get(&row) else {
                continue;
            };
            let camera = &recon.cameras[recon.images[image].camera_index as usize];
            let ray = camera.pixel_to_ray(uv[0], uv[1]);
            let world = recon.images[image].quaternion_wxyz.inverse()
                * Vector3::new(ray[0], ray[1], ray[2]);
            let norm = world.norm();
            if norm > 0.0 && !norm.is_nan() {
                mean += world / norm;
            }
        }
        let norm = mean.norm();
        if norm > 0.0 && !norm.is_nan() {
            out.insert(p, mean / norm);
        }
    }
    out
}

/// One target's observations of points at infinity, as
/// `(held-out world bearing, camera ray)` pairs — the rotation-only path's
/// whole input.
fn gather_bearings(
    target_rows: &[usize],
    observed: &[usize],
    pixel_of: &HashMap<usize, [f64; 2]>,
    bearing_of: &HashMap<usize, Vector3<f64>>,
    camera: &CameraIntrinsics,
) -> Vec<BearingPair> {
    let mut out = Vec::new();
    for (k, &row) in target_rows.iter().enumerate() {
        let Some(&bearing) = bearing_of.get(&observed[k]) else {
            continue;
        };
        let Some(uv) = pixel_of.get(&row) else {
            continue;
        };
        let ray = camera.pixel_to_ray(uv[0], uv[1]);
        let ray = Vector3::new(ray[0], ray[1], ray[2]);
        let rn = ray.norm();
        if rn > 0.0 {
            out.push((bearing, ray / rn));
        }
    }
    out
}

/// The outcome of one pose estimate, before it reaches the reconstruction.
struct Estimate {
    rotation: UnitQuaternion<f64>,
    translation: Vector3<f64>,
    rotation_only: bool,
    correspondences: usize,
    inliers: usize,
    inlier_fraction: f64,
    accepted: bool,
    refusal: Option<String>,
}

impl Estimate {
    /// A target neither path has support for: too few held-out finite points for
    /// the finite path, and too few bearings for the rotation-only one. Its
    /// stored pose stands, and the hold-out is still what the derived
    /// reconstruction shows for the points it observes.
    fn no_support(
        recon: &SfmrReconstruction,
        image_index: usize,
        finite: usize,
        bearings: usize,
    ) -> Self {
        Estimate {
            rotation: recon.images[image_index].quaternion_wxyz,
            translation: recon.images[image_index].translation_xyz,
            rotation_only: false,
            correspondences: finite,
            inliers: 0,
            inlier_fraction: 0.0,
            accepted: false,
            refusal: Some(format!(
                "no support: {finite} held-out finite point{} and {bearings} bearing{}",
                if finite == 1 { "" } else { "s" },
                if bearings == 1 { "" } else { "s" }
            )),
        }
    }
}

/// The finite path: [`resect_images_batch`] over every target of a shared
/// camera, against the held-out positions.
///
/// The observation arrays it is handed are the targets' correspondences plus —
/// for the covisibility ranking its neighbour-initialized fallback uses — the
/// non-target images' own observations of the same points. Point indexes double
/// as cluster ids, so the primitive's `points` row for a cluster is that point's
/// held-out position. Targets are grouped by camera because the primitive
/// resects one camera model at a time; each image's RANSAC is seeded from its
/// own index, so the grouping does not change any answer.
fn finite_estimates(
    recon: &SfmrReconstruction,
    targets: &[usize],
    pairs_of: &HashMap<usize, Vec<Correspondence>>,
    posed_others: &[bool],
    gathered: &[usize],
    pixel_of: &HashMap<usize, [f64; 2]>,
    options: &ResectImageOptions,
) -> HashMap<usize, Estimate> {
    let mut out = HashMap::new();
    if targets.is_empty() {
        return out;
    }

    // Held-out positions of every point any target is scored against — shared
    // by all the groups below, so a point means the same thing to each.
    let mut points = vec![[f64::NAN; 3]; recon.points.len()];
    for t in targets {
        for &(p, _, world) in &pairs_of[t] {
            points[p] = world;
        }
    }

    let posed_indexes: Vec<u32> = (0..recon.images.len() as u32)
        .filter(|&i| posed_others[i as usize])
        .collect();
    let posed_quaternions: Vec<[f64; 4]> = posed_indexes
        .iter()
        .map(|&i| {
            let q = recon.images[i as usize].quaternion_wxyz.into_inner();
            [q.w, q.i, q.j, q.k]
        })
        .collect();
    let posed_translations: Vec<[f64; 3]> = posed_indexes
        .iter()
        .map(|&i| {
            let t = recon.images[i as usize].translation_xyz;
            [t.x, t.y, t.z]
        })
        .collect();

    // Group the targets by camera model, keeping the caller's order within each
    // group.
    let mut cameras: Vec<u32> = targets
        .iter()
        .map(|&t| recon.images[t].camera_index)
        .collect();
    cameras.sort_unstable();
    cameras.dedup();

    for camera_index in cameras {
        let group: Vec<usize> = targets
            .iter()
            .copied()
            .filter(|&t| recon.images[t].camera_index == camera_index)
            .collect();
        let camera = &recon.cameras[camera_index as usize];

        // (cluster, image, pixel) rows, sorted by cluster: the group's
        // correspondences, and every non-target posed image's observation of a
        // point that has a held-out position.
        let mut rows: Vec<(u32, u32, [f64; 2])> = Vec::new();
        for &t in &group {
            rows.extend(
                pairs_of[&t]
                    .iter()
                    .map(|&(p, uv, _)| (p as u32, t as u32, uv)),
            );
        }
        for &row in gathered {
            let image = recon.tracks[row].image_index as usize;
            let p = recon.tracks[row].point_index as usize;
            if !posed_others[image] || !points[p][0].is_finite() {
                continue;
            }
            rows.push((p as u32, image as u32, pixel_of[&row]));
        }
        rows.sort_by_key(|&(cluster, image, _)| (cluster, image));

        let cluster_indexes: Vec<u32> = rows.iter().map(|r| r.0).collect();
        let image_indexes: Vec<u32> = rows.iter().map(|r| r.1).collect();
        let positions_xy: Vec<[f64; 2]> = rows.iter().map(|r| r.2).collect();
        let image_list: Vec<u32> = group.iter().map(|&t| t as u32).collect();

        let batch = resect_images_batch(
            &cluster_indexes,
            &image_indexes,
            &positions_xy,
            camera,
            &points,
            &image_list,
            &posed_quaternions,
            &posed_translations,
            &posed_indexes,
            &options.resect,
        );

        for (slot, &t) in group.iter().enumerate() {
            let q = batch.quaternions_wxyz[slot];
            let rotation = UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]));
            let translation = Vector3::new(
                batch.translations[slot][0],
                batch.translations[slot][1],
                batch.translations[slot][2],
            );
            let pairs = &pairs_of[&t];
            let inliers = pairs
                .iter()
                .filter(|&&(_, uv, world)| {
                    let local = rotation * Vector3::new(world[0], world[1], world[2]) + translation;
                    camera
                        .ray_to_pixel([local.x, local.y, local.z])
                        .is_some_and(|(u, v)| (u - uv[0]).hypot(v - uv[1]) < INLIER_PX)
                })
                .count();
            let accepted = batch.accepted[slot];
            out.insert(
                t,
                Estimate {
                    rotation,
                    translation,
                    rotation_only: false,
                    correspondences: pairs.len(),
                    inliers,
                    inlier_fraction: batch.inlier_fractions[slot],
                    accepted,
                    refusal: (!accepted).then(|| {
                        format!(
                            "inlier fraction {:.2} below the {:.2} gate",
                            batch.inlier_fractions[slot], options.resect.accept_gate
                        )
                    }),
                },
            );
        }
    }
    out
}

/// The rotation-only path: closed-form absolute orientation between the
/// target's observed rays and the held-out bearings of the points at infinity
/// it observes, trimmed and iterated. The translation is left at its stored
/// value.
///
/// Refuses when the bearings span no angle a pixel of this camera could
/// resolve — a spread below the camera's own per-pixel angle is not a spread,
/// and Wahba's problem is undetermined for parallel bearings.
fn rotation_estimate(
    recon: &SfmrReconstruction,
    image_index: usize,
    bearings: &[BearingPair],
    camera: &CameraIntrinsics,
    options: &ResectImageOptions,
) -> Estimate {
    let stored = &recon.images[image_index];
    let degenerate = |n: usize| Estimate {
        rotation: stored.quaternion_wxyz,
        translation: stored.translation_xyz,
        rotation_only: true,
        correspondences: n,
        inliers: 0,
        inlier_fraction: 0.0,
        accepted: false,
        refusal: Some("the bearings span no measurable angle".to_string()),
    };

    let world: Vec<Vector3<f64>> = bearings.iter().map(|b| b.0).collect();
    let rays: Vec<Vector3<f64>> = bearings.iter().map(|b| b.1).collect();
    // The angular bound the pixel inlier bound is worth on this camera, and the
    // floor the bearing spread has to clear: one pixel's worth of angle at the
    // focal the model is carrying.
    let (fx, fy) = camera.focal_lengths();
    let pixel_angle = 1.0 / fx.max(fy).max(1.0);
    let n = bearings.len();
    if bearing_span(&world) <= pixel_angle {
        return degenerate(n);
    }

    let mut keep: Vec<usize> = (0..n).collect();
    let Some(mut rotation) = kabsch(&world, &rays, &keep) else {
        return degenerate(n);
    };
    for _ in 1..ROTATION_TRIM_ROUNDS {
        let mut ranked: Vec<(f64, usize)> = (0..n)
            .map(|i| (angle_between(&(rotation * world[i]), &rays[i]), i))
            .collect();
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let take = ((n as f64 * ROTATION_KEEP_FRACTION).round() as usize).max(MIN_BEARINGS);
        keep = ranked.iter().take(take.min(n)).map(|r| r.1).collect();
        keep.sort_unstable();
        let Some(refit) = kabsch(&world, &rays, &keep) else {
            break;
        };
        rotation = refit;
    }

    let tolerance = INLIER_PX * pixel_angle;
    let inliers = (0..n)
        .filter(|&i| angle_between(&(rotation * world[i]), &rays[i]) < tolerance)
        .count();
    let inlier_fraction = inliers as f64 / n as f64;
    let accepted = inlier_fraction >= options.resect.accept_gate;
    let rotation = UnitQuaternion::from_rotation_matrix(
        &nalgebra::Rotation3::from_matrix_unchecked(orthonormalized(&rotation)),
    );
    Estimate {
        rotation,
        translation: stored.translation_xyz,
        rotation_only: true,
        correspondences: n,
        inliers,
        inlier_fraction,
        accepted,
        refusal: (!accepted).then(|| {
            format!(
                "bearing inlier fraction {inlier_fraction:.2} below the {:.2} gate",
                options.resect.accept_gate
            )
        }),
    }
}

/// The largest angle any bearing makes with the set's mean direction — the
/// spread a rotation fit has to work with.
fn bearing_span(bearings: &[Vector3<f64>]) -> f64 {
    let mut mean = Vector3::zeros();
    for b in bearings {
        mean += b;
    }
    let norm = mean.norm();
    if norm <= 0.0 || norm.is_nan() {
        // The bearings cancel, which is spread rather than the lack of it.
        return std::f64::consts::PI;
    }
    let mean = mean / norm;
    bearings
        .iter()
        .map(|b| angle_between(b, &mean))
        .fold(0.0, f64::max)
}

/// Angle between two vectors that are already unit length.
fn angle_between(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    a.dot(b).clamp(-1.0, 1.0).acos()
}

/// The nearest rotation matrix, so a product of SVD factors is a rotation to
/// `nalgebra`'s satisfaction rather than to `f64`'s.
fn orthonormalized(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => {
            let d = (u * v_t).determinant().signum();
            u * Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, d) * v_t
        }
        _ => *m,
    }
}

/// The capture's own length unit: the median over images of that image's median
/// camera-to-structure distance.
///
/// `None` for a reconstruction with no finite structure to measure against — a
/// rotation-only one, where every displacement is unitless.
fn scene_scale(recon: &SfmrReconstruction) -> Option<f64> {
    let mut per_image: Vec<Vec<f64>> = vec![Vec::new(); recon.images.len()];
    for obs in recon.tracks.iter() {
        let point = &recon.points[obs.point_index as usize];
        if point.is_at_infinity() {
            continue;
        }
        let image = &recon.images[obs.image_index as usize];
        let d = (point.position - image.camera_center()).norm();
        if d.is_finite() && d > 0.0 {
            per_image[obs.image_index as usize].push(d);
        }
    }
    let mut medians: Vec<f64> = per_image
        .iter_mut()
        .filter(|d| !d.is_empty())
        .map(|d| median(d))
        .collect();
    (!medians.is_empty()).then(|| median(&mut medians))
}

/// The median of a slice, by sorting it in place. Lower of the two middles on an
/// even count, so the answer is a member of the sample rather than a mean of
/// two.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[(values.len() - 1) / 2]
}

/// Record what this resection was, in the derived reconstruction's metadata, so
/// a later save carries provenance.
fn write_provenance(
    recon: &mut SfmrReconstruction,
    reports: &[ResectImageReport],
    totals: &ResectTotals,
) {
    recon.metadata.operation = "explorer_resect".to_string();
    let images: Vec<serde_json::Value> = reports
        .iter()
        .map(|r| {
            serde_json::json!({
                "image": r.image_name,
                "rotation_only": r.rotation_only,
                "correspondences": r.correspondences,
                "inliers": r.inliers,
                "inlier_fraction": r.inlier_fraction,
                "accepted": r.accepted,
            })
        })
        .collect();
    recon.metadata.tool_options.insert(
        "resect_image".to_string(),
        serde_json::json!({
            "correspondence_source": reports.first().map(|r| r.source).unwrap_or(""),
            "images": images,
            "accepted": totals.accepted,
            "refused": totals.refused,
            "inlier_fraction": totals.inlier_fraction,
        }),
    );
    recon.metadata.point_count = recon.points.len() as u32;
    recon.metadata.infinity_point_count = recon.infinity_point_count as u32;
    recon.metadata.observation_count = recon.tracks.len() as u32;
}
