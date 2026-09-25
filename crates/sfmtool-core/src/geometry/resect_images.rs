// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Re-estimate a set of images' poses against structure that did not depend on
//! any of them.
//!
//! [`resect_images`] is the whole mechanism of `specs/gui/edits/resect-image.md`:
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
//! Two correspondence sources (see [`ResectSource`]): the reconstruction's
//! tracks alone, or the tracks together with the clusters of a cluster-patches
//! `.matches` file, each cluster used as a track of its own. The two sets go to
//! the estimate side by side and are never joined to each other. The tracks
//! lead: when a target has at least three finite track pairs, the pose
//! hypotheses are drawn from the tracks alone and the clusters only support
//! them.
//!
//! Deterministic: the finite path seeds each target's RANSAC as a pure function
//! of `(seed, image index)`, the rotation-only path is a closed-form fit with a
//! fixed trimming schedule, and every gather below walks the reconstruction in
//! storage order.

use std::collections::{HashMap, HashSet};

use nalgebra::{Point3, UnitQuaternion, Vector3};
use rayon::prelude::*;

use sfmtool_matches_format::MatchesData;

use crate::camera::report::angle_between;
use crate::camera::CameraIntrinsics;
use crate::geometry::batch_resection::ResectOptions;
use crate::geometry::focal_vote::column_scan::kabsch;
use crate::geometry::reconstruction_growth::per_image_seed;
use crate::geometry::rotation::orthonormalized;
use crate::numeric::median_in_place;
use crate::reconstruction::triangulation::triangulate_batch;
use crate::reconstruction::{
    ObservationSource, ReconstructionError, SfmrImage, SfmrReconstruction,
};

use finite::{Pair, Source, World};

mod clusters;
mod finite;

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

/// Inlier bound on a pair's residual, in pixels: the batch-registration
/// primitive's own (`reconstruction_growth::INLIER_PX`). The reported inlier
/// count and the fraction the gate is applied to are the same measurement.
// This bound also decides which clusters count as inliers. Its use on clusters
// has been checked by measurement on two reconstructions only, the seoul_bull
// ground truth and Kerry Park, and it needs evaluating across a large number of
// solves before it is relied on. The form to evaluate it against is a bound
// derived from the reconstruction's own data, for example one scaled from the
// reprojection residuals of its tracks, so a cluster is held to the
// consistency its tracks show.
const INLIER_PX: f64 = 3.0;

/// One bearing correspondence of the rotation-only path: the point's held-out
/// world direction, and the unit ray the target sees it along.
type BearingPair = (Vector3<f64>, Vector3<f64>);

/// Where a target image's 2D–3D pairs come from.
#[derive(Clone, Copy)]
pub enum ResectSource<'a> {
    /// The reconstruction's tracks: each target's own observations, joined to
    /// the held-out positions and bearings of the points the set observes.
    Tracks,
    /// The tracks, and beside them the clusters of a parsed cluster-patches
    /// `.matches` file, each cluster used as a track of its own: placed by
    /// triangulating its kept members in the non-target posed images that
    /// have tracks, and paired with its kept member's refined position in the
    /// target. The file must carry the clusters section, the cluster-patches
    /// section and the member positions. Clusters feed the pose estimate only:
    /// they create no points and are not re-triangulated.
    TracksAndClusters(&'a MatchesData),
}

impl ResectSource<'_> {
    /// The provenance string recorded in the derived reconstruction's metadata.
    pub fn name(&self) -> &'static str {
        match self {
            ResectSource::Tracks => "tracks",
            ResectSource::TracksAndClusters(_) => "tracks_and_clusters",
        }
    }
}

/// Settings of [`resect_images`].
#[derive(Clone, Debug)]
pub struct ResectImageOptions {
    /// The batch-registration primitive's options, read here as: the floor of
    /// finite pairs below which the finite path is unavailable, the
    /// acceptance gate on the inlier fraction over all pairs, and the RANSAC
    /// seed.
    pub resect: ResectOptions,
    /// The largest distance, in pixels, a cluster's counted member may lie
    /// from the reprojection of the cluster's triangulated position into that
    /// member's image. A cluster with any member farther away gives no pair.
    /// Only read under [`ResectSource::TracksAndClusters`].
    pub max_cluster_residual_px: f64,
}

/// Default of [`ResectImageOptions::max_cluster_residual_px`], chosen by
/// measurement on two reconstructions (the table is in
/// `specs/gui/edits/resect-image.md`, "Correspondence sources").
// The value was chosen by measurement on two reconstructions only, the
// seoul_bull ground truth and Kerry Park. It needs evaluating across a large
// number of solves before it is relied on. The form to evaluate it against is a
// threshold derived from the reconstruction's own data, for example one scaled
// from the reprojection residuals of its tracks, so a cluster is held to the
// consistency its tracks show.
pub const DEFAULT_MAX_CLUSTER_RESIDUAL_PX: f64 = 1.5;

impl Default for ResectImageOptions {
    fn default() -> Self {
        Self {
            resect: ResectOptions::default(),
            max_cluster_residual_px: DEFAULT_MAX_CLUSTER_RESIDUAL_PX,
        }
    }
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
    /// Pairs the estimate saw: `track_correspondences +
    /// cluster_correspondences`.
    pub correspondences: usize,
    /// Of those, the pairs from the tracks, finite and at infinity.
    pub track_correspondences: usize,
    /// Of the track pairs, the bearings: tracks whose point is at infinity.
    /// On the rotation-only path every track pair is one.
    pub bearing_correspondences: usize,
    /// Of those, the pairs from the clusters. Zero on the rotation-only path,
    /// which reads the tracks' bearings only.
    pub cluster_correspondences: usize,
    /// How many of them the resected pose puts within the 3 px inlier bound
    /// (`INLIER_PX`). A bearing's residual is its angle times the camera's
    /// focal length.
    pub inliers: usize,
    /// Of the inliers, the ones from the tracks, finite and at infinity.
    pub track_inliers: usize,
    /// Of the track inliers, the bearings.
    pub bearing_inliers: usize,
    /// Of the inliers, the ones from the clusters.
    pub cluster_inliers: usize,
    /// Clusters with at least one kept member in this image. Zero when the
    /// source is the tracks alone.
    pub clusters_considered: usize,
    /// Of those, the ones the member rules set aside: more than one kept
    /// member in this image, or kept members in fewer than two non-target
    /// posed images.
    pub clusters_skipped: usize,
    /// Of those, the ones with kept members in two or more non-target posed
    /// images, but in fewer than two images that have tracks. A member in an
    /// image with no track observation does not count.
    pub clusters_untracked: usize,
    /// Of those, the ones whose members in tracked images did not
    /// triangulate.
    pub clusters_failed: usize,
    /// Of those, the ones whose triangulated position lies farther than
    /// [`ResectImageOptions::max_cluster_residual_px`] from one of the members
    /// it was triangulated from. The rest, `clusters_considered -
    /// clusters_skipped - clusters_untracked - clusters_failed -
    /// clusters_inconsistent`, each gave this image one pair.
    pub clusters_inconsistent: usize,
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
    /// Track pairs summed over the targets' estimates.
    pub track_correspondences: usize,
    /// Bearings summed over the targets' estimates.
    pub bearing_correspondences: usize,
    /// Cluster pairs summed over the targets' estimates.
    pub cluster_correspondences: usize,
    /// Inliers summed over the targets' estimates.
    pub inliers: usize,
    /// Track inliers summed over the targets' estimates.
    pub track_inliers: usize,
    /// Bearing inliers summed over the targets' estimates.
    pub bearing_inliers: usize,
    /// Cluster inliers summed over the targets' estimates.
    pub cluster_inliers: usize,
    /// [`ResectImageReport::clusters_untracked`] summed over the targets.
    pub clusters_untracked: usize,
    /// [`ResectImageReport::clusters_inconsistent`] summed over the targets.
    pub clusters_inconsistent: usize,
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
    /// The cluster-patches file cannot serve as a cluster source: it lacks the
    /// clusters section, the cluster-patches section, or the member positions.
    Clusters(String),
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
            ResectImageError::Clusters(m) => write!(f, "{m}"),
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
/// The whole mechanism of `specs/gui/edits/resect-image.md`:
///
/// 1. **Held-out structure.** Every finite point any target observes that keeps
///    at least two *non-target* observations is re-triangulated from those
///    alone, at the non-target images' stored poses; the stored position is
///    discarded. A point with fewer than two non-target observations has no
///    held-out position and is excluded from the estimates. A point at infinity
///    is a direction, which one rotation already fixes, so its held-out bearing
///    is the mean of the non-target observations' world rays.
/// 2. **Pose estimates.** A target's track pairs are its observations of
///    points with a held-out position or a held-out bearing. Under
///    [`ResectSource::TracksAndClusters`] its cluster pairs stand beside them
///    (see the `clusters` submodule). A target with at least
///    `ResectOptions::min_obs` finite pairs (tracks and clusters) takes the
///    finite path (the `finite` submodule): RANSAC P3P whose minimal samples
///    are the finite track pairs when there are at least three, and every
///    finite pair otherwise, scored over all pairs including the bearings, then
///    a trimmed refinement. A target below that floor takes the rotation-only
///    path: its rotation is fit in closed form to the bearings (trimmed,
///    iterated) and its translation is left at its stored value. Each estimate
///    is accepted or refused on `ResectOptions::accept_gate`, independently of
///    the others; a refusal keeps that image's stored pose and reports itself
///    rather than failing the call.
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
    let count = recon.image_table.images.len();
    if image_indexes.is_empty() {
        return Err(ResectImageError::NoTargets);
    }
    let posed: Vec<bool> = recon.image_table.images.iter().map(is_posed).collect();
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
    // `tracks` is sorted by point then image, so each target's rows are
    // ascending in point index.
    let mut target_rows: HashMap<usize, Vec<usize>> =
        image_indexes.iter().map(|&t| (t, Vec::new())).collect();
    // Which images have at least one track observation: the clusters count
    // members in these images only.
    let mut tracked = vec![false; count];
    for (row, obs) in recon.point_set.tracks.iter().enumerate() {
        let image = obs.image_index as usize;
        tracked[image] = true;
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
                    .map(|&row| recon.point_set.tracks[row].point_index as usize)
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
        gathered.extend(
            recon.point_set.observation_offsets[p]..recon.point_set.observation_offsets[p + 1],
        );
    }
    gathered.sort_unstable();
    gathered.dedup();
    let pixels = observation_pixels(recon, &gathered)?;
    let pixel_of: HashMap<usize, [f64; 2]> = gathered.iter().copied().zip(pixels).collect();

    // ── Step 1: held-out structure ─────────────────────────────────────────
    let finite_observed: Vec<usize> = all_observed
        .iter()
        .copied()
        .filter(|&p| !recon.point_set.points[p].is_at_infinity())
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

    // ── Step 2: the pose estimates ─────────────────────────────────────────
    // The tracks' pairs: each target's observations of the points that have a
    // held-out position or a held-out bearing, in point order.
    let mut pairs_of: HashMap<usize, Vec<Pair>> = HashMap::new();
    for &t in image_indexes {
        let camera = &recon.image_table.cameras[recon.image_table.images[t].camera_index as usize];
        let rows = &target_rows[&t];
        let points = &observed[&t];
        let mut out = Vec::with_capacity(rows.len());
        for (k, &row) in rows.iter().enumerate() {
            let p = points[k];
            let world = if let Some(&x) = held_out.get(&p) {
                (Source::Track, World::Point(x))
            } else if let Some(&d) = bearing_of.get(&p) {
                (Source::Bearing, World::Direction(d))
            } else {
                continue;
            };
            out.extend(Pair::new(world.0, pixel_of[&row], world.1, camera));
        }
        pairs_of.insert(t, out);
    }
    // The clusters' pairs, appended beside them.
    let cluster_support = match source {
        ResectSource::Tracks => None,
        ResectSource::TracksAndClusters(matches) => Some(clusters::cluster_support(
            recon,
            image_indexes,
            &is_target,
            &posed_others,
            &tracked,
            matches,
            options.max_cluster_residual_px,
        )?),
    };
    if let Some(support) = &cluster_support {
        for &t in image_indexes {
            let camera =
                &recon.image_table.cameras[recon.image_table.images[t].camera_index as usize];
            let pairs = pairs_of.get_mut(&t).expect("every target has pairs");
            pairs.extend(
                support[&t]
                    .pairs
                    .iter()
                    .filter_map(|&(uv, x)| Pair::new(Source::Cluster, uv, World::Point(x), camera)),
            );
        }
    }

    // Each target is estimated on its own, seeded from its own index, so the
    // order they run in changes no answer.
    let estimates: HashMap<usize, Estimate> = image_indexes
        .par_iter()
        .map(|&t| {
            let camera =
                &recon.image_table.cameras[recon.image_table.images[t].camera_index as usize];
            let pairs = &pairs_of[&t];
            let finite = pairs.iter().filter(|p| p.source != Source::Bearing).count();
            let bearings: Vec<BearingPair> = pairs
                .iter()
                .filter_map(|p| match p.world {
                    World::Direction(d) => Some((d, p.ray)),
                    World::Point(_) => None,
                })
                .collect();
            let estimate = if finite >= options.resect.min_obs {
                finite_estimate(recon, t, pairs, camera, options)
            } else if bearings.len() >= MIN_BEARINGS {
                rotation_estimate(recon, t, &bearings, camera, options)
            } else {
                Estimate::no_support(recon, t, Counts::of(pairs, |_| true))
            };
            (t, estimate)
        })
        .collect();

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
        .filter(|&p| !recon.point_set.points[p].is_at_infinity())
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
    let mut out = recon.clone_for_edit();
    for &t in image_indexes {
        let estimate = &estimates[&t];
        if estimate.accepted {
            out.image_table.images[t].quaternion_wxyz = estimate.rotation;
            out.image_table.images[t].translation_xyz = estimate.translation;
        }
    }
    let mut retriangulated: HashSet<usize> = HashSet::new();
    let mut drop_mask = vec![false; recon.point_set.points.len()];
    for &p in &finite_observed {
        match (refit.get(&p), held_out.get(&p)) {
            (Some(&position), _) => {
                out.point_set.points[p].position =
                    Point3::new(position[0], position[1], position[2]);
                retriangulated.insert(p);
            }
            (None, Some(&position)) => {
                out.point_set.points[p].position =
                    Point3::new(position[0], position[1], position[2]);
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
            let stored = &recon.image_table.images[t];
            let (rotation, translation) = if estimate.accepted {
                (estimate.rotation, estimate.translation)
            } else {
                (stored.quaternion_wxyz, stored.translation_xyz)
            };
            let centre_delta =
                (world_centre(&rotation, &translation) - stored.camera_center()).norm();
            let mine = &observed[&t];
            let from_clusters = cluster_support
                .as_ref()
                .map(|support| support[&t].clone())
                .unwrap_or_default();
            ResectImageReport {
                image_index: t,
                image_name: stored.name.clone(),
                source: source.name(),
                rotation_only: estimate.rotation_only,
                correspondences: estimate.pairs.total(),
                track_correspondences: estimate.pairs.tracks,
                bearing_correspondences: estimate.pairs.bearings,
                cluster_correspondences: estimate.pairs.clusters,
                inliers: estimate.inliers.total(),
                track_inliers: estimate.inliers.tracks,
                bearing_inliers: estimate.inliers.bearings,
                cluster_inliers: estimate.inliers.clusters,
                clusters_considered: from_clusters.considered,
                clusters_skipped: from_clusters.skipped,
                clusters_untracked: from_clusters.untracked,
                clusters_failed: from_clusters.failed,
                clusters_inconsistent: from_clusters.inconsistent,
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
        track_correspondences: reports.iter().map(|r| r.track_correspondences).sum(),
        bearing_correspondences: reports.iter().map(|r| r.bearing_correspondences).sum(),
        cluster_correspondences: reports.iter().map(|r| r.cluster_correspondences).sum(),
        inliers,
        track_inliers: reports.iter().map(|r| r.track_inliers).sum(),
        bearing_inliers: reports.iter().map(|r| r.bearing_inliers).sum(),
        cluster_inliers: reports.iter().map(|r| r.cluster_inliers).sum(),
        clusters_untracked: reports.iter().map(|r| r.clusters_untracked).sum(),
        clusters_inconsistent: reports.iter().map(|r| r.clusters_inconsistent).sum(),
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

/// Why an in-place resection produced no value.
///
/// Two kinds, because the caller reports them the same way and decides on
/// neither: the resection could not be attempted at all, or it was attempted and
/// its estimate was refused. The second is not an error of [`resect_images`],
/// which reports it through [`ResectImageReport::refusal`] and still hands back
/// a reconstruction carrying the stored pose; it is an error *here*, because
/// installing that reconstruction as the image's own next value would move the
/// structure while leaving the pose the estimate declined to re-state.
#[derive(Debug)]
pub enum ResectInPlaceError {
    /// The resection could not be attempted; see [`ResectImageError`].
    Resect(ResectImageError),
    /// The estimate was refused, carrying the reason
    /// [`ResectImageReport::refusal`] gives.
    Refused(String),
}

impl std::fmt::Display for ResectInPlaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResectInPlaceError::Resect(e) => write!(f, "{e}"),
            ResectInPlaceError::Refused(reason) => write!(f, "{reason}"),
        }
    }
}

impl std::error::Error for ResectInPlaceError {}

impl From<ResectImageError> for ResectInPlaceError {
    fn from(value: ResectImageError) -> Self {
        ResectInPlaceError::Resect(value)
    }
}

/// One image's resection as the value to install in place of `recon`, rather
/// than as a reconstruction to stand beside it.
///
/// The estimate is [`resect_images`] on the one-element target set, so the pose
/// and the structure that come back are exactly that call's; what this adds is
/// the rule an in-place caller needs and a comparison caller does not.
/// **A refused estimate yields no value.** [`resect_images`] hands a refusal
/// back as a held-out re-triangulation that can be looked at beside the
/// original; the same reconstruction installed *as* the original would be a
/// version that moved the points and left the pose alone.
///
/// The returned report is the target's own, and carries no refusal.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::geometry::{
///     resect_image_in_place, ResectImageOptions, ResectSource,
/// };
/// # fn run(recon: &sfmtool_core::SfmrReconstruction)
/// # -> Result<(), Box<dyn std::error::Error>> {
/// let (next, report) = resect_image_in_place(
///     recon,
///     7,
///     ResectSource::Tracks,
///     &ResectImageOptions::default(),
/// )?;
/// assert!(report.refusal.is_none());
/// assert_eq!(next.image_count(), recon.image_count());
/// # Ok(())
/// # }
/// ```
pub fn resect_image_in_place(
    recon: &SfmrReconstruction,
    image: usize,
    source: ResectSource<'_>,
    options: &ResectImageOptions,
) -> Result<(SfmrReconstruction, ResectImageReport), ResectInPlaceError> {
    let mut resected = resect_images(recon, &[image], source, options)?;
    let report = resected.reports.pop().expect("one target, one report");
    match report.refusal {
        Some(reason) => Err(ResectInPlaceError::Refused(reason)),
        None => Ok((resected.reconstruction, report)),
    }
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
/// An inline keypoint column answers directly -- always for `embedded_patches`,
/// and for a `sift_files` reconstruction that carries the optional copy, whose
/// coordinates are what that reconstruction says its observations are. Failing
/// that, a `sift_files` reconstruction resolves each row through the images'
/// `.sift` companions, read once per image touched rather than once per row.
fn observation_pixels(
    recon: &SfmrReconstruction,
    rows: &[usize],
) -> Result<Vec<[f64; 2]>, ReconstructionError> {
    if let Some(keypoints_xy) = recon.keypoints_xy() {
        return Ok(rows
            .iter()
            .map(|&row| [keypoints_xy[[row, 0]] as f64, keypoints_xy[[row, 1]] as f64])
            .collect());
    }
    match &recon.point_set.observations {
        ObservationSource::EmbeddedPatches { .. } => {
            unreachable!("embedded_patches always carries keypoints_xy, handled above")
        }
        ObservationSource::SiftFiles {
            feature_indexes, ..
        } => {
            let mut images: Vec<usize> = rows
                .iter()
                .map(|&row| recon.point_set.tracks[row].image_index as usize)
                .collect();
            images.sort_unstable();
            images.dedup();
            let mut positions: HashMap<usize, Vec<[f32; 2]>> = HashMap::new();
            for image in images {
                positions.insert(image, read_sift_positions(recon, image)?);
            }
            rows.iter()
                .map(|&row| {
                    let image = recon.point_set.tracks[row].image_index as usize;
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
    let count = recon.point_set.max_track_feature_index[image] as usize + 1;
    sfmtool_sift_format::read_sift_positions(&path, count).map_err(|e| {
        ReconstructionError::SiftRead {
            path,
            source: e.to_string(),
        }
    })
}

/// Ray-midpoint triangulation of `points`, one entry per input point.
///
/// `contributes` is the per-image mask of whose observations are read — the
/// hold-out passes the non-target posed images, the re-triangulation at the new
/// poses passes every posed image. `replace` substitutes a pose per image, so
/// the same gather serves both. The placement rules are
/// [`triangulate_groups`]'s.
fn triangulate_points(
    recon: &SfmrReconstruction,
    points: &[usize],
    pixel_of: &HashMap<usize, [f64; 2]>,
    contributes: &[bool],
    replace: &[Option<Pose>],
) -> Vec<Option<[f64; 3]>> {
    let groups: Vec<Vec<(usize, [f64; 2])>> = points
        .iter()
        .map(|&p| {
            (recon.point_set.observation_offsets[p]..recon.point_set.observation_offsets[p + 1])
                .filter_map(|row| {
                    let image = recon.point_set.tracks[row].image_index as usize;
                    if !contributes[image] {
                        return None;
                    }
                    pixel_of.get(&row).map(|&uv| (image, uv))
                })
                .collect()
        })
        .collect();
    triangulate_groups(recon, &groups, replace)
}

/// Ray-midpoint triangulation of each group of `(image, pixel)` sightings, one
/// entry per group, at the images' stored poses unless `replace` substitutes
/// one.
///
/// The one set of placement rules every triangulation of the resection uses —
/// the tracks' hold-out, their re-triangulation at the new poses, and the
/// clusters' placement. A group is placed only when at least two rays survive,
/// the solve puts it in front of every one of them, and its depth is
/// observable at all (parallel rays leave the normal matrix rank-deficient,
/// which the triangulation reports as an infinite condition number).
fn triangulate_groups(
    recon: &SfmrReconstruction,
    groups: &[Vec<(usize, [f64; 2])>],
    replace: &[Option<Pose>],
) -> Vec<Option<[f64; 3]>> {
    let mut dirs: Vec<Vector3<f64>> = Vec::new();
    let mut centers: Vec<Point3<f64>> = Vec::new();
    let mut offsets: Vec<usize> = vec![0];
    for group in groups {
        for &(image, uv) in group {
            let (rotation, translation) = match replace[image] {
                Some(pose) => pose,
                None => (
                    recon.image_table.images[image].quaternion_wxyz,
                    recon.image_table.images[image].translation_xyz,
                ),
            };
            let camera =
                &recon.image_table.cameras[recon.image_table.images[image].camera_index as usize];
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
        if !recon.point_set.points[p].is_at_infinity() {
            continue;
        }
        let mut mean = Vector3::zeros();
        for row in
            recon.point_set.observation_offsets[p]..recon.point_set.observation_offsets[p + 1]
        {
            let image = recon.point_set.tracks[row].image_index as usize;
            if !posed_others[image] {
                continue;
            }
            let Some(uv) = pixel_of.get(&row) else {
                continue;
            };
            let camera =
                &recon.image_table.cameras[recon.image_table.images[image].camera_index as usize];
            let ray = camera.pixel_to_ray(uv[0], uv[1]);
            let world = recon.image_table.images[image].quaternion_wxyz.inverse()
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

/// Pairs counted by source. `tracks` counts every track pair, finite and at
/// infinity; `bearings` counts the ones at infinity.
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    tracks: usize,
    bearings: usize,
    clusters: usize,
}

impl Counts {
    /// The pairs of `pairs` that `keep` selects, counted by source.
    fn of(pairs: &[Pair], keep: impl Fn(usize) -> bool) -> Self {
        let mut counts = Counts::default();
        for (k, pair) in pairs.iter().enumerate() {
            if !keep(k) {
                continue;
            }
            match pair.source {
                Source::Track => counts.tracks += 1,
                Source::Bearing => {
                    counts.tracks += 1;
                    counts.bearings += 1;
                }
                Source::Cluster => counts.clusters += 1,
            }
        }
        counts
    }

    fn total(&self) -> usize {
        self.tracks + self.clusters
    }
}

/// The outcome of one pose estimate, before it reaches the reconstruction.
struct Estimate {
    rotation: UnitQuaternion<f64>,
    translation: Vector3<f64>,
    rotation_only: bool,
    /// The pairs the estimate saw.
    pairs: Counts,
    /// Of those, the ones the estimated pose puts within `INLIER_PX`.
    inliers: Counts,
    inlier_fraction: f64,
    accepted: bool,
    refusal: Option<String>,
}

impl Estimate {
    /// An estimate that leaves the stored pose standing, with its reason.
    fn refused(
        recon: &SfmrReconstruction,
        image_index: usize,
        rotation_only: bool,
        pairs: Counts,
        reason: String,
    ) -> Self {
        Estimate {
            rotation: recon.image_table.images[image_index].quaternion_wxyz,
            translation: recon.image_table.images[image_index].translation_xyz,
            rotation_only,
            pairs,
            inliers: Counts::default(),
            inlier_fraction: 0.0,
            accepted: false,
            refusal: Some(reason),
        }
    }

    /// A target neither path has support for: too few finite correspondences
    /// for the finite path, and too few bearings for the rotation-only one. Its
    /// stored pose stands, and the hold-out is still what the derived
    /// reconstruction shows for the points it observes.
    fn no_support(recon: &SfmrReconstruction, image_index: usize, pairs: Counts) -> Self {
        let tracks = pairs.tracks - pairs.bearings;
        let clusters = pairs.clusters;
        let finite = tracks + clusters;
        let bearings = pairs.bearings;
        let reason = format!(
            "no support: {finite} finite correspondence{} ({tracks} from tracks, \
             {clusters} from clusters) and {bearings} bearing{}",
            if finite == 1 { "" } else { "s" },
            if bearings == 1 { "" } else { "s" }
        );
        Estimate::refused(recon, image_index, false, pairs, reason)
    }
}

/// The finite path for one target: the `finite` submodule's estimate, gated on
/// `ResectOptions::accept_gate`.
fn finite_estimate(
    recon: &SfmrReconstruction,
    image_index: usize,
    pairs: &[Pair],
    camera: &CameraIntrinsics,
    options: &ResectImageOptions,
) -> Estimate {
    let counts = Counts::of(pairs, |_| true);
    let seed = per_image_seed(options.resect.seed, image_index as u32);
    let fit = match finite::estimate(camera, pairs, seed) {
        Ok(fit) => fit,
        Err(reason) => return Estimate::refused(recon, image_index, false, counts, reason),
    };
    let inliers = Counts::of(pairs, |k| fit.inliers[k]);
    let inlier_fraction = inliers.total() as f64 / counts.total() as f64;
    let accepted = inlier_fraction >= options.resect.accept_gate;
    Estimate {
        rotation: fit.pose.0,
        translation: fit.pose.1,
        rotation_only: false,
        pairs: counts,
        inliers,
        inlier_fraction,
        accepted,
        refusal: (!accepted).then(|| {
            format!(
                "inlier fraction {inlier_fraction:.2} below the {:.2} gate",
                options.resect.accept_gate
            )
        }),
    }
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
    let n = bearings.len();
    let only_bearings = |k: usize| Counts {
        tracks: k,
        bearings: k,
        clusters: 0,
    };
    let degenerate = || {
        Estimate::refused(
            recon,
            image_index,
            true,
            only_bearings(n),
            "the bearings span no measurable angle".to_string(),
        )
    };

    let world: Vec<Vector3<f64>> = bearings.iter().map(|b| b.0).collect();
    let rays: Vec<Vector3<f64>> = bearings.iter().map(|b| b.1).collect();
    // The angular bound the pixel inlier bound is worth on this camera, and the
    // floor the bearing spread has to clear: one pixel's worth of angle at the
    // focal the model is carrying.
    let pixel_angle = 1.0 / finite::pixels_per_radian(camera);
    if bearing_span(&world) <= pixel_angle {
        return degenerate();
    }

    let mut keep: Vec<usize> = (0..n).collect();
    let Some(mut rotation) = kabsch(&world, &rays, &keep) else {
        return degenerate();
    };
    for _ in 1..ROTATION_TRIM_ROUNDS {
        let mut ranked: Vec<(f64, usize)> = (0..n)
            .map(|i| {
                (
                    angle_between((rotation * world[i]).into(), rays[i].into()),
                    i,
                )
            })
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
        .filter(|&i| angle_between((rotation * world[i]).into(), rays[i].into()) < tolerance)
        .count();
    let inlier_fraction = inliers as f64 / n as f64;
    let accepted = inlier_fraction >= options.resect.accept_gate;
    let rotation = UnitQuaternion::from_rotation_matrix(
        &nalgebra::Rotation3::from_matrix_unchecked(orthonormalized(&rotation)),
    );
    Estimate {
        rotation,
        translation: recon.image_table.images[image_index].translation_xyz,
        rotation_only: true,
        pairs: only_bearings(n),
        inliers: only_bearings(inliers),
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
        .map(|b| angle_between((*b).into(), mean.into()))
        .fold(0.0, f64::max)
}

/// The capture's own length unit: the median over images of that image's median
/// camera-to-structure distance.
///
/// `None` for a reconstruction with no finite structure to measure against — a
/// rotation-only one, where every displacement is unitless.
///
/// Crate-visible because every edit that reports a camera displacement reports
/// it in this unit: the resection here, and the moved camera in
/// [`move_camera`](crate::reconstruction::move_camera::move_camera).
pub(crate) fn scene_scale(recon: &SfmrReconstruction) -> Option<f64> {
    let mut per_image: Vec<Vec<f64>> = vec![Vec::new(); recon.image_table.images.len()];
    for obs in recon.point_set.tracks.iter() {
        let point = &recon.point_set.points[obs.point_index as usize];
        if point.is_at_infinity() {
            continue;
        }
        let image = &recon.image_table.images[obs.image_index as usize];
        let d = (point.position - image.camera_center()).norm();
        if d.is_finite() && d > 0.0 {
            per_image[obs.image_index as usize].push(d);
        }
    }
    let mut medians: Vec<f64> = per_image
        .iter_mut()
        .filter(|d| !d.is_empty())
        .map(|d| median_in_place(d))
        .collect();
    (!medians.is_empty()).then(|| median_in_place(&mut medians))
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
                "track_correspondences": r.track_correspondences,
                "bearing_correspondences": r.bearing_correspondences,
                "cluster_correspondences": r.cluster_correspondences,
                "inliers": r.inliers,
                "track_inliers": r.track_inliers,
                "bearing_inliers": r.bearing_inliers,
                "cluster_inliers": r.cluster_inliers,
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
    recon.metadata.point_count = recon.point_set.points.len() as u32;
    recon.metadata.infinity_point_count = recon.point_set.infinity_point_count as u32;
    recon.metadata.observation_count = recon.point_set.tracks.len() as u32;
}
