// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The one rule that decides whether a bench track's rays fix a point or only a
//! bearing.
//!
//! `specs/core/bench/editable-track.md` § "Finite points and bearings" is the
//! design. Every bench step that triangulates -- the track-stage
//! [`fit`](super::fit::fit) and the cluster-to-track upgrade
//! ([`set_stage`](super::stage::set_stage())) -- ends at
//! [`classify_track_rays`], so the two cannot come to disagree about which
//! representation one set of sightings has earned.
//!
//! The criterion is not the bench's. It is the point-or-bearing
//! likelihood-ratio test of `specs/core/reconstruction/batch-triangulation-api.md`
//! § "Point or bearing": [`bearing_score`] and [`is_finite`] decide, at the
//! reconstruction's measured noise level, and the plain least-squares point fit
//! places a finite track by the consumer rule [`fit_usable_point`] carries, the
//! same rule reclassifying a whole reconstruction
//! ([`classify_points_at_infinity`](crate::SfmrReconstruction::classify_points_at_infinity))
//! places its points by. What the bench adds is what a fit writes when the
//! rule leaves a track with no representation that describes every sighting,
//! since a fit has to write something.

use std::fmt;

use nalgebra::{Matrix2x3, Point3, Vector3};

use crate::analysis::point_or_bearing::{fit_usable_point, is_usable_point};
use crate::patch::normal_refine::ProjectedImage;
use crate::readable::Readable;
use crate::reconstruction::triangulation::{bearing_score, is_finite, observed_ray};

/// Which outcome of the test settled a track, and so what the sentence beside
/// the verdict says.
///
/// The first two are finite verdicts the point fit placed; the next two are
/// bearing verdicts; the last four are the consumer rule's, where the verdict
/// and what the rays can be stored as disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationReason {
    /// Finite: the depth score reached the threshold, and the point fit placed
    /// a usable point.
    ScoreCleared,
    /// Finite: the midpoint bound reached the threshold where the depth score
    /// did not -- rays spread over a wide angle, where the bearing describes
    /// nothing -- and the point fit placed a usable point.
    MidpointBoundCleared,
    /// A bearing: the bearing's own cost is under the threshold, and no depth
    /// can lower a cost by more than the cost itself.
    BearingCostBelowThreshold,
    /// A bearing: neither the depth score nor the midpoint bound reached the
    /// threshold.
    ScoreBelowThreshold,
    /// A finite verdict whose point fit gave no usable point, written at the
    /// place the track held because that place is usable: in front of every
    /// observing camera and no nearer one than the minimum depth.
    /// Reclassification keeps such a stored point as it is.
    HeldPointKept,
    /// A finite verdict, written as the bearing: the point fit gave no usable
    /// point (none, one behind a camera or nearer a camera centre than the
    /// minimum depth, or one whose `Λ` falls short of the threshold).
    NoUsablePoint,
    /// A bearing verdict, written as the point fit's usable point: the bearing
    /// is behind one of the observing cameras, so it describes no sighting
    /// there.
    BearingBehindCamera,
    /// Neither a usable point nor a bearing in front of every observing camera
    /// describes the sightings, so the track keeps the coordinate it had, or
    /// takes the bearing when it had none. A sighting that looks the other way
    /// is the person's to turn out.
    LeftUnusable,
}

impl ClassificationReason {
    /// The snake-case name the bindings and the wire carry.
    pub fn name(self) -> &'static str {
        match self {
            Self::ScoreCleared => "score_cleared",
            Self::MidpointBoundCleared => "midpoint_bound_cleared",
            Self::BearingCostBelowThreshold => "bearing_cost_below_threshold",
            Self::ScoreBelowThreshold => "score_below_threshold",
            Self::HeldPointKept => "held_point_kept",
            Self::NoUsablePoint => "no_usable_point",
            Self::BearingBehindCamera => "bearing_behind_camera",
            Self::LeftUnusable => "left_unusable",
        }
    }
}

/// What one set of a track's rays resolves to, and the numbers behind the call.
///
/// [`Self::coordinate`] is what the track stores: a world point when
/// [`Self::at_infinity`] is false, and the unit bearing direction when it is
/// true. That is the `.sfmr` rule for a `w = 0` row, so a caller writes this
/// straight into a position and a `w` without asking which it has.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackClassification {
    /// Whether the rays fix a bearing only, which is `w == 0`.
    pub at_infinity: bool,
    /// The coordinate the track takes: the placed point when finite, the unit
    /// direction when at infinity.
    pub coordinate: Point3<f64>,
    /// Which outcome settled it.
    pub reason: ClassificationReason,
    /// The per-axis pixel noise the rays were weighted by: the reconstruction's
    /// measured reprojection noise, unless a caller gave its own.
    pub sigma_px: f64,
    /// The threshold the score, the bound and `Λ` were judged against.
    pub threshold: f64,
    /// The bearing's cost in units of the noise. `Λ` cannot exceed it.
    pub bearing_cost: f64,
    /// The score statistic for a depth at the bearing.
    pub depth_score: f64,
    /// The cost reduction at the weighted linear midpoint, a lower bound on
    /// `Λ`.
    pub midpoint_bound: f64,
    /// `Λ` of the plain least-squares point fit, `NaN` where no fit ran (a
    /// bearing verdict whose bearing is in front of every camera).
    pub depth_likelihood_ratio: f64,
    /// The distance from the observing cameras' centroid to the coordinate the
    /// track takes, in scene units; `NaN` for a bearing.
    pub distance: f64,
    /// The minimum depth a placed point keeps from every observing camera's
    /// centre, in scene units.
    pub min_depth: f64,
    /// How many rays the test read.
    pub num_views: usize,
    /// The widest angle between any two of the rays, in degrees. Not a test --
    /// it is the number a person reads a near-parallel track by, and it is the
    /// one Track View already shows for a committed track.
    pub max_pair_angle_deg: f64,
}

impl fmt::Display for TrackClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Every number goes through `Readable`: a caller's `sigma_px`, or a
        // score at a keypoint-resolution noise level, can be far outside the
        // range `{:.1}` prints in a few characters.
        let [x, y, z] = [self.coordinate.x, self.coordinate.y, self.coordinate.z].map(Readable);
        let [score, bound, ratio, cost, threshold] = [
            self.depth_score,
            self.midpoint_bound,
            self.depth_likelihood_ratio,
            self.bearing_cost,
            self.threshold,
        ]
        .map(Readable);
        if self.at_infinity {
            write!(f, "at infinity along ({x:.4}, {y:.4}, {z:.4})")?;
        } else {
            write!(f, "finite at ({x:.4}, {y:.4}, {z:.4})")?;
        }
        write!(f, ": ")?;
        match self.reason {
            ClassificationReason::ScoreCleared => write!(
                f,
                "depth score {score:.1} over the {threshold:.0} threshold, likelihood ratio \
                 {ratio:.1}"
            )?,
            ClassificationReason::MidpointBoundCleared => write!(
                f,
                "midpoint bound {bound:.1} over the {threshold:.0} threshold (depth score \
                 {score:.1}), likelihood ratio {ratio:.1}"
            )?,
            ClassificationReason::BearingCostBelowThreshold => write!(
                f,
                "the bearing's cost {cost:.1} is under the {threshold:.0} threshold, so no depth \
                 can clear it"
            )?,
            ClassificationReason::ScoreBelowThreshold => write!(
                f,
                "depth score {score:.1} and midpoint bound {bound:.1} under the {threshold:.0} \
                 threshold"
            )?,
            ClassificationReason::HeldPointKept => write!(
                f,
                "depth score {score:.1} (bound {bound:.1}) clears the {threshold:.0} threshold; \
                 the point fit gave no usable point, so the track keeps the usable place it held"
            )?,
            ClassificationReason::NoUsablePoint => write!(
                f,
                "depth score {score:.1} (bound {bound:.1}) clears the {threshold:.0} threshold, \
                 but the point fit gave no usable point"
            )?,
            ClassificationReason::BearingBehindCamera => write!(
                f,
                "depth score {score:.1} under the {threshold:.0} threshold, but the bearing is \
                 behind an observing camera; placed by a point fit with likelihood ratio \
                 {ratio:.1}"
            )?,
            ClassificationReason::LeftUnusable => write!(
                f,
                "neither a usable point nor the bearing describes every sighting \
                 (depth score {score:.1}, threshold {threshold:.0})"
            )?,
        }
        if !self.at_infinity && self.distance.is_finite() {
            write!(
                f,
                ", {:.3} from the observing cameras",
                Readable(self.distance)
            )?;
        }
        write!(
            f,
            ", at {:.3} px noise over {} rays up to {:.3} deg apart",
            Readable(self.sigma_px),
            self.num_views,
            Readable(self.max_pair_angle_deg)
        )
    }
}

/// One track's rays in world, each weighted for the test.
///
/// Built once per triangulation and handed to both the solve and the
/// classification, so the two read one set of rays.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackRays {
    /// Unit world-space ray per sighting.
    pub dirs: Vec<Vector3<f64>>,
    /// The camera centre each was cast from.
    pub centers: Vec<Point3<f64>>,
    /// Each ray's 2×3 world-frame noise weight at [`Self::sigma_px`], as
    /// [`observed_ray`] builds it from the lens the sighting was measured
    /// through.
    pub weights: Vec<Matrix2x3<f64>>,
    /// The per-axis pixel noise the weights are over.
    pub sigma_px: f64,
}

impl TrackRays {
    /// The rays of `sightings`, each a pixel and the index of the image in
    /// `images` it was seen in, weighted at `sigma_px`.
    ///
    /// A sighting gives no ray when its image is past `images`, its pixel is
    /// not finite, or [`observed_ray`] declines it (a pixel outside the camera
    /// model's domain, or a `sigma_px` that is not finite and positive), which
    /// is how the reconstruction-level test treats an observation too.
    pub fn of_sightings(
        sightings: &[([f64; 2], usize)],
        images: &[ProjectedImage<'_>],
        sigma_px: f64,
    ) -> Self {
        let mut rays = Self {
            dirs: Vec::with_capacity(sightings.len()),
            centers: Vec::with_capacity(sightings.len()),
            weights: Vec::with_capacity(sightings.len()),
            sigma_px,
        };
        for &(pixel, image) in sightings {
            let Some(view) = images.get(image) else {
                continue;
            };
            if !(pixel[0].is_finite() && pixel[1].is_finite()) {
                continue;
            }
            let rotation = view.cam_from_world.rotation.as_nalgebra();
            let Some(ray) = observed_ray(view.camera, rotation, pixel, sigma_px) else {
                continue;
            };
            rays.dirs.push(ray.dir);
            rays.centers
                .push(view.cam_from_world.inverse_translation_origin());
            rays.weights.push(ray.weight);
        }
        rays
    }

    /// How many rays there are.
    pub fn len(&self) -> usize {
        self.dirs.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.dirs.is_empty()
    }
}

/// Decide whether `rays` fix a point or only a bearing, and where the track
/// stands: the point-or-bearing test, and the consumer rule that places a
/// finite verdict.
///
/// `held` is the coordinate the track has and whether it is a bearing, `None`
/// for a track that has none yet (a cluster being upgraded). A finite `held`
/// is where the point fit starts, so a refit of a track that has not moved
/// converges where it stands; it is also what the track keeps when nothing
/// describes its sightings. `min_depth` is the reconstruction's
/// [`min_point_depth`](crate::SfmrReconstruction::min_point_depth), and
/// `threshold` the one the score, the bound and the fit's `Λ` are judged
/// against, by default
/// [`DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD`](crate::reconstruction::triangulation::DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD).
///
/// - **A finite verdict** ([`is_finite`]) is placed by [`fit_usable_point`],
///   from `held` when it is finite. A usable point is written; otherwise a
///   usable held place is kept, as reclassification keeps a usable stored
///   point ([`ClassificationReason::HeldPointKept`]), and with none the track
///   is a bearing ([`ClassificationReason::NoUsablePoint`]).
/// - **A bearing verdict** writes the score's closed-form bearing, when it is
///   in front of every observing camera.
/// - **A bearing behind a camera** describes no sighting there. The track is
///   then placed by the point fit, with no bar on its `Λ` since the verdict
///   asked for none ([`ClassificationReason::BearingBehindCamera`]), and when
///   that gives no usable point either it keeps `held`, or takes the bearing
///   when it has none ([`ClassificationReason::LeftUnusable`]).
///
/// `None` when the rays are too few for the test (fewer than two, or weights
/// so large the bearing's cost overflows).
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::classify::{classify_track_rays, TrackRays};
/// # use sfmtool_core::reconstruction::triangulation::DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD;
/// # fn run(
/// #     sightings: &[([f64; 2], usize)],
/// #     images: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// #     edited: &sfmtool_core::EditedReconstruction,
/// # ) -> Result<(), String> {
/// let sigma_px = edited.base_reprojection_noise_px()?;
/// let rays = TrackRays::of_sightings(sightings, images, sigma_px);
/// if let Some(call) = classify_track_rays(
///     &rays,
///     None,
///     edited.base_min_point_depth(),
///     DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD,
/// ) {
///     println!("{call}"); // "finite at (…): depth score 132.0 over the 25 threshold, …"
/// }
/// # Ok(())
/// # }
/// ```
pub fn classify_track_rays(
    rays: &TrackRays,
    held: Option<(Point3<f64>, bool)>,
    min_depth: f64,
    threshold: f64,
) -> Option<TrackClassification> {
    let score = bearing_score(&rays.dirs, &rays.centers, &rays.weights)?;
    let bearing = Point3::from(score.bearing);
    let held_point = held
        .filter(|&(x, at_infinity)| !at_infinity && x.coords.iter().all(|c| c.is_finite()))
        .map(|(x, _)| x);
    let fit_from_held = |bar: f64| {
        fit_usable_point(
            &rays.dirs,
            &rays.centers,
            &rays.weights,
            held_point,
            min_depth,
            bar,
        )
    };

    let mut likelihood_ratio = f64::NAN;
    let (at_infinity, coordinate, reason) = if is_finite(&score, threshold) {
        let placed = fit_from_held(threshold);
        likelihood_ratio = placed.fit.map_or(f64::NAN, |f| f.depth_likelihood_ratio);
        match placed.point {
            Some(x) => (
                false,
                x,
                if score.depth_score >= threshold {
                    ClassificationReason::ScoreCleared
                } else {
                    ClassificationReason::MidpointBoundCleared
                },
            ),
            // Reclassification keeps a usable stored point it agrees with, and
            // so does the bench when the fit from it found nothing better.
            None if held_point
                .is_some_and(|x| is_usable_point(&x, &rays.dirs, &rays.centers, min_depth)) =>
            {
                (
                    false,
                    held_point.expect("checked above"),
                    ClassificationReason::HeldPointKept,
                )
            }
            None if score.bearing_in_front_of_all_cameras => {
                (true, bearing, ClassificationReason::NoUsablePoint)
            }
            None => left_unusable(held, bearing),
        }
    } else if score.bearing_in_front_of_all_cameras {
        (
            true,
            bearing,
            if score.bearing_cost < threshold {
                ClassificationReason::BearingCostBelowThreshold
            } else {
                ClassificationReason::ScoreBelowThreshold
            },
        )
    } else {
        let placed = fit_from_held(0.0);
        likelihood_ratio = placed.fit.map_or(f64::NAN, |f| f.depth_likelihood_ratio);
        match placed.point {
            Some(x) => (false, x, ClassificationReason::BearingBehindCamera),
            None => left_unusable(held, bearing),
        }
    };

    let distance = if at_infinity {
        f64::NAN
    } else {
        let mut sum = Vector3::zeros();
        for c in &rays.centers {
            sum += c.coords;
        }
        (coordinate.coords - sum / rays.centers.len() as f64).norm()
    };
    Some(TrackClassification {
        at_infinity,
        coordinate,
        reason,
        sigma_px: rays.sigma_px,
        threshold,
        bearing_cost: score.bearing_cost,
        depth_score: score.depth_score,
        midpoint_bound: score.midpoint_bound,
        depth_likelihood_ratio: likelihood_ratio,
        distance,
        min_depth,
        num_views: score.num_views,
        max_pair_angle_deg: max_pair_angle_deg(&rays.dirs),
    })
}

/// What a track with no representation that describes every sighting writes:
/// the coordinate it held, or the bearing when it held none.
fn left_unusable(
    held: Option<(Point3<f64>, bool)>,
    bearing: Point3<f64>,
) -> (bool, Point3<f64>, ClassificationReason) {
    match held {
        Some((x, at_infinity)) if x.coords.iter().all(|c| c.is_finite()) => {
            (at_infinity, x, ClassificationReason::LeftUnusable)
        }
        _ => (true, bearing, ClassificationReason::LeftUnusable),
    }
}

/// The widest angle between any two of `dirs`, in degrees.
///
/// Quadratic in the track length, which is what a bench track is: tens of
/// sightings, once per fit.
fn max_pair_angle_deg(dirs: &[Vector3<f64>]) -> f64 {
    let mut widest = 0.0_f64;
    for (i, a) in dirs.iter().enumerate() {
        for b in &dirs[i + 1..] {
            let cos = a.dot(b).clamp(-1.0, 1.0);
            widest = widest.max(cos.acos().to_degrees());
        }
    }
    widest
}
