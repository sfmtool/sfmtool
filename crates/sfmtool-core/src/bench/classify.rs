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
//! The criterion is not the bench's own. It is
//! [`classify_rays_at_infinity`], the same per-track test
//! [`classify_points_at_infinity`](crate::SfmrReconstruction::classify_points_at_infinity)
//! reclassifies a whole reconstruction with and
//! [`find_points_at_infinity`](crate::SfmrReconstruction::find_points_at_infinity)
//! admits discovered tracks on, called here over one track's rays with the same
//! defaults. What the bench adds is the third answer's disposal: a
//! reconstruction pass leaves an *indeterminate* track alone, because leaving it
//! alone is an option when the pass is relabel-only, and a fit has to write
//! something. A track whose baseline cannot resolve a scene-scale depth is
//! written as the bearing it is.
//!
//! **And the bench checks the criterion's answer against the sightings.** The
//! criterion is a statement about *observability* -- whether the geometry could
//! resolve a depth -- and on an ill-conditioned solve it can clear its own bar
//! on a depth that nothing in the photographs supports: the least-squares
//! midpoint of near-parallel, slightly inconsistent rays lands wherever the
//! inconsistency throws it, and then reprojects nowhere near the sightings it was
//! solved from. So both candidates are scored on the one thing the person can
//! see -- the rms pixel distance from each sighting to where the candidate
//! projects in its own photograph -- and the criterion's answer stands only if
//! that agrees. See [`RESIDUAL_MARGIN`].

use std::fmt;

use nalgebra::{Point3, Vector3};

use crate::analysis::infinity::{
    camera_extents, classify_rays_at_infinity, Classification, DEFAULT_INVERSE_DEPTH_Z_CUTOFF,
    DEFAULT_NOISE_FLOOR_PX,
};
use crate::patch::normal_refine::ProjectedImage;

/// The measurement noise a bench classification assumes at each sighting, in
/// source-image px.
///
/// The reconstruction's own floor, unchanged: the per-ray angular noise is this
/// over the observing camera's focal length, which is what turns a spread of
/// rays into a depth uncertainty. A bench track's sightings are localized by the
/// same kernels the embed pass runs, so they carry the same floor.
pub const DEFAULT_CLASSIFY_NOISE_FLOOR_PX: f64 = DEFAULT_NOISE_FLOOR_PX;

/// The inverse-depth z-score a track's depth has to reach to be written as a
/// finite point.
///
/// [`DEFAULT_INVERSE_DEPTH_Z_CUTOFF`], the reconstruction's own bar.
pub const DEFAULT_CLASSIFY_Z_CUTOFF: f64 = DEFAULT_INVERSE_DEPTH_Z_CUTOFF;

/// How much better a finite point has to explain the sightings than the bearing
/// does before the depth is believed: its rms reprojection residual must be
/// under this fraction of the bearing's, **and** under it by more than the noise
/// floor.
///
/// A fraction rather than a difference because the comparison has no natural
/// scale -- a scene metre is a pixel count that depends on the lens and the
/// depth -- and a two-sided test because neither half alone is enough. Without
/// the fraction a residual of 0.4 px would beat one of 1.5 px and the depth
/// would be believed on a pixel of noise; without the noise-floor term a
/// residual of 0.05 px would beat one of 0.07 px, which is two roundings of the
/// same answer.
///
/// **Why the bearing is the default and the finite point has to earn it.** The
/// finite candidate has three degrees of freedom against the bearing's two, and
/// neither is fitted to minimise pixel error -- so a finite point that fits the
/// sightings *slightly* better has bought that with its extra freedom, while one
/// that fits them clearly better has found a depth. `0.8` asks for a fifth off
/// the residual, which is far more than a degree of freedom buys on a track
/// whose parallax is real, and far less than the several-fold gap a genuine
/// finite point shows against a bearing that cannot bend toward it.
pub const RESIDUAL_MARGIN: f64 = 0.8;

/// Which of the criterion's three tests settled a track, and so what the
/// sentence beside the verdict says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationReason {
    /// The solve is conditioned well enough that the depth is observable without
    /// a noise model at all, which is the criterion's cheap pre-filter.
    WellConditioned,
    /// The depth stands clear of its own uncertainty: the inverse-depth z-score
    /// reached the bar.
    DepthResolved,
    /// The depth does not stand clear of its own uncertainty: the z-score fell
    /// short of the bar, or the solve was degenerate or behind a camera.
    DepthUnresolved,
    /// The observing baseline cannot place a point even at the capture's own
    /// scale, so neither answer is earned on the numbers and the honest one is
    /// the bearing.
    BaselineTooShort,
    /// The criterion found the depth observable, and the triangulated point then
    /// reprojected no better than the bearing: an ill-conditioned midpoint that
    /// landed wherever the rays' inconsistency threw it. The bearing stands.
    ///
    /// The two residuals are on the classification
    /// ([`TrackClassification::finite_rms_px`] and
    /// [`TrackClassification::bearing_rms_px`]) rather than in here, the same
    /// way [`Self::DepthResolved`] names the test and
    /// [`TrackClassification::inverse_depth_z`] carries its number.
    FiniteDoesNotExplainTheSightings,
    /// The criterion found the depth unobservable, and the bearing then
    /// reprojected clearly worse than the triangulated point: whatever the
    /// conditioning says, the photographs place the track at a depth.
    BearingDoesNotExplainTheSightings,
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
    /// The coordinate the track takes: the triangulated point when finite, the
    /// unit direction when at infinity.
    pub coordinate: Point3<f64>,
    /// Which test settled it.
    pub reason: ClassificationReason,
    /// The triangulation's condition number.
    pub condition_number: f64,
    /// The inverse-depth z-score, `NaN` where the pre-filter settled the call
    /// before the noise model was consulted.
    pub inverse_depth_z: f64,
    /// The bar that z-score was judged against.
    pub inverse_depth_z_cutoff: f64,
    /// The farthest depth this track's geometry can tell from infinity, in world
    /// units. `NaN` where the pre-filter settled the call.
    pub resolvable_distance: f64,
    /// The distance the resolvable one had to reach: the camera cloud's extent,
    /// which is the scale of the region the capture explored.
    pub finite_horizon: f64,
    /// The widest angle between any two of the rays, in degrees. Not a test --
    /// it is the number a person reads a near-parallel track by, and it is the
    /// one the Point Track Detail panel already shows for a committed track.
    pub max_pair_angle_deg: f64,
    /// Rms distance, in px, from each sighting to where the **triangulated
    /// point** projects in its own photograph. `NaN` when no sighting's view
    /// could be projected into.
    pub finite_rms_px: f64,
    /// Rms distance, in px, from each sighting to where the **bearing** projects
    /// in its own photograph.
    pub bearing_rms_px: f64,
    /// The fraction of `bearing_rms_px` that `finite_rms_px` had to come under
    /// for the depth to be believed. [`RESIDUAL_MARGIN`] unless a caller moved
    /// it.
    pub residual_margin: f64,
}

impl TrackClassification {
    /// Whether the finite point explains the sightings clearly better than the
    /// bearing does: under the margin's fraction of its residual, and under it by
    /// more than one sighting's worth of noise.
    ///
    /// The one comparison, written once, because it is asked in both directions:
    /// a finite answer stands only when this holds, and a bearing answer stands
    /// only when it does not.
    fn finite_explains_better(&self, noise_floor_px: f64) -> bool {
        self.finite_rms_px.is_finite()
            && self.bearing_rms_px.is_finite()
            && self.finite_rms_px < self.residual_margin * self.bearing_rms_px
            && self.finite_rms_px + noise_floor_px < self.bearing_rms_px
    }
}

impl fmt::Display for TrackClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = self.coordinate;
        if self.at_infinity {
            write!(f, "at infinity along ({:.4}, {:.4}, {:.4})", c.x, c.y, c.z)?;
        } else {
            write!(f, "finite at ({:.4}, {:.4}, {:.4})", c.x, c.y, c.z)?;
        }
        write!(f, ": ")?;
        match self.reason {
            ClassificationReason::WellConditioned => write!(
                f,
                "condition number {:.0} under the {:.0} bar",
                self.condition_number,
                crate::analysis::infinity::CONDITION_NUMBER_PREFILTER
            )?,
            ClassificationReason::DepthResolved => write!(
                f,
                "inverse-depth z {:.2} over the {:.2} bar",
                self.inverse_depth_z, self.inverse_depth_z_cutoff
            )?,
            ClassificationReason::DepthUnresolved => write!(
                f,
                "inverse-depth z {:.2} under the {:.2} bar",
                self.inverse_depth_z, self.inverse_depth_z_cutoff
            )?,
            ClassificationReason::BaselineTooShort => write!(
                f,
                "the baseline tells depth apart from infinity only to {:.3}, short of \
                 the {:.3} the capture spans",
                self.resolvable_distance, self.finite_horizon
            )?,
            ClassificationReason::FiniteDoesNotExplainTheSightings => write!(
                f,
                "finite point would have {:.1} px rms against the bearing's {:.1} px",
                self.finite_rms_px, self.bearing_rms_px
            )?,
            ClassificationReason::BearingDoesNotExplainTheSightings => write!(
                f,
                "bearing would have {:.1} px rms against the point's {:.1} px",
                self.bearing_rms_px, self.finite_rms_px
            )?,
        }
        // The two residuals are the evidence a person reads the call by, so they
        // are in every sentence and not only in the two the data check settled.
        if !matches!(
            self.reason,
            ClassificationReason::FiniteDoesNotExplainTheSightings
                | ClassificationReason::BearingDoesNotExplainTheSightings
        ) {
            write!(
                f,
                ", rms {:.1} px finite against {:.1} px as a bearing",
                self.finite_rms_px, self.bearing_rms_px
            )?;
        }
        write!(f, ", rays up to {:.3} deg apart", self.max_pair_angle_deg)
    }
}

/// One track's rays in world, and the lens each was measured through.
///
/// Built once per triangulation and handed to both the solve and the
/// classification, so the two read one set of rays.
pub struct TrackRays {
    /// Unit world-space ray per sighting.
    pub dirs: Vec<Vector3<f64>>,
    /// The camera centre each was cast from.
    pub centers: Vec<Point3<f64>>,
    /// The largest focal length of the camera each was measured through, in px,
    /// which is what turns the pixel noise floor into an angular one.
    pub focal_max: Vec<f64>,
    /// The pixel each ray was cast from, in its own photograph. What the
    /// residual check scores a candidate against, because it is what a person
    /// looking at that photograph can see.
    pub pixels: Vec<[f64; 2]>,
    /// The image each ray was cast in, as an index into the views.
    pub views: Vec<usize>,
}

impl TrackRays {
    /// How many rays there are.
    pub fn len(&self) -> usize {
        self.dirs.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.dirs.is_empty()
    }
}

/// Decide whether `rays` fix a point or only a bearing: the reconstruction's own
/// criterion, checked against what the two candidates do to the sightings.
///
/// `noise_floor_px` is the per-sighting measurement noise, `z_cutoff` the
/// inverse-depth z-score a depth has to reach and `residual_margin` the fraction
/// of the bearing's residual a finite point has to come under; all three default
/// to the constants at the top of this module, the first two being the
/// reconstruction's own. `images` supplies the photographs the residuals are
/// measured in and the camera cloud the `finite_horizon` is measured over: the
/// extent of **every** image's centre, not just the observing ones, because the
/// horizon is the scale of the region the capture explored and a track observed
/// by three neighbouring frames is still a track in that capture.
///
/// **The criterion answers first and the sightings have the last word.** The
/// criterion says whether the depth is *observable*; the residuals say whether
/// the depth it found is *there*. Where they disagree the answer is the one the
/// photographs support, and the reason names that
/// ([`ClassificationReason::FiniteDoesNotExplainTheSightings`] and its
/// counterpart), because a call overturned on the pixels is a different thing to
/// report than a call the conditioning settled.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::classify::{classify_track_rays, DEFAULT_CLASSIFY_NOISE_FLOOR_PX,
/// #     DEFAULT_CLASSIFY_Z_CUTOFF, RESIDUAL_MARGIN};
/// # fn run(
/// #     rays: &sfmtool_core::bench::classify::TrackRays,
/// #     images: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) {
/// let call = classify_track_rays(
///     rays,
///     images,
///     DEFAULT_CLASSIFY_NOISE_FLOOR_PX,
///     DEFAULT_CLASSIFY_Z_CUTOFF,
///     RESIDUAL_MARGIN,
/// );
/// println!("{call}");   // "at infinity along (…): finite point would have 15.1 px rms …"
/// # }
/// ```
pub fn classify_track_rays(
    rays: &TrackRays,
    images: &[ProjectedImage<'_>],
    noise_floor_px: f64,
    z_cutoff: f64,
    residual_margin: f64,
) -> TrackClassification {
    let centers: Vec<Point3<f64>> = images
        .iter()
        .map(|view| view.cam_from_world.inverse_translation_origin())
        .collect();
    let finite_horizon = camera_extents(&centers);
    let sigma_rad: Vec<f64> = rays
        .focal_max
        .iter()
        .map(|&f| if f > 0.0 { noise_floor_px / f } else { 0.0 })
        .collect();
    let rc = classify_rays_at_infinity(
        &rays.dirs,
        &rays.centers,
        &sigma_rad,
        z_cutoff,
        finite_horizon,
    );
    // Both candidates, always: the triangulated point the criterion judged, and
    // the bearing it would fall back to. Which the criterion picked decides
    // nothing here -- the residuals are a reading of the data and not of the
    // answer, so they are the same two numbers either way.
    let finite_point = rc.point;
    let finite_rms_px = residual_rms_px(rays, images, &finite_point, 1.0);
    let bearing_rms_px = residual_rms_px(rays, images, &rc.bearing, 0.0);

    // The criterion leaves the noise-calibrated diagnostics `NaN` when its cheap
    // condition-number pre-filter settled the call, which is exactly how a
    // caller tells the two finite paths apart.
    let (at_infinity, reason) = match rc.class {
        Classification::Finite(_) => (
            false,
            if rc.inverse_depth_z.is_nan() {
                ClassificationReason::WellConditioned
            } else {
                ClassificationReason::DepthResolved
            },
        ),
        Classification::Infinity(_) => (true, ClassificationReason::DepthUnresolved),
        // A reconstruction pass can leave an indeterminate track alone; a fit has
        // to write something, and what the numbers say is that this track's
        // depth is not observable.
        Classification::Indeterminate => (true, ClassificationReason::BaselineTooShort),
    };
    let mut call = TrackClassification {
        at_infinity,
        coordinate: if at_infinity {
            rc.bearing
        } else {
            finite_point
        },
        reason,
        condition_number: rc.condition_number,
        inverse_depth_z: rc.inverse_depth_z,
        inverse_depth_z_cutoff: z_cutoff,
        resolvable_distance: rc.resolvable_distance,
        finite_horizon,
        max_pair_angle_deg: max_pair_angle_deg(&rays.dirs),
        finite_rms_px,
        bearing_rms_px,
        residual_margin,
    };

    // The data check, in both directions. A finite answer stands only where the
    // point explains the sightings clearly better than the bearing; a bearing
    // answer stands only where it does not.
    let finite_wins = call.finite_explains_better(noise_floor_px);
    match (call.at_infinity, finite_wins) {
        (false, false) => {
            call.at_infinity = true;
            call.coordinate = rc.bearing;
            call.reason = ClassificationReason::FiniteDoesNotExplainTheSightings;
        }
        (true, true) => {
            call.at_infinity = false;
            call.coordinate = finite_point;
            call.reason = ClassificationReason::BearingDoesNotExplainTheSightings;
        }
        _ => {}
    }
    call
}

/// Rms distance, in px, from each sighting to where the homogeneous candidate
/// `(coordinate, w)` projects in that sighting's own photograph.
///
/// The residual is [`observation_metrics`](super::evaluate::observation_metrics)'
/// own first number -- the one the *Error* column shows and the one a commit
/// stores -- so what the classification compares is what the panel then
/// tabulates. A sighting whose view the candidate does not project into scores
/// nothing rather than zero; with none left the answer is `NaN`, which
/// `finite_explains_better` reads as "no evidence" and refuses to act on.
fn residual_rms_px(
    rays: &TrackRays,
    images: &[ProjectedImage<'_>],
    coordinate: &Point3<f64>,
    w: f64,
) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for (k, &view) in rays.views.iter().enumerate() {
        let Some(image) = images.get(view) else {
            continue;
        };
        let (error, _) = super::evaluate::observation_metrics(image, coordinate, w, rays.pixels[k]);
        if error.is_finite() {
            sum += error * error;
            count += 1;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    (sum / count as f64).sqrt()
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
