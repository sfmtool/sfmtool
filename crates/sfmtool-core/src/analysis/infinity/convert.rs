// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Finite ↔ infinity point conversions for [`SfmrReconstruction`].
//!
//! A point at infinity (`w = 0`) is a feature track whose observation rays are
//! parallel to within measurement noise — distant content whose depth the SfM
//! solve cannot pin down. The two conversions here move points across that
//! boundary; see `specs/formats/sfmr-file-format.md` for the theory.
//!
//! [`SfmrReconstruction::classify_points_at_infinity`] decides each point with
//! the point-or-bearing test and moves it the way its verdict says, in either
//! direction. [`SfmrReconstruction::materialize_points_at_infinity`] places
//! every point at infinity at a finite depth for a consumer that cannot store
//! `w = 0`; that depth is supplied, not measured, so the two are not inverses.
//!
//! [`classify_rays_at_infinity`] is the older inverse-depth z rule, which
//! discovery and the bench still decide with.

use std::sync::Arc;

use nalgebra::{Point3, Vector3};
use ndarray::{Array2, Array4, Axis};

use crate::analysis::point_or_bearing::{track_rays, PointOrBearingError};
use crate::analysis::reprojection_noise::ReprojectionNoise;
use crate::reconstruction::data::{count_points_at_infinity, observation_reprojection_error};
use crate::reconstruction::triangulation::{
    bearing_score_batch, depth_uncertainty_batch, fit_point_and_bearing, is_finite,
    triangulate_batch, PointBearingFitOptions, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD,
};
use crate::reconstruction::SfmrReconstruction;

/// The minimum depth of a promoted point, as a fraction of the reconstruction's
/// scene scale: the median distance from a finite point to the centre of a
/// camera observing it (the camera extents when no finite point is observed).
///
/// [`SfmrReconstruction::classify_points_at_infinity`] promotes a point at
/// infinity only to a fitted position at least this far from every observing
/// camera centre. In near-camera geometry the point fit can end almost on a
/// camera's centre, which fits the rays but is no depth to store. On the Kerry
/// Park ground truth, the seoul bull ground truth and a seoul bull solve, the
/// stored finite point nearest a camera sits at 13%, 42% and 39% of that
/// median, so 1% leaves an order of magnitude below any stored point.
pub const DEFAULT_MIN_DEPTH_FRACTION: f64 = 0.01;

/// What [`SfmrReconstruction::classify_points_at_infinity`] did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InfinityReclassification {
    /// The noise level the points were scored at: the caller's, or the
    /// measured one. `None` when none was given and none could be measured
    /// (no observation of a finite point), in which case nothing was decided
    /// and every count is 0.
    pub sigma_px: Option<f64>,
    /// The measurement behind `sigma_px` when the pass measured it, with its
    /// observation and outlier counts; `None` when the caller gave one.
    pub noise: Option<ReprojectionNoise>,
    /// Points at infinity stored as finite points.
    pub promoted: usize,
    /// Finite points stored as points at infinity: those with a bearing
    /// verdict, and those with a finite verdict whose stored position was not
    /// one to keep and whose point fit gave none either.
    pub demoted: usize,
    /// Scored points whose verdict agrees with how they were stored, left as
    /// they were.
    pub kept: usize,
    /// Finite points with a finite verdict whose stored position was not one
    /// to keep (behind an observing camera, or nearer one's centre than the
    /// minimum depth), moved to the point fit started from it.
    pub refitted: usize,
    /// Finite points with a bearing verdict left finite because the bearing is
    /// behind one of their observing cameras.
    pub bearing_behind_camera: usize,
    /// Points at infinity with a finite verdict left at infinity because the
    /// point fit gave no usable point, or a `Λ` under the threshold.
    pub no_usable_point: usize,
    /// Finite points with a finite verdict whose stored position was not one
    /// to keep, whose point fit gave none either, and whose bearing is behind
    /// an observing camera: no representation describes every sighting, so
    /// the point is left at its stored position, unusable as it is. A caller
    /// that can prune sightings or drop points is the one to act on them.
    pub left_unusable: usize,
    /// Points with fewer than two usable rays, left as they were.
    pub unscored: usize,
}

/// Patch-frame adjustment for a point crossing the finite/infinity boundary.
///
/// A finite point's patch half-vectors are world-unit in-plane extents; a
/// point at infinity anchors its patch on a unit *direction*, with `u`, `v`
/// tangent to the direction sphere and `u × v` along `−d` (see the patch
/// frame section of `specs/formats/sfmr-file-format.md`). Without an
/// adjustment a demoted patch keeps world-sized vectors on the unit sphere —
/// orders of magnitude larger than its own anchor.
enum PatchFix {
    /// Finite → infinity: divide the extents by the demotion-time distance
    /// (world → angular, preserving apparent size), project the frame onto
    /// the tangent plane of `dir`, and enforce `u × v` along `−dir`.
    /// Degenerate projections (the surfel plane was parallel to `dir`) clear
    /// the patch instead.
    Demote { inv_dist: f64, dir: Vector3<f64> },
    /// Infinity → finite: angular extents scale by the placement distance
    /// (the stored tangent frame becomes a fronto-parallel world surfel).
    Scale(f64),
    /// The frame is meaningless: zero it (and its bitmap row).
    Clear,
}

/// Free the constraint of every point a representation change touched, `fixes`
/// being the patch-fix list both passes already keep of exactly those points.
///
/// A ranged or held point states something about a coordinate the caller owns;
/// a pass that rewrites the coordinate has ended that statement, so the honest
/// outcome is to release the point rather than leave a distance describing where
/// it used to be. Freeing an already-free point is a no-op, so this touches only
/// what a caller constrained.
fn release_touched_constraints(recon: &mut SfmrReconstruction, fixes: &[(usize, PatchFix)]) {
    if let Some(constraints) = recon.point_set.point_constraints.as_mut() {
        for (pidx, _) in fixes {
            constraints.free(*pidx);
        }
    }
}

/// Apply every fix in `fixes` to `recon`'s patch columns.
fn apply_patch_fixes(recon: &mut SfmrReconstruction, fixes: Vec<(usize, PatchFix)>) {
    if fixes.is_empty() {
        return;
    }
    // The bitmaps are shared behind an `Arc`, so this takes an owned array
    // out, edits that, and wraps a new one back in -- one array copy for the
    // whole pass at worst, and none when the pointer was unique, rather than a
    // write through a value another reconstruction may hold.
    let mut patch_u = recon.point_set.patch_u_halfvec_xyz.take();
    let mut patch_v = recon.point_set.patch_v_halfvec_xyz.take();
    let mut patch_bitmaps = recon
        .point_set
        .patch_bitmaps_y_x_rgba
        .take()
        .map(Arc::unwrap_or_clone);
    for (pidx, fix) in fixes {
        apply_patch_fix(&mut patch_u, &mut patch_v, &mut patch_bitmaps, pidx, fix);
    }
    recon.point_set.patch_u_halfvec_xyz = patch_u;
    recon.point_set.patch_v_halfvec_xyz = patch_v;
    recon.point_set.patch_bitmaps_y_x_rgba = patch_bitmaps.map(Arc::new);
}

fn apply_patch_fix(
    u: &mut Option<Array2<f32>>,
    v: &mut Option<Array2<f32>>,
    bitmaps: &mut Option<Array4<u8>>,
    pidx: usize,
    fix: PatchFix,
) {
    let clear = |u: &mut Option<Array2<f32>>,
                 v: &mut Option<Array2<f32>>,
                 bitmaps: &mut Option<Array4<u8>>| {
        for arr in [u, v] {
            if let Some(a) = arr.as_mut() {
                a.row_mut(pidx).fill(0.0);
            }
        }
        if let Some(b) = bitmaps.as_mut() {
            b.index_axis_mut(Axis(0), pidx).fill(0);
        }
    };
    let read = |a: &Option<Array2<f32>>| -> Option<Vector3<f64>> {
        a.as_ref().map(|a| {
            Vector3::new(
                f64::from(a[[pidx, 0]]),
                f64::from(a[[pidx, 1]]),
                f64::from(a[[pidx, 2]]),
            )
        })
    };
    let write = |a: &mut Option<Array2<f32>>, w: Vector3<f64>| {
        if let Some(a) = a.as_mut() {
            a[[pidx, 0]] = w.x as f32;
            a[[pidx, 1]] = w.y as f32;
            a[[pidx, 2]] = w.z as f32;
        }
    };
    match fix {
        PatchFix::Scale(s) => {
            for arr in [u, v] {
                if let Some(a) = arr.as_mut() {
                    for x in a.row_mut(pidx) {
                        *x = (f64::from(*x) * s) as f32;
                    }
                }
            }
        }
        PatchFix::Clear => clear(u, v, bitmaps),
        PatchFix::Demote { inv_dist, dir } => {
            let (Some(u0), Some(v0)) = (read(u), read(v)) else {
                return;
            };
            if u0 == Vector3::zeros() {
                return; // no-patch row stays the no-patch row
            }
            let mut ut = (u0 - u0.dot(&dir) * dir) * inv_dist;
            let mut vt = (v0 - v0.dot(&dir) * dir) * inv_dist;
            let cross = ut.cross(&vt);
            let max_sq = ut.norm_squared().max(vt.norm_squared());
            if max_sq == 0.0 || cross.norm_squared() < 1e-12 * max_sq * max_sq {
                clear(u, v, bitmaps);
                return;
            }
            if cross.dot(&dir) > 0.0 {
                std::mem::swap(&mut ut, &mut vt);
            }
            write(u, ut);
            write(v, vt);
        }
    }
}

/// The camera-cloud centroid — the reference origin both conversions measure
/// point distances from (matching the materialisation placement geometry).
fn camera_cloud_centroid(centers: &[Point3<f64>]) -> Point3<f64> {
    if centers.is_empty() {
        return Point3::origin();
    }
    let mut sum = Vector3::zeros();
    for c in centers {
        sum += c.coords;
    }
    Point3::from(sum / centers.len() as f64)
}

/// SIFT keypoint localisation noise floor (pixels), for the z rule of
/// [`classify_rays_at_infinity`].
///
/// A caller of the z rule estimates a track's measurement noise from its
/// reprojection error but never lets it fall below this floor: a short track is
/// triangulated to fit its few observations almost exactly regardless of depth
/// conditioning, so its reprojection error under-states the true noise.
pub const DEFAULT_NOISE_FLOOR_PX: f64 = 1.0;

/// Provisional inverse-depth z-score cutoff: a track whose `depth / σ_depth`
/// falls below this is statistically indistinguishable from infinity and is
/// classified as a `w = 0` point. (KerryPark360 populations: genuine z ≈ 62 vs
/// discovered z ≈ 3.) The scale-free z-score is the decision variable; final
/// calibration on larger captures is deferred — see the spec's open questions.
pub const DEFAULT_INVERSE_DEPTH_Z_CUTOFF: f64 = 4.0;

/// Cheap geometric pre-filter on the condition number of the normal matrix `A`.
/// A track this well-conditioned has an observable depth and is finite without
/// computing the noise-calibrated z-score; the z-score is only consulted in the
/// ill-conditioned regime above this. (KerryPark360 medians: genuine 82 vs
/// degenerate 89,599.) Note the condition number scales with track length, so
/// it is a pre-filter, not the decision variable.
pub const CONDITION_NUMBER_PREFILTER: f64 = 1e4;

/// What a track's rays resolve to.
#[derive(Debug, Clone, Copy)]
pub enum Classification {
    /// Triangulated finite point.
    Finite(Point3<f64>),
    /// Point at infinity — a unit bearing direction.
    Infinity(Point3<f64>),
    /// The baseline could not place a point even at `finite_horizon`, so neither
    /// finite nor infinity is earned (see `classify_rays_at_infinity`).
    Indeterminate,
}

/// A track's classification plus the diagnostics behind it, kept for debug
/// review of the points that get dropped.
#[derive(Debug, Clone, Copy)]
pub struct RayClassification {
    pub class: Classification,
    /// The least-squares point the rays triangulated to, whatever [`Self::class`]
    /// made of it.
    ///
    /// Beside the class rather than only inside `Classification::Finite`, because
    /// a caller that refused the depth may still want to know what was refused:
    /// the bench scores this point and [`Self::bearing`] against the sightings and
    /// keeps whichever explains them, which needs both candidates in hand.
    pub point: Point3<f64>,
    pub condition_number: f64,
    pub resolvable_distance: f64,
    pub inverse_depth_z: f64,
    pub bearing: Point3<f64>,
    pub num_views: usize,
}

/// Spatial extent (bounding-box diagonal) of a set of camera centers — the
/// scale of the region a capture explored, and the default `finite_horizon`.
pub fn camera_extents(centers: &[Point3<f64>]) -> f64 {
    let Some(first) = centers.first() else {
        return 0.0;
    };
    let mut lo = first.coords;
    let mut hi = first.coords;
    for c in centers {
        lo = lo.inf(&c.coords);
        hi = hi.sup(&c.coords);
    }
    (hi - lo).norm()
}

/// Classify one track from its observation rays into finite / at-infinity /
/// indeterminate.
///
/// `dirs` are the unit world-space rays (at least one), `centers` the matching
/// camera centers, and `sigma_rad` the per-ray angular noise (`noise_px / fᵢ`).
/// The decision:
///
/// - A clearly well-conditioned, in-front solve (condition number below
///   [`CONDITION_NUMBER_PREFILTER`]) is **finite** — no noise model needed.
/// - Otherwise, if the geometry cannot resolve a point even at `finite_horizon`
///   (`resolvable_distance < finite_horizon`), the call is **indeterminate**:
///   the baseline is too small to tell a scene-scale finite point from infinity.
/// - With adequate baseline, a degenerate/behind solve or an inverse-depth
///   z-score below `z_cutoff` is **at infinity** (a `w = 0` bearing direction —
///   the mean of the rays, or the first ray if they cancel exactly); else
///   **finite**.
pub fn classify_rays_at_infinity(
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    sigma_rad: &[f64],
    z_cutoff: f64,
    finite_horizon: f64,
) -> RayClassification {
    let offsets = [0usize, dirs.len()];
    let tri = triangulate_batch(dirs, centers, &offsets)
        .pop()
        .expect("one track");

    // The direction to store for a w = 0 point: the bearing mean of the rays,
    // or the first ray if they cancel exactly (degenerate; unreachable for
    // genuine near-parallel infinity tracks, whose rays sum to ≈ K·d ≠ 0).
    let bearing = {
        let mut sum = Vector3::zeros();
        for d in dirs {
            sum += d;
        }
        let norm = sum.norm();
        if norm > 0.0 {
            Point3::from(sum / norm)
        } else {
            Point3::from(dirs[0])
        }
    };
    let num_views = dirs.len();

    // A rank-deficient solve (parallel rays → infinite condition number), a
    // point behind a camera, or a non-finite point cannot be a physical finite
    // point.
    let geometrically_finite = tri.in_front_of_all_cameras
        && tri.condition_number.is_finite()
        && tri.point.coords.iter().all(|c| c.is_finite());

    // Cheap pre-filter: a well-conditioned, in-front depth is finite without the
    // noise-calibrated test (and has ample baseline, so never indeterminate).
    if geometrically_finite && tri.condition_number < CONDITION_NUMBER_PREFILTER {
        return RayClassification {
            class: Classification::Finite(tri.point),
            point: tri.point,
            condition_number: tri.condition_number,
            resolvable_distance: f64::NAN,
            inverse_depth_z: f64::NAN,
            bearing,
            num_views,
        };
    }

    // Degenerate / near-parallel / behind regime: needs the noise-calibrated
    // diagnostics. `resolvable_distance` is depth-independent, so it stays valid
    // even when the solved point (and hence `inverse_depth_z`) is noise.
    let du = depth_uncertainty_batch(&[tri], dirs, centers, &offsets, sigma_rad)
        .pop()
        .expect("one track");
    let class = if du.resolvable_distance < finite_horizon {
        // Baseline can't reach the required distance — can't adjudicate.
        Classification::Indeterminate
    } else if !geometrically_finite || du.inverse_depth_z < z_cutoff {
        Classification::Infinity(bearing)
    } else {
        Classification::Finite(tri.point)
    };
    RayClassification {
        class,
        point: tri.point,
        condition_number: tri.condition_number,
        resolvable_distance: du.resolvable_distance,
        inverse_depth_z: du.inverse_depth_z,
        bearing,
        num_views,
    }
}

impl SfmrReconstruction {
    /// Largest focal length (pixels) for each image's camera.
    fn per_image_focal_max(&self) -> Vec<f64> {
        self.image_table
            .images
            .iter()
            .map(|im| {
                let (fx, fy) = self.image_table.cameras[im.camera_index as usize].focal_lengths();
                fx.max(fy)
            })
            .collect()
    }

    /// Decide every point of the reconstruction with the point-or-bearing test,
    /// and store each one the way its verdict says, returning a new
    /// reconstruction and what changed.
    ///
    /// Every point, finite or at infinity, is scored with
    /// [`bearing_score_batch`] on the rays of its observed pixels at the noise
    /// level `sigma_px` (the reconstruction's
    /// [`reprojection_noise`](Self::reprojection_noise) when `None`), and
    /// [`is_finite`] at [`DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD`] gives the
    /// verdict. Where the verdict and the stored representation agree the point
    /// is left as it is. Where they disagree:
    ///
    /// - **Demotion.** A finite point with a bearing verdict becomes `w = 0` at
    ///   the closed-form bearing of its score. When that bearing is behind one
    ///   of the observing cameras it describes no sighting there, and the point
    ///   is left as the solve produced it.
    /// - **Promotion.** A point at infinity with a finite verdict is placed by
    ///   [`fit_point_and_bearing`] with plain least squares and becomes
    ///   `w = 1` there, when the fit gives a usable point: one in front of every
    ///   observing camera and no closer to any of their centres than the
    ///   minimum depth, [`DEFAULT_MIN_DEPTH_FRACTION`] of the reconstruction's
    ///   scene scale, with a likelihood ratio `Λ` that itself reaches the
    ///   threshold. Otherwise it stays a bearing.
    /// - **A finite point stored where no point can be.** A finite verdict
    ///   keeps a finite point where it is stored when that position is usable
    ///   in the same sense. When it is not (behind a camera, or on top of one,
    ///   as when a run of frames collapsed onto one centre), the point fit is
    ///   run from the stored position and the point moved to its result if
    ///   that is usable; if not, the track is treated as a bearing and demoted
    ///   as above. When that bearing is behind a camera too, the point is left
    ///   at its stored position and counted in `left_unusable`.
    ///
    /// A point that changes gets its normal zeroed and its normal confidence
    /// set to 0 (a finite point's zero normal is filled in from its viewing
    /// directions when the file is written), its error recomputed against its
    /// observed pixels in the new representation, its patch frame converted
    /// between world and angular extents by its distance from the camera-cloud
    /// centroid, and its constraint released. A point with fewer than two
    /// usable rays is not scored and stays as it is. The pass removes no
    /// point and adds none.
    ///
    /// When `sigma_px` is `None` and the reconstruction has no observation of a
    /// finite point to measure the noise from, nothing is decided: the result
    /// is the reconstruction unchanged, with a summary whose `sigma_px` is
    /// `None`.
    ///
    /// # Errors
    ///
    /// [`PointOrBearingError::InvalidNoiseLevel`] for a `sigma_px` that is not
    /// finite and positive, and [`PointOrBearingError::Reconstruction`] when the
    /// observations' pixels cannot be read (a `sift_files` reconstruction
    /// without its inline keypoints whose `.sift` files are not in the
    /// workspace).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sfmtool_core::SfmrReconstruction;
    /// # fn run(recon: &SfmrReconstruction) -> Result<(), Box<dyn std::error::Error>> {
    /// let (reclassified, summary) = recon.classify_points_at_infinity(None)?;
    /// println!(
    ///     "{} promoted, {} demoted at {:?} px",
    ///     summary.promoted, summary.demoted, summary.sigma_px
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn classify_points_at_infinity(
        &self,
        sigma_px: Option<f64>,
    ) -> Result<(Self, InfinityReclassification), PointOrBearingError> {
        if let Some(s) = sigma_px {
            if !(s.is_finite() && s > 0.0) {
                return Err(PointOrBearingError::InvalidNoiseLevel(s));
            }
        }
        let mut summary = InfinityReclassification::default();

        let pixels = self.observation_pixels(&vec![true; self.image_table.images.len()])?;
        let sigma = match sigma_px {
            Some(s) => s,
            None => {
                let noise = self.reprojection_noise_from_pixels(&pixels);
                let measured = noise.sigma_px;
                summary.noise = Some(noise);
                match measured {
                    Some(s) => s,
                    None => return Ok((self.clone(), summary)),
                }
            }
        };
        summary.sigma_px = Some(sigma);

        let point_count = self.point_set.points.len();
        let mut observations = Vec::with_capacity(self.point_set.tracks.len());
        let mut offsets = Vec::with_capacity(point_count + 1);
        offsets.push(0);
        for p in 0..point_count {
            let start = self.point_set.observation_offsets[p];
            for (k, obs) in self.observations_for_point(p).iter().enumerate() {
                let row = start + k;
                observations.push((
                    obs.image_index as usize,
                    [pixels[[row, 0]] as f64, pixels[[row, 1]] as f64],
                ));
            }
            offsets.push(observations.len());
        }
        let rays = track_rays(&self.image_table, &observations, &offsets, sigma);
        let scores = bearing_score_batch(&rays.dirs, &rays.centers, &rays.offsets, &rays.weights);

        let centers: Vec<Point3<f64>> = self
            .image_table
            .images
            .iter()
            .map(|im| im.camera_center())
            .collect();
        let origin = camera_cloud_centroid(&centers);
        let min_depth = DEFAULT_MIN_DEPTH_FRACTION * self.scene_scale_for_min_depth(&centers);
        let plain = PointBearingFitOptions {
            soft_l1_scale: None,
            ..PointBearingFitOptions::default()
        };

        let mut recon = self.clone_for_edit();
        let mut patch_fixes: Vec<(usize, PatchFix)> = Vec::new();
        for (p, score) in scores.iter().enumerate() {
            let Some(score) = score else {
                summary.unscored += 1;
                continue;
            };
            let (lo, hi) = (rays.offsets[p], rays.offsets[p + 1]);
            // A position to store: in front of every observing camera, along
            // the ray its pixel gives, and no nearer any camera centre than the
            // minimum depth.
            let usable = |x: &Point3<f64>| {
                x.coords.iter().all(|c| c.is_finite())
                    && rays.dirs[lo..hi]
                        .iter()
                        .zip(&rays.centers[lo..hi])
                        .all(|(d, c)| (x - c).dot(d) > 0.0 && (x - c).norm() >= min_depth)
            };
            // The plain least-squares point fit, from `start` when given, kept
            // when it gives a usable point and its `Λ` itself clears the
            // threshold: the score is a prediction at `ρ = 0`, and where the
            // geometry is degenerate it can exceed the `Λ` the fit finds.
            let fit_usable = |start: Option<Point3<f64>>| {
                fit_point_and_bearing(
                    &rays.dirs[lo..hi],
                    &rays.centers[lo..hi],
                    &rays.weights[lo..hi],
                    start,
                    None,
                    &plain,
                )
                .filter(|f| {
                    f.in_front_of_all_cameras
                        && f.depth_likelihood_ratio >= DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD
                })
                .and_then(|f| f.point)
                .filter(&usable)
            };
            let finite = is_finite(score, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD);
            let point = &mut recon.point_set.points[p];

            if point.is_at_infinity() {
                if !finite {
                    summary.kept += 1;
                    continue;
                }
                let Some(x) = fit_usable(None) else {
                    summary.no_usable_point += 1;
                    continue;
                };
                point.position = x;
                point.w = 1.0;
                summary.promoted += 1;
                patch_fixes.push((p, PatchFix::Scale((x - origin).norm())));
            } else {
                // A finite verdict keeps a finite point where it is stored when
                // that position is usable. One that is not (behind a camera, or
                // on top of one) is re-fitted from where it is stored, and is
                // demoted when that gives no usable point either.
                let held = point.position;
                let refit = if !finite {
                    None
                } else if usable(&held) {
                    summary.kept += 1;
                    continue;
                } else {
                    fit_usable(Some(held))
                };
                if let Some(x) = refit {
                    point.position = x;
                    summary.refitted += 1;
                    // The patch was solved at a position that was no depth.
                    patch_fixes.push((p, PatchFix::Clear));
                } else {
                    if !score.bearing_in_front_of_all_cameras {
                        // Neither a point nor the bearing describes every
                        // sighting: the point stays where it is stored.
                        if finite {
                            summary.left_unusable += 1;
                        } else {
                            summary.bearing_behind_camera += 1;
                        }
                        continue;
                    }
                    let dir = score.bearing.normalize();
                    // The patch's world extents become angular ones at the
                    // distance the point was stored at, unless that position
                    // was no real depth (on or next to a camera), which leaves
                    // no extent to convert.
                    let dist = (held - origin).norm();
                    let near_camera = rays.centers[lo..hi]
                        .iter()
                        .any(|c| (held - c).norm() < min_depth);
                    let fix = if dist > 0.0 && dist.is_finite() && !near_camera {
                        PatchFix::Demote {
                            inv_dist: 1.0 / dist,
                            dir,
                        }
                    } else {
                        PatchFix::Clear
                    };
                    point.position = Point3::from(dir);
                    point.w = 0.0;
                    summary.demoted += 1;
                    patch_fixes.push((p, fix));
                }
            }
            point.normal = Vector3::zeros();
            point.error = self.observed_error(p, &point.position, point.w == 0.0, &pixels);
        }

        // A changed point's normal was just zeroed, so its confidence drops to
        // zero too: the format keeps the two coherent for `w = 0` rows, and a
        // promoted point's zero normal is a placeholder until one is fitted.
        if let Some(confidence) = recon.point_set.normal_confidence.as_mut() {
            for (pidx, _) in &patch_fixes {
                confidence[*pidx] = 0;
            }
        }

        // A changed point is no longer the point its constraint described: a
        // ranged one sits at a distance this pass just declared unmeasurable or
        // replaced, and a held coordinate has been overwritten. Release both,
        // the same rule a dropped reference image gets: a statement the
        // reconstruction has moved out from under is not one to keep.
        release_touched_constraints(&mut recon, &patch_fixes);
        apply_patch_fixes(&mut recon, patch_fixes);
        recon.point_set.infinity_point_count = count_points_at_infinity(&recon.point_set.points);
        Ok((recon, summary))
    }

    /// The length the minimum depth of a promoted point is a fraction of: the
    /// median distance from a finite point to the centre of a camera observing
    /// it, or, with no finite point observed, the camera extents.
    fn scene_scale_for_min_depth(&self, centers: &[Point3<f64>]) -> f64 {
        let points = &self.point_set.points;
        let mut distances: Vec<f64> = self
            .point_set
            .tracks
            .iter()
            .filter_map(|obs| {
                let point = &points[obs.point_index as usize];
                (!point.is_at_infinity())
                    .then(|| (point.position - centers[obs.image_index as usize]).norm())
            })
            .filter(|d| d.is_finite())
            .collect();
        if distances.is_empty() {
            return camera_extents(centers);
        }
        let mid = distances.len() / 2;
        let (_, &mut median, _) = distances.select_nth_unstable_by(mid, f64::total_cmp);
        median
    }

    /// The mean reprojection error of point `p`'s observations against
    /// `pixels` (rows as `tracks`), were the point at `position` (a direction
    /// when `at_infinity`). Observations with no pixel or no projection are
    /// left out, and with none left the error is 0, as `recompute_point_errors`
    /// gives.
    fn observed_error(
        &self,
        p: usize,
        position: &Point3<f64>,
        at_infinity: bool,
        pixels: &Array2<f32>,
    ) -> f32 {
        let start = self.point_set.observation_offsets[p];
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for (k, obs) in self.observations_for_point(p).iter().enumerate() {
            let pixel = [pixels[[start + k, 0]] as f64, pixels[[start + k, 1]] as f64];
            if !(pixel[0].is_finite() && pixel[1].is_finite()) {
                continue;
            }
            let image = &self.image_table.images[obs.image_index as usize];
            if let Some(e) = observation_reprojection_error(
                &image.quaternion_wxyz,
                &image.translation_xyz,
                &self.image_table.cameras[image.camera_index as usize],
                position,
                at_infinity,
                pixel,
            ) {
                if e.is_finite() {
                    sum += e;
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            (sum / count as f64) as f32
        }
    }

    /// Materialise every point at infinity as a finite point, returning a new
    /// reconstruction.
    ///
    /// A `w = 0` point has no depth to recover, so this does not triangulate.
    /// It places the point along its stored direction, at the camera-cloud
    /// centroid plus a distance `t · d`. `t` is the largest per-camera
    /// distance beyond which the materialised point's parallax falls below one
    /// pixel (`fᵢ · r⊥ᵢ`, the focal length times the camera-to-origin offset
    /// perpendicular to the direction) — far enough to be faithful in every
    /// camera, no farther. Finite points are left unchanged, as is any
    /// malformed `w = 0` point whose stored direction has zero length.
    ///
    /// The result exists for consumers that cannot represent `w = 0` (COLMAP
    /// export, a finite-only solver). It is not the inverse of
    /// [`Self::classify_points_at_infinity`].
    pub fn materialize_points_at_infinity(&self) -> Self {
        if self.image_table.images.is_empty() {
            return self.clone();
        }
        let centers: Vec<Point3<f64>> = self
            .image_table
            .images
            .iter()
            .map(|im| im.camera_center())
            .collect();
        let focal_max = self.per_image_focal_max();

        // Reference origin: the camera-cloud centroid.
        let origin = camera_cloud_centroid(&centers);

        // Fallback placement distance when the pixel-differential geometry is
        // degenerate (every camera lies on the line origin + t·d).
        let cloud_radius = centers
            .iter()
            .map(|c| (c.coords - origin.coords).norm())
            .fold(0.0_f64, f64::max)
            .max(1.0);

        let mut recon = self.clone_for_edit();
        let mut patch_fixes: Vec<(usize, PatchFix)> = Vec::new();
        for (pidx, pt) in recon.point_set.points.iter_mut().enumerate() {
            if !pt.is_at_infinity() {
                continue;
            }
            // The stored coordinate is meant to be a unit direction;
            // renormalise defensively so the placement geometry below holds
            // even if the input drifted off the unit sphere. A zero-norm
            // direction is malformed — leave that point untouched.
            let d = pt.position.coords;
            let d_norm = d.norm();
            if d_norm == 0.0 {
                continue;
            }
            let d = d / d_norm;

            let mut t = 0.0_f64;
            for o in self.observations_for_point(pidx) {
                let img = o.image_index as usize;
                let r = origin.coords - centers[img].coords;
                let r_perp = (r - r.dot(&d) * d).norm();
                t = t.max(focal_max[img] * r_perp);
            }
            if !t.is_finite() || t <= 0.0 {
                t = cloud_radius;
            }

            pt.position = Point3::from(origin.coords + t * d);
            pt.w = 1.0;
            // Angular patch extents on the direction sphere become world-unit
            // extents at the placement depth (the inverse of the demotion
            // rescale in `classify_points_at_infinity`).
            patch_fixes.push((pidx, PatchFix::Scale(t)));
        }

        // Promotion rewrites the point's representation, so its constraint is
        // released for the reason the demotion above releases one.
        release_touched_constraints(&mut recon, &patch_fixes);

        apply_patch_fixes(&mut recon, patch_fixes);
        recon.point_set.infinity_point_count = count_points_at_infinity(&recon.point_set.points);
        recon
    }
}

#[cfg(test)]
mod tests;
