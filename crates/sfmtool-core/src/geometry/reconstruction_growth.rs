// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Reconstruction growth ([`grow_reconstruction`]). See
//! `specs/core/reconstruction-growth.md`; the standalone batch registration
//! primitive lives in [`super::batch_resection`].
//!
//! [`grow_reconstruction`] registers the un-posed images of a cluster-track
//! set against a seeded reconstruction: repeatedly pose the un-posed image
//! with the most observations of valid points (RANSAC P3P polished by trimmed
//! pose-only refinement, with covisible-neighbour inits as the fallback),
//! triangulate clusters as they gain posed views, and interleave bounded
//! staged bundle adjustments — until no further image clears its acceptance
//! gate. Per-adjustment cost is bounded by construction: the frontier window
//! (`ba_window`) refines only the most-recently-posed cameras, periodic
//! anchor adjustments (`anchor_every`) refine a covisibility-spread subset of
//! all posed cameras to pull back drift, and `ba_cluster_cap` restricts the
//! adjustments to the best clusters by span. After every adjustment, points
//! for clusters outside the adjustment's observation set are re-triangulated
//! from the full observation set at the updated poses (the adjustment wipes
//! what it was not given; without the refill the next-best-view count stalls
//! at the adjustment-set boundary). A finishing adjustment releases the focal
//! on a covisibility-spread subset, followed by a re-triangulation at the
//! released focal.
//!
//! Images whose resection falls below `accept_gate ×` the median
//! accepted-so-far inlier fraction are deferred rather than rejected; one
//! adjustment + refill re-arms them, and if growth still stalls the strongest
//! deferred candidate is force-accepted without building points from it,
//! verified through one adjustment, and kept only if its inliers rose into
//! the accepted band (its P3P consensus clusters promoted whole into the
//! adjustment set, its non-consensus observations quarantined). A rejected
//! force-accept restores the prior poses, structure, and adjustment set.
//!
//! Everything runs in the canonical camera frame (the camera looks along
//! `−Z`); observations are full-pixel positions and poses are world-to-camera
//! `x_cam = R·X + t`. Degenerate inputs (no seed poses, no triangulable
//! clusters, every image below `min_obs`) return the input state with empty
//! growth, not an error.

use std::collections::BTreeSet;

use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};

use crate::camera::CameraModel;
use crate::features::cluster_match::covisibility::ClusterCovisibility;
use crate::geometry::absolute_pose::{estimate_absolute_pose, AbsolutePoseOptions};
use crate::geometry::bundle_adjust::{bundle_adjust, BaSchedule, DEFAULT_SCHEDULE};
use crate::geometry::pose_refine::refine_absolute_pose;
use crate::reconstruction::triangulation::triangulate_batch;
use crate::CameraIntrinsics;

// ── Tuning (see the spec) ────────────────────────────────────────────────────

/// The growth adjustments' staged schedule (the finishing adjustment uses
/// [`DEFAULT_SCHEDULE`]).
const GROW_SCHEDULE: [BaSchedule; 2] = [
    BaSchedule {
        trim_px: 30.0,
        loss_scale: 3.0,
    },
    BaSchedule {
        trim_px: 8.0,
        loss_scale: 1.5,
    },
];
/// LM iteration budget per adjustment round.
const BA_MAX_ITERS: usize = 60;
/// Trim survivors a point needs to stay in an adjustment.
const BA_MIN_TRACK: usize = 2;
/// Degenerate floor for the adjustments (state passes through below it).
const BA_MIN_OBS: usize = 12;
/// Pixel inlier bound for the P3P RANSAC, converted to an angular threshold
/// via the camera's mean focal length.
const P3P_MAX_ERROR_PX: f64 = 4.0;
/// A growth P3P consensus below this falls through to the neighbour-init
/// fallback.
const P3P_MIN_CONSENSUS_GROW: usize = 12;
/// Trim rounds for the pose-only refinement polish.
const REFINE_TRIM_ROUNDS: usize = 5;
/// Fraction of observations retained per trim round.
const REFINE_KEEP_FRACTION: f64 = 0.6;
/// Final-inlier pixel threshold: scores resections and the growth gates.
const INLIER_PX: f64 = 3.0;
/// The neighbour-init fallback stops at the first init clearing this
/// all-observation inlier fraction.
const FALLBACK_EARLY_INLIER: f64 = 0.4;
/// Covisible-neighbour inits tried by the growth fallback.
const GROW_FALLBACK_INITS: usize = 3;
/// A force-accepted image with a P3P consensus is kept when at least this
/// fraction of its consensus observations survive the verification
/// adjustment (the registration claim is those observations, not the junk).
const CONSENSUS_SURVIVAL: f64 = 0.5;
/// Camera cap for the periodic anchor adjustments' covisibility-spread
/// subset.
const ANCHOR_CAP: usize = 150;
/// Camera cap for the finishing (focal-release) adjustment's
/// covisibility-spread subset.
const FINISH_CAP: usize = 120;

/// Options for [`grow_reconstruction`].
#[derive(Clone, Debug)]
pub struct GrowOptions {
    /// Number of most-recently-posed cameras each growth adjustment refines
    /// (registration order, force-rejected images removed). 0 refines every
    /// posed camera.
    pub ba_window: usize,
    /// Every `anchor_every`-th growth adjustment refines a
    /// covisibility-spread subset of all posed cameras (capped
    /// [`ANCHOR_CAP`]) instead of the frontier. 0 never anchors.
    pub anchor_every: usize,
    /// Restrict the adjustments to the best-`cap` clusters by span
    /// (resection, triangulation, and the next-best-view count always see
    /// every cluster). 0 admits all.
    pub ba_cluster_cap: usize,
    /// An image needs at least this many observations of valid points to be
    /// a growth candidate.
    pub min_obs: usize,
    /// Defer an image whose inlier fraction falls below `accept_gate ×` the
    /// median accepted-so-far fraction.
    pub accept_gate: f64,
    /// Seed for the P3P RANSAC; same inputs + seed give identical output.
    pub seed: u64,
}

impl Default for GrowOptions {
    fn default() -> Self {
        Self {
            ba_window: 0,
            anchor_every: 0,
            ba_cluster_cap: 0,
            min_obs: 8,
            accept_gate: 0.35,
            seed: 0,
        }
    }
}

/// Result of [`grow_reconstruction`]. Arrays cover every image / cluster the
/// inputs reference; un-posed images carry the identity pose (see `posed`).
#[derive(Clone, Debug)]
pub struct ReconstructionGrowth {
    /// World-to-camera rotations (WXYZ), one per image.
    pub quaternions_wxyz: Vec<[f64; 4]>,
    /// World-to-camera translations, one per image.
    pub translations: Vec<[f64; 3]>,
    /// Which images are posed (seed and accepted growth).
    pub posed: Vec<bool>,
    /// World points indexed by cluster id (`NaN` where never triangulated).
    pub points: Vec<[f64; 3]>,
    /// The shared focal after the finishing release (the input focal when
    /// growth was empty or the model is not SIMPLE_PINHOLE).
    pub focal: f64,
    /// Per-observation residual norm at the final state; `+∞` where invalid
    /// (un-posed image, absent point, or failed projection).
    pub residual_norms: Vec<f64>,
}

// ── Small helpers ────────────────────────────────────────────────────────────

/// SplitMix64 scramble (the deterministic seed mixer; mirrors
/// `absolute_pose`).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Per-image RANSAC seed: a pure function of `(seed, image index)`, so the
/// batch registration is order-free and parallel-deterministic.
pub(crate) fn per_image_seed(seed: u64, image: u32) -> u64 {
    let mut s = seed ^ (image as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
    splitmix64(&mut s)
}

/// numpy-style median (mean of the middle two for even counts).
fn median(values: &[f64]) -> f64 {
    let mut s = values.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// The camera at focal `f` (identity for every model but SIMPLE_PINHOLE,
/// matching the bundle adjustment's focal handling).
fn cam_with_focal(cam: &CameraIntrinsics, f: f64) -> CameraIntrinsics {
    let mut out = cam.clone();
    if let CameraModel::SimplePinhole { focal_length, .. } = &mut out.model {
        *focal_length = f;
    }
    out
}

/// The P3P RANSAC's angular inlier threshold from the pixel bound.
fn angular_threshold(cam: &CameraIntrinsics) -> f64 {
    let (fx, fy) = cam.focal_lengths();
    (P3P_MAX_ERROR_PX / (0.5 * (fx + fy))).atan()
}

/// Reprojection residual norm of one observation under `(r, t)`; `None` when
/// the point is non-finite or outside the model domain.
fn residual_norm(
    cam: &CameraIntrinsics,
    r: &UnitQuaternion<f64>,
    t: &Vector3<f64>,
    x: &[f64; 3],
    uv: &[f64; 2],
) -> Option<f64> {
    if !x[0].is_finite() || !x[1].is_finite() || !x[2].is_finite() {
        return None;
    }
    let c = r * Vector3::new(x[0], x[1], x[2]) + t;
    let (u, v) = cam.ray_to_pixel([c.x, c.y, c.z])?;
    Some((u - uv[0]).hypot(v - uv[1]))
}

/// Fraction of the given observations within [`INLIER_PX`] under `(r, t)`.
fn inlier_fraction_of(
    cam: &CameraIntrinsics,
    r: &UnitQuaternion<f64>,
    t: &Vector3<f64>,
    world: &[[f64; 3]],
    uv: &[[f64; 2]],
) -> f64 {
    if uv.is_empty() {
        return 0.0;
    }
    let n_in = uv
        .iter()
        .zip(world)
        .filter(|(o, x)| residual_norm(cam, r, t, x, o).is_some_and(|rn| rn < INLIER_PX))
        .count();
    n_in as f64 / uv.len() as f64
}

/// One image's gathered 2D-3D candidates: observation row indexes, observed
/// pixels, world points, and unit bearings.
type Gathered = (Vec<usize>, Vec<[f64; 2]>, Vec<[f64; 3]>, Vec<Vector3<f64>>);

/// One image's estimate-then-refine resection ladder.
pub(crate) struct ResectOutcome {
    pub(crate) rotation: UnitQuaternion<f64>,
    pub(crate) translation: Vector3<f64>,
    /// All-observation inlier fraction at [`INLIER_PX`] over the supplied
    /// correspondences.
    pub(crate) inlier_fraction: f64,
    /// Indexes (into the supplied correspondence list) of the P3P consensus,
    /// when the minimal path won.
    pub(crate) consensus: Option<Vec<usize>>,
}

/// Resect one image against known structure: RANSAC P3P over the 2D-3D
/// candidates polished by trimmed pose-only refinement on the consensus
/// subset; when the minimal estimate fails (or its consensus is below
/// `min_consensus`), trimmed refinement from each `init_poses` entry, best
/// all-observation inlier fraction wins (stopping at the first init clearing
/// [`FALLBACK_EARLY_INLIER`]). `None` when both paths are unavailable.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resect_one(
    cam: &CameraIntrinsics,
    uv: &[[f64; 2]],
    world: &[[f64; 3]],
    bearings: &[Vector3<f64>],
    min_consensus: usize,
    seed: u64,
    init_poses: &[(UnitQuaternion<f64>, Vector3<f64>)],
) -> Option<ResectOutcome> {
    if uv.len() >= 3 {
        let opts = AbsolutePoseOptions {
            max_angular_error: angular_threshold(cam),
            seed,
            ..Default::default()
        };
        let points: Vec<Point3<f64>> = world
            .iter()
            .map(|x| Point3::new(x[0], x[1], x[2]))
            .collect();
        if let Some(est) = estimate_absolute_pose(bearings, &points, &opts) {
            let rows: Vec<usize> = (0..uv.len()).filter(|&k| est.inliers[k]).collect();
            if rows.len() >= min_consensus {
                // Polish on the consensus subset only (mostly inliers), then
                // score against ALL supplied observations so the gate stays
                // coherent with the fallback path and the median-based bar.
                let uv_c: Vec<[f64; 2]> = rows.iter().map(|&k| uv[k]).collect();
                let world_c: Vec<[f64; 3]> = rows.iter().map(|&k| world[k]).collect();
                let refined = refine_absolute_pose(
                    cam,
                    &uv_c,
                    &world_c,
                    &est.rotation,
                    &est.translation,
                    REFINE_TRIM_ROUNDS,
                    REFINE_KEEP_FRACTION,
                    INLIER_PX,
                );
                let inl =
                    inlier_fraction_of(cam, &refined.rotation, &refined.translation, world, uv);
                return Some(ResectOutcome {
                    rotation: refined.rotation,
                    translation: refined.translation,
                    inlier_fraction: inl,
                    consensus: Some(rows),
                });
            }
        }
    }

    let mut best: Option<ResectOutcome> = None;
    for (r0, t0) in init_poses {
        let refined = refine_absolute_pose(
            cam,
            uv,
            world,
            r0,
            t0,
            REFINE_TRIM_ROUNDS,
            REFINE_KEEP_FRACTION,
            INLIER_PX,
        );
        let inl = refined.inlier_fraction;
        if best.as_ref().is_none_or(|b| inl > b.inlier_fraction) {
            best = Some(ResectOutcome {
                rotation: refined.rotation,
                translation: refined.translation,
                inlier_fraction: inl,
                consensus: None,
            });
        }
        if inl > FALLBACK_EARLY_INLIER {
            break;
        }
    }
    best
}

/// Triangulate every cluster that lacks a finite point but has posed
/// observations (ray-midpoint batch triangulation; fewer than two posed
/// observations leave the row `NaN`). Existing finite points are untouched —
/// this is the post-adjustment refill: the adjustment re-triangulates only
/// the observations it was given, wiping every other cluster's point.
#[allow(clippy::too_many_arguments)]
fn fill_new_points(
    cam: &CameraIntrinsics,
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    posed: &[bool],
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &mut [[f64; 3]],
) {
    let mut dirs = Vec::new();
    let mut centers = Vec::new();
    let mut offsets = Vec::new();
    let mut track_cid: Vec<usize> = Vec::new();
    // Cluster ids are nondecreasing, so a linear scan groups the rows.
    for (k, &cid) in cluster_indexes.iter().enumerate() {
        let cid = cid as usize;
        if points[cid][0].is_finite() {
            continue;
        }
        let i = image_indexes[k] as usize;
        if !posed[i] {
            continue;
        }
        if track_cid.last() != Some(&cid) {
            offsets.push(dirs.len());
            track_cid.push(cid);
        }
        let r_inv = quats[i].inverse();
        let d = cam.pixel_to_ray(positions_xy[k][0], positions_xy[k][1]);
        dirs.push(r_inv * Vector3::new(d[0], d[1], d[2]));
        centers.push(Point3::from(-(r_inv * trans[i])));
    }
    offsets.push(dirs.len());

    let tris = triangulate_batch(&dirs, &centers, &offsets);
    for (t, tri) in tris.iter().enumerate() {
        if offsets[t + 1] - offsets[t] >= 2 {
            points[track_cid[t]] = [tri.point.x, tri.point.y, tri.point.z];
        }
    }
}

/// The dense covisibility over the cluster runs, when the image count fits
/// the dense bound (`None` beyond it: neighbour ranking and the spread
/// subsets then degrade to registration-order fallbacks).
pub(crate) fn build_covisibility(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    n_img: usize,
    n_cl: usize,
) -> Option<ClusterCovisibility> {
    let mut starts = vec![0u32; n_cl + 1];
    for &c in cluster_indexes {
        starts[c as usize + 1] += 1;
    }
    for c in 0..n_cl {
        starts[c + 1] += starts[c];
    }
    ClusterCovisibility::from_clusters(&starts, image_indexes, None, n_img).ok()
}

/// Per-observation adjustment-set mask under `ba_cluster_cap`: the best-`cap`
/// clusters by span (distinct observing images, ties broken by ascending
/// cluster id). All-true when the cap is 0 or not binding.
fn ba_cluster_mask(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    n_cl: usize,
    cap: usize,
) -> Vec<bool> {
    let n_obs = cluster_indexes.len();
    if cap == 0 || cap >= n_cl {
        return vec![true; n_obs];
    }
    let mut span = vec![0u32; n_cl];
    let mut run_start = 0usize;
    let mut imgs: Vec<u32> = Vec::new();
    while run_start < n_obs {
        let cid = cluster_indexes[run_start];
        let mut run_end = run_start + 1;
        while run_end < n_obs && cluster_indexes[run_end] == cid {
            run_end += 1;
        }
        imgs.clear();
        imgs.extend_from_slice(&image_indexes[run_start..run_end]);
        imgs.sort_unstable();
        imgs.dedup();
        span[cid as usize] = imgs.len() as u32;
        run_start = run_end;
    }
    let mut order: Vec<u32> = (0..n_cl as u32).collect();
    order.sort_unstable_by(|&a, &b| span[b as usize].cmp(&span[a as usize]).then(a.cmp(&b)));
    let mut keep_cl = vec![false; n_cl];
    for &c in order.iter().take(cap) {
        keep_cl[c as usize] = true;
    }
    cluster_indexes
        .iter()
        .map(|&c| keep_cl[c as usize])
        .collect()
}

/// Pack the final state into the result struct, computing per-observation
/// residual norms from the FULL observation set at the final camera.
#[allow(clippy::too_many_arguments)]
fn build_result(
    cam_final: &CameraIntrinsics,
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    posed: &[bool],
    points: Vec<[f64; 3]>,
    focal: f64,
) -> ReconstructionGrowth {
    let residual_norms: Vec<f64> = cluster_indexes
        .iter()
        .zip(image_indexes)
        .zip(positions_xy)
        .map(|((&cid, &img), uv)| {
            if !posed[img as usize] {
                return f64::INFINITY;
            }
            residual_norm(
                cam_final,
                &quats[img as usize],
                &trans[img as usize],
                &points[cid as usize],
                uv,
            )
            .unwrap_or(f64::INFINITY)
        })
        .collect();
    let quaternions_wxyz = quats
        .iter()
        .map(|q| {
            let q = q.into_inner();
            [q.w, q.i, q.j, q.k]
        })
        .collect();
    let translations = trans.iter().map(|t| [t.x, t.y, t.z]).collect();
    ReconstructionGrowth {
        quaternions_wxyz,
        translations,
        posed: posed.to_vec(),
        points,
        focal,
        residual_norms,
    }
}

// ── grow_reconstruction ──────────────────────────────────────────────────────

/// Grow a seeded reconstruction to full registration by next-best-view
/// resection, incremental triangulation, and bounded staged adjustments. See
/// `specs/core/reconstruction-growth.md` and the module docs for the
/// mechanism.
///
/// `cluster_indexes` must be nondecreasing (each distinct cluster is a
/// contiguous run); `image_indexes` and `positions_xy` are the image id and
/// full-pixel keypoint position per observation. `quaternions_wxyz` /
/// `translations` / `posed_indexes` are parallel seed-pose arrays. All images
/// share `camera`; growth runs at its focal, and the finishing adjustment
/// releases the focal for SIMPLE_PINHOLE.
///
/// Deterministic: same inputs and options give identical output.
#[allow(clippy::too_many_arguments)]
pub fn grow_reconstruction(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    camera: &CameraIntrinsics,
    quaternions_wxyz: &[[f64; 4]],
    translations: &[[f64; 3]],
    posed_indexes: &[u32],
    options: &GrowOptions,
) -> ReconstructionGrowth {
    let n_obs = cluster_indexes.len();
    assert_eq!(
        image_indexes.len(),
        n_obs,
        "cluster_indexes and image_indexes length mismatch"
    );
    assert_eq!(
        positions_xy.len(),
        n_obs,
        "cluster_indexes and positions_xy length mismatch"
    );
    let n_seed = posed_indexes.len();
    assert_eq!(
        quaternions_wxyz.len(),
        n_seed,
        "quaternions_wxyz and posed_indexes length mismatch"
    );
    assert_eq!(
        translations.len(),
        n_seed,
        "translations and posed_indexes length mismatch"
    );
    assert!(
        cluster_indexes.windows(2).all(|w| w[0] <= w[1]),
        "cluster_indexes must be nondecreasing"
    );

    let n_img = image_indexes
        .iter()
        .chain(posed_indexes)
        .map(|&i| i as usize + 1)
        .max()
        .unwrap_or(0);
    let n_cl = cluster_indexes.last().map(|&c| c as usize + 1).unwrap_or(0);
    let f0 = camera.focal_lengths().0;

    // Seed state. Un-posed images carry the identity pose until accepted.
    let mut quats = vec![UnitQuaternion::identity(); n_img];
    let mut trans = vec![Vector3::zeros(); n_img];
    let mut posed = vec![false; n_img];
    for (k, &i) in posed_indexes.iter().enumerate() {
        let q = quaternions_wxyz[k];
        quats[i as usize] =
            UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]));
        trans[i as usize] =
            Vector3::new(translations[k][0], translations[k][1], translations[k][2]);
        posed[i as usize] = true;
    }
    // Registration order: seed images ascending, then acceptance order.
    let mut posed_order: Vec<usize> = (0..n_img).filter(|&i| posed[i]).collect();

    // Per-image observation rows, the covisibility, the adjustment-set
    // cluster mask, and the (fixed-focal) observation bearings.
    let mut image_obs: Vec<Vec<usize>> = vec![Vec::new(); n_img];
    for (k, &i) in image_indexes.iter().enumerate() {
        image_obs[i as usize].push(k);
    }
    let covis = build_covisibility(cluster_indexes, image_indexes, n_img, n_cl);
    let mut ba_mask = ba_cluster_mask(cluster_indexes, image_indexes, n_cl, options.ba_cluster_cap);
    let bearings: Vec<Vector3<f64>> = positions_xy
        .iter()
        .map(|p| {
            let d = camera.pixel_to_ray(p[0], p[1]);
            Vector3::new(d[0], d[1], d[2])
        })
        .collect();

    // Initial structure from the seed poses.
    let mut points = vec![[f64::NAN; 3]; n_cl];
    fill_new_points(
        camera,
        cluster_indexes,
        image_indexes,
        positions_xy,
        &posed,
        &quats,
        &trans,
        &mut points,
    );

    // ── Growth loop ──────────────────────────────────────────────────────
    let ba_every = (n_img / 10).clamp(3, 8);
    let mut ba_calls = 0usize;
    let mut since_ba = 0usize;
    let mut accepted_inl: Vec<f64> = Vec::new();
    let mut blocked: BTreeSet<usize> = BTreeSet::new();
    let mut force_tried: BTreeSet<usize> = BTreeSet::new();
    let mut ba_retry = true;
    let mut grown = 0usize;

    // One bounded growth adjustment + the full-set refill. Macro-free helper
    // closures fight the borrow checker here, so the adjustment is a local
    // function over explicit state.
    #[allow(clippy::too_many_arguments)]
    fn run_grow_ba(
        camera: &CameraIntrinsics,
        cluster_indexes: &[u32],
        image_indexes: &[u32],
        positions_xy: &[[f64; 2]],
        quats: &mut [UnitQuaternion<f64>],
        trans: &mut [Vector3<f64>],
        points: &mut [[f64; 3]],
        posed: &[bool],
        posed_order: &[usize],
        ba_mask: &[bool],
        covis: Option<&ClusterCovisibility>,
        options: &GrowOptions,
        ba_calls: &mut usize,
    ) {
        *ba_calls += 1;
        let n_img = posed.len();
        let n_posed = posed.iter().filter(|&&p| p).count();

        // Camera subset: the frontier window, or (every anchor_every-th
        // call) a covisibility-spread subset of all posed cameras.
        let mut win = vec![false; n_img];
        let mut windowed = false;
        if options.ba_window > 0 {
            let anchor = options.anchor_every > 0
                && ba_calls.is_multiple_of(options.anchor_every)
                && n_posed > options.ba_window;
            if let Some(cv) = if anchor { covis } else { None } {
                for &i in &cv.thin_to(n_posed.min(ANCHOR_CAP)) {
                    if posed[i as usize] {
                        win[i as usize] = true;
                        windowed = true;
                    }
                }
            } else if posed_order.len() > options.ba_window {
                for &i in &posed_order[posed_order.len() - options.ba_window..] {
                    win[i] = true;
                    windowed = true;
                }
            }
        }

        let mut obs_img: Vec<u32> = Vec::new();
        let mut obs_pt: Vec<u32> = Vec::new();
        let mut uv: Vec<[f64; 2]> = Vec::new();
        for k in 0..cluster_indexes.len() {
            let i = image_indexes[k] as usize;
            let c = cluster_indexes[k] as usize;
            if !posed[i] || !points[c][0].is_finite() || !ba_mask[k] {
                continue;
            }
            if windowed && !win[i] {
                continue;
            }
            obs_img.push(i as u32);
            obs_pt.push(c as u32);
            uv.push(positions_xy[k]);
        }
        bundle_adjust(
            camera,
            quats,
            trans,
            points,
            &uv,
            &obs_img,
            &obs_pt,
            None,
            false,
            &GROW_SCHEDULE,
            BA_MAX_ITERS,
            BA_MIN_TRACK,
            BA_MIN_OBS,
        );
        // The adjustment re-triangulates only the observations it was given,
        // wiping every other cluster's point — refill from the full
        // observation set at the updated poses, or the next-best-view count
        // sees only adjustment-set connectivity and growth stalls at its
        // boundary.
        fill_new_points(
            camera,
            cluster_indexes,
            image_indexes,
            positions_xy,
            posed,
            quats,
            trans,
            points,
        );
    }

    // Gathered 2D-3D candidates of one image (rows with a finite point).
    let gather = |i: usize, image_obs: &[Vec<usize>], points: &[[f64; 3]]| -> Gathered {
        let mut rows = Vec::new();
        let mut uv = Vec::new();
        let mut world = Vec::new();
        let mut brs = Vec::new();
        for &k in &image_obs[i] {
            let c = cluster_indexes[k] as usize;
            if points[c][0].is_finite() {
                rows.push(k);
                uv.push(positions_xy[k]);
                world.push(points[c]);
                brs.push(bearings[k]);
            }
        }
        (rows, uv, world, brs)
    };

    // Most-covisible posed neighbours' poses as fallback inits.
    let fallback_inits = |i: usize,
                          cap: usize,
                          posed: &[bool],
                          quats: &[UnitQuaternion<f64>],
                          trans: &[Vector3<f64>]|
     -> Vec<(UnitQuaternion<f64>, Vector3<f64>)> {
        let posed_idx: Vec<u32> = (0..n_img as u32).filter(|&j| posed[j as usize]).collect();
        let mut sel: Vec<u32> = match &covis {
            Some(c) => c
                .rank_by_covisibility(i as u32, &posed_idx)
                .into_iter()
                .take(cap)
                .collect(),
            None => Vec::new(),
        };
        if sel.is_empty() {
            sel = posed_idx.iter().take(1).copied().collect();
        }
        sel.iter()
            .map(|&j| (quats[j as usize], trans[j as usize]))
            .collect()
    };

    // All-observation inlier fraction of one image at the current state
    // (denominator: its observations of finite points).
    let image_inl = |i: usize,
                     quats: &[UnitQuaternion<f64>],
                     trans: &[Vector3<f64>],
                     points: &[[f64; 3]]|
     -> f64 {
        let mut n_tot = 0usize;
        let mut n_in = 0usize;
        for &k in &image_obs[i] {
            let c = cluster_indexes[k] as usize;
            if !points[c][0].is_finite() {
                continue;
            }
            n_tot += 1;
            if residual_norm(camera, &quats[i], &trans[i], &points[c], &positions_xy[k])
                .is_some_and(|rn| rn < INLIER_PX)
            {
                n_in += 1;
            }
        }
        if n_tot == 0 {
            0.0
        } else {
            n_in as f64 / n_tot as f64
        }
    };

    if n_seed > 0 {
        loop {
            // Next-best-view: most observations of currently-valid points.
            let mut cnt_all = vec![0usize; n_img];
            for (k, &i) in image_indexes.iter().enumerate() {
                let i = i as usize;
                if !posed[i] && points[cluster_indexes[k] as usize][0].is_finite() {
                    cnt_all[i] += 1;
                }
            }
            if cnt_all.iter().all(|&c| c == 0) {
                break;
            }
            let mut best_i = 0usize;
            let mut best_cnt = 0usize;
            for (i, &c) in cnt_all.iter().enumerate() {
                let c = if blocked.contains(&i) { 0 } else { c };
                if c > best_cnt {
                    best_cnt = c;
                    best_i = i;
                }
            }
            let i = best_i;

            if best_cnt < options.min_obs {
                // Every eligible image is deferred or too weak. One
                // adjustment + refill pass may repair the frontier;
                // afterwards the deferred images get a second chance.
                if !blocked.is_empty() && ba_retry {
                    ba_retry = false;
                    blocked.clear();
                    run_grow_ba(
                        camera,
                        cluster_indexes,
                        image_indexes,
                        positions_xy,
                        &mut quats,
                        &mut trans,
                        &mut points,
                        &posed,
                        &posed_order,
                        &ba_mask,
                        covis.as_ref(),
                        options,
                        &mut ba_calls,
                    );
                    since_ba = 0;
                    continue;
                }
                // Verified force-accept: pose the strongest deferred
                // candidate WITHOUT building points from it, adjust, and
                // keep it only if its inliers rose into the accepted band.
                let trial: Option<usize> = blocked
                    .iter()
                    .filter(|&&j| !force_tried.contains(&j) && cnt_all[j] >= options.min_obs)
                    .max_by_key(|&&j| (cnt_all[j], std::cmp::Reverse(j)))
                    .copied();
                let Some(j) = trial else {
                    break;
                };
                force_tried.insert(j);
                blocked.remove(&j);

                let (rows, uv_j, world_j, brs_j) = gather(j, &image_obs, &points);
                let inits = fallback_inits(j, GROW_FALLBACK_INITS, &posed, &quats, &trans);
                let Some(outcome) = resect_one(
                    camera,
                    &uv_j,
                    &world_j,
                    &brs_j,
                    P3P_MIN_CONSENSUS_GROW,
                    options.seed,
                    &inits,
                ) else {
                    continue;
                };

                // Snapshot for the restore-on-reject contract.
                let quats_saved = quats.clone();
                let trans_saved = trans.clone();
                let points_saved = points.clone();
                let ba_mask_saved = ba_mask.clone();
                let since_ba_saved = since_ba;

                quats[j] = outcome.rotation;
                trans[j] = outcome.translation;
                posed[j] = true;
                posed_order.push(j);
                // A P3P-registered image's observations are mostly wrong
                // matches; anchor it on its own verified evidence: the WHOLE
                // consensus clusters enter the adjustment set (all members'
                // observations, so the inter-round retriangulation keeps
                // their points), its non-consensus observations leave it.
                let consensus_rows: Option<Vec<usize>> = outcome
                    .consensus
                    .as_ref()
                    .map(|cons| cons.iter().map(|&r| rows[r]).collect());
                if let Some(cons_rows) = &consensus_rows {
                    let mut cons_cl = vec![false; n_cl];
                    for &k in cons_rows {
                        cons_cl[cluster_indexes[k] as usize] = true;
                    }
                    for (k, m) in ba_mask.iter_mut().enumerate() {
                        if cons_cl[cluster_indexes[k] as usize] {
                            *m = true;
                        }
                    }
                    for &k in &image_obs[j] {
                        ba_mask[k] = false;
                    }
                    for &k in cons_rows {
                        ba_mask[k] = true;
                    }
                }
                run_grow_ba(
                    camera,
                    cluster_indexes,
                    image_indexes,
                    positions_xy,
                    &mut quats,
                    &mut trans,
                    &mut points,
                    &posed,
                    &posed_order,
                    &ba_mask,
                    covis.as_ref(),
                    options,
                    &mut ba_calls,
                );
                since_ba = 0;

                let inl_after = image_inl(j, &quats, &trans, &points);
                let bar = if accepted_inl.is_empty() {
                    0.0
                } else {
                    options.accept_gate * median(&accepted_inl)
                };
                // Verification: the all-observation inlier bar, OR — for a
                // P3P-registered image whose observations are mostly wrong
                // matches — survival of the consensus set through the
                // adjustment (the registration claim is those observations).
                let surv = consensus_rows.as_ref().map(|cons_rows| {
                    let n_in = cons_rows
                        .iter()
                        .filter(|&&k| {
                            residual_norm(
                                camera,
                                &quats[j],
                                &trans[j],
                                &points[cluster_indexes[k] as usize],
                                &positions_xy[k],
                            )
                            .is_some_and(|rn| rn < INLIER_PX)
                        })
                        .count();
                    n_in as f64 / cons_rows.len().max(1) as f64
                });
                if inl_after >= bar || surv.is_some_and(|s| s >= CONSENSUS_SURVIVAL) {
                    accepted_inl.push(inl_after.max(bar));
                    grown += 1;
                    fill_new_points(
                        camera,
                        cluster_indexes,
                        image_indexes,
                        positions_xy,
                        &posed,
                        &quats,
                        &trans,
                        &mut points,
                    );
                    ba_retry = true;
                    blocked.clear();
                } else {
                    // Rejected: restore the prior poses, structure, and
                    // adjustment set; the image stays un-posed for good.
                    posed[j] = false;
                    if posed_order.last() == Some(&j) {
                        posed_order.pop();
                    }
                    quats = quats_saved;
                    trans = trans_saved;
                    points = points_saved;
                    ba_mask = ba_mask_saved;
                    since_ba = since_ba_saved;
                }
                continue;
            }

            // Normal candidate: estimate-then-refine, then the acceptance
            // gate against the accepted-so-far median.
            let (_rows, uv_i, world_i, brs_i) = gather(i, &image_obs, &points);
            let inits = fallback_inits(i, GROW_FALLBACK_INITS, &posed, &quats, &trans);
            let Some(outcome) = resect_one(
                camera,
                &uv_i,
                &world_i,
                &brs_i,
                P3P_MIN_CONSENSUS_GROW,
                options.seed,
                &inits,
            ) else {
                blocked.insert(i);
                continue;
            };
            if !accepted_inl.is_empty()
                && outcome.inlier_fraction < options.accept_gate * median(&accepted_inl)
            {
                // Deferred, not rejected: another chance after the frontier
                // improves.
                blocked.insert(i);
                continue;
            }
            accepted_inl.push(outcome.inlier_fraction);
            quats[i] = outcome.rotation;
            trans[i] = outcome.translation;
            posed[i] = true;
            posed_order.push(i);
            ba_retry = true;
            grown += 1;
            fill_new_points(
                camera,
                cluster_indexes,
                image_indexes,
                positions_xy,
                &posed,
                &quats,
                &trans,
                &mut points,
            );
            since_ba += 1;
            if since_ba >= ba_every {
                since_ba = 0;
                run_grow_ba(
                    camera,
                    cluster_indexes,
                    image_indexes,
                    positions_xy,
                    &mut quats,
                    &mut trans,
                    &mut points,
                    &posed,
                    &posed_order,
                    &ba_mask,
                    covis.as_ref(),
                    options,
                    &mut ba_calls,
                );
            }
        }
    }

    // ── Finishing: release the focal on a covisibility-spread subset ────
    if grown == 0 {
        // Degenerate inputs / empty growth: the input state passes through.
        return build_result(
            camera,
            cluster_indexes,
            image_indexes,
            positions_xy,
            &quats,
            &trans,
            &posed,
            points,
            f0,
        );
    }

    let n_posed = posed.iter().filter(|&&p| p).count();
    let sub: Vec<bool> = match &covis {
        Some(cv) if n_posed > FINISH_CAP => {
            let mut s = vec![false; n_img];
            for &i in &cv.thin_to(FINISH_CAP) {
                if posed[i as usize] {
                    s[i as usize] = true;
                }
            }
            s
        }
        _ => posed.clone(),
    };
    let mut obs_img: Vec<u32> = Vec::new();
    let mut obs_pt: Vec<u32> = Vec::new();
    let mut uv: Vec<[f64; 2]> = Vec::new();
    for k in 0..n_obs {
        let i = image_indexes[k] as usize;
        let c = cluster_indexes[k] as usize;
        if sub[i] && points[c][0].is_finite() && ba_mask[k] {
            obs_img.push(i as u32);
            obs_pt.push(c as u32);
            uv.push(positions_xy[k]);
        }
    }
    let ba = bundle_adjust(
        camera,
        &mut quats,
        &mut trans,
        &mut points,
        &uv,
        &obs_img,
        &obs_pt,
        None,
        true,
        &DEFAULT_SCHEDULE,
        BA_MAX_ITERS,
        BA_MIN_TRACK,
        BA_MIN_OBS,
    );
    let focal = ba.focal;
    let cam_final = cam_with_focal(camera, focal);
    // Re-triangulation at the released focal (the finishing adjustment wiped
    // every cluster outside its observation set).
    fill_new_points(
        &cam_final,
        cluster_indexes,
        image_indexes,
        positions_xy,
        &posed,
        &quats,
        &trans,
        &mut points,
    );

    build_result(
        &cam_final,
        cluster_indexes,
        image_indexes,
        positions_xy,
        &quats,
        &trans,
        &posed,
        points,
        focal,
    )
}

#[cfg(test)]
mod tests;
