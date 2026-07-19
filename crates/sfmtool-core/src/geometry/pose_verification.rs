// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Displacement-neighborhood pose verification and repair
//! ([`verify_poses`] / [`repair_poses`]). See
//! `specs/core/pose-verification.md`.
//!
//! Detects and repairs misregistered cameras in a reconstruction without a
//! reference solve, an image ordering, or a motion model. The ruler is the
//! [`DisplacementNeighborhood`] substrate — which images are near-duplicate
//! viewpoints of which, measured by keypoint displacement, computed once from
//! the 2D cluster tracks — and the screens hold the current poses against it:
//!
//! - **Screen A (self-resection):** re-resect every registered camera's own
//!   observations against the shared structure with
//!   [`crate::geometry::batch_resection::resect_images_batch`]; a camera whose
//!   pose cannot be re-derived from its own 2D-3D support is flagged.
//! - **Screen B (measured-versus-posed relative rotation):** for each
//!   registered camera and each of its `nearest` low-displacement neighbours
//!   (the low-parallax regime, where the conjugate-homography model holds),
//!   estimate the homography over the pair's shared-cluster correspondences,
//!   extract `R = K⁻¹HK` (orthonormalized, conjugated to the canonical frame
//!   by `S = diag(1, −1, −1)`), and compare with the pose-implied relative
//!   rotation. The per-image score is the **median** angular discrepancy over
//!   its neighbours; flag at or above a threshold.
//!
//! Two properties are load-bearing: the comparison is restricted to
//! low-displacement neighbours (at wider baselines the displacement carries
//! parallax and the small-rotation model misattributes it) and the
//! aggregation is a per-image median (a single discrepant pair is noise or
//! parallax; a misregistered camera is implicated consistently by every
//! neighbour that overlaps it).
//!
//! **Repair:** a flagged camera is re-initialized from its top-2 `nearest`
//! registered neighbours — chordal mean of their rotations, mean of their
//! centres — then trimmed pose-only refinement against the current structure.
//! A repair is accepted only when the all-observation inlier fraction reaches
//! `max(floor, before + margin)`: an "improvement" below the absolute floor
//! means the camera's neighbourhood structure is itself broken, which
//! pose-only repair cannot fix. Rejected repairs leave the pose untouched and
//! the flag standing.
//!
//! Everything runs in the canonical camera frame (the camera looks along
//! `−Z`); poses are world-to-camera `x_cam = R·X + t`. Both screens are
//! read-only on the observation data, images are independent and run in
//! parallel, and every random draw derives deterministically from the input
//! seed — identical inputs reproduce identical output bit for bit.

use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};
use rayon::prelude::*;

use crate::features::cluster_match::covisibility::DisplacementNeighborhood;
use crate::geometry::batch_resection::{resect_images_batch, ResectOptions};
use crate::geometry::homography_estimation::{estimate_homography, HomographyOptions};
use crate::geometry::pose_refine::refine_absolute_pose;
use crate::geometry::reconstruction_growth::per_image_seed;
use crate::CameraIntrinsics;

// ── Tuning (see the spec) ────────────────────────────────────────────────────

/// Final-inlier pixel bound shared by the screens and the repair acceptance
/// (matches the growth kernel's [`INLIER_PX`]).
const INLIER_PX: f64 = 3.0;
/// Trim rounds for the repair's pose-only refinement.
const REFINE_TRIM_ROUNDS: usize = 5;
/// Fraction of observations retained per trim round.
const REFINE_KEEP_FRACTION: f64 = 0.6;
/// Registered neighbours a repair blends its initial pose from.
const REPAIR_INIT_NEIGHBORS: usize = 2;

/// Options for [`verify_poses`].
#[derive(Clone, Debug)]
pub struct VerifyOptions {
    /// Screen A: skip a camera with fewer observations of valid points than
    /// this (it cannot be re-derived and is flagged).
    pub resect_min_obs: usize,
    /// Screen A: a re-resection at or above this all-observation inlier
    /// fraction clears the screen.
    pub resect_accept_gate: f64,
    /// Screen B: lowest-displacement registered neighbours examined per
    /// camera.
    pub max_neighbors: usize,
    /// Substrate shared-cluster floor for a pair to count as a neighbour.
    pub min_shared: u32,
    /// Screen B: a pair below this many shared-cluster correspondences is
    /// skipped.
    pub min_pair_correspondences: usize,
    /// Screen B: a homography supported by fewer inliers than this is
    /// skipped.
    pub min_h_inliers: usize,
    /// Screen B: a camera needs at least this many neighbour measurements to
    /// be scored (below it the screen abstains).
    pub min_rotation_measurements: usize,
    /// Screen B: flag at or above this median angular discrepancy (degrees).
    pub rotation_threshold_deg: f64,
    /// Base seed for the per-image resection RANSAC and the per-pair
    /// homography RANSAC (each draw is a pure function of the seed and the
    /// image indexes involved).
    pub seed: u64,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        Self {
            resect_min_obs: 8,
            resect_accept_gate: 0.30,
            max_neighbors: 4,
            min_shared: 50,
            min_pair_correspondences: 30,
            min_h_inliers: 20,
            min_rotation_measurements: 2,
            rotation_threshold_deg: 3.0,
            seed: 0,
        }
    }
}

/// Result of [`verify_poses`]; every array is aligned with the input
/// `posed_indexes`.
#[derive(Clone, Debug)]
pub struct PoseVerification {
    /// Screen A: the camera failed self-resection.
    pub resect_flags: Vec<bool>,
    /// Screen A score: the re-resection's all-observation inlier fraction.
    pub resect_inlier_fractions: Vec<f64>,
    /// Screen B: the camera's median measured-versus-posed relative-rotation
    /// discrepancy is at or above the threshold.
    pub rotation_flags: Vec<bool>,
    /// Screen B score: the median discrepancy (degrees); `NaN` when the
    /// screen abstained (fewer than `min_rotation_measurements` usable
    /// neighbours).
    pub rotation_scores_deg: Vec<f64>,
    /// Union of both screens.
    pub flagged: Vec<bool>,
}

/// Options for [`repair_poses`].
#[derive(Clone, Debug)]
pub struct RepairOptions {
    /// The verification pass that selects the repair candidates.
    pub verify: VerifyOptions,
    /// Skip a flagged camera with fewer observations of valid points than
    /// this (the flag stands).
    pub min_obs: usize,
    /// Absolute inlier-fraction floor a repair must reach.
    pub inlier_floor: f64,
    /// Improvement margin over the pre-repair inlier fraction a repair must
    /// reach.
    pub inlier_margin: f64,
}

impl Default for RepairOptions {
    fn default() -> Self {
        Self {
            verify: VerifyOptions::default(),
            min_obs: 12,
            inlier_floor: 0.10,
            inlier_margin: 0.05,
        }
    }
}

/// Result of [`repair_poses`]; every array is aligned with the input
/// `posed_indexes`.
#[derive(Clone, Debug)]
pub struct PoseRepair {
    /// The verification that selected the repair candidates.
    pub verification: PoseVerification,
    /// Updated world-to-camera rotations (WXYZ); unrepaired cameras carry
    /// their input pose.
    pub quaternions_wxyz: Vec<[f64; 4]>,
    /// Updated world-to-camera translations.
    pub translations: Vec<[f64; 3]>,
    /// Whether the camera's repair was accepted.
    pub repaired: Vec<bool>,
    /// All-observation inlier fraction before the repair attempt; `NaN`
    /// where no attempt ran (unflagged, too few neighbours, too few
    /// observations).
    pub inlier_before: Vec<f64>,
    /// All-observation inlier fraction the attempted repair reached; `NaN`
    /// where no attempt ran.
    pub inlier_after: Vec<f64>,
}

// ── Small numeric helpers ────────────────────────────────────────────────────

/// Nearest rotation to `m` (polar factor `U Vᵀ` of the SVD). A negative
/// determinant flips the sign of the whole matrix — a homography is defined
/// up to scale *including sign*, so `M ≈ −R` must come back as `R` (mirrors
/// `rotation_init`). `None` for a non-finite or degenerate input.
fn polar_rotation(m: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = m.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let p = u * v_t;
    if !p.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(if p.determinant() < 0.0 { -p } else { p })
}

/// Rotation angle of `r` in radians.
fn rotation_angle(r: &Matrix3<f64>) -> f64 {
    (((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)).acos()
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

/// Unit quaternion from a WXYZ row.
fn quat_of(q: &[f64; 4]) -> UnitQuaternion<f64> {
    UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]))
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
        .filter(|(o, x)| {
            let c = r * Vector3::new(x[0], x[1], x[2]) + t;
            cam.ray_to_pixel([c.x, c.y, c.z])
                .is_some_and(|(u, v)| (u - o[0]).hypot(v - o[1]) < INLIER_PX)
        })
        .count();
    n_in as f64 / uv.len() as f64
}

// ── Shared assembly ──────────────────────────────────────────────────────────

/// Per-image `(cluster id, observation row)` lists, sorted by cluster id with
/// duplicate cluster observations deduplicated (last wins).
fn image_cluster_rows(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    n_img: usize,
) -> Vec<Vec<(u32, usize)>> {
    let mut lists: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n_img];
    // Cluster ids are nondecreasing, so per-image lists build sorted.
    for (k, (&c, &i)) in cluster_indexes.iter().zip(image_indexes).enumerate() {
        let list = &mut lists[i as usize];
        match list.last_mut() {
            Some(last) if last.0 == c => last.1 = k,
            _ => list.push((c, k)),
        }
    }
    lists
}

/// Merge-join of two images' cluster lists: the shared-cluster observation
/// rows `(row in a, row in b)`.
fn shared_rows(lists: &[Vec<(u32, usize)>], a: usize, b: usize) -> Vec<(usize, usize)> {
    let (la, lb) = (&lists[a], &lists[b]);
    let mut out = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < la.len() && j < lb.len() {
        match la[i].0.cmp(&lb[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push((la[i].1, lb[j].1));
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Registered `nearest` neighbours of `subject`: partners at or above the
/// shared-cluster floor with a registered pose, sorted by ascending mean
/// displacement (ties: ascending index), truncated to `k`.
fn nearest_registered(
    neighborhood: &DisplacementNeighborhood,
    subject: u32,
    posed: &[bool],
    k: usize,
    min_shared: u32,
) -> Vec<u32> {
    let mut ranked: Vec<(f64, u32)> = neighborhood
        .neighbors(subject)
        .filter(|&(j, shared, _)| shared >= min_shared && j != subject && posed[j as usize])
        .map(|(j, _, d)| (d, j))
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    ranked.truncate(k);
    ranked.into_iter().map(|(_, j)| j).collect()
}

/// Validated shared shape of the two kernels' inputs. Returns `n_img`.
#[allow(clippy::too_many_arguments)]
fn validate_inputs(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    points: &[[f64; 3]],
    quaternions_wxyz: &[[f64; 4]],
    translations: &[[f64; 3]],
    posed_indexes: &[u32],
    neighborhood: &DisplacementNeighborhood,
) -> usize {
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
    assert!(
        cluster_indexes.windows(2).all(|w| w[0] <= w[1]),
        "cluster_indexes must be nondecreasing"
    );
    if let Some(&max_c) = cluster_indexes.iter().max() {
        assert!(
            (max_c as usize) < points.len(),
            "cluster index {max_c} out of range for {} points",
            points.len()
        );
    }
    let n_posed = posed_indexes.len();
    assert_eq!(
        quaternions_wxyz.len(),
        n_posed,
        "quaternions_wxyz and posed_indexes length mismatch"
    );
    assert_eq!(
        translations.len(),
        n_posed,
        "translations and posed_indexes length mismatch"
    );
    let n_img = image_indexes
        .iter()
        .chain(posed_indexes)
        .map(|&i| i as usize + 1)
        .max()
        .unwrap_or(0);
    assert!(
        neighborhood.num_images() >= n_img,
        "neighborhood covers {} images but the observations reference {n_img}",
        neighborhood.num_images()
    );
    n_img
}

// ── Screen B ─────────────────────────────────────────────────────────────────

/// Measured relative rotation `b → a` in the canonical frame via the
/// conjugate homography over the pair's shared-cluster correspondences, or
/// `None` when the pair has too few correspondences, the homography fails, or
/// its consensus is below the inlier floor.
#[allow(clippy::too_many_arguments)]
fn measured_relative_rotation(
    lists: &[Vec<(u32, usize)>],
    positions_xy: &[[f64; 2]],
    k_m: &Matrix3<f64>,
    k_inv: &Matrix3<f64>,
    a: u32,
    b: u32,
    options: &VerifyOptions,
) -> Option<Matrix3<f64>> {
    let shared = shared_rows(lists, a as usize, b as usize);
    if shared.len() < options.min_pair_correspondences {
        return None;
    }
    let xa: Vec<[f64; 2]> = shared.iter().map(|&(ra, _)| positions_xy[ra]).collect();
    let xb: Vec<[f64; 2]> = shared.iter().map(|&(_, rb)| positions_xy[rb]).collect();
    let h_opts = HomographyOptions {
        min_inliers: options.min_h_inliers,
        seed: per_image_seed(per_image_seed(options.seed, a), b),
        ..Default::default()
    };
    // H maps b's pixels onto a's, so K⁻¹ H K is the optical-frame rotation
    // b → a; conjugation by S = diag(1, −1, −1) moves it to the canonical
    // frame.
    let hest = estimate_homography(&xb, &xa, &h_opts)?;
    let m = k_inv * hest.h_matrix * k_m;
    let r_opt = polar_rotation(&m)?;
    let s = Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0));
    Some(s * r_opt * s)
}

// ── The kernels ──────────────────────────────────────────────────────────────

/// Verify the registered cameras' poses against the displacement
/// neighborhood. See the module docs and `specs/core/pose-verification.md`.
///
/// `cluster_indexes` must be nondecreasing (each distinct cluster is a
/// contiguous run); `image_indexes` and `positions_xy` are the image id and
/// full-pixel keypoint position per observation; `points` is the current
/// structure indexed by cluster id (`NaN` rows invalid);
/// `quaternions_wxyz` / `translations` / `posed_indexes` are the registered
/// world-to-camera poses; `neighborhood` is the substrate over the same image
/// index space (build it with
/// [`DisplacementNeighborhood::from_clusters`] or reload it with
/// [`DisplacementNeighborhood::from_arrays`]).
///
/// Both screens are read-only; cameras are independent and run in parallel;
/// identical inputs and seed reproduce identical output bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn verify_poses(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    camera: &CameraIntrinsics,
    points: &[[f64; 3]],
    quaternions_wxyz: &[[f64; 4]],
    translations: &[[f64; 3]],
    posed_indexes: &[u32],
    neighborhood: &DisplacementNeighborhood,
    options: &VerifyOptions,
) -> PoseVerification {
    let n_img = validate_inputs(
        cluster_indexes,
        image_indexes,
        positions_xy,
        points,
        quaternions_wxyz,
        translations,
        posed_indexes,
        neighborhood,
    );

    // Screen A: batch self-resection of every registered camera against the
    // shared structure, its own current pose available as a fallback init.
    let resect = resect_images_batch(
        cluster_indexes,
        image_indexes,
        positions_xy,
        camera,
        points,
        posed_indexes,
        quaternions_wxyz,
        translations,
        posed_indexes,
        &ResectOptions {
            min_obs: options.resect_min_obs,
            accept_gate: options.resect_accept_gate,
            seed: options.seed,
        },
    );
    let resect_flags: Vec<bool> = resect.accepted.iter().map(|&a| !a).collect();

    // Screen B: measured-versus-posed relative rotation over each camera's
    // lowest-displacement registered neighbours.
    let mut posed = vec![false; n_img];
    let mut rot_of_img = vec![Matrix3::<f64>::identity(); n_img];
    for (k, &i) in posed_indexes.iter().enumerate() {
        posed[i as usize] = true;
        rot_of_img[i as usize] = quat_of(&quaternions_wxyz[k])
            .to_rotation_matrix()
            .into_inner();
    }
    let lists = image_cluster_rows(cluster_indexes, image_indexes, n_img);
    let (fx, fy) = camera.focal_lengths();
    let (cx, cy) = camera.principal_point();
    let k_m = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    let k_inv = Matrix3::new(
        1.0 / fx,
        0.0,
        -cx / fx,
        0.0,
        1.0 / fy,
        -cy / fy,
        0.0,
        0.0,
        1.0,
    );

    let scored: Vec<(f64, bool)> = posed_indexes
        .par_iter()
        .map(|&i| {
            let mut errs: Vec<f64> = Vec::new();
            for j in nearest_registered(
                neighborhood,
                i,
                &posed,
                options.max_neighbors,
                options.min_shared,
            ) {
                let Some(r_meas) =
                    measured_relative_rotation(&lists, positions_xy, &k_m, &k_inv, i, j, options)
                else {
                    continue;
                };
                let r_pose = rot_of_img[i as usize] * rot_of_img[j as usize].transpose();
                errs.push(rotation_angle(&(r_meas * r_pose.transpose())).to_degrees());
            }
            if errs.len() >= options.min_rotation_measurements {
                let score = median(&errs);
                (score, score >= options.rotation_threshold_deg)
            } else {
                (f64::NAN, false)
            }
        })
        .collect();
    let rotation_scores_deg: Vec<f64> = scored.iter().map(|&(s, _)| s).collect();
    let rotation_flags: Vec<bool> = scored.iter().map(|&(_, f)| f).collect();

    let flagged = resect_flags
        .iter()
        .zip(&rotation_flags)
        .map(|(&a, &b)| a || b)
        .collect();
    PoseVerification {
        resect_flags,
        resect_inlier_fractions: resect.inlier_fractions,
        rotation_flags,
        rotation_scores_deg,
        flagged,
    }
}

/// Verify, then repair the flagged cameras against the displacement
/// neighborhood. See the module docs and `specs/core/pose-verification.md`.
///
/// Runs [`verify_poses`], then walks the flagged cameras in ascending image
/// order: re-initialize from the top-2 `nearest` registered neighbours
/// (chordal mean of their rotations, mean of their centres, `t = −R·c`),
/// trimmed pose-only refinement against the current structure, and accept
/// only when the all-observation inlier fraction reaches
/// `max(inlier_floor, before + inlier_margin)`. Accepted repairs update the
/// working poses (later repairs see them); rejected repairs leave the pose
/// untouched and the flag standing. Deterministic for identical inputs and
/// seed.
#[allow(clippy::too_many_arguments)]
pub fn repair_poses(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    camera: &CameraIntrinsics,
    points: &[[f64; 3]],
    quaternions_wxyz: &[[f64; 4]],
    translations: &[[f64; 3]],
    posed_indexes: &[u32],
    neighborhood: &DisplacementNeighborhood,
    options: &RepairOptions,
) -> PoseRepair {
    let verification = verify_poses(
        cluster_indexes,
        image_indexes,
        positions_xy,
        camera,
        points,
        quaternions_wxyz,
        translations,
        posed_indexes,
        neighborhood,
        &options.verify,
    );

    let n_posed = posed_indexes.len();
    let n_img = image_indexes
        .iter()
        .chain(posed_indexes)
        .map(|&i| i as usize + 1)
        .max()
        .unwrap_or(0);

    // Working per-image pose state (accepted repairs feed later inits).
    let mut posed = vec![false; n_img];
    let mut quats = vec![UnitQuaternion::identity(); n_img];
    let mut trans = vec![Vector3::zeros(); n_img];
    let mut slot_of_img = vec![usize::MAX; n_img];
    for (k, &i) in posed_indexes.iter().enumerate() {
        posed[i as usize] = true;
        quats[i as usize] = quat_of(&quaternions_wxyz[k]);
        trans[i as usize] =
            Vector3::new(translations[k][0], translations[k][1], translations[k][2]);
        slot_of_img[i as usize] = k;
    }

    // Per-image observation rows of valid points.
    let mut image_obs: Vec<Vec<usize>> = vec![Vec::new(); n_img];
    for (k, &i) in image_indexes.iter().enumerate() {
        if points[cluster_indexes[k] as usize][0].is_finite() {
            image_obs[i as usize].push(k);
        }
    }

    let mut repaired = vec![false; n_posed];
    let mut inlier_before = vec![f64::NAN; n_posed];
    let mut inlier_after = vec![f64::NAN; n_posed];
    // Outputs start as the input rows so unrepaired cameras pass through
    // bit for bit (the working arrays above re-normalize quaternions).
    let mut quaternions_out: Vec<[f64; 4]> = quaternions_wxyz.to_vec();
    let mut translations_out: Vec<[f64; 3]> = translations.to_vec();

    // Ascending image order for a deterministic repair sequence.
    let mut flagged_images: Vec<u32> = posed_indexes
        .iter()
        .enumerate()
        .filter(|&(k, _)| verification.flagged[k])
        .map(|(_, &i)| i)
        .collect();
    flagged_images.sort_unstable();

    for &i in &flagged_images {
        let iu = i as usize;
        let near = nearest_registered(
            neighborhood,
            i,
            &posed,
            REPAIR_INIT_NEIGHBORS,
            options.verify.min_shared,
        );
        if near.len() < REPAIR_INIT_NEIGHBORS {
            continue;
        }
        // Chordal mean of the neighbours' rotations, mean of their centres.
        let mut sum = Matrix3::<f64>::zeros();
        let mut c_mean = Vector3::<f64>::zeros();
        for &j in &near {
            let ju = j as usize;
            sum += quats[ju].to_rotation_matrix().into_inner();
            c_mean += -(quats[ju].inverse() * trans[ju]);
        }
        c_mean /= near.len() as f64;
        let Some(r0) = polar_rotation(&sum) else {
            continue;
        };
        let t0 = -(r0 * c_mean);

        let rows = &image_obs[iu];
        if rows.len() < options.min_obs {
            continue;
        }
        let uv: Vec<[f64; 2]> = rows.iter().map(|&k| positions_xy[k]).collect();
        let world: Vec<[f64; 3]> = rows
            .iter()
            .map(|&k| points[cluster_indexes[k] as usize])
            .collect();

        let slot = slot_of_img[iu];
        let inl0 = inlier_fraction_of(camera, &quats[iu], &trans[iu], &world, &uv);
        let refined = refine_absolute_pose(
            camera,
            &uv,
            &world,
            &UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(r0)),
            &t0,
            REFINE_TRIM_ROUNDS,
            REFINE_KEEP_FRACTION,
            INLIER_PX,
        );
        let inl1 = refined.inlier_fraction;
        inlier_before[slot] = inl0;
        inlier_after[slot] = inl1;
        // Absolute floor + real margin: a 0% → 0% "improvement" is float
        // noise, and a repair that cannot reach minimal support means the
        // segment's structure is broken (needs re-pose + re-triangulation,
        // not pose-only repair).
        if inl1 >= (inl0 + options.inlier_margin).max(options.inlier_floor) {
            quats[iu] = refined.rotation;
            trans[iu] = refined.translation;
            let q = refined.rotation.into_inner();
            quaternions_out[slot] = [q.w, q.i, q.j, q.k];
            translations_out[slot] = [
                refined.translation.x,
                refined.translation.y,
                refined.translation.z,
            ];
            repaired[slot] = true;
        }
    }

    PoseRepair {
        verification,
        quaternions_wxyz: quaternions_out,
        translations: translations_out,
        repaired,
        inlier_before,
        inlier_after,
    }
}

#[cfg(test)]
mod tests;
