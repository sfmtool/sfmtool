// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for displacement-neighborhood pose verification and repair
//! (``sfmtool._sfmtool.geometry.verify_poses`` / ``repair_poses``; see
//! ``specs/core/pose-verification.md``).
//!
//! The substrate travels as its compact serialization — the parallel per-pair
//! arrays ``pair_i`` / ``pair_j`` / ``pair_count`` / ``pair_mean_disp``
//! produced by ``ClusterCovisibility.neighborhood_arrays()`` (or persisted by
//! an earlier stage) — so one computation serves a multi-stage pipeline.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::borrow::Cow;

use sfmtool_core::features::cluster_match::covisibility::DisplacementNeighborhood;
use sfmtool_core::geometry::pose_verification::{
    repair_poses as core_repair, verify_poses as core_verify, PoseVerification, RepairOptions,
    VerifyOptions,
};

use crate::geometry::reconstruction_growth::{read_observations, read_rows};
use crate::geometry::PyCameraIntrinsics;

/// Everything both kernels share, validated: observations, points, poses,
/// and the reloaded substrate.
struct VerifyInputs {
    clusters: Vec<u32>,
    images: Vec<u32>,
    positions: Vec<[f64; 2]>,
    points: Vec<[f64; 3]>,
    quats: Vec<[f64; 4]>,
    trans: Vec<[f64; 3]>,
    posed_idx: Vec<u32>,
    neighborhood: DisplacementNeighborhood,
}

#[allow(clippy::too_many_arguments)]
fn read_verify_inputs(
    cluster_indexes: &PyReadonlyArray1<'_, u32>,
    image_indexes: &PyReadonlyArray1<'_, u32>,
    positions_xy: &PyReadonlyArray2<'_, f64>,
    points: &PyReadonlyArray2<'_, f64>,
    quaternions_wxyz: &PyReadonlyArray2<'_, f64>,
    translations: &PyReadonlyArray2<'_, f64>,
    posed_indexes: &PyReadonlyArray1<'_, u32>,
    pair_i: &PyReadonlyArray1<'_, u32>,
    pair_j: &PyReadonlyArray1<'_, u32>,
    pair_count: &PyReadonlyArray1<'_, u32>,
    pair_mean_disp: &PyReadonlyArray1<'_, f64>,
) -> PyResult<VerifyInputs> {
    let (clusters, images, positions) =
        read_observations(cluster_indexes, image_indexes, positions_xy)?;
    let points = read_rows::<3>(points, "points")?;
    if let Some(&max_c) = clusters.iter().max() {
        if max_c as usize >= points.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cluster index {max_c} out of range for {} points",
                points.len()
            )));
        }
    }
    let quats = read_rows::<4>(quaternions_wxyz, "quaternions_wxyz")?;
    let trans = read_rows::<3>(translations, "translations")?;
    let posed_idx = to_contiguous!(posed_indexes).into_owned();
    if quats.len() != posed_idx.len() || trans.len() != posed_idx.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "quaternions_wxyz, translations, and posed_indexes must share n_posed",
        ));
    }

    let pi: Cow<'_, [u32]> = to_contiguous!(pair_i);
    let pj: Cow<'_, [u32]> = to_contiguous!(pair_j);
    let pc: Cow<'_, [u32]> = to_contiguous!(pair_count);
    let pd: Cow<'_, [f64]> = to_contiguous!(pair_mean_disp);
    let n_img = images
        .iter()
        .chain(posed_idx.iter())
        .chain(pi.iter())
        .chain(pj.iter())
        .map(|&i| i as usize + 1)
        .max()
        .unwrap_or(0);
    let neighborhood = DisplacementNeighborhood::from_arrays(&pi, &pj, &pc, &pd, n_img)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(VerifyInputs {
        clusters,
        images,
        positions,
        points,
        quats,
        trans,
        posed_idx,
        neighborhood,
    })
}

/// Pack a [`PoseVerification`] into the shared result dict.
fn verification_dict<'py>(py: Python<'py>, out: &PoseVerification) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("resect_flags", PyArray1::from_slice(py, &out.resect_flags))?;
    d.set_item(
        "resect_inlier_fractions",
        PyArray1::from_slice(py, &out.resect_inlier_fractions),
    )?;
    d.set_item(
        "rotation_flags",
        PyArray1::from_slice(py, &out.rotation_flags),
    )?;
    d.set_item(
        "rotation_scores_deg",
        PyArray1::from_slice(py, &out.rotation_scores_deg),
    )?;
    d.set_item("flagged", PyArray1::from_slice(py, &out.flagged))?;
    Ok(d)
}

/// Verify the registered cameras' poses against the displacement
/// neighborhood (see ``specs/core/pose-verification.md``).
///
/// Screen A re-resects every registered camera's own observations against
/// the shared structure (``resect_images_batch``); a camera whose pose
/// cannot be re-derived — no acceptable consensus — is flagged. Screen B
/// compares, per camera, the relative rotation measured from its
/// lowest-displacement registered neighbours' shared-cluster correspondences
/// (conjugate homography, ``R = K^-1 H K`` orthonormalized and conjugated to
/// the canonical frame) with the pose-implied one; the per-image score is
/// the median discrepancy over its neighbours, flagged at or above the
/// threshold. Only low-displacement neighbours are measured (parallax breaks
/// the homography model) and the median makes a single discrepant pair
/// harmless. Read-only; cameras run in parallel; identical inputs and seed
/// reproduce identical output bit for bit.
///
/// Args:
///     cluster_indexes: (n_obs,) uint32 cluster id per observation,
///         nondecreasing.
///     image_indexes: (n_obs,) uint32 image id per observation.
///     positions_xy: (n_obs, 2) float64 full-pixel keypoint positions.
///     camera: Shared ``CameraIntrinsics``.
///     points: (n_clusters, 3) float64 world points (NaN rows are invalid).
///     quaternions_wxyz: (n_posed, 4) float64 registered world-to-camera
///         rotations.
///     translations: (n_posed, 3) float64 registered translations.
///     posed_indexes: (n_posed,) uint32 image ids of the registered poses.
///     pair_i: (n_pairs,) uint32 substrate pair first image.
///     pair_j: (n_pairs,) uint32 substrate pair second image.
///     pair_count: (n_pairs,) uint32 shared-cluster count per pair.
///     pair_mean_disp: (n_pairs,) float64 mean keypoint displacement per
///         pair (pixels). The four pair arrays are the substrate's compact
///         serialization (``ClusterCovisibility.neighborhood_arrays()``).
///     resect_min_obs: Screen A skips (and flags) a camera with fewer
///         observations of valid points than this (default 8).
///     resect_accept_gate: Screen A clears a camera at or above this
///         all-observation inlier fraction (default 0.30).
///     max_neighbors: Screen B examines at most this many nearest
///         registered neighbours per camera (default 4).
///     min_shared: Substrate shared-cluster floor for a pair to count as a
///         neighbour (default 50).
///     min_pair_correspondences: Screen B skips a pair below this many
///         shared-cluster correspondences (default 30).
///     min_h_inliers: Screen B skips a homography supported by fewer
///         inliers than this (default 20).
///     min_rotation_measurements: Screen B abstains for a camera with fewer
///         usable neighbour measurements than this (default 2).
///     rotation_threshold_deg: Screen B flags at or above this median
///         discrepancy in degrees (default 3.0).
///     seed: Base RANSAC seed (default 0).
///
/// Returns:
///     A dict aligned with ``posed_indexes``: ``{"resect_flags" (n_posed,)
///     bool, "resect_inlier_fractions" (n_posed,), "rotation_flags"
///     (n_posed,) bool, "rotation_scores_deg" (n_posed,) with NaN where
///     screen B abstained, "flagged" (n_posed,) bool}`` (``flagged`` is the
///     union of both screens).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (cluster_indexes, image_indexes, positions_xy, camera, points, quaternions_wxyz, translations, posed_indexes, pair_i, pair_j, pair_count, pair_mean_disp, *, resect_min_obs=8, resect_accept_gate=0.30, max_neighbors=4, min_shared=50, min_pair_correspondences=30, min_h_inliers=20, min_rotation_measurements=2, rotation_threshold_deg=3.0, seed=0))]
pub fn verify_poses<'py>(
    py: Python<'py>,
    cluster_indexes: PyReadonlyArray1<'py, u32>,
    image_indexes: PyReadonlyArray1<'py, u32>,
    positions_xy: PyReadonlyArray2<'py, f64>,
    camera: PyRef<'py, PyCameraIntrinsics>,
    points: PyReadonlyArray2<'py, f64>,
    quaternions_wxyz: PyReadonlyArray2<'py, f64>,
    translations: PyReadonlyArray2<'py, f64>,
    posed_indexes: PyReadonlyArray1<'py, u32>,
    pair_i: PyReadonlyArray1<'py, u32>,
    pair_j: PyReadonlyArray1<'py, u32>,
    pair_count: PyReadonlyArray1<'py, u32>,
    pair_mean_disp: PyReadonlyArray1<'py, f64>,
    resect_min_obs: usize,
    resect_accept_gate: f64,
    max_neighbors: usize,
    min_shared: u32,
    min_pair_correspondences: usize,
    min_h_inliers: usize,
    min_rotation_measurements: usize,
    rotation_threshold_deg: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let inputs = read_verify_inputs(
        &cluster_indexes,
        &image_indexes,
        &positions_xy,
        &points,
        &quaternions_wxyz,
        &translations,
        &posed_indexes,
        &pair_i,
        &pair_j,
        &pair_count,
        &pair_mean_disp,
    )?;
    let cam = camera.inner.clone();
    let options = VerifyOptions {
        resect_min_obs,
        resect_accept_gate,
        max_neighbors,
        min_shared,
        min_pair_correspondences,
        min_h_inliers,
        min_rotation_measurements,
        rotation_threshold_deg,
        seed,
    };
    let out = py.detach(move || {
        core_verify(
            &inputs.clusters,
            &inputs.images,
            &inputs.positions,
            &cam,
            &inputs.points,
            &inputs.quats,
            &inputs.trans,
            &inputs.posed_idx,
            &inputs.neighborhood,
            &options,
        )
    });
    verification_dict(py, &out)
}

/// Verify, then repair the flagged cameras (see
/// ``specs/core/pose-verification.md``).
///
/// Runs the two verification screens (see ``verify_poses``), then walks the
/// flagged cameras in ascending image order: re-initialize from the top-2
/// nearest registered neighbours (chordal mean of their rotations, mean of
/// their centres, ``t = -R @ c``), trimmed pose-only refinement against the
/// current structure, and accept only when the all-observation inlier
/// fraction reaches ``max(inlier_floor, before + inlier_margin)``. Accepted
/// repairs update the working poses (later repairs see them); rejected
/// repairs leave the pose untouched — bit for bit — and the flag standing.
///
/// Args:
///     (all ``verify_poses`` arguments, plus:)
///     min_obs: Skip a flagged camera with fewer observations of valid
///         points than this (default 12).
///     inlier_floor: Absolute inlier-fraction floor a repair must reach
///         (default 0.10).
///     inlier_margin: Improvement margin over the pre-repair inlier
///         fraction (default 0.05).
///
/// Returns:
///     The ``verify_poses`` dict plus ``{"quaternions_wxyz" (n_posed, 4),
///     "translations" (n_posed, 3), "repaired" (n_posed,) bool,
///     "inlier_before" (n_posed,), "inlier_after" (n_posed,)}`` — the
///     inlier arrays are NaN where no repair was attempted.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (cluster_indexes, image_indexes, positions_xy, camera, points, quaternions_wxyz, translations, posed_indexes, pair_i, pair_j, pair_count, pair_mean_disp, *, resect_min_obs=8, resect_accept_gate=0.30, max_neighbors=4, min_shared=50, min_pair_correspondences=30, min_h_inliers=20, min_rotation_measurements=2, rotation_threshold_deg=3.0, seed=0, min_obs=12, inlier_floor=0.10, inlier_margin=0.05))]
pub fn repair_poses<'py>(
    py: Python<'py>,
    cluster_indexes: PyReadonlyArray1<'py, u32>,
    image_indexes: PyReadonlyArray1<'py, u32>,
    positions_xy: PyReadonlyArray2<'py, f64>,
    camera: PyRef<'py, PyCameraIntrinsics>,
    points: PyReadonlyArray2<'py, f64>,
    quaternions_wxyz: PyReadonlyArray2<'py, f64>,
    translations: PyReadonlyArray2<'py, f64>,
    posed_indexes: PyReadonlyArray1<'py, u32>,
    pair_i: PyReadonlyArray1<'py, u32>,
    pair_j: PyReadonlyArray1<'py, u32>,
    pair_count: PyReadonlyArray1<'py, u32>,
    pair_mean_disp: PyReadonlyArray1<'py, f64>,
    resect_min_obs: usize,
    resect_accept_gate: f64,
    max_neighbors: usize,
    min_shared: u32,
    min_pair_correspondences: usize,
    min_h_inliers: usize,
    min_rotation_measurements: usize,
    rotation_threshold_deg: f64,
    seed: u64,
    min_obs: usize,
    inlier_floor: f64,
    inlier_margin: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let inputs = read_verify_inputs(
        &cluster_indexes,
        &image_indexes,
        &positions_xy,
        &points,
        &quaternions_wxyz,
        &translations,
        &posed_indexes,
        &pair_i,
        &pair_j,
        &pair_count,
        &pair_mean_disp,
    )?;
    let cam = camera.inner.clone();
    let options = RepairOptions {
        verify: VerifyOptions {
            resect_min_obs,
            resect_accept_gate,
            max_neighbors,
            min_shared,
            min_pair_correspondences,
            min_h_inliers,
            min_rotation_measurements,
            rotation_threshold_deg,
            seed,
        },
        min_obs,
        inlier_floor,
        inlier_margin,
    };
    let out = py.detach(move || {
        core_repair(
            &inputs.clusters,
            &inputs.images,
            &inputs.positions,
            &cam,
            &inputs.points,
            &inputs.quats,
            &inputs.trans,
            &inputs.posed_idx,
            &inputs.neighborhood,
            &options,
        )
    });

    let d = verification_dict(py, &out.verification)?;
    let q_rows: Vec<Vec<f64>> = out.quaternions_wxyz.iter().map(|q| q.to_vec()).collect();
    let t_rows: Vec<Vec<f64>> = out.translations.iter().map(|t| t.to_vec()).collect();
    d.set_item(
        "quaternions_wxyz",
        PyArray2::from_vec2(py, &q_rows)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
    )?;
    d.set_item(
        "translations",
        PyArray2::from_vec2(py, &t_rows)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
    )?;
    d.set_item("repaired", PyArray1::from_slice(py, &out.repaired))?;
    d.set_item(
        "inlier_before",
        PyArray1::from_slice(py, &out.inlier_before),
    )?;
    d.set_item("inlier_after", PyArray1::from_slice(py, &out.inlier_after))?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(verify_poses, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(repair_poses, m)?)?;
    Ok(())
}
