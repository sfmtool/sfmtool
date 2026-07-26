// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Binding for the cluster match census
//! (``sfmtool._sfmtool.analysis.cluster_census``; see
//! ``specs/core/cluster-census.md``).

use std::borrow::Cow;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::analysis::cluster_census::{
    cluster_census as core_census, CensusParams, CensusReport,
};

use crate::geometry::reconstruction_growth::{read_observations, read_rows};
use crate::geometry::PyCameraIntrinsics;

/// Score a candidate solve against the raw correspondence evidence it did not
/// consume (see ``specs/core/cluster-census.md``).
///
/// A reconstruction can be internally consistent and wrong — a viewpoint group glued
/// at the wrong relative pose, or poses and structure bent to absorb a wrong
/// focal — and neither shows up in its own reprojection error. The census
/// partitions the posed images into viewpoint groups by greedy-modularity
/// communities of the *raw* cluster-covisibility graph, triangulates every raw
/// cluster at the candidate poses, and reports, per group pair, the Wilson
/// lower bound of the fraction of eligible high-parallax bridge clusters the
/// candidate cannot satisfy. The score is the maximum over pairs, so a fine
/// partition cannot dilute one bad seam. Trust is data-derived: a cluster
/// counts as a genuine correspondence when its warp-consistency residual is
/// within the P95 of the residuals of the clusters the candidate *does*
/// satisfy.
///
/// Fewer than two groups means the capture has no group structure to census —
/// ``n_groups < 2`` with score 0 is *unverifiable*, which callers must read as
/// "no evidence", not "clean".
///
/// Deterministic: identical inputs give identical output.
///
/// Args:
///     cluster_indexes: (n_obs,) uint32 raw cluster id per observation,
///         nondecreasing (each cluster is a contiguous run).
///     image_indexes: (n_obs,) uint32 image id per observation.
///     positions_xy: (n_obs, 2) float64 full-pixel observation positions.
///     cluster_warp_consistency: (n_clusters,) float64 matching-time
///         warp-consistency residual per cluster (lower is better; non-finite
///         for a cluster that never entered the consistency fit).
///     camera: Shared ``CameraIntrinsics`` of the candidate.
///     quaternions_wxyz: (n_posed, 4) float64 world-to-camera rotations
///         (canonical frame, the camera looks along −Z).
///     translations: (n_posed, 3) float64 world-to-camera translations.
///     posed_indexes: (n_posed,) uint32 image ids of the candidate's poses.
///     sat_px: Satisfied bar on a cluster's median reprojection residual
///         (default 2.0).
///     hi_parallax_deg: Parallax floor for a bridge to carry constraint
///         (default 5.0).
///     warp_percentile: Percentile of the satisfied clusters'
///         warp-consistency residuals that becomes the eligibility
///         threshold (default 95.0).
///     wilson_z: Wilson bound z (default 1.96, the 95 % bound).
///
/// Returns:
///     A dict ``{"score": float, "n_groups": int, "group_of" (n_img,) int32
///     with -1 for an unposed image, "pairs": list of
///     ``{"group_a", "group_b", "n_eligible_hi", "n_unsatisfied_hi",
///     "wilson_lb"}`` ascending by group pair, "sat_pct": float,
///     "group_consistency": None}``. ``group_consistency`` is reserved for
///     the phase-2 companion and is always ``None`` today.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    cluster_indexes,
    image_indexes,
    positions_xy,
    cluster_warp_consistency,
    camera,
    quaternions_wxyz,
    translations,
    posed_indexes,
    *,
    sat_px=2.0,
    hi_parallax_deg=5.0,
    warp_percentile=95.0,
    wilson_z=1.96,
))]
pub fn cluster_census<'py>(
    py: Python<'py>,
    cluster_indexes: PyReadonlyArray1<'py, u32>,
    image_indexes: PyReadonlyArray1<'py, u32>,
    positions_xy: PyReadonlyArray2<'py, f64>,
    cluster_warp_consistency: PyReadonlyArray1<'py, f64>,
    camera: PyRef<'py, PyCameraIntrinsics>,
    quaternions_wxyz: PyReadonlyArray2<'py, f64>,
    translations: PyReadonlyArray2<'py, f64>,
    posed_indexes: PyReadonlyArray1<'py, u32>,
    sat_px: f64,
    hi_parallax_deg: f64,
    warp_percentile: f64,
    wilson_z: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (clusters, images, positions) =
        read_observations(&cluster_indexes, &image_indexes, &positions_xy)?;
    let warp = to_contiguous!(cluster_warp_consistency).into_owned();
    let quats = read_rows::<4>(&quaternions_wxyz, "quaternions_wxyz")?;
    let trans = read_rows::<3>(&translations, "translations")?;
    let posed_idx = to_contiguous!(posed_indexes).into_owned();
    if quats.len() != posed_idx.len() || trans.len() != posed_idx.len() {
        return Err(PyValueError::new_err(
            "quaternions_wxyz, translations, and posed_indexes must share n_posed",
        ));
    }
    if !(0.0..=100.0).contains(&warp_percentile) {
        return Err(PyValueError::new_err(
            "warp_percentile must lie in [0, 100]",
        ));
    }
    // A NaN threshold would leak through every comparison below as "no
    // cluster qualifies" and report a clean 0.0 — the one silent failure the
    // census must never produce.
    if !(sat_px.is_finite() && sat_px > 0.0) {
        return Err(PyValueError::new_err("sat_px must be finite and positive"));
    }
    if !(hi_parallax_deg.is_finite() && hi_parallax_deg >= 0.0) {
        return Err(PyValueError::new_err(
            "hi_parallax_deg must be finite and non-negative",
        ));
    }
    if !(wilson_z.is_finite() && wilson_z >= 0.0) {
        return Err(PyValueError::new_err(
            "wilson_z must be finite and non-negative",
        ));
    }

    let cam = camera.inner.clone();
    let params = CensusParams {
        sat_px,
        hi_parallax_deg,
        warp_percentile,
        wilson_z,
    };
    let report: CensusReport = py
        .detach(move || {
            core_census(
                &clusters, &images, &positions, &warp, &cam, &quats, &trans, &posed_idx, &params,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let pairs = PyList::empty(py);
    for pair in &report.pairs {
        let d = PyDict::new(py);
        d.set_item("group_a", pair.group_a)?;
        d.set_item("group_b", pair.group_b)?;
        d.set_item("n_eligible_hi", pair.n_eligible_hi)?;
        d.set_item("n_unsatisfied_hi", pair.n_unsatisfied_hi)?;
        d.set_item("wilson_lb", pair.wilson_lb)?;
        pairs.append(d)?;
    }

    let out = PyDict::new(py);
    out.set_item("score", report.score)?;
    out.set_item("n_groups", report.n_groups)?;
    out.set_item("group_of", PyArray1::from_slice(py, &report.group_of))?;
    out.set_item("pairs", pairs)?;
    out.set_item("sat_pct", report.sat_pct)?;
    // Phase 2: the group-consistency companion is specified but not
    // implemented.
    out.set_item("group_consistency", py.None())?;
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cluster_census, m)?)?;
    Ok(())
}
