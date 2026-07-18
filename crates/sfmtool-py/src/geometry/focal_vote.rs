// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the structure-free focal vote
//! (``sfmtool._sfmtool.geometry.focal_vote``; see ``specs/core/focal-vote.md``).

use std::borrow::Cow;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::geometry::focal_vote::focal_vote as core_focal_vote;

/// Estimate a shared focal length from cluster-track observations without any
/// reconstruction (see ``specs/core/focal-vote.md``).
///
/// Image pairs drawn from the cluster tracks each cast one focal vote through
/// whichever estimator their geometry can observe — the Bougnoux focal of a
/// fundamental matrix (parallax-rich pairs) or rotation self-calibration of a
/// parallax-free homography (far-field pairs) — and the consensus focal is the
/// median of the winning family. No structure is estimated, so the vote cannot
/// be biased by the depth/focal (bas-relief) compensation of structure-based
/// focal estimation.
///
/// Args:
///     cluster_indexes: (n_obs,) uint32 cluster id per observation,
///         nondecreasing (each distinct cluster is a contiguous run).
///     image_indexes: (n_obs,) uint32 image id per observation.
///     positions_xy: (n_obs, 2) float64 full-pixel keypoint positions.
///     width: Shared image width; the principal point is the image centre.
///     height: Shared image height.
///     seed: SplitMix64 seed for the sampled pair-table pass and the RANSAC
///         estimators; same inputs + seed => bit-identical output (default 0).
///
/// Returns:
///     A dict mirroring the output table: ``{"focal_px": float | None,
///     "family": "Epipolar" | "Rotation" | None, "epipolar_focal_px":
///     float | None, "rotation_focal_px": float | None, "n_epipolar": int,
///     "n_rotation": int, "parallax_poverty": float}``.
#[pyfunction]
#[pyo3(signature = (cluster_indexes, image_indexes, positions_xy, width, height, *, seed=0))]
pub fn focal_vote<'py>(
    py: Python<'py>,
    cluster_indexes: PyReadonlyArray1<'py, u32>,
    image_indexes: PyReadonlyArray1<'py, u32>,
    positions_xy: PyReadonlyArray2<'py, f64>,
    width: u32,
    height: u32,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    if positions_xy.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "positions_xy must have shape (n_obs, 2)",
        ));
    }
    let n_obs = cluster_indexes.shape()[0];
    if image_indexes.shape()[0] != n_obs || positions_xy.shape()[0] != n_obs {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cluster_indexes, image_indexes, and positions_xy must share n_obs",
        ));
    }

    let clusters = to_contiguous!(cluster_indexes);
    let images = to_contiguous!(image_indexes);
    let pos_flat = to_contiguous!(positions_xy);
    let positions: Vec<[f64; 2]> = pos_flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect();

    // Cluster ids must be nondecreasing (contiguous runs).
    if clusters.windows(2).any(|w| w[1] < w[0]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cluster_indexes must be nondecreasing",
        ));
    }

    let result =
        py.detach(move || core_focal_vote(&clusters, &images, &positions, width, height, seed));

    let d = PyDict::new(py);
    d.set_item("focal_px", result.focal_px)?;
    d.set_item("family", result.family.map(|f| f.as_str()))?;
    d.set_item("epipolar_focal_px", result.epipolar_focal_px)?;
    d.set_item("rotation_focal_px", result.rotation_focal_px)?;
    d.set_item("n_epipolar", result.n_epipolar)?;
    d.set_item("n_rotation", result.n_rotation)?;
    d.set_item("parallax_poverty", result.parallax_poverty)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(focal_vote, m)?)?;
    Ok(())
}
