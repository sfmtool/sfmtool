// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Binding for the far-field rotation initialization
//! (``sfmtool._sfmtool.geometry.rotation_init``; see
//! ``specs/core/rotation-init.md``).

use std::borrow::Cow;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::geometry::rotation_init as core_rotation_init;

/// Build an initial multi-camera reconstruction from cluster tracks by
/// far-field rotation initialization (see ``specs/core/rotation-init.md``).
///
/// Parallax-free (far-field) correspondences fix rotations between image
/// pairs through conjugate homographies ``H = K R K^-1`` (spanning-tree
/// propagation plus chordal-mean rotation averaging over the largest
/// connected component); parallax-bearing (near-field) correspondences then
/// supply the metric side — a linear seed baseline, triangulated structure,
/// and rotation-locked translation growth — finishing with one staged bundle
/// adjustment at fixed ``f0`` (the far-field clusters modeled at infinity).
/// Poses are world-to-camera in the canonical frame (the camera looks along
/// −Z); the seed pair's baseline defines unit scale.
///
/// Args:
///     cluster_indexes: (n_obs,) uint32 cluster id per observation,
///         nondecreasing (each distinct cluster is a contiguous run).
///     image_indexes: (n_obs,) uint32 image id per observation.
///     positions_xy: (n_obs, 2) float64 full-pixel keypoint positions.
///     width: Shared image width; the principal point is the image centre.
///     height: Shared image height.
///     f0: Shared focal in pixels (typically a focal-vote consensus).
///     seed: SplitMix64 seed for the homography RANSAC; same inputs + seed
///         => bit-identical output (default 0).
///     min_images: Fail when the largest rotation component has fewer images
///         than this (default 8).
///     max_images: Core size budget for the translation growth (default 14).
///
/// Returns:
///     A dict ``{"image_indexes" (n_posed,) uint32, "quaternions_wxyz"
///     (n_posed, 4), "translations" (n_posed, 3), "points" (n_clusters, 3)
///     with NaN where absent, "inlier_fractions" (n_posed,),
///     "far_cluster_indexes" (m,) uint32}``, or ``None`` when no rotation
///     edge validates, the component is too small, or the seed fails its
///     cheirality floor. Rows of ``points`` listed in
///     ``far_cluster_indexes`` are unit world-frame directions (usable as
///     ``bundle_adjust``'s ``point_at_infinity`` rows); other finite rows
///     are triangulated positions.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (cluster_indexes, image_indexes, positions_xy, width, height, f0, *, seed=0, min_images=8, max_images=14))]
pub fn rotation_init<'py>(
    py: Python<'py>,
    cluster_indexes: PyReadonlyArray1<'py, u32>,
    image_indexes: PyReadonlyArray1<'py, u32>,
    positions_xy: PyReadonlyArray2<'py, f64>,
    width: u32,
    height: u32,
    f0: f64,
    seed: u64,
    min_images: usize,
    max_images: usize,
) -> PyResult<Option<Bound<'py, PyDict>>> {
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

    let result = py.detach(move || {
        core_rotation_init(
            &clusters, &images, &positions, width, height, f0, seed, min_images, max_images,
        )
    });
    let Some(out) = result else {
        return Ok(None);
    };

    let q_rows: Vec<Vec<f64>> = out.quaternions_wxyz.iter().map(|q| q.to_vec()).collect();
    let t_rows: Vec<Vec<f64>> = out.translations.iter().map(|t| t.to_vec()).collect();
    let p_rows: Vec<Vec<f64>> = out.points.iter().map(|p| p.to_vec()).collect();

    let d = PyDict::new(py);
    d.set_item("image_indexes", PyArray1::from_vec(py, out.image_indexes))?;
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
    d.set_item(
        "points",
        PyArray2::from_vec2(py, &p_rows)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
    )?;
    d.set_item(
        "inlier_fractions",
        PyArray1::from_vec(py, out.inlier_fractions),
    )?;
    d.set_item(
        "far_cluster_indexes",
        PyArray1::from_vec(py, out.far_cluster_indexes),
    )?;
    Ok(Some(d))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(rotation_init, m)?)?;
    Ok(())
}
