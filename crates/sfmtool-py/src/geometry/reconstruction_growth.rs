// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for reconstruction growth and batch registration
//! (``sfmtool._sfmtool.geometry.grow_reconstruction`` /
//! ``resect_images_batch``; see ``specs/core/reconstruction-growth.md``).

use std::borrow::Cow;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::geometry::batch_resection::{resect_images_batch as core_resect, ResectOptions};
use sfmtool_core::geometry::reconstruction_growth::{
    grow_reconstruction as core_grow, GrowOptions,
};

use crate::geometry::PyCameraIntrinsics;

/// Validate an `(n, w)` float64 array and return its rows as fixed arrays.
pub(crate) fn read_rows<const W: usize>(
    arr: &PyReadonlyArray2<'_, f64>,
    name: &str,
) -> PyResult<Vec<[f64; W]>> {
    if arr.shape()[1] != W {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have shape (n, {W}), got (n, {})",
            arr.shape()[1]
        )));
    }
    let flat = to_contiguous!(arr);
    Ok(flat
        .chunks_exact(W)
        .map(|c| {
            let mut row = [0.0; W];
            row.copy_from_slice(c);
            row
        })
        .collect())
}

/// Validated observation arrays: cluster ids, image ids, and positions.
pub(crate) type Observations = (Vec<u32>, Vec<u32>, Vec<[f64; 2]>);

/// Shared observation-array validation: parallel lengths and nondecreasing
/// cluster ids.
pub(crate) fn read_observations(
    cluster_indexes: &PyReadonlyArray1<'_, u32>,
    image_indexes: &PyReadonlyArray1<'_, u32>,
    positions_xy: &PyReadonlyArray2<'_, f64>,
) -> PyResult<Observations> {
    let n_obs = cluster_indexes.shape()[0];
    if image_indexes.shape()[0] != n_obs || positions_xy.shape()[0] != n_obs {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cluster_indexes, image_indexes, and positions_xy must share n_obs",
        ));
    }
    let clusters = to_contiguous!(cluster_indexes).into_owned();
    let images = to_contiguous!(image_indexes).into_owned();
    let positions = read_rows::<2>(positions_xy, "positions_xy")?;
    if clusters.windows(2).any(|w| w[1] < w[0]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cluster_indexes must be nondecreasing",
        ));
    }
    Ok((clusters, images, positions))
}

/// Grow a seeded reconstruction to full registration
/// (see ``specs/core/reconstruction-growth.md``).
///
/// Registers the un-posed images of a cluster-track set against the seed
/// poses: next-best-view resection (RANSAC P3P polished by trimmed pose-only
/// refinement, covisible-neighbour inits as the fallback), incremental
/// triangulation, and bounded staged bundle adjustments, until no further
/// image clears its acceptance gate. Deferred images are re-armed by one
/// adjustment pass and then verified force-accept. A finishing adjustment
/// releases the shared focal (SIMPLE_PINHOLE) on a covisibility-spread
/// subset, followed by re-triangulation at the released focal. Poses are
/// world-to-camera in the canonical frame (the camera looks along −Z);
/// observations are full-pixel positions.
///
/// Args:
///     cluster_indexes: (n_obs,) uint32 cluster id per observation,
///         nondecreasing (each distinct cluster is a contiguous run).
///     image_indexes: (n_obs,) uint32 image id per observation.
///     positions_xy: (n_obs, 2) float64 full-pixel keypoint positions.
///     camera: Shared ``CameraIntrinsics``; growth runs at its focal.
///     quaternions_wxyz: (n_seed, 4) float64 seed world-to-camera rotations.
///     translations: (n_seed, 3) float64 seed world-to-camera translations.
///     posed_indexes: (n_seed,) uint32 image ids of the seed poses.
///     ba_window: Most-recently-posed cameras each growth adjustment refines
///         (registration order); 0 refines every posed camera (default 0).
///     anchor_every: Every ``anchor_every``-th growth adjustment refines a
///         covisibility-spread subset of all posed cameras (capped ~150)
///         instead of the frontier; 0 never anchors (default 0).
///     ba_cluster_cap: Restrict the adjustments to the best-``cap`` clusters
///         by span; 0 admits all (default 0).
///     min_obs: An image needs at least this many observations of valid
///         points to be a growth candidate (default 8).
///     accept_gate: Defer an image whose inlier fraction falls below
///         ``accept_gate ×`` the median accepted-so-far fraction
///         (default 0.35).
///     seed: RANSAC seed; same inputs + seed give identical output
///         (default 0).
///
/// Returns:
///     A dict ``{"quaternions_wxyz" (n_img, 4), "translations" (n_img, 3),
///     "posed" (n_img,) bool, "points" (n_clusters, 3) with NaN where never
///     triangulated, "focal" float, "residual_norms" (n_obs,) with inf where
///     invalid}``. Un-posed images carry the identity pose. Degenerate
///     inputs (no seed poses, no triangulable clusters, every image below
///     ``min_obs``) return the input state with empty growth, not an error.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (cluster_indexes, image_indexes, positions_xy, camera, quaternions_wxyz, translations, posed_indexes, *, ba_window=0, anchor_every=0, ba_cluster_cap=0, min_obs=8, accept_gate=0.35, seed=0))]
pub fn grow_reconstruction<'py>(
    py: Python<'py>,
    cluster_indexes: PyReadonlyArray1<'py, u32>,
    image_indexes: PyReadonlyArray1<'py, u32>,
    positions_xy: PyReadonlyArray2<'py, f64>,
    camera: PyRef<'py, PyCameraIntrinsics>,
    quaternions_wxyz: PyReadonlyArray2<'py, f64>,
    translations: PyReadonlyArray2<'py, f64>,
    posed_indexes: PyReadonlyArray1<'py, u32>,
    ba_window: usize,
    anchor_every: usize,
    ba_cluster_cap: usize,
    min_obs: usize,
    accept_gate: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let (clusters, images, positions) =
        read_observations(&cluster_indexes, &image_indexes, &positions_xy)?;
    let quats = read_rows::<4>(&quaternions_wxyz, "quaternions_wxyz")?;
    let trans = read_rows::<3>(&translations, "translations")?;
    let posed_idx = to_contiguous!(posed_indexes).into_owned();
    if quats.len() != posed_idx.len() || trans.len() != posed_idx.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "quaternions_wxyz, translations, and posed_indexes must share n_seed",
        ));
    }

    let cam = camera.inner.clone();
    let options = GrowOptions {
        ba_window,
        anchor_every,
        ba_cluster_cap,
        min_obs,
        accept_gate,
        seed,
    };
    let out = py.detach(move || {
        core_grow(
            &clusters, &images, &positions, &cam, &quats, &trans, &posed_idx, &options,
        )
    });

    let q_rows: Vec<Vec<f64>> = out.quaternions_wxyz.iter().map(|q| q.to_vec()).collect();
    let t_rows: Vec<Vec<f64>> = out.translations.iter().map(|t| t.to_vec()).collect();
    let p_rows: Vec<Vec<f64>> = out.points.iter().map(|p| p.to_vec()).collect();

    let d = PyDict::new(py);
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
    d.set_item("posed", PyArray1::from_slice(py, &out.posed))?;
    d.set_item(
        "points",
        PyArray2::from_vec2(py, &p_rows)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
    )?;
    d.set_item("focal", out.focal)?;
    d.set_item("residual_norms", PyArray1::from_vec(py, out.residual_norms))?;
    Ok(d)
}

/// Pose-only resection of many images against fixed structure
/// (see ``specs/core/reconstruction-growth.md``).
///
/// Each requested image is registered independently — no adjustment, no
/// cross-image coupling — and the images run in parallel. Per image: gather
/// its observations of clusters with a finite ``points`` row (below
/// ``min_obs``, skip), estimate by RANSAC P3P polished by trimmed pose-only
/// refinement on the consensus subset, and when the minimal estimate fails
/// fall back to trimmed refinement initialized from the poses of the image's
/// most-covisible registered neighbours (``posed_quaternions_wxyz`` /
/// ``posed_translations`` / ``posed_indexes``; without them the fallback is
/// unavailable). Score by the all-observation inlier fraction at 3 px;
/// accept at or above ``accept_gate``. Each image's RANSAC is seeded as a
/// pure function of ``(seed, image index)``, so the parallel execution is
/// deterministic and a one-image call matches its batch row bit for bit.
///
/// Args:
///     cluster_indexes: (n_obs,) uint32 cluster id per observation,
///         nondecreasing.
///     image_indexes: (n_obs,) uint32 image id per observation.
///     positions_xy: (n_obs, 2) float64 full-pixel keypoint positions.
///     camera: Shared ``CameraIntrinsics``.
///     points: (n_clusters, 3) float64 world points (NaN rows are invalid).
///     image_list: (m,) uint32 image ids to register.
///     posed_quaternions_wxyz: Optional (n_posed, 4) float64 rotations of
///         already-registered images, for the fallback inits.
///     posed_translations: Optional (n_posed, 3) float64 translations.
///     posed_indexes: Optional (n_posed,) uint32 image ids of the registered
///         poses.
///     min_obs: Skip an image with fewer observations of valid points than
///         this (default 8).
///     accept_gate: Accept an image at or above this all-observation inlier
///         fraction (default 0.30).
///     seed: Base RANSAC seed (default 0).
///
/// Returns:
///     A dict ``{"quaternions_wxyz" (m, 4), "translations" (m, 3),
///     "inlier_fractions" (m,), "accepted" (m,) bool}`` aligned with
///     ``image_list``; skipped or failed images carry the identity pose and
///     ``accepted = False``.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (cluster_indexes, image_indexes, positions_xy, camera, points, image_list, *, posed_quaternions_wxyz=None, posed_translations=None, posed_indexes=None, min_obs=8, accept_gate=0.30, seed=0))]
pub fn resect_images_batch<'py>(
    py: Python<'py>,
    cluster_indexes: PyReadonlyArray1<'py, u32>,
    image_indexes: PyReadonlyArray1<'py, u32>,
    positions_xy: PyReadonlyArray2<'py, f64>,
    camera: PyRef<'py, PyCameraIntrinsics>,
    points: PyReadonlyArray2<'py, f64>,
    image_list: PyReadonlyArray1<'py, u32>,
    posed_quaternions_wxyz: Option<PyReadonlyArray2<'py, f64>>,
    posed_translations: Option<PyReadonlyArray2<'py, f64>>,
    posed_indexes: Option<PyReadonlyArray1<'py, u32>>,
    min_obs: usize,
    accept_gate: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let (clusters, images, positions) =
        read_observations(&cluster_indexes, &image_indexes, &positions_xy)?;
    let pts = read_rows::<3>(&points, "points")?;
    if let Some(&max_c) = clusters.iter().max() {
        if max_c as usize >= pts.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cluster index {max_c} out of range for {} points",
                pts.len()
            )));
        }
    }
    let img_list = to_contiguous!(image_list).into_owned();

    let posed_given = [
        posed_quaternions_wxyz.is_some(),
        posed_translations.is_some(),
        posed_indexes.is_some(),
    ];
    if posed_given.iter().any(|&g| g) && !posed_given.iter().all(|&g| g) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "posed_quaternions_wxyz, posed_translations, and posed_indexes must be \
             given together",
        ));
    }
    let posed_quats = match &posed_quaternions_wxyz {
        Some(a) => read_rows::<4>(a, "posed_quaternions_wxyz")?,
        None => Vec::new(),
    };
    let posed_trans = match &posed_translations {
        Some(a) => read_rows::<3>(a, "posed_translations")?,
        None => Vec::new(),
    };
    let posed_idx = match &posed_indexes {
        Some(a) => to_contiguous!(a).into_owned(),
        None => Vec::new(),
    };
    if posed_quats.len() != posed_idx.len() || posed_trans.len() != posed_idx.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "posed_quaternions_wxyz, posed_translations, and posed_indexes must share \
             n_posed",
        ));
    }

    let cam = camera.inner.clone();
    let options = ResectOptions {
        min_obs,
        accept_gate,
        seed,
    };
    let out = py.detach(move || {
        core_resect(
            &clusters,
            &images,
            &positions,
            &cam,
            &pts,
            &img_list,
            &posed_quats,
            &posed_trans,
            &posed_idx,
            &options,
        )
    });

    let q_rows: Vec<Vec<f64>> = out.quaternions_wxyz.iter().map(|q| q.to_vec()).collect();
    let t_rows: Vec<Vec<f64>> = out.translations.iter().map(|t| t.to_vec()).collect();
    let d = PyDict::new(py);
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
        "inlier_fractions",
        PyArray1::from_vec(py, out.inlier_fractions),
    )?;
    d.set_item("accepted", PyArray1::from_slice(py, &out.accepted))?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(grow_reconstruction, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(resect_images_batch, m)?)?;
    Ok(())
}
