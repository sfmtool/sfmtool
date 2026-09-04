// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the source-cluster join: which clusters of a selection a
//! member's admission never held, banded by feature radius.

use numpy::{
    PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::analysis::source_clusters::{
    assign_bands as core_assign_bands, source_clusters as core_source_clusters, MemberIdentity,
    SourceSelection,
};

/// Every cluster of a selection the member's placed frames see that its
/// admission never held, with each one's radius and radius band.
///
/// The identity between a member observation and a selection row is exact and
/// needs no geometry: both carry the image index and the feature index, so a
/// member row is a selection row by ``(image, feature)``. A member row takes the
/// FIRST selection row carrying its key, and that row's cluster is admitted. A
/// candidate is a cluster at least two placed frames see that no member row
/// admitted.
///
/// A cluster's radius is its widest row's: half the refine radius times the sum
/// of that row's stored affine's two column norms.
///
/// Args:
///     cluster_starts: (n_cluster + 1,) uint32 CSR boundaries.
///     member_images: (n_member,) uint32 image index per selection row.
///     member_features: (n_member,) uint32 feature index per selection row.
///     member_positions: (n_member, 2) float64 absolute keypoint positions.
///     member_affine_shapes: (n_member, 2, 2) float64 absolute affine shapes.
///     refine_radius: The radius the shapes are expressed against.
///     n_images: How many images the selection's table names.
///     obs_image: (n_obs,) uint32 image index per member observation.
///     obs_feature: (n_obs,) uint32 feature index per member observation.
///     frames: (n_frames,) uint32 placed image indices.
///     band_edges: (n_band + 1,) float64, DECREASING, in units of the
///         admission floor; ``band_edges[k]`` is band ``k``'s upper bound and
///         ``band_edges[k + 1]`` its lower one, half open.
///
/// Returns:
///     A dict with ``n_file_clusters``, ``n_admitted``, ``n_rows_matched``,
///     ``admission_radius`` (n_admitted,), ``admission_floor_px`` (float, NaN
///     where the admission is empty), ``candidates`` (n_cand,) uint32 ascending,
///     ``candidate_radius`` (n_cand,), ``candidate_band`` (n_cand,) int64,
///     and the selected observations ``obs_cluster`` / ``obs_image`` /
///     ``obs_feature`` (uint32), ``obs_uv`` (n_sel, 2) and ``obs_shape``
///     (n_sel, 2, 2), all in selection row order.
#[pyfunction]
#[pyo3(signature = (
    cluster_starts,
    member_images,
    member_features,
    member_positions,
    member_affine_shapes,
    refine_radius,
    n_images,
    obs_image,
    obs_feature,
    frames,
    band_edges,
))]
#[allow(clippy::too_many_arguments)]
pub fn source_clusters<'py>(
    py: Python<'py>,
    cluster_starts: PyReadonlyArray1<'py, u32>,
    member_images: PyReadonlyArray1<'py, u32>,
    member_features: PyReadonlyArray1<'py, u32>,
    member_positions: PyReadonlyArray2<'py, f64>,
    member_affine_shapes: PyReadonlyArray3<'py, f64>,
    refine_radius: f64,
    n_images: usize,
    obs_image: PyReadonlyArray1<'py, u32>,
    obs_feature: PyReadonlyArray1<'py, u32>,
    frames: PyReadonlyArray1<'py, u32>,
    band_edges: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let n_member = member_images.shape()[0];
    if member_features.shape()[0] != n_member {
        return Err(PyValueError::new_err(
            "member_images and member_features must share the same length",
        ));
    }
    if member_positions.shape() != [n_member, 2] {
        return Err(PyValueError::new_err(
            "member_positions must have shape (n_member, 2)",
        ));
    }
    if member_affine_shapes.shape() != [n_member, 2, 2] {
        return Err(PyValueError::new_err(
            "member_affine_shapes must have shape (n_member, 2, 2)",
        ));
    }
    if obs_image.shape()[0] != obs_feature.shape()[0] {
        return Err(PyValueError::new_err(
            "obs_image and obs_feature must share the same length",
        ));
    }
    if cluster_starts.shape()[0] == 0 {
        return Err(PyValueError::new_err(
            "cluster_starts must carry at least one boundary",
        ));
    }

    let starts = to_contiguous!(cluster_starts);
    let mimg = to_contiguous!(member_images);
    let mfeat = to_contiguous!(member_features);
    let mpos = to_contiguous!(member_positions);
    let mshp = to_contiguous!(member_affine_shapes);
    let oimg = to_contiguous!(obs_image);
    let ofeat = to_contiguous!(obs_feature);
    let fr = to_contiguous!(frames);
    let edges = to_contiguous!(band_edges);

    let mut prev = 0u32;
    for (k, &s) in starts.iter().enumerate() {
        if s < prev {
            return Err(PyValueError::new_err(format!(
                "cluster_starts must be non-decreasing, got {s} after {prev} at index {k}"
            )));
        }
        prev = s;
    }
    if prev as usize > n_member {
        return Err(PyValueError::new_err(format!(
            "cluster_starts[-1] = {prev} exceeds the {n_member} selection rows"
        )));
    }

    let out = py.detach(|| {
        core_source_clusters(
            SourceSelection {
                cluster_starts: &starts,
                member_images: &mimg,
                member_features: &mfeat,
                member_positions: &mpos,
                member_affine_shapes: &mshp,
                refine_radius,
                n_images,
            },
            MemberIdentity {
                obs_image: &oimg,
                obs_feature: &ofeat,
            },
            &fr,
            &edges,
        )
    });

    let n_sel = out.obs_cluster.len();
    let dict = PyDict::new(py);
    dict.set_item("n_file_clusters", out.n_file_clusters)?;
    dict.set_item("n_admitted", out.admission_radius.len())?;
    dict.set_item("n_rows_matched", out.n_rows_matched)?;
    dict.set_item("admission_floor_px", out.admission_floor_px)?;
    dict.set_item(
        "admission_radius",
        PyArray1::from_vec(py, out.admission_radius),
    )?;
    dict.set_item("candidates", PyArray1::from_vec(py, out.candidates))?;
    dict.set_item(
        "candidate_radius",
        PyArray1::from_vec(py, out.candidate_radius),
    )?;
    dict.set_item("candidate_band", PyArray1::from_vec(py, out.candidate_band))?;
    dict.set_item("obs_cluster", PyArray1::from_vec(py, out.obs_cluster))?;
    dict.set_item("obs_image", PyArray1::from_vec(py, out.obs_image))?;
    dict.set_item("obs_feature", PyArray1::from_vec(py, out.obs_feature))?;
    // Flat then reshaped: a row-of-rows conversion allocates one Vec per
    // observation, which on a whole capture's selection costs more than the
    // join itself.
    let uv = PyArray1::from_vec(py, out.obs_uv)
        .reshape([n_sel, 2])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    dict.set_item("obs_uv", uv)?;
    let shape = PyArray1::from_vec(py, out.obs_shape)
        .reshape([n_sel, 2, 2])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    dict.set_item("obs_shape", shape)?;
    dict.set_item("n_selected", n_sel)?;
    Ok(dict.into_any().unbind())
}

/// The radius band each value falls in, or ``-1`` past the last edge.
///
/// Half open, in units of ``floor``: band ``k`` holds the values in
/// ``[floor * band_edges[k + 1], floor * band_edges[k])``. ``band_edges`` runs
/// DECREASING and its first entry may be infinite, which leaves the top band
/// open above the floor.
///
/// Args:
///     radius: (n,) float64 values to band.
///     floor: The unit the bands are measured in.
///     band_edges: (n_band + 1,) float64, decreasing.
///
/// Returns:
///     (n,) int64 band index per value.
#[pyfunction]
pub fn assign_bands<'py>(
    py: Python<'py>,
    radius: PyReadonlyArray1<'py, f64>,
    floor: f64,
    band_edges: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let r = to_contiguous!(radius);
    let e = to_contiguous!(band_edges);
    Ok(PyArray1::from_vec(py, core_assign_bands(&r, floor, &e)))
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(source_clusters, m)?)?;
    m.add_function(wrap_pyfunction!(assign_bands, m)?)?;
    Ok(())
}
