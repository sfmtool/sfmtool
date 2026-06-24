// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for batch triangulation with observability diagnostics.

use std::borrow::Cow;

use nalgebra::{Point3, Vector3};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::reconstruction::triangulation::triangulate_batch as core_triangulate_batch;

/// Triangulate a batch of tracks from world-space rays, returning each track's
/// least-squares point and the observability diagnostics the solve computes.
///
/// Tracks are flattened CSR-style: track ``t`` owns ``dirs[offsets[t]:offsets[t+1]]``
/// and the matching ``centers``.
///
/// Args:
///     dirs: Unit world-space rays, shape ``(T, 3)`` float64.
///     centers: Matching camera centers, shape ``(T, 3)`` float64.
///     offsets: CSR track boundaries, shape ``(M + 1,)`` int64.
///
/// Returns:
///     A dict of arrays: ``points`` ``(M, 3)`` float64, ``eigenvalues``
///     ``(M, 3)`` float64 (ascending, of ``A = Σ(I − dᵢdᵢᵀ)``),
///     ``condition_number`` ``(M,)`` float64, ``in_front_of_all_cameras``
///     ``(M,)`` bool.
#[pyfunction]
pub fn triangulate_batch(
    py: Python<'_>,
    dirs: PyReadonlyArray2<f64>,
    centers: PyReadonlyArray2<f64>,
    offsets: PyReadonlyArray1<i64>,
) -> PyResult<Py<PyAny>> {
    let t = dirs.shape()[0];
    if centers.shape()[0] != t {
        return Err(PyValueError::new_err(
            "dirs and centers must have the same length",
        ));
    }

    let dirs_data = to_contiguous!(dirs);
    let centers_data = to_contiguous!(centers);
    let offsets_data = to_contiguous!(offsets);

    let dirs_vec: Vec<Vector3<f64>> = dirs_data
        .chunks_exact(3)
        .map(|c| Vector3::new(c[0], c[1], c[2]))
        .collect();
    let centers_vec: Vec<Point3<f64>> = centers_data
        .chunks_exact(3)
        .map(|c| Point3::new(c[0], c[1], c[2]))
        .collect();

    // Validate the CSR offsets up front: they must be non-negative,
    // non-decreasing, and end within the ray count. Otherwise the core would
    // index `dirs`/`centers` out of bounds and panic on user input.
    let mut offsets_vec: Vec<usize> = Vec::with_capacity(offsets_data.len());
    let mut prev: i64 = 0;
    for (k, &o) in offsets_data.iter().enumerate() {
        if o < 0 {
            return Err(PyValueError::new_err(format!(
                "offsets must be non-negative, got {o} at index {k}"
            )));
        }
        if o < prev {
            return Err(PyValueError::new_err(format!(
                "offsets must be non-decreasing, got {o} after {prev} at index {k}"
            )));
        }
        prev = o;
        offsets_vec.push(o as usize);
    }
    if let Some(&last) = offsets_vec.last() {
        if last > t {
            return Err(PyValueError::new_err(format!(
                "offsets[-1] = {last} exceeds the number of rays {t}"
            )));
        }
    }

    let tris = core_triangulate_batch(&dirs_vec, &centers_vec, &offsets_vec);

    let points: Vec<Vec<f64>> = tris
        .iter()
        .map(|tri| vec![tri.point.x, tri.point.y, tri.point.z])
        .collect();
    let eigenvalues: Vec<Vec<f64>> = tris.iter().map(|tri| tri.eigenvalues.to_vec()).collect();
    let condition: Vec<f64> = tris.iter().map(|tri| tri.condition_number).collect();
    let in_front: Vec<bool> = tris.iter().map(|tri| tri.in_front_of_all_cameras).collect();

    let dict = PyDict::new(py);
    dict.set_item(
        "points",
        PyArray2::from_vec2(py, &points).map_err(|e| PyValueError::new_err(e.to_string()))?,
    )?;
    dict.set_item(
        "eigenvalues",
        PyArray2::from_vec2(py, &eigenvalues).map_err(|e| PyValueError::new_err(e.to_string()))?,
    )?;
    dict.set_item("condition_number", PyArray1::from_vec(py, condition))?;
    dict.set_item("in_front_of_all_cameras", PyArray1::from_vec(py, in_front))?;
    Ok(dict.into_any().unbind())
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(triangulate_batch, m)?)?;
    Ok(())
}
