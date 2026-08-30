// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for per-edge baseline directions read off ray coplanarity
//! with the rotations held.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::geometry::baseline_direction::{
    baseline_directions as core_baseline_directions, BaselineTrim,
};

/// The baseline direction of every edge of a graph, from ray coplanarity.
///
/// With both rotations held, the baseline ``b = c_j - c_i`` is coplanar with
/// every point's two world rays: ``b . (u_i x u_j) = 0``, so ``b`` is the null
/// space of the matrix whose rows are those unit normals. A row whose parallax
/// is inside ``tol_rad`` carries no baseline and is DROPPED, not down-weighted.
/// The null space is refit on its own best fraction for ``rounds`` rounds, and
/// its sign is fixed by cheirality.
///
/// Edges are flattened CSR-style: edge ``e`` owns rows
/// ``offsets[e]:offsets[e+1]`` of ``rays_i`` and ``rays_j``.
///
/// Args:
///     rays_i: (n_row, 3) float64 unit world rays of the first frame.
///     rays_j: (n_row, 3) float64 unit world rays of the second frame.
///     offsets: (n_edge + 1,) int64 CSR edge boundaries.
///     tol_rad: The angular bound below which a row states no baseline.
///     rounds: Refit rounds over the retained rows.
///     keep_fraction: The fraction of retained rows each round keeps, never
///         fewer than three.
///
/// Returns:
///     A dict of arrays, one entry per edge: ``stated`` (n_edge,) bool, False
///     where fewer than three rows cleared the bound and every other field is
///     meaningless; ``direction`` (n_edge, 3) float64; ``n_rows`` and
///     ``n_used`` (n_edge,) int64; ``condition``, ``parallax_median_deg``,
///     ``parallax_max_deg``, ``cheiral_fraction`` and ``residual_median_rad``
///     (n_edge,) float64.
#[pyfunction]
#[pyo3(signature = (rays_i, rays_j, offsets, tol_rad, rounds, keep_fraction))]
pub fn baseline_directions<'py>(
    py: Python<'py>,
    rays_i: PyReadonlyArray2<'py, f64>,
    rays_j: PyReadonlyArray2<'py, f64>,
    offsets: PyReadonlyArray1<'py, i64>,
    tol_rad: f64,
    rounds: usize,
    keep_fraction: f64,
) -> PyResult<Py<PyAny>> {
    if rays_i.shape()[1] != 3 || rays_j.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "rays_i and rays_j must have shape (n_row, 3)",
        ));
    }
    let n_row = rays_i.shape()[0];
    if rays_j.shape()[0] != n_row {
        return Err(PyValueError::new_err(
            "rays_i and rays_j must share the same length",
        ));
    }
    let ui = to_contiguous!(rays_i);
    let uj = to_contiguous!(rays_j);
    let off = to_contiguous!(offsets);

    let mut bounds: Vec<usize> = Vec::with_capacity(off.len());
    let mut prev = 0i64;
    for (k, &o) in off.iter().enumerate() {
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
        bounds.push(o as usize);
    }
    if prev as usize > n_row {
        return Err(PyValueError::new_err(format!(
            "offsets[-1] = {prev} exceeds the number of rows {n_row}"
        )));
    }

    let trim = BaselineTrim {
        tol_rad,
        rounds,
        keep_fraction,
    };
    let out = py.detach(|| core_baseline_directions(&ui, &uj, &bounds, trim));

    let n = out.len();
    let mut stated = Vec::with_capacity(n);
    let mut direction = Vec::with_capacity(n * 3);
    let mut n_rows = Vec::with_capacity(n);
    let mut n_used = Vec::with_capacity(n);
    let mut condition = Vec::with_capacity(n);
    let mut par_med = Vec::with_capacity(n);
    let mut par_max = Vec::with_capacity(n);
    let mut cheiral = Vec::with_capacity(n);
    let mut resid = Vec::with_capacity(n);
    for edge in &out {
        match edge {
            None => {
                stated.push(false);
                direction.extend_from_slice(&[f64::NAN; 3]);
                n_rows.push(0i64);
                n_used.push(0i64);
                condition.push(f64::NAN);
                par_med.push(f64::NAN);
                par_max.push(f64::NAN);
                cheiral.push(f64::NAN);
                resid.push(f64::NAN);
            }
            Some(b) => {
                stated.push(true);
                direction.extend_from_slice(&b.direction);
                n_rows.push(b.n_rows as i64);
                n_used.push(b.n_used as i64);
                condition.push(b.condition);
                par_med.push(b.parallax_median_deg);
                par_max.push(b.parallax_max_deg);
                cheiral.push(b.cheiral_fraction);
                resid.push(b.residual_median_rad);
            }
        }
    }

    let dict = PyDict::new(py);
    dict.set_item("stated", PyArray1::from_vec(py, stated))?;
    dict.set_item(
        "direction",
        PyArray1::from_vec(py, direction)
            .reshape([n, 3])
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    )?;
    dict.set_item("n_rows", PyArray1::from_vec(py, n_rows))?;
    dict.set_item("n_used", PyArray1::from_vec(py, n_used))?;
    dict.set_item("condition", PyArray1::from_vec(py, condition))?;
    dict.set_item("parallax_median_deg", PyArray1::from_vec(py, par_med))?;
    dict.set_item("parallax_max_deg", PyArray1::from_vec(py, par_max))?;
    dict.set_item("cheiral_fraction", PyArray1::from_vec(py, cheiral))?;
    dict.set_item("residual_median_rad", PyArray1::from_vec(py, resid))?;
    Ok(dict.into_any().unbind())
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(baseline_directions, m)?)?;
    Ok(())
}
