// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the adjacency surfel normal fit.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::analysis::adjacency_surfel_normals::{
    estimate_adjacency_surfel_normals as core_estimate_adjacency_surfel_normals,
    AdjacencySurfelParams, ExtraNeighbours,
};

/// Turn the caller-supplied `{point: (k, 3) array}` mapping into CSR over the
/// cloud.
fn extras_to_csr(extras: &Bound<'_, PyDict>, n_points: usize) -> PyResult<ExtraNeighbours> {
    let mut rows: Vec<Vec<[f64; 3]>> = vec![Vec::new(); n_points];
    for (key, value) in extras.iter() {
        let p: usize = key
            .extract()
            .map_err(|_| PyValueError::new_err("extras keys must be non-negative point indices"))?;
        if p >= n_points {
            return Err(PyValueError::new_err(format!(
                "extras contains point {p}, out of range for {n_points} points"
            )));
        }
        let block: PyReadonlyArray2<f64> = value.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "extras[{p}] must be a (k, 3) float64 array of neighbour positions"
            ))
        })?;
        if block.shape()[1] != 3 {
            return Err(PyValueError::new_err(format!(
                "extras[{p}] must have shape (k, 3)"
            )));
        }
        let data = to_contiguous!(block);
        rows[p].extend(data.as_chunks::<3>().0.iter().map(|c| [c[0], c[1], c[2]]));
    }

    let mut out = ExtraNeighbours {
        offsets: Vec::with_capacity(n_points + 1),
        positions: Vec::new(),
    };
    out.offsets.push(0);
    for row in rows {
        out.positions.extend_from_slice(&row);
        out.offsets.push(out.positions.len() as u32);
    }
    Ok(out)
}

/// Fit a surfel normal at each selected point from the directions to its
/// image-space neighbours.
///
/// The plane passes through the point itself and is fitted on the unit
/// directions to its adjacency-graph neighbours, so each neighbour contributes
/// its angular deviation once rather than in proportion to its distance. A
/// Tukey-redescending IRLS loop discards neighbours that belong to another
/// surface. Nothing is ever substituted for a normal that cannot be fitted:
/// unselected points, and selected points with fewer than two usable
/// neighbours, come back ``NaN`` with ``determined = False``.
///
/// Args:
///     positions: Per point, shape ``(P, 3)`` float64.
///     offsets: Adjacency CSR row boundaries, shape ``(P + 1,)`` uint32.
///     neighbours: Adjacency CSR neighbour indices, shape ``(E,)`` uint32.
///     view_dirs: Per point, shape ``(P, 3)`` float64 — the reference direction
///         the normal's sign and the sector basis are taken from, typically the
///         mean unit direction toward the observing cameras.
///     selected: Per point, shape ``(P,)`` bool.
///     extras: Optional ``{point index: (k, 3) float64 array}`` of synthesized
///         neighbour positions. They enter the fit exactly like graph
///         neighbours; entries for unselected points are ignored.
///     irls_iters: IRLS passes before the final solve.
///     tukey_c: Tukey biweight tuning constant.
///     sigma_floor_deg: Floor on the robust scale, as an angle.
///     n_sectors: Equal tangent sectors the angular coverage is measured in.
///     det_n_eff: ``determined`` floor on the effective neighbour count.
///     det_sectors: ``determined`` floor on the occupied sector count.
///     det_aniso: ``determined`` floor on the in-plane anisotropy.
///
/// Returns:
///     A dict of dense per-point arrays: ``normals`` ``(P, 3)`` float64, the
///     float64 diagnostics ``n_eff``, ``anisotropy``, ``sectors``,
///     ``sigma_deg``, ``resid_deg`` and ``n_support``, and the bool verdict
///     ``determined``.
#[pyfunction]
#[pyo3(signature = (
    positions,
    offsets,
    neighbours,
    view_dirs,
    selected,
    *,
    extras=None,
    irls_iters=3,
    tukey_c=4.685,
    sigma_floor_deg=2.0,
    n_sectors=8,
    det_n_eff=4.0,
    det_sectors=3,
    det_aniso=0.10,
))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_adjacency_surfel_normals(
    py: Python<'_>,
    positions: PyReadonlyArray2<f64>,
    offsets: PyReadonlyArray1<u32>,
    neighbours: PyReadonlyArray1<u32>,
    view_dirs: PyReadonlyArray2<f64>,
    selected: PyReadonlyArray1<bool>,
    extras: Option<&Bound<'_, PyDict>>,
    irls_iters: u32,
    tukey_c: f64,
    sigma_floor_deg: f64,
    n_sectors: u32,
    det_n_eff: f64,
    det_sectors: u32,
    det_aniso: f64,
) -> PyResult<Py<PyAny>> {
    let n_points = positions.shape()[0];

    if positions.shape()[1] != 3 {
        return Err(PyValueError::new_err("positions must have shape (P, 3)"));
    }
    if view_dirs.shape() != [n_points, 3] {
        return Err(PyValueError::new_err(
            "view_dirs must have shape (P, 3) matching positions",
        ));
    }
    if selected.shape()[0] != n_points {
        return Err(PyValueError::new_err(format!(
            "selected has {} entries but positions has {n_points}",
            selected.shape()[0]
        )));
    }
    if n_points > 0 && offsets.shape()[0] != n_points + 1 {
        return Err(PyValueError::new_err(format!(
            "offsets has {} entries but positions has {n_points}; expected {}",
            offsets.shape()[0],
            n_points + 1
        )));
    }

    let positions_data = to_contiguous!(positions);
    let offsets_data = to_contiguous!(offsets);
    let neighbours_data = to_contiguous!(neighbours);
    let view_data = to_contiguous!(view_dirs);
    let selected_data = to_contiguous!(selected);

    if n_points > 0 {
        if offsets_data.windows(2).any(|w| w[0] > w[1]) {
            return Err(PyValueError::new_err("offsets must be non-decreasing"));
        }
        let last = *offsets_data.last().expect("offsets is non-empty") as usize;
        if last != neighbours_data.len() {
            return Err(PyValueError::new_err(format!(
                "offsets ends at {last} but neighbours has {} entries",
                neighbours_data.len()
            )));
        }
        if let Some(&bad) = neighbours_data.iter().find(|&&q| q as usize >= n_points) {
            return Err(PyValueError::new_err(format!(
                "neighbours contains {bad}, out of range for {n_points} points"
            )));
        }
    }

    let point_positions: Vec<[f64; 3]> = positions_data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let point_view_dirs: Vec<[f64; 3]> = view_data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    let extra_neighbours = match extras {
        Some(dict) => extras_to_csr(dict, n_points)?,
        None => ExtraNeighbours::none(),
    };

    let params = AdjacencySurfelParams {
        irls_iters,
        tukey_c,
        sigma_floor_deg,
        n_sectors,
        det_n_eff,
        det_sectors,
        det_aniso,
    };

    let fit = py.detach(|| {
        core_estimate_adjacency_surfel_normals(
            &point_positions,
            &offsets_data,
            &neighbours_data,
            &point_view_dirs,
            &selected_data,
            &extra_neighbours,
            &params,
        )
    });

    // Built flat and reshaped so an empty cloud still comes back as `(0, 3)`.
    let normals_flat: Vec<f64> = fit.normals.iter().flatten().copied().collect();
    let normals = PyArray1::from_vec(py, normals_flat).reshape([n_points, 3])?;

    let dict = PyDict::new(py);
    dict.set_item("normals", normals)?;
    dict.set_item("n_eff", PyArray1::from_vec(py, fit.n_eff))?;
    dict.set_item("anisotropy", PyArray1::from_vec(py, fit.anisotropy))?;
    dict.set_item("sectors", PyArray1::from_vec(py, fit.sectors))?;
    dict.set_item("sigma_deg", PyArray1::from_vec(py, fit.sigma_deg))?;
    dict.set_item("resid_deg", PyArray1::from_vec(py, fit.resid_deg))?;
    dict.set_item("n_support", PyArray1::from_vec(py, fit.n_support))?;
    dict.set_item("determined", PyArray1::from_vec(py, fit.determined))?;
    Ok(dict.into_any().unbind())
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_adjacency_surfel_normals, m)?)?;
    Ok(())
}
