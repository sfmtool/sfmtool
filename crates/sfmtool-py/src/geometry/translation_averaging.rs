// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for camera centres from pairwise baselines: the centre
//! solve, the reading the directions alone carry, the relative lengths the
//! two-view depths state and the orientation bit cheirality settles.

use numpy::PyUntypedArrayMethods;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::geometry::translation_averaging::{
    average_translations as core_average, direction_reading as core_direction_reading,
    orientation_reading as core_orientation_reading, relative_lengths as core_relative_lengths,
    AveragingCensus, DepthRows, OrientationRays, TranslationGraph, CG_STEPS, CG_TOL, IRLS_ROUNDS,
    LENGTH_IRLS_ROUNDS, MIN_TIED_ROWS,
};

/// How far a direction row may drift from unit length before the binding
/// refuses it. The rows come out of a null-space read, so they are unit to
/// round-off; anything past this is a caller mistake rather than arithmetic.
const UNIT_TOL: f64 = 1e-9;

/// The `(m, 2)` edge array as two index vectors, bounds-checked.
fn edge_indices(
    edges: &PyReadonlyArray2<'_, i64>,
    n_frames: usize,
) -> PyResult<(Vec<u32>, Vec<u32>)> {
    if edges.shape()[1] != 2 {
        return Err(PyValueError::new_err("edges must have shape (n_edge, 2)"));
    }
    let view = edges.as_array();
    let m = view.shape()[0];
    let mut ii = Vec::with_capacity(m);
    let mut jj = Vec::with_capacity(m);
    for e in 0..m {
        let (a, b) = (view[[e, 0]], view[[e, 1]]);
        for v in [a, b] {
            if v < 0 || v as usize >= n_frames {
                return Err(PyValueError::new_err(format!(
                    "edge index {v} at row {e} is outside n_frames = {n_frames}"
                )));
            }
        }
        if a == b {
            return Err(PyValueError::new_err(format!(
                "edge row {e} joins frame {a} to itself"
            )));
        }
        ii.push(a as u32);
        jj.push(b as u32);
    }
    Ok((ii, jj))
}

/// The `(m, 3)` direction array, flattened and checked for unit rows.
fn unit_directions(directions: &PyReadonlyArray2<'_, f64>, m: usize) -> PyResult<Vec<f64>> {
    if directions.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "directions must have shape (n_edge, 3)",
        ));
    }
    if directions.shape()[0] != m {
        return Err(PyValueError::new_err(
            "directions must have one row per edge",
        ));
    }
    let view = directions.as_array();
    let mut out = Vec::with_capacity(3 * m);
    for e in 0..m {
        let row = [view[[e, 0]], view[[e, 1]], view[[e, 2]]];
        let norm = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
        if !norm.is_finite() || (norm - 1.0).abs() > UNIT_TOL {
            return Err(PyValueError::new_err(format!(
                "direction row {e} has norm {norm}, which is not unit"
            )));
        }
        out.extend_from_slice(&row);
    }
    Ok(out)
}

/// One per-edge float array, checked for length.
fn per_edge<'py>(name: &str, values: &PyReadonlyArray1<'py, f64>, m: usize) -> PyResult<Vec<f64>> {
    if values.shape()[0] != m {
        return Err(PyValueError::new_err(format!(
            "{name} must have one entry per edge"
        )));
    }
    Ok(values.as_array().iter().copied().collect())
}

/// The census as the dict Python reads.
fn census_dict<'py>(py: Python<'py>, c: &AveragingCensus) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("lam_max", c.lam_max)?;
    dict.set_item("lam1_rel", c.lam1_rel)?;
    dict.set_item("lam2_rel", c.lam2_rel)?;
    dict.set_item("gap", c.gap)?;
    dict.set_item("n_null", c.n_null)?;
    dict.set_item("n_loose", c.n_loose)?;
    dict.set_item("n_free", c.n_free)?;
    dict.set_item("read_off_null", c.read_off_null)?;
    dict.set_item("n_lengths", c.n_lengths)?;
    dict.set_item("solved", c.solved)?;
    Ok(dict)
}

/// Camera centres from pairwise baselines, by weighted linear averaging.
///
/// Minimizes ``sum_ij w_ij || P_ij (c_j - c_i) ||^2 + a_ij (d_ij . (c_j - c_i)
/// - s L_ij)^2`` under the scale gauge ``sum_ij w_ij d_ij . (c_j - c_i) =
/// sum_ij w_ij`` and the shift gauge ``sum_j c_j = 0``, with the scale ``s``
/// that turns the relative lengths into distances eliminated in closed form.
/// The constellation is what the resulting form sends to zero, so the answer
/// is read off the form's null space where that null space is one dimension no
/// frame owns half of, and from the range solution otherwise.
///
/// ``rounds`` rounds of reweighting follow, each charging an edge's direction
/// residual and length slip against the median of that half over the graph.
/// Reweighting never raises a weight above the one passed in.
///
/// Args:
///     edges: (n_edge, 2) int64 frame indices, ``i`` then ``j``.
///     directions: (n_edge, 3) float64 unit direction from ``c_i`` to ``c_j``.
///     weights: (n_edge,) float64 direction weights.
///     lengths: (n_edge,) float64 relative baseline lengths on one common
///         scale, NaN where the edge states none. None is no length anywhere.
///     length_weights: (n_edge,) float64 length weights; an entry at or below
///         zero states no length. None is no length anywhere.
///     n_frames: How many frames the edge indices address.
///     rounds: Reweighting rounds.
///
/// Returns:
///     ``(centres, lambda, residual, census)``: ``centres`` (n_frames, 3)
///     float64 mean-centred at the gauge scale, ``lambda`` (n_edge,) the
///     projected baseline length of each edge (negative where the
///     constellation placed it backwards), ``residual`` (n_edge,) the part of
///     each baseline the direction says should not be there, and ``census`` a
///     dict of ``lam_max``, ``lam1_rel``, ``lam2_rel``, ``gap``, ``n_null``,
///     ``n_loose``, ``n_free``, ``read_off_null``, ``n_lengths`` and
///     ``solved``. The three arrays are empty when ``solved`` is False, which
///     is the graph stating no baseline at all.
#[pyfunction]
#[pyo3(signature = (edges, directions, weights, lengths=None, length_weights=None, *, n_frames, rounds=IRLS_ROUNDS))]
#[allow(clippy::too_many_arguments)]
pub fn average_translations<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    directions: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    lengths: Option<PyReadonlyArray1<'py, f64>>,
    length_weights: Option<PyReadonlyArray1<'py, f64>>,
    n_frames: usize,
    rounds: usize,
) -> PyResult<Py<PyAny>> {
    let (ii, jj) = edge_indices(&edges, n_frames)?;
    let m = ii.len();
    let d = unit_directions(&directions, m)?;
    let w = per_edge("weights", &weights, m)?;
    let ell = lengths
        .as_ref()
        .map(|v| per_edge("lengths", v, m))
        .transpose()?;
    let aa = length_weights
        .as_ref()
        .map(|v| per_edge("length_weights", v, m))
        .transpose()?;
    if rounds == 0 {
        return Err(PyValueError::new_err("rounds must be at least one"));
    }

    let out = py.detach(|| {
        core_average(
            TranslationGraph {
                edge_i: &ii,
                edge_j: &jj,
                directions: &d,
                weights: &w,
                lengths: ell.as_deref(),
                length_weights: aa.as_deref(),
                n_frames,
            },
            rounds,
        )
    });

    let flat: Vec<f64> = out.centres.iter().flatten().copied().collect();
    let centres = PyArray1::from_vec(py, flat)
        .reshape([out.centres.len(), 3])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let census = census_dict(py, &out.census)?;
    Ok((
        centres,
        PyArray1::from_vec(py, out.lambda),
        PyArray1::from_vec(py, out.residual),
        census,
    )
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

/// What the DIRECTIONS alone determine, before any length is read in.
///
/// The same form :func:`average_translations` builds, at the weights the edges
/// came with and with the length half empty, decomposed once and not
/// reweighted. It states what the graph's geometry determines on its own,
/// which is a property of the capture rather than of a solve: ``n_free > 0``
/// is a colinear path, whose spacing the directions leave undetermined.
///
/// Args:
///     edges: (n_edge, 2) int64 frame indices.
///     directions: (n_edge, 3) float64 unit directions.
///     weights: (n_edge,) float64 direction weights.
///     n_frames: How many frames the edge indices address.
///
/// Returns:
///     The census dict alone, with ``n_lengths`` zero and ``solved`` False
///     (no solve was posed).
#[pyfunction]
#[pyo3(signature = (edges, directions, weights, n_frames))]
pub fn direction_reading<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    directions: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    n_frames: usize,
) -> PyResult<Py<PyAny>> {
    let (ii, jj) = edge_indices(&edges, n_frames)?;
    let m = ii.len();
    let d = unit_directions(&directions, m)?;
    let w = per_edge("weights", &weights, m)?;
    let read = py.detach(|| {
        core_direction_reading(TranslationGraph {
            edge_i: &ii,
            edge_j: &jj,
            directions: &d,
            weights: &w,
            lengths: None,
            length_weights: None,
            n_frames,
        })
    });
    Ok(census_dict(py, &read)?.into_any().unbind())
}

/// One row index array, bounds-checked against `n`.
fn row_indices(name: &str, values: &PyReadonlyArray1<'_, i64>, n: usize) -> PyResult<Vec<u32>> {
    let mut out = Vec::with_capacity(values.shape()[0]);
    for (r, &v) in values.as_array().iter().enumerate() {
        if v < 0 || v as usize >= n {
            return Err(PyValueError::new_err(format!(
                "{name}[{r}] = {v} is outside [0, {n})"
            )));
        }
        out.push(v as u32);
    }
    Ok(out)
}

/// Relative baseline lengths, from the depths each pair's own solve implies.
///
/// The whole graph is one fit of ``log z(edge, frame, point) = D(frame, point)
/// - x(edge)``, with ``x`` the log baseline length and ``D`` the log world
/// depth. The ``D`` are eliminated rather than solved for, so the operator is
/// a pass over the rows and never a matrix, and the system is solved by
/// preconditioned conjugate gradient.
///
/// A row is TIED when another edge saw the same point from the same frame;
/// only tied rows relate one baseline to another. An edge with fewer than
/// ``min_tied`` tied rows states no length and comes back NaN, so a centre
/// solve constrains its direction only.
///
/// Args:
///     edge_of_row: (n_row,) int64 edge each depth came from.
///     frame_of_row: (n_row,) int64 frame each depth was read from.
///     point_of_row: (n_row,) int64 point each row saw.
///     depth_of_row: (n_row,) float64 depth, in units of its edge's baseline.
///         Must be positive: the caller supplies the rows already filtered.
///     n_edges: How many edges the rows index.
///     rounds: Rounds of the row reweighting.
///     min_tied: The fewest tied rows an edge needs to state a length.
///
/// Returns:
///     ``(lengths, scatter, n_tied)``: (n_edges,) float64 relative lengths
///     gauged to a median log length of zero (NaN where none is stated),
///     (n_edges,) float64 median absolute log residual of each edge's own rows
///     (NaN where the edge has none in the fit), and (n_edges,) int64 tied-row
///     counts.
#[pyfunction]
#[pyo3(signature = (edge_of_row, frame_of_row, point_of_row, depth_of_row, n_edges, rounds=LENGTH_IRLS_ROUNDS, min_tied=MIN_TIED_ROWS))]
#[allow(clippy::too_many_arguments)]
pub fn relative_lengths<'py>(
    py: Python<'py>,
    edge_of_row: PyReadonlyArray1<'py, i64>,
    frame_of_row: PyReadonlyArray1<'py, i64>,
    point_of_row: PyReadonlyArray1<'py, i64>,
    depth_of_row: PyReadonlyArray1<'py, f64>,
    n_edges: usize,
    rounds: usize,
    min_tied: usize,
) -> PyResult<Py<PyAny>> {
    let n_row = edge_of_row.shape()[0];
    for (name, len) in [
        ("frame_of_row", frame_of_row.shape()[0]),
        ("point_of_row", point_of_row.shape()[0]),
        ("depth_of_row", depth_of_row.shape()[0]),
    ] {
        if len != n_row {
            return Err(PyValueError::new_err(format!(
                "{name} must have one entry per row, got {len} against {n_row}"
            )));
        }
    }
    let ee = row_indices("edge_of_row", &edge_of_row, n_edges)?;
    let ff = row_indices("frame_of_row", &frame_of_row, usize::MAX)?;
    let pp = row_indices("point_of_row", &point_of_row, usize::MAX)?;
    let mut zz = Vec::with_capacity(n_row);
    for (r, &z) in depth_of_row.as_array().iter().enumerate() {
        if !z.is_finite() || z <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "depth_of_row[{r}] = {z} is not positive"
            )));
        }
        zz.push(z);
    }

    let out = py.detach(|| {
        core_relative_lengths(
            DepthRows {
                edge_of_row: &ee,
                frame_of_row: &ff,
                point_of_row: &pp,
                depth_of_row: &zz,
                n_edges,
            },
            rounds,
            min_tied,
        )
    });
    Ok((
        PyArray1::from_vec(py, out.lengths),
        PyArray1::from_vec(py, out.scatter),
        PyArray1::from_vec(py, out.n_tied),
    )
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

/// The one bit the pairwise directions cannot state.
///
/// Pairwise directions determine the constellation only up to the point
/// reflection ``c -> -c``, because the form the averaging builds is quadratic
/// in the centres and does not contain them. For every point with two or more
/// rays the point is solved at the given centres by the least-squares midpoint
/// over its rays and each ray's depth is read; the reading is the
/// parallax-weighted vote ``sum_points theta_widest * (n_front - n_behind)``,
/// so a point inside ``angular_bound`` (a bearing whose depth sign is a coin
/// toss) contributes nothing beyond its own small angle. ``angw < 0`` says the
/// constellation should be reflected. The reading is exactly antisymmetric
/// under ``c -> -c``.
///
/// Args:
///     centres: (n_frame, 3) float64 camera centres.
///     rays_world: (n_ray, 3) float64 unit world rays.
///     point_of_ray: (n_ray,) int64 point each ray saw.
///     frame_of_ray: (n_ray,) int64 frame each ray was seen from.
///     angular_bound: The angle, in radians, below which a point's cheirality
///         statement is a coin toss.
///
/// Returns:
///     A dict with ``angw``, ``obs_front``, ``obs_total``, ``obs_frac``,
///     ``angw_per_obs``, ``margin_frac`` and the point census ``pts``,
///     ``thin`` and ``behind``.
#[pyfunction]
#[pyo3(signature = (centres, rays_world, point_of_ray, frame_of_ray, angular_bound))]
pub fn orientation_reading<'py>(
    py: Python<'py>,
    centres: PyReadonlyArray2<'py, f64>,
    rays_world: PyReadonlyArray2<'py, f64>,
    point_of_ray: PyReadonlyArray1<'py, i64>,
    frame_of_ray: PyReadonlyArray1<'py, i64>,
    angular_bound: f64,
) -> PyResult<Py<PyAny>> {
    if centres.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "centres must have shape (n_frame, 3)",
        ));
    }
    if rays_world.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "rays_world must have shape (n_ray, 3)",
        ));
    }
    let n_frame = centres.shape()[0];
    let n_ray = rays_world.shape()[0];
    if point_of_ray.shape()[0] != n_ray || frame_of_ray.shape()[0] != n_ray {
        return Err(PyValueError::new_err(
            "point_of_ray and frame_of_ray must have one entry per ray",
        ));
    }
    let cs: Vec<f64> = centres.as_array().iter().copied().collect();
    let rw: Vec<f64> = rays_world.as_array().iter().copied().collect();
    let pof = row_indices("point_of_ray", &point_of_ray, usize::MAX)?;
    let fof = row_indices("frame_of_ray", &frame_of_ray, n_frame)?;

    let out = py.detach(|| {
        core_orientation_reading(
            OrientationRays {
                centres: &cs,
                rays_world: &rw,
                point_of_ray: &pof,
                frame_of_ray: &fof,
            },
            angular_bound,
        )
    });
    let dict = PyDict::new(py);
    dict.set_item("angw", out.angw)?;
    dict.set_item("obs_front", out.obs_front)?;
    dict.set_item("obs_total", out.obs_total)?;
    dict.set_item("obs_frac", out.obs_frac)?;
    dict.set_item("angw_per_obs", out.angw_per_obs)?;
    dict.set_item("margin_frac", out.margin_frac)?;
    dict.set_item("pts", out.points)?;
    dict.set_item("thin", out.thin)?;
    dict.set_item("behind", out.behind)?;
    Ok(dict.into_any().unbind())
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(average_translations, m)?)?;
    m.add_function(wrap_pyfunction!(direction_reading, m)?)?;
    m.add_function(wrap_pyfunction!(relative_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(orientation_reading, m)?)?;
    m.add("TRANSLATION_IRLS_ROUNDS", IRLS_ROUNDS)?;
    m.add("LENGTH_IRLS_ROUNDS", LENGTH_IRLS_ROUNDS)?;
    m.add("LENGTH_CG_STEPS", CG_STEPS)?;
    m.add("LENGTH_CG_TOL", CG_TOL)?;
    m.add("LENGTH_MIN_TIED_ROWS", MIN_TIED_ROWS)?;
    Ok(())
}
