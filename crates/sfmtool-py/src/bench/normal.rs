// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the bench's three normal steps: *Fit Normal*
//! (`fit_normal`), *Finite Diff Normal* and today's *Grid Plane Normal*
//! (`finite_difference_normal` with its two layouts), and the cell-registered
//! grid-plane normal (`grid_plane_normal`).
//!
//! Each takes a track-stage track and gives back the turned track and a report,
//! as every other step does, and refuses with the core's sentence.

use std::sync::Arc;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::bench::{
    finite_difference_normal as core_finite_difference_normal, fit_normal as core_fit_normal,
    grid_plane_normal as core_grid_plane_normal, FiniteDifferenceOptions, FitNormalOptions,
    GridPlaneAcceptance, GridPlaneCells, GridPlaneIteration, GridPlaneNormalOptions,
    GridPlaneOutcome, NormalEstimate, NormalReport, PieceLayout,
};
use sfmtool_core::patch::cell_plane_normals::{CellPlaneStatus, NormalDeterminacy};
use sfmtool_core::patch::cluster_refine::{CellStatus, LoopStop};
use sfmtool_core::progress::Progress;
use sfmtool_matches_format::ClusterCellStatus;

use super::{
    evaluate_report_dict, fit_options, refused, tilt_report_dict, views_of, PyEditableTrack,
};
use crate::patches::views::{resolve_pyramids, PosedViews};
use crate::reconstruction::edited::PyEditedReconstruction;

/// The word for why a cell loop stopped.
fn loop_stop_name(stop: LoopStop) -> &'static str {
    match stop {
        LoopStop::NotRun => "not_run",
        LoopStop::Converged => "converged",
        LoopStop::Cap => "cap",
        LoopStop::Rejected => "rejected",
        LoopStop::Oscillation => "oscillation",
        LoopStop::Measured => "measured",
    }
}

/// The word for a grid-plane outcome.
fn outcome_name(outcome: GridPlaneOutcome) -> &'static str {
    match outcome {
        GridPlaneOutcome::Turned => "turned",
        GridPlaneOutcome::Undetermined => "undetermined",
        GridPlaneOutcome::Unstable => "unstable",
    }
}

/// A cell's registration status as the `.matches` cell-status legend names it;
/// `CellStatus`'s discriminants are that legend's codes.
fn cell_status_name(status: CellStatus) -> &'static str {
    ClusterCellStatus::NAMES[status as usize]
}

/// Nine per-cell words, `[row * 3 + col]`, as a nested `[row][col]` list.
fn word_grid<'py>(py: Python<'py>, words: [&'static str; 9]) -> PyResult<Bound<'py, PyList>> {
    let rows: Vec<Vec<&str>> = words.chunks(3).map(|r| r.to_vec()).collect();
    PyList::new(py, rows)
}

/// The word for which axes of a normal the cells fixed.
fn determinacy_name(determinacy: NormalDeterminacy) -> &'static str {
    match determinacy {
        NormalDeterminacy::BothAxes => "both_axes",
        NormalDeterminacy::OneAxis { .. } => "one_axis",
        NormalDeterminacy::None => "none",
    }
}

/// The dict form of one render of the loop.
fn iteration_dict<'py>(
    py: Python<'py>,
    entry: &GridPlaneIteration,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("score", entry.score)?;
    d.set_item("determinacy", determinacy_name(entry.determinacy))?;
    d.set_item("in_plane", entry.in_plane)?;
    d.set_item("normal_precision_deg", entry.normal_precision_deg)?;
    d.set_item("gated", entry.gated)?;
    d.set_item(
        "proposed_normal",
        entry
            .proposed_normal
            .map(|n| PyArray1::from_vec(py, n.to_vec())),
    )?;
    d.set_item("largest_move_px", entry.largest_move_px)?;
    d.set_item("taken", entry.taken)?;
    Ok(d)
}

/// The acceptance rule `word` names.
fn parse_acceptance(word: &str) -> PyResult<GridPlaneAcceptance> {
    match word {
        "windowed_zncc" => Ok(GridPlaneAcceptance::WindowedZncc),
        "none" => Ok(GridPlaneAcceptance::None),
        other => Err(PyValueError::new_err(format!(
            "unknown acceptance: {other:?} (expected windowed_zncc|none)"
        ))),
    }
}

/// The dict form of the cells' report.
fn cells_dict<'py>(py: Python<'py>, cells: &GridPlaneCells) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", "grid_plane_cells")?;
    d.set_item("outcome", outcome_name(cells.outcome))?;
    d.set_item("stop", loop_stop_name(cells.stop))?;
    let free_axis = match cells.determinacy {
        NormalDeterminacy::OneAxis { free_axis } => Some(free_axis),
        NormalDeterminacy::BothAxes | NormalDeterminacy::None => None,
    };
    d.set_item("determinacy", determinacy_name(cells.determinacy))?;
    d.set_item(
        "free_axis",
        free_axis.map(|a| PyArray1::from_vec(py, a.to_vec())),
    )?;
    d.set_item(
        "cell_status",
        word_grid(
            py,
            cells
                .cell_status
                .map(|s| CellPlaneStatus::NAMES[s as usize]),
        )?,
    )?;
    d.set_item(
        "cell_rays",
        Array2::from_shape_vec((3, 3), cells.cell_rays.to_vec())
            .expect("nine values in a 3x3")
            .into_pyarray(py),
    )?;
    d.set_item(
        "cell_weight",
        Array2::from_shape_vec((3, 3), cells.cell_weight.to_vec())
            .expect("nine values in a 3x3")
            .into_pyarray(py),
    )?;
    d.set_item("in_plane", cells.in_plane())?;
    let view_cells = PyList::empty(py);
    for row in &cells.view_cells {
        view_cells.append(word_grid(py, row.map(cell_status_name))?)?;
    }
    d.set_item("view_cells", view_cells)?;
    d.set_item("plane_rms", cells.plane_rms)?;
    d.set_item("plane_offset", cells.plane_offset)?;
    d.set_item("working_resolution", cells.working_resolution)?;
    d.set_item("iterations", cells.iterations)?;
    d.set_item("score_before", cells.score_before)?;
    d.set_item("score_after", cells.score_after)?;
    d.set_item("normal_precision_deg", cells.normal_precision_deg)?;
    d.set_item("view_baseline_deg", cells.view_baseline_deg)?;
    let trace = PyList::empty(py);
    for entry in &cells.trace {
        trace.append(iteration_dict(py, entry)?)?;
    }
    d.set_item("trace", trace)?;
    Ok(d)
}

/// The dict form of where a normal came from, `kind` naming which estimate.
fn estimate_dict<'py>(py: Python<'py>, estimate: &NormalEstimate) -> PyResult<Bound<'py, PyDict>> {
    let d = match estimate {
        &NormalEstimate::Photometric {
            before,
            after,
            views,
        } => {
            let d = PyDict::new(py);
            d.set_item("kind", "photometric")?;
            d.set_item("before", before)?;
            d.set_item("after", after)?;
            d.set_item("views", views)?;
            d
        }
        &NormalEstimate::FiniteDifference {
            pieces,
            u,
            v,
            both_axes,
        } => {
            let d = PyDict::new(py);
            d.set_item("kind", "finite_difference")?;
            d.set_item("pieces", pieces)?;
            d.set_item("u", u)?;
            d.set_item("v", v)?;
            d.set_item("both_axes", both_axes)?;
            d
        }
        &NormalEstimate::GridPlane {
            pieces,
            fitted,
            both_axes,
            off_plane,
        } => {
            let d = PyDict::new(py);
            d.set_item("kind", "grid_plane")?;
            d.set_item("pieces", pieces)?;
            d.set_item("fitted", fitted)?;
            d.set_item("both_axes", both_axes)?;
            d.set_item("off_plane", off_plane)?;
            d
        }
        NormalEstimate::GridPlaneCells(cells) => cells_dict(py, cells)?,
    };
    Ok(d)
}

/// The dict form of one normal step's report.
fn normal_report_dict<'py>(py: Python<'py>, report: &NormalReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let n = report.tilt.normal;
    d.set_item("normal", PyArray1::from_vec(py, vec![n.x, n.y, n.z]))?;
    d.set_item("estimate", estimate_dict(py, &report.estimate)?)?;
    d.set_item("tilt", tilt_report_dict(py, &report.tilt)?)?;
    d.set_item("evaluate", evaluate_report_dict(py, &report.evaluate)?)?;
    d.set_item("text", report.to_string())?;
    Ok(d)
}

/// The layout `word` names.
fn parse_layout(word: &str) -> PyResult<PieceLayout> {
    match word {
        "cross" => Ok(PieceLayout::Cross),
        "grid" => Ok(PieceLayout::Grid),
        other => Err(PyValueError::new_err(format!(
            "unknown layout: {other:?} (expected cross|grid)"
        ))),
    }
}

/// Turn a track-stage `track`'s patch to the normal its ``in`` sightings agree
/// on best, by the photometric normal search: the bench's *Fit Normal*.
///
/// The centre does not move. The step ends as a fit does, reading the turned
/// track back with its bitmap rendered again. ``images`` and the four
/// keywords are :func:`fit`'s; the keywords set the reading the step ends with.
///
/// Returns ``(EditableTrack, report)``; the report is described under
/// :func:`grid_plane_normal`, with an ``estimate`` of ``kind``
/// ``"photometric"`` carrying ``before`` and ``after`` (the consensus ZNCC at
/// the normal the patch had and at the one found) and ``views``. Raises
/// ``ValueError`` with the reason when the step is refused.
#[pyfunction]
#[pyo3(signature = (
    track,
    edited,
    images,
    *,
    max_seed_offset_px = None,
    max_cache_bytes = None,
    sigma_px = None,
    depth_likelihood_ratio_threshold = None,
))]
#[allow(clippy::too_many_arguments)]
fn fit_normal(
    py: Python<'_>,
    track: &PyEditableTrack,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    max_seed_offset_px: Option<f64>,
    max_cache_bytes: Option<usize>,
    sigma_px: Option<f64>,
    depth_likelihood_ratio_threshold: Option<f64>,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let views = views_of(&posed, &pyramids);
    let options = FitNormalOptions {
        fit: fit_options(
            max_seed_offset_px,
            max_cache_bytes,
            sigma_px,
            depth_likelihood_ratio_threshold,
        ),
        ..FitNormalOptions::default()
    };
    let (next, report) = core_fit_normal(
        &track.inner,
        &edited.inner,
        &views,
        &options,
        &Progress::none(),
    )
    .map_err(refused)?;
    let d = normal_report_dict(py, &report)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// Turn a track-stage `track`'s patch to the plane through where smaller
/// pieces of it fit: the bench's *Finite Diff Normal* (``layout="cross"``)
/// and today's *Grid Plane Normal* (``layout="grid"``).
///
/// ``pieces`` (2 to 8, default 2) is how many pieces the patch is cut into
/// along each axis: a row of them through the centre along each axis for
/// ``"cross"``, ``pieces x pieces`` tiling the patch for ``"grid"``.
/// ``overlap`` (0 to 0.9, default 0) is how much two neighbouring pieces
/// overlap, as a fraction of a piece's side. Track View's Grid Plane Normal
/// runs ``layout="grid", pieces=2, overlap=0.0``. Each piece is given a full
/// :func:`fit`; a piece whose fit fails or which the fit puts at infinity is
/// left out. The centre does not move. ``images`` and the four keywords are
/// :func:`fit`'s, and set both the pieces' fits and the reading the step ends
/// with.
///
/// Returns ``(EditableTrack, report)``; the report is described under
/// :func:`grid_plane_normal`, with an ``estimate`` of ``kind``
/// ``"finite_difference"`` (``pieces``, ``u`` and ``v``, the pieces fitted
/// along each axis, and ``both_axes``) or ``"grid_plane"`` (``pieces``,
/// ``fitted``, ``both_axes`` and ``off_plane``, the rms distance of the fitted
/// centres from their plane in half-lengths, ``NaN`` with no plane). Raises
/// ``ValueError`` with the reason when the step is refused, including too few
/// pieces fitting to fix a line.
#[pyfunction]
#[pyo3(signature = (
    track,
    edited,
    images,
    *,
    layout = "cross",
    pieces = 2,
    overlap = 0.0,
    max_seed_offset_px = None,
    max_cache_bytes = None,
    sigma_px = None,
    depth_likelihood_ratio_threshold = None,
))]
#[allow(clippy::too_many_arguments)]
fn finite_difference_normal(
    py: Python<'_>,
    track: &PyEditableTrack,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    layout: &str,
    pieces: usize,
    overlap: f64,
    max_seed_offset_px: Option<f64>,
    max_cache_bytes: Option<usize>,
    sigma_px: Option<f64>,
    depth_likelihood_ratio_threshold: Option<f64>,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let options = FiniteDifferenceOptions {
        layout: parse_layout(layout)?,
        pieces,
        overlap,
        fit: fit_options(
            max_seed_offset_px,
            max_cache_bytes,
            sigma_px,
            depth_likelihood_ratio_threshold,
        ),
    };
    // Refuse the settings before the photographs are decoded.
    options.check().map_err(refused)?;
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let views = views_of(&posed, &pyramids);
    let (next, report) = core_finite_difference_normal(
        &track.inner,
        &edited.inner,
        &views,
        &options,
        &Progress::none(),
    )
    .map_err(refused)?;
    let d = normal_report_dict(py, &report)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// Turn a track-stage `track`'s patch to the plane through the nine cells of
/// its ZNCC grid, each registered in every ``in`` view.
///
/// Each ``in`` view is rendered once per iteration through the patch's plane,
/// and each cell is registered against the reference view's by a shift search
/// within ``cell_shift_bound_px`` (grid px of the working grid, default 2.0).
/// A cell is refused in a view when its peak is flatter than
/// ``min_cell_curvature`` (ZNCC per grid px², default 0.02), its ZNCC at the
/// optimum is below ``min_cell_zncc`` (default 0.8), its best shift lies on
/// the bound, or the view's robust affine fit gives it no weight. The
/// surviving cells are triangulated and a weighted, robust plane fitted
/// through them; the patch is turned toward it and measured again, until a
/// turn moves no cell centre by more than ``update_tolerance_px`` (grid px,
/// default 0.05) or ``max_iterations`` renders (default 5) are made.
/// ``acceptance`` is the rule a rendered turn is kept by:
/// ``"windowed_zncc"`` (the default), kept only when the views' mean windowed
/// ZNCC against the reference falls by no more than a small tolerance and
/// not below its starting value, or ``"none"``, every turn kept.
/// ``max_normal_precision_deg`` (default ``None``) is a bar on the plane's
/// predicted precision in degrees: a plane predicted less precise proposes
/// no turn, and the step ends ``"undetermined"``. ``working_resolution`` is
/// the side of the working grid in samples; the
/// default ``None`` follows the reference view's footprint of the patch,
/// about one sample per photograph pixel, clamped to ``R`` to ``2R`` for
/// ``R`` the bitmap's resolution. Every default is the Rust step's.
///
/// Only a loop that converges with at least one axis fixed turns the patch
/// (``outcome`` ``"turned"``); every other ending leaves the normal as it was
/// given and is reported, not refused. The centre does not move. ``images``
/// and the last four keywords are :func:`fit`'s, and set the reading the step
/// ends with.
///
/// Returns ``(EditableTrack, report)``. Every normal step's report carries
/// ``normal`` (the one the patch now shows), ``estimate`` (a dict whose
/// ``kind`` names where the normal came from), ``tilt`` (the turn, as
/// :func:`tilt_patch` reports it), ``evaluate`` (the reading the step ended
/// with, as :func:`evaluate` reports it) and ``text`` (the sentence the viewer
/// shows). Here ``estimate`` has ``kind`` ``"grid_plane_cells"`` and carries
/// ``outcome`` (``"turned"``, ``"undetermined"`` or ``"unstable"``), ``stop``
/// (``"not_run"``, ``"converged"``, ``"cap"``, ``"rejected"``,
/// ``"oscillation"`` or ``"measured"``), ``determinacy`` (``"both_axes"``,
/// ``"one_axis"`` or ``"none"``) with ``free_axis`` (a unit 3-vector under
/// one axis, else ``None``), ``cell_status`` (``[row][col]`` words of the
/// plane fit: ``"in_plane"``, ``"plane_outlier"``, ``"positioned"``,
/// ``"too_few_rays"``, ``"narrow_baseline"``, ``"behind_camera"``),
/// ``cell_rays`` (``(3, 3)`` uint32) and ``cell_weight`` (``(3, 3)``
/// float64), ``in_plane`` (how many cells the plane gave weight),
/// ``view_cells`` (per observation, ``[row][col]`` registration words:
/// ``"fitted"``, ``"refused_curvature"``, ``"refused_zncc"``,
/// ``"not_attempted"``, ``"refused_bound"``, ``"refused_outlier"``),
/// ``plane_rms`` and ``plane_offset`` (world units, ``NaN`` where there is no
/// plane), ``working_resolution``, ``iterations`` (renders made), and
/// ``score_before`` and ``score_after`` (the acceptance score at the start
/// and at the last render kept), ``normal_precision_deg`` (the last kept
/// render's plane's predicted precision in degrees, along its weaker in-plane
/// axis under both axes and along the fixed one under one axis, ``NaN``
/// with no axis fixed), ``view_baseline_deg`` (the largest angle between two
/// ``in`` sightings' directions from the patch centre) and ``trace``, one
/// dict per render in order with ``score``, ``determinacy``, ``in_plane``,
/// ``normal_precision_deg``, ``gated`` (the precision bar stopped its turn),
/// ``proposed_normal`` (the normal its turn reaches, or ``None``),
/// ``largest_move_px`` (grid px, ``NaN`` with no turn) and ``taken``.
/// Raises ``ValueError`` with the reason when the step is refused.
#[pyfunction]
#[pyo3(signature = (
    track,
    edited,
    images,
    *,
    working_resolution = None,
    cell_shift_bound_px = None,
    min_cell_zncc = None,
    min_cell_curvature = None,
    update_tolerance_px = None,
    max_iterations = None,
    acceptance = "windowed_zncc",
    max_normal_precision_deg = None,
    max_seed_offset_px = None,
    max_cache_bytes = None,
    sigma_px = None,
    depth_likelihood_ratio_threshold = None,
))]
#[allow(clippy::too_many_arguments)]
fn grid_plane_normal(
    py: Python<'_>,
    track: &PyEditableTrack,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    working_resolution: Option<u32>,
    cell_shift_bound_px: Option<f32>,
    min_cell_zncc: Option<f32>,
    min_cell_curvature: Option<f32>,
    update_tolerance_px: Option<f32>,
    max_iterations: Option<u8>,
    acceptance: &str,
    max_normal_precision_deg: Option<f64>,
    max_seed_offset_px: Option<f64>,
    max_cache_bytes: Option<usize>,
    sigma_px: Option<f64>,
    depth_likelihood_ratio_threshold: Option<f64>,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let defaults = GridPlaneNormalOptions::default();
    let options = GridPlaneNormalOptions {
        working_resolution,
        cell_shift_bound_px: cell_shift_bound_px.unwrap_or(defaults.cell_shift_bound_px),
        min_cell_zncc: min_cell_zncc.unwrap_or(defaults.min_cell_zncc),
        min_cell_curvature: min_cell_curvature.unwrap_or(defaults.min_cell_curvature),
        update_tolerance_px: update_tolerance_px.unwrap_or(defaults.update_tolerance_px),
        max_iterations: max_iterations.unwrap_or(defaults.max_iterations),
        acceptance: parse_acceptance(acceptance)?,
        max_normal_precision_deg,
        fit: fit_options(
            max_seed_offset_px,
            max_cache_bytes,
            sigma_px,
            depth_likelihood_ratio_threshold,
        ),
        ..defaults
    };
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let views = views_of(&posed, &pyramids);
    let (next, report) = core_grid_plane_normal(
        &track.inner,
        &edited.inner,
        &views,
        &options,
        &Progress::none(),
    )
    .map_err(refused)?;
    let d = normal_report_dict(py, &report)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// Register the normal steps on the `sfmtool.bench` submodule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_normal, m)?)?;
    m.add_function(wrap_pyfunction!(finite_difference_normal, m)?)?;
    m.add_function(wrap_pyfunction!(grid_plane_normal, m)?)?;
    Ok(())
}
