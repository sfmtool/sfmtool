// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the bench's normal steps: *Fit Normal*
//! (`fit_normal`), *Finite Diff Normal* and today's *Grid Plane Normal*
//! (`finite_difference_normal` with its two layouts), the cell-registered
//! grid-plane normal (`grid_plane_normal`) and the grown-plane normal
//! (`grown_plane_normal`).
//!
//! Each takes a track-stage track and gives back the turned track and a report,
//! as every other step does, and refuses with the core's sentence.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::bench::{
    finite_difference_normal as core_finite_difference_normal, fit_normal as core_fit_normal,
    grid_plane_normal as core_grid_plane_normal, grown_plane_normal as core_grown_plane_normal,
    FiniteDifferenceOptions, FitNormalOptions, GridPlaneAcceptance, GridPlaneCells,
    GridPlaneDetail, GridPlaneFitModel, GridPlaneIteration, GridPlaneNormalOptions,
    GridPlaneOutcome, GridPlaneRender, GridPlaneTile, GridPlaneView, GrownPlaneAttempt,
    GrownPlaneNormalOptions, GrownPlaneOutcome, GrownPlanePair, GrownPlanePick, GrownPlaneRefusal,
    GrownPlaneSequence, GrownPlaneStart, GrownPlaneTiltStart, GrownPlaneTryOutcome, GrownPlaneView,
    NormalEstimate, NormalReport, PieceLayout,
};
use sfmtool_core::patch::cell_plane_normals::{CellPlaneStatus, NormalDeterminacy};
use sfmtool_core::patch::cluster_refine::{CellStatus, LoopStop};
use sfmtool_core::progress::{Event, Progress};
use sfmtool_matches_format::ClusterCellStatus;

use super::{
    evaluate_report_dict, fit_options, refused, tilt_report_dict, views_of, PyEditableTrack,
};
use crate::patches::views::{resolve_pyramids, PosedViews};
use crate::phase_record::PhaseRecord;
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

/// Nine per-cell numbers, `[row * 3 + col]`, as a `(3, 3)` array.
fn cell_grid<T: Copy>(values: [T; 9]) -> Array2<T> {
    Array2::from_shape_vec((3, 3), values.to_vec()).expect("nine values in a 3x3")
}

/// Nine per-cell vectors of `N`, `[row * 3 + col]`, as a `(3, 3, N)` array.
fn cell_vectors<const N: usize>(values: [[f64; N]; 9]) -> Array3<f64> {
    Array3::from_shape_vec((3, 3, N), values.iter().flatten().copied().collect())
        .expect("nine vectors in a 3x3")
}

/// The free axis of a one-axis verdict, `None` otherwise.
fn free_axis_of(determinacy: NormalDeterminacy) -> Option<[f64; 3]> {
    match determinacy {
        NormalDeterminacy::OneAxis { free_axis } => Some(free_axis),
        NormalDeterminacy::BothAxes | NormalDeterminacy::None => None,
    }
}

/// The word for a view's affine-fit model.
fn fit_model_name(model: GridPlaneFitModel) -> &'static str {
    match model {
        GridPlaneFitModel::Affine => "affine",
        GridPlaneFitModel::Similarity => "similarity",
        GridPlaneFitModel::Shift => "shift",
    }
}

/// A tile's samples as a `(side, side, channels)` float32 array.
fn tile_array<'py>(py: Python<'py>, tile: &GridPlaneTile) -> Bound<'py, numpy::PyArray3<f32>> {
    let (side, channels) = (tile.side as usize, tile.channels as usize);
    Array3::from_shape_vec((side, side, channels), tile.samples.clone())
        .expect("side x side x channels samples")
        .into_pyarray(py)
}

/// The dict form of one non-reference view at one render.
fn view_detail_dict<'py>(py: Python<'py>, view: &GridPlaneView) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("observation", view.tile.observation)?;
    d.set_item("image", view.tile.image)?;
    d.set_item("tile", tile_array(py, &view.tile))?;
    d.set_item("score", view.score)?;
    let cells = &view.cells;
    d.set_item(
        "whole_shift",
        cell_vectors(cells.map(|c| c.whole_shift.map_or([f64::NAN; 2], |w| w.map(|v| v as f64))))
            .into_pyarray(py),
    )?;
    d.set_item(
        "shift",
        cell_vectors(cells.map(|c| c.shift)).into_pyarray(py),
    )?;
    d.set_item("zncc", cell_grid(cells.map(|c| c.zncc)).into_pyarray(py))?;
    d.set_item(
        "curvature",
        cell_grid(cells.map(|c| c.curvature)).into_pyarray(py),
    )?;
    d.set_item(
        "reading",
        word_grid(py, cells.map(|c| cell_status_name(c.reading)))?,
    )?;
    d.set_item(
        "status",
        word_grid(py, cells.map(|c| cell_status_name(c.status)))?,
    )?;
    d.set_item(
        "fit_weight",
        cell_grid(cells.map(|c| c.fit_weight)).into_pyarray(py),
    )?;
    let fit = match &view.fit {
        Some(fit) => {
            let f = PyDict::new(py);
            f.set_item("model", fit_model_name(fit.model))?;
            f.set_item(
                "a",
                Array2::from_shape_vec((2, 2), fit.a.iter().flatten().copied().collect())
                    .expect("a 2x2")
                    .into_pyarray(py),
            )?;
            f.set_item("b", PyArray1::from_vec(py, fit.b.to_vec()))?;
            Some(f)
        }
        None => None,
    };
    d.set_item("fit", fit)?;
    Ok(d)
}

/// The dict form of one render of the detail record, with the layout every
/// render shares.
fn render_detail_dict<'py>(
    py: Python<'py>,
    detail: &GridPlaneDetail,
    render: &GridPlaneRender,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let layout = &detail.layout;
    d.set_item("working_resolution", layout.working_resolution)?;
    d.set_item("margin", layout.margin)?;
    d.set_item("reach", layout.reach)?;
    d.set_item(
        "cell_bounds",
        PyArray1::from_vec(py, layout.cell_bounds.to_vec()),
    )?;
    d.set_item(
        "cell_centres",
        cell_vectors(layout.cell_centres).into_pyarray(py),
    )?;

    let patch_frame = PyDict::new(py);
    patch_frame.set_item(
        "center",
        PyArray1::from_vec(py, render.patch_frame.center.to_vec()),
    )?;
    patch_frame.set_item(
        "u_axis",
        PyArray1::from_vec(py, render.patch_frame.u_axis.to_vec()),
    )?;
    patch_frame.set_item(
        "v_axis",
        PyArray1::from_vec(py, render.patch_frame.v_axis.to_vec()),
    )?;
    patch_frame.set_item(
        "normal",
        PyArray1::from_vec(py, render.patch_frame.normal.to_vec()),
    )?;
    patch_frame.set_item(
        "half_extent",
        PyArray1::from_vec(py, render.patch_frame.half_extent.to_vec()),
    )?;
    d.set_item("patch_frame", patch_frame)?;

    let reference = match &render.reference {
        Some(tile) => {
            let r = PyDict::new(py);
            r.set_item("observation", tile.observation)?;
            r.set_item("image", tile.image)?;
            r.set_item("tile", tile_array(py, tile))?;
            Some(r)
        }
        None => None,
    };
    d.set_item("reference", reference)?;

    let views = PyList::empty(py);
    for view in &render.views {
        views.append(view_detail_dict(py, view)?)?;
    }
    d.set_item("views", views)?;

    let cells = &render.cells;
    let c = PyDict::new(py);
    c.set_item(
        "status",
        word_grid(
            py,
            std::array::from_fn(|k| CellPlaneStatus::NAMES[cells[k].status as usize]),
        )?,
    )?;
    c.set_item(
        "position",
        cell_vectors(std::array::from_fn(|k| cells[k].position)).into_pyarray(py),
    )?;
    for (name, value) in [
        ("residual_px", std::array::from_fn(|k| cells[k].residual_px)),
        ("sigma_px", std::array::from_fn(|k| cells[k].sigma_px)),
        (
            "normal_variance",
            std::array::from_fn(|k| cells[k].normal_variance),
        ),
        (
            "precision_weight",
            std::array::from_fn(|k| cells[k].precision_weight),
        ),
        (
            "tukey_weight",
            std::array::from_fn(|k| cells[k].tukey_weight),
        ),
        ("weight", std::array::from_fn(|k| cells[k].weight)),
    ] {
        c.set_item(name, cell_grid::<f64>(value).into_pyarray(py))?;
    }
    c.set_item(
        "ray_count",
        cell_grid(std::array::from_fn(|k| cells[k].rays.len() as u32)).into_pyarray(py),
    )?;
    d.set_item("cells", c)?;

    // Every ray of every cell, one row each, with the cell it belongs to.
    let rays: Vec<(usize, &sfmtool_core::bench::GridPlaneRay)> = cells
        .iter()
        .enumerate()
        .flat_map(|(k, cell)| cell.rays.iter().map(move |ray| (k, ray)))
        .collect();
    let n = rays.len();
    let r = PyDict::new(py);
    r.set_item(
        "cell",
        Array1::from_iter(rays.iter().map(|(k, _)| *k as u32)).into_pyarray(py),
    )?;
    r.set_item(
        "observation",
        Array1::from_iter(rays.iter().map(|(_, ray)| ray.observation as u64)).into_pyarray(py),
    )?;
    r.set_item(
        "image",
        Array1::from_iter(rays.iter().map(|(_, ray)| ray.image)).into_pyarray(py),
    )?;
    let rows = |f: &dyn Fn(&sfmtool_core::bench::GridPlaneRay) -> Vec<f64>, width: usize| {
        Array2::from_shape_vec(
            (n, width),
            rays.iter().flat_map(|(_, ray)| f(ray)).collect(),
        )
        .expect("one row per ray")
    };
    r.set_item("pixel", rows(&|ray| ray.pixel.to_vec(), 2).into_pyarray(py))?;
    r.set_item(
        "origin",
        rows(&|ray| ray.origin.to_vec(), 3).into_pyarray(py),
    )?;
    r.set_item(
        "direction",
        rows(&|ray| ray.direction.to_vec(), 3).into_pyarray(py),
    )?;
    r.set_item(
        "floor_px",
        Array1::from_iter(rays.iter().map(|(_, ray)| ray.floor_px)).into_pyarray(py),
    )?;
    d.set_item("rays", r)?;

    let plane = &render.plane;
    let iteration = &render.iteration;
    let p = PyDict::new(py);
    p.set_item("centroid", PyArray1::from_vec(py, plane.centroid.to_vec()))?;
    p.set_item(
        "fitted_normal",
        PyArray1::from_vec(py, plane.fitted_normal.to_vec()),
    )?;
    p.set_item("normal", PyArray1::from_vec(py, plane.normal.to_vec()))?;
    p.set_item("determinacy", determinacy_name(plane.determinacy))?;
    p.set_item(
        "free_axis",
        free_axis_of(plane.determinacy).map(|a| PyArray1::from_vec(py, a.to_vec())),
    )?;
    p.set_item(
        "scatter_eigenvalues",
        PyArray1::from_vec(py, plane.scatter_eigenvalues.to_vec()),
    )?;
    p.set_item("anisotropy", plane.anisotropy)?;
    p.set_item("n_eff", plane.n_eff)?;
    p.set_item("plane_rms", plane.plane_rms)?;
    p.set_item(
        "precision_deg",
        PyArray1::from_vec(py, plane.precision_deg.to_vec()),
    )?;
    p.set_item("score", iteration.score)?;
    p.set_item("normal_precision_deg", iteration.normal_precision_deg)?;
    p.set_item("gated", iteration.gated)?;
    p.set_item(
        "proposed_normal",
        iteration
            .proposed_normal
            .map(|n| PyArray1::from_vec(py, n.to_vec())),
    )?;
    p.set_item("largest_move_px", iteration.largest_move_px)?;
    p.set_item("taken", iteration.taken)?;
    d.set_item("plane", p)?;
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
    let free_axis = free_axis_of(cells.determinacy);
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
    if let Some(detail) = &cells.detail {
        let renders = PyList::empty(py);
        for render in &detail.renders {
            renders.append(render_detail_dict(py, detail, render)?)?;
        }
        d.set_item("detail", renders)?;
    }
    Ok(d)
}

/// The word for how the grown-plane step ended.
fn grown_outcome_name(outcome: GrownPlaneOutcome) -> &'static str {
    match outcome {
        GrownPlaneOutcome::Turned => "turned",
        GrownPlaneOutcome::NoneJoined => "none_joined",
        GrownPlaneOutcome::WeakTurn => "weak_turn",
    }
}

/// The word for which plane a grown-plane growth started from.
fn tilt_start_name(from: GrownPlaneTiltStart) -> &'static str {
    match from {
        GrownPlaneTiltStart::Starting => "starting",
        GrownPlaneTiltStart::Combined => "combined",
    }
}

/// The word for how a grown-plane view's first window found its shift.
fn start_name(start: GrownPlaneStart) -> &'static str {
    match start {
        GrownPlaneStart::NoSearch => "no_search",
        GrownPlaneStart::Keypoint => "keypoint",
        GrownPlaneStart::Widened => "widened",
        GrownPlaneStart::NoKeypoint => "no_keypoint",
    }
}

/// The word for why a grown-plane growth did not join.
fn refusal_name(refusal: GrownPlaneRefusal) -> &'static str {
    match refusal {
        GrownPlaneRefusal::UnderFloor => "under_floor",
        GrownPlaneRefusal::TooFar => "too_far",
        GrownPlaneRefusal::HeldUnderFloor => "held_under_floor",
        GrownPlaneRefusal::ViewLeftOut => "view_left_out",
    }
}

/// The word for what became of one try of a view.
fn try_outcome_name(outcome: GrownPlaneTryOutcome) -> &'static str {
    match outcome {
        GrownPlaneTryOutcome::Matched => "matched",
        GrownPlaneTryOutcome::LeftOut => "left_out",
    }
}

/// A fixed-length vector as a float64 array, `None` staying `None`.
fn maybe_vector<'py, const N: usize>(
    py: Python<'py>,
    value: Option<[f64; N]>,
) -> Option<Bound<'py, PyArray1<f64>>> {
    value.map(|v| PyArray1::from_vec(py, v.to_vec()))
}

/// The dict form of one growth of one view.
fn attempt_dict<'py>(py: Python<'py>, attempt: &GrownPlaneAttempt) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("start", start_name(attempt.start))?;
    d.set_item("tilt_start", tilt_start_name(attempt.tilt_start))?;
    d.set_item(
        "stage_zncc",
        PyArray1::from_vec(py, attempt.stage_zncc.clone()),
    )?;
    d.set_item("shift", PyArray1::from_vec(py, attempt.shift.to_vec()))?;
    d.set_item("normal", PyArray1::from_vec(py, attempt.normal.to_vec()))?;
    d.set_item("grid_zncc", attempt.grid_zncc)?;
    d.set_item("sigma_deg", attempt.sigma_deg)?;
    d.set_item("refusal", attempt.refusal.map(refusal_name))?;
    Ok(d)
}

/// The dict form of what the grown-plane step made of one observation,
/// `kind` naming which.
fn grown_view_dict<'py>(py: Python<'py>, view: &GrownPlaneView) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match view {
        GrownPlaneView::Reference { keypoint } => {
            d.set_item("kind", "reference")?;
            d.set_item("keypoint", maybe_vector(py, *keypoint))?;
        }
        GrownPlaneView::Joined {
            start,
            shift,
            grid_zncc,
            distance,
            held_zncc,
            was,
            keypoint,
        } => {
            d.set_item("kind", "joined")?;
            d.set_item("start", start_name(*start))?;
            d.set_item("shift", PyArray1::from_vec(py, shift.to_vec()))?;
            d.set_item("grid_zncc", *grid_zncc)?;
            d.set_item("distance", *distance)?;
            d.set_item("held_zncc", *held_zncc)?;
            d.set_item("was", maybe_vector(py, *was))?;
            d.set_item("keypoint", maybe_vector(py, *keypoint))?;
        }
        GrownPlaneView::ShiftOnly {
            start,
            shift,
            grid_zncc,
            was,
            keypoint,
        } => {
            d.set_item("kind", "shift_only")?;
            d.set_item("start", start_name(*start))?;
            d.set_item("shift", PyArray1::from_vec(py, shift.to_vec()))?;
            d.set_item("grid_zncc", *grid_zncc)?;
            d.set_item("was", maybe_vector(py, *was))?;
            d.set_item("keypoint", maybe_vector(py, *keypoint))?;
        }
        GrownPlaneView::LeftOut {
            reason,
            distance,
            held_zncc,
        } => {
            d.set_item("kind", "left_out")?;
            d.set_item("reason", refusal_name(*reason))?;
            d.set_item("distance", *distance)?;
            d.set_item("held_zncc", *held_zncc)?;
        }
        GrownPlaneView::NotAttempted => d.set_item("kind", "not_attempted")?,
    }
    Ok(d)
}

/// The dict form of one pair of views matched against each other.
fn pair_dict<'py>(py: Python<'py>, pair: &GrownPlanePair) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("observations", pair.observations.to_vec())?;
    d.set_item("images", pair.images.to_vec())?;
    d.set_item("angle_deg", pair.angle_deg)?;
    d.set_item("normal", PyArray1::from_vec(py, pair.normal.to_vec()))?;
    d.set_item(
        "relative_shift",
        PyArray1::from_vec(py, pair.relative_shift.to_vec()),
    )?;
    d.set_item("zncc", pair.zncc)?;
    d.set_item("sigma_deg", pair.sigma_deg)?;
    d.set_item("distance", pair.distance)?;
    d.set_item("kept", pair.refusal.is_none())?;
    d.set_item("refusal", pair.refusal.map(refusal_name))?;
    Ok(d)
}

/// The dict form of one pick of the selection.
fn pick_dict<'py>(py: Python<'py>, pick: &GrownPlanePick) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("observation", pick.observation)?;
    d.set_item("image", pick.image)?;
    d.set_item("angle_deg", pick.angle_deg)?;
    d.set_item(
        "offset_direction",
        PyArray1::from_vec(py, pick.offset_direction.to_vec()),
    )?;
    d.set_item("scale_px", pick.scale_px)?;
    d.set_item("predicted_weak", pick.predicted_weak)?;
    d.set_item("matched", pick.matched)?;
    d.set_item("rung", pick.rung)?;
    d.set_item("sigma_after_deg", pick.sigma_after_deg)?;
    Ok(d)
}

/// The dict form of the grown-plane step's sequence of views.
fn sequence_dict<'py>(
    py: Python<'py>,
    sequence: &GrownPlaneSequence,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", "grown_plane")?;
    d.set_item("reference", sequence.reference)?;
    d.set_item("working_resolution", sequence.working_resolution)?;
    d.set_item("window_half_widths", sequence.window_half_widths.clone())?;
    let views = PyList::empty(py);
    for view in &sequence.views {
        views.append(grown_view_dict(py, view)?)?;
    }
    d.set_item("views", views)?;
    let trace = PyList::empty(py);
    for entry in &sequence.trace {
        let t = PyDict::new(py);
        t.set_item("observation", entry.observation)?;
        t.set_item("image", entry.image)?;
        t.set_item("angle_deg", entry.angle_deg)?;
        let attempts = PyList::empty(py);
        for attempt in &entry.attempts {
            attempts.append(attempt_dict(py, attempt)?)?;
        }
        t.set_item("attempts", attempts)?;
        t.set_item("outcome", try_outcome_name(entry.outcome))?;
        t.set_item("combined_normal", maybe_vector(py, entry.combined_normal))?;
        t.set_item("combined_sigma_deg", entry.combined_sigma_deg)?;
        trace.append(t)?;
    }
    d.set_item("trace", trace)?;
    let picks = PyList::empty(py);
    for pick in &sequence.picks {
        picks.append(pick_dict(py, pick)?)?;
    }
    d.set_item("picks", picks)?;
    let pairs = PyList::empty(py);
    for pair in &sequence.pairs {
        pairs.append(pair_dict(py, pair)?)?;
    }
    d.set_item("pairs", pairs)?;
    d.set_item("outcome", grown_outcome_name(sequence.outcome))?;
    d.set_item("normal", maybe_vector(py, sequence.normal))?;
    d.set_item("turn_deg", sequence.turn_deg)?;
    d.set_item("turn_sigma_ratio", sequence.turn_sigma_ratio())?;
    let precision = match &sequence.precision {
        Some(p) => {
            let q = PyDict::new(py);
            q.set_item("sigma_weak_deg", p.sigma_weak_deg)?;
            q.set_item("sigma_strong_deg", p.sigma_strong_deg)?;
            q.set_item("discounted_sigma_weak_deg", p.discounted_sigma_weak_deg)?;
            q.set_item("discounted_sigma_strong_deg", p.discounted_sigma_strong_deg)?;
            q.set_item("low_confidence", p.low_confidence)?;
            Some(q)
        }
        None => None,
    };
    d.set_item("precision", precision)?;
    d.set_item("turned", sequence.turned())?;
    d.set_item("joined", sequence.joined())?;
    d.set_item("left_out", sequence.left_out())?;
    d.set_item("shift_only", sequence.shift_only())?;
    d.set_item("matched", sequence.matched())?;
    d.set_item("pairs_kept", sequence.pairs_kept())?;
    d.set_item("written", sequence.written)?;
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
        NormalEstimate::GrownPlane(sequence) => sequence_dict(py, sequence)?,
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
/// optimum is below ``min_cell_zncc`` (default 0.6), its best shift lies on
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
///
/// With ``record_detail=True`` (default ``False``; the result is the same
/// either way) ``estimate`` also carries ``detail``, one dict per render in
/// the trace's order, holding what the step read and decided there. Every
/// dict repeats the layout: ``working_resolution`` (``R``), ``margin`` (a
/// view tile is ``R + 2 margin`` a side), ``reach`` (the whole-shift search's
/// half-width), ``cell_bounds`` (``(4,)``: cell ``(row, col)`` covers columns
/// ``cell_bounds[col]:cell_bounds[col + 1]`` and the same rows of the
/// reference tile) and ``cell_centres`` (``(3, 3, 2)``, ``[x, y]`` grid px
/// centred on the grid). ``patch_frame`` is the patch frame the render was made at
/// (``center``, ``u_axis``, ``v_axis``, ``normal``, ``half_extent``).
/// ``reference`` is ``None`` or a dict with ``observation``, ``image`` and
/// ``tile``, the ``(R, R, channels)`` float32 render the templates are cut
/// from, ``NaN`` off the photograph. ``views`` lists each other ``in`` view
/// rendered, with ``observation``, ``image``, ``tile`` (``(R + 2 margin, R +
/// 2 margin, channels)``), ``score`` (its windowed ZNCC term of the
/// acceptance score, or ``None``), per cell as ``(3, 3)`` arrays
/// ``whole_shift`` (``(3, 3, 2)``, ``NaN`` where the search did not finish),
/// ``shift`` (``(3, 3, 2)`` sub-pixel, ``NaN`` with no peak), ``zncc``,
/// ``curvature`` and ``fit_weight`` (the cell's IRLS weight in the view's
/// fit), ``reading`` and ``status`` (``[row][col]`` registration words
/// before and after the view's fit), and ``fit`` (``None`` or ``model``,
/// ``"affine"``, ``"similarity"`` or ``"shift"``, with ``a`` ``(2, 2)`` and
/// ``b`` ``(2,)``). ``cells`` holds ``status`` (plane-fit words), ``position``
/// (``(3, 3, 3)``), and ``(3, 3)`` arrays ``residual_px``, ``sigma_px``,
/// ``normal_variance``, ``precision_weight``, ``tukey_weight``, ``weight``
/// and ``ray_count``. ``rays`` lists every cell's rays, one row each:
/// ``cell`` (``row * 3 + col``), ``observation``, ``image``, ``pixel``
/// (``(n, 2)``), ``origin`` and ``direction`` (``(n, 3)``) and ``floor_px``.
/// ``plane`` holds ``centroid``, ``fitted_normal`` (the least-variance
/// direction, whatever the verdict), ``normal`` (the verdict's), the
/// ``determinacy`` word with ``free_axis``, ``scatter_eigenvalues``
/// (ascending, the in-plane scatter the verdict reads), ``anisotropy``,
/// ``n_eff``, ``plane_rms``, ``precision_deg`` (``(2,)``, weaker and stronger
/// axis), and the render's trace entry: ``score``,
/// ``normal_precision_deg``, ``gated``, ``proposed_normal``,
/// ``largest_move_px`` and ``taken``.
///
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
    record_detail = false,
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
    record_detail: bool,
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
        record_detail,
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

/// Turn a track-stage `track`'s patch to the one plane its ``in`` views agree
/// on, fitted directly to the photographs.
///
/// The reference and the working grid are :func:`grid_plane_normal`'s. A view
/// is grown against the reference from the starting plane: a
/// ``spot_window``-sample window (default 7) at the grid's centre starts from
/// the shift the view's keypoint implies, with no search; where that growth
/// is refused, it finds the shift by a search within ``keypoint_search_px``
/// (default 3.0) of that shift, widened to ``spot_search_px`` (default 12.0) when no shift reaches
/// ``min_keypoint_zncc`` (default 0.7) or the best is on the edge, in steps
/// of ``spot_search_step_px`` (default 0.5); the window then grows through
/// half-widths of ``window_fractions`` of the grid's side (default
/// ``[1/8, 1/5, 1/3, 1/2]``, ``1/2`` covering the whole grid), with tilt and
/// shift fitted by Levenberg-Marquardt at each size (``max_iterations``,
/// default 20, ``initial_damping``, default 1e-3, ``min_improvement``,
/// default 1e-6, and the Jacobian steps ``jacobian_tilt_step``, default 1e-3,
/// and ``jacobian_shift_step_px``, default 0.05). Each growth gives the
/// view's own plane and its information about the tilt.
///
/// The views grown are a selection, picked one at a time near to far by
/// their angle from the reference, in rungs, until the combined weakest ``σ``
/// is at most ``target_sigma_deg`` (default 0.5) or ``max_selected_views``
/// (default 6) have passed the floor: a growth whose own score reaches
/// ``min_held_zncc`` (default 0.5). The first rung is the view nearest
/// ``first_rung_deg`` (default 15.0) within ``first_rung_min_deg`` (default
/// 10.0) to ``first_rung_max_deg`` (default 20.0), grown from the starting
/// plane; each later rung aims at ``rung_ratio`` (default 1.6) times the
/// widest angle kept, out to the widest views, and within a rung the pick is
/// the view whose information, predicted from its geometry, most raises the
/// weakest measured so far. Each later pick is grown from the combined plane
/// so far and, where that is refused, from the starting plane. Their planes are combined by a robust
/// information-weighted mean, never by fitting the views together, leaving
/// out the furthest estimate past ``max_agreement_distance`` (default 3.0)
/// by the Mahalanobis distance until none is, and then any view under
/// ``min_held_zncc`` at the mean. Up to ``max_pairs`` (default 3) of the
/// widest pairs of kept views are then matched against each other and
/// combined with the views' planes by the robust mean, and the shift alone
/// of every view not picked is fitted at the final plane. ``zncc_tolerance``
/// (default 0.01) turns information into a covariance, ``min_coverage``
/// (default 0.6) is the share of a ZNCC's samples that must land on the
/// photograph, and
/// ``low_confidence_sigma_deg`` (default 2.0) is the weakest-direction ``σ``,
/// with each pair's information halved, above which the plane is reported as
/// low confidence. A turn from the starting normal of at least
/// ``weak_turn_deg`` (default 70.0) that is under ``min_turn_sigma_ratio``
/// (default 3.0) times the weakest-direction ``σ`` (every kept estimate's
/// information summed; a ``σ`` that is not finite counts as no evidence) is
/// a large turn on weak evidence, and is not made. ``working_resolution`` is
/// :func:`grid_plane_normal`'s.
/// Every default is the Rust step's, and a setting outside what the step
/// takes is refused before the photographs are decoded.
///
/// The patch is turned to the final plane whenever a view joined and the
/// turn is not a large one on weak evidence, and the reference's, each
/// joined view's and each shift-only view's that passed the floor keypoints
/// are written; when no view joined, or the turn is a large one on weak
/// evidence, the track is given back as it was, read again, and in the
/// second case no view has its shift alone fitted. The centre
/// does not move. ``images`` and the last four keywords are :func:`fit`'s,
/// and set the reading the step ends with.
///
/// Returns ``(EditableTrack, report)``, the report described under
/// :func:`grid_plane_normal`. Here ``estimate`` has ``kind`` ``"grown_plane"``
/// and carries ``reference`` (the reference observation, or
/// ``None`` when none was found and nothing was tried),
/// ``working_resolution`` (``0`` with no reference), ``window_half_widths``
/// (the stages' windows, grid px), ``outcome`` (``"turned"``,
/// ``"none_joined"`` when no view joined, or ``"weak_turn"`` when the plane
/// found was a large turn on weak evidence and the start was kept),
/// ``normal`` (the final plane's outward normal, reported under
/// ``"weak_turn"`` too, or ``None`` when no view joined), ``turn_deg`` (the
/// turn from the start to that plane, the report's ``tilt`` degrees it gives
/// or would have given, or ``None``), ``turn_sigma_ratio`` (``turn_deg``
/// over ``sigma_weak_deg``, 0 where that is not finite, or ``None``),
/// ``precision`` (``None`` when no
/// view joined, else ``sigma_weak_deg`` and ``sigma_strong_deg`` from every
/// kept estimate's information summed, ``discounted_sigma_weak_deg`` and
/// ``discounted_sigma_strong_deg`` with each kept pair's halved, and
/// ``low_confidence``), ``turned``, ``joined``, ``left_out``, ``shift_only``,
/// ``matched`` (views grown against the reference), ``pairs_kept``,
/// ``written`` (keypoints written, the reference's among them), ``views``,
/// ``trace``, ``picks`` and ``pairs``.
///
/// ``views`` has one dict per observation, ``kind`` naming what the step made
/// of it: ``"reference"`` with ``keypoint`` (the one written, or ``None``);
/// ``"joined"`` with ``start`` (``"no_search"``, ``"keypoint"``,
/// ``"widened"`` or ``"no_keypoint"``), ``distance`` and ``held_zncc`` (its
/// plane's distance from the final plane and its score there), ``shift``
/// (``(2,)``, image px, moved to the final plane) and
/// ``grid_zncc`` at the final plane, ``was`` (the keypoint before the step,
/// or ``None``) and ``keypoint`` (the one written, or ``None`` off the
/// photograph); ``"shift_only"`` with ``start``, ``shift``, ``grid_zncc``,
/// ``was`` and ``keypoint`` (``None`` under the floor or off the photograph);
/// ``"left_out"`` with ``reason`` (``"under_floor"``, ``"too_far"`` or
/// ``"held_under_floor"``), ``distance`` and ``held_zncc``; or
/// ``"not_attempted"``. ``trace`` has one dict per try in the order made:
/// ``observation``, ``image``, ``angle_deg`` (from the reference, seen from
/// the patch centre), ``outcome`` (``"matched"`` or ``"left_out"``),
/// ``combined_normal`` (the plane of the views picked so far, this one among
/// them, or ``None`` for a try that left the view out or came after the
/// selection), ``combined_sigma_deg`` (``NaN`` then) and ``attempts``, each
/// with ``start``, ``stage_zncc`` (a float64 array, one score per stage),
/// ``shift``, ``normal`` (the growth's own plane), ``grid_zncc``,
/// ``sigma_deg`` (its own weakest ``σ``), ``refusal`` (``None`` when it
/// was taken, else ``"under_floor"``) and ``tilt_start`` (``"starting"`` or
/// ``"combined"``, the plane its windows started from). ``picks`` has one dict per pick of
/// the selection: ``observation``, ``image``,
/// ``angle_deg``, ``offset_direction`` (``(2,)``, in the starting patch frame),
/// ``scale_px``, ``predicted_weak`` (the smallest eigenvalue of the
/// information predicted with it added, ``NaN`` for a pick made by angle
/// alone, in the first rung), ``matched``, ``rung`` (the rung it was picked
/// for, from 0) and ``sigma_after_deg`` (``NaN`` when it was not
/// matched).
/// ``pairs`` has one dict per pair: ``observations``, ``images``,
/// ``angle_deg`` (between the two views), ``normal``, ``relative_shift``,
/// ``zncc``, ``sigma_deg``, ``distance``, ``kept`` and ``refusal`` (a
/// ``reason`` word or ``"view_left_out"``).
///
/// With ``record_phases=True`` (default ``False``; the result is the same
/// either way) the step runs with a detailed progress record, and the report
/// also carries ``phases``: one dict per phase the step closed, in the order
/// entered, a phase before the phases inside it, each with ``name``,
/// ``depth``, ``note`` (what the phase did, or ``None``), ``start_s`` (seconds
/// from the beginning of the call to the phase's entry), ``duration_s``
/// (seconds) and ``messages`` (a list of dicts with ``level`` and ``text``).
/// Without it the report has no ``phases``.
///
/// Raises ``ValueError`` with the reason when the step is refused, including a
/// setting it does not take.
#[pyfunction]
#[pyo3(signature = (
    track,
    edited,
    images,
    *,
    working_resolution = None,
    spot_window = None,
    window_fractions = None,
    keypoint_search_px = None,
    min_keypoint_zncc = None,
    spot_search_px = None,
    spot_search_step_px = None,
    zncc_tolerance = None,
    max_agreement_distance = None,
    min_held_zncc = None,
    low_confidence_sigma_deg = None,
    weak_turn_deg = None,
    min_turn_sigma_ratio = None,
    max_pairs = None,
    target_sigma_deg = None,
    max_selected_views = None,
    first_rung_deg = None,
    first_rung_min_deg = None,
    first_rung_max_deg = None,
    rung_ratio = None,
    min_coverage = None,
    max_iterations = None,
    initial_damping = None,
    min_improvement = None,
    jacobian_tilt_step = None,
    jacobian_shift_step_px = None,
    max_seed_offset_px = None,
    max_cache_bytes = None,
    sigma_px = None,
    depth_likelihood_ratio_threshold = None,
    record_phases = false,
))]
#[allow(clippy::too_many_arguments)]
fn grown_plane_normal(
    py: Python<'_>,
    track: &PyEditableTrack,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    working_resolution: Option<u32>,
    spot_window: Option<u32>,
    window_fractions: Option<Vec<f64>>,
    keypoint_search_px: Option<f64>,
    min_keypoint_zncc: Option<f64>,
    spot_search_px: Option<f64>,
    spot_search_step_px: Option<f64>,
    zncc_tolerance: Option<f64>,
    max_agreement_distance: Option<f64>,
    min_held_zncc: Option<f64>,
    low_confidence_sigma_deg: Option<f64>,
    weak_turn_deg: Option<f64>,
    min_turn_sigma_ratio: Option<f64>,
    max_pairs: Option<usize>,
    target_sigma_deg: Option<f64>,
    max_selected_views: Option<usize>,
    first_rung_deg: Option<f64>,
    first_rung_min_deg: Option<f64>,
    first_rung_max_deg: Option<f64>,
    rung_ratio: Option<f64>,
    min_coverage: Option<f64>,
    max_iterations: Option<u32>,
    initial_damping: Option<f64>,
    min_improvement: Option<f64>,
    jacobian_tilt_step: Option<f64>,
    jacobian_shift_step_px: Option<f64>,
    max_seed_offset_px: Option<f64>,
    max_cache_bytes: Option<usize>,
    sigma_px: Option<f64>,
    depth_likelihood_ratio_threshold: Option<f64>,
    record_phases: bool,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let defaults = GrownPlaneNormalOptions::default();
    let options = GrownPlaneNormalOptions {
        working_resolution,
        spot_window: spot_window.unwrap_or(defaults.spot_window),
        window_fractions: window_fractions.unwrap_or_else(|| defaults.window_fractions.clone()),
        keypoint_search_px: keypoint_search_px.unwrap_or(defaults.keypoint_search_px),
        min_keypoint_zncc: min_keypoint_zncc.unwrap_or(defaults.min_keypoint_zncc),
        spot_search_px: spot_search_px.unwrap_or(defaults.spot_search_px),
        spot_search_step_px: spot_search_step_px.unwrap_or(defaults.spot_search_step_px),
        zncc_tolerance: zncc_tolerance.unwrap_or(defaults.zncc_tolerance),
        max_agreement_distance: max_agreement_distance.unwrap_or(defaults.max_agreement_distance),
        min_held_zncc: min_held_zncc.unwrap_or(defaults.min_held_zncc),
        low_confidence_sigma_deg: low_confidence_sigma_deg
            .unwrap_or(defaults.low_confidence_sigma_deg),
        weak_turn_deg: weak_turn_deg.unwrap_or(defaults.weak_turn_deg),
        min_turn_sigma_ratio: min_turn_sigma_ratio.unwrap_or(defaults.min_turn_sigma_ratio),
        max_pairs: max_pairs.unwrap_or(defaults.max_pairs),
        target_sigma_deg: target_sigma_deg.unwrap_or(defaults.target_sigma_deg),
        max_selected_views: max_selected_views.unwrap_or(defaults.max_selected_views),
        first_rung_deg: first_rung_deg.unwrap_or(defaults.first_rung_deg),
        first_rung_min_deg: first_rung_min_deg.unwrap_or(defaults.first_rung_min_deg),
        first_rung_max_deg: first_rung_max_deg.unwrap_or(defaults.first_rung_max_deg),
        rung_ratio: rung_ratio.unwrap_or(defaults.rung_ratio),
        min_coverage: min_coverage.unwrap_or(defaults.min_coverage),
        max_iterations: max_iterations.unwrap_or(defaults.max_iterations),
        initial_damping: initial_damping.unwrap_or(defaults.initial_damping),
        min_improvement: min_improvement.unwrap_or(defaults.min_improvement),
        jacobian_tilt_step: jacobian_tilt_step.unwrap_or(defaults.jacobian_tilt_step),
        jacobian_shift_step_px: jacobian_shift_step_px.unwrap_or(defaults.jacobian_shift_step_px),
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
    let record = PhaseRecord::new();
    let sink = |event: Event<'_>| record.sink(event);
    let progress = if record_phases {
        Progress::to(&sink).detailed(true)
    } else {
        Progress::none()
    };
    let (next, report) =
        core_grown_plane_normal(&track.inner, &edited.inner, &views, &options, &progress)
            .map_err(refused)?;
    let d = normal_report_dict(py, &report)?;
    if record_phases {
        d.set_item("phases", record.to_list(py)?)?;
    }
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
    m.add_function(wrap_pyfunction!(grown_plane_normal, m)?)?;
    Ok(())
}
