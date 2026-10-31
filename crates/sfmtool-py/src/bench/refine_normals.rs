// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for refining the normal of every finite point of a
//! reconstruction, or of a list of points: the bench's *Grown Plane Normal*
//! and *Commit* of each point, written back as one new value.

use std::sync::atomic::Ordering;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::bench::{
    refine_normals as core_refine_normals, NormalError, NormalLeftUnchanged, RefineNormalsOptions,
    RefineNormalsOutcome, RefineNormalsReport,
};
use sfmtool_core::progress::{Event, Progress};

use super::normal::GrownPlaneKeywords;
use super::{refused, views_of};
use crate::patches::views::{resolve_pyramids, PosedViews};
use crate::{ProgressCounter, PySfmrReconstruction};

/// Turn the patch of every finite point of `recon`, or of every finite point
/// of ``point_indexes``, to its grown-plane normal, and give back a new
/// reconstruction that holds the answers.
///
/// For each point this does what :func:`grown_plane_normal` and :func:`commit`
/// do to a track put on the bench from that point with every observation
/// ``in`` and pinned: one plane is fitted to the photographs the point is seen
/// in, the patch is turned about its centre to face along that plane's normal,
/// its bitmap is rendered again and each observation's readings are taken
/// again. The point's centre does not move and no keypoint moves. No index
/// moves: the value returned has the input's points at the input's indexes,
/// and each track keeps its images. `recon` itself is not changed. A caller
/// holding an :class:`~sfmtool.reconstruction.EditedReconstruction` folds it
/// first with its ``materialize``.
///
/// A point at infinity is skipped. Held and ranged points are refined and keep
/// their constraint. A point the step leaves unchanged or refuses, or renders
/// no bitmap for in a value that stores one per point, keeps every column it
/// had. ``point_indexes`` lists the points to refine (``None``, the default,
/// refines every finite point); a point not listed keeps every column it had,
/// and a point listed twice is refined once.
///
/// ``images`` is one decoded image per image of `recon`, or a prebuilt
/// :class:`ImagePyramidSet`, as :func:`fit` takes them. The other keywords are
/// :func:`grown_plane_normal`'s, with its defaults. ``progress``, a
/// :class:`~sfmtool.ProgressCounter`, climbs while the run goes toward the
/// number of points asked for: those listed, each counted once, or every
/// point, those at infinity included. The points are refined in parallel with
/// the GIL released.
///
/// Returns ``(SfmrReconstruction, report)``. The report carries the counts
/// ``read`` (every point but those at infinity and those not asked for),
/// ``turned``, ``unchanged`` (turned points whose record agreed with the
/// stored one, so they were not rewritten), ``low_confidence`` (turned points
/// whose plane was found with low confidence), ``none_joined`` and
/// ``weak_turn`` (points whose normal the step left unchanged, because no view
/// joined or because the turn was a large one on weak evidence), ``kept``
/// (points the step refused or rendered no bitmap for), ``at_infinity`` and
/// ``not_asked``; ``median_turn_deg`` (over the turned points; ``nan`` where
/// none was turned); ``mean_agreement`` (over the turned points with an
/// agreement; ``nan`` where there are none); and ``points``, a list with one
/// dict per point in index order. Each has ``index`` and ``outcome``:
/// ``"turned"``, ``"none_joined"``, ``"weak_turn"``, ``"not_refined"``,
/// ``"no_bitmap"``, ``"at_infinity"`` or ``"not_asked"``. A turned point adds
/// ``turn_deg``, ``joined`` (the views that joined the plane),
/// ``low_confidence``, ``unchanged`` and ``agreement`` (the mean ``grid_zncc``
/// at the plane of the views that joined it or had their shift fitted at it,
/// ``nan`` when none has a score). A ``"weak_turn"`` point adds ``turn_deg``
/// and ``agreement``, the turn not made and the agreement at the plane not
/// turned to. A point the step refused adds ``error``, the refusal's name
/// (``"too_few_observations"``, ...), and ``reason``, its sentence.
///
/// Raises ``ValueError`` with the reason when the run as a whole is refused: a
/// ``sift_files`` reconstruction, one with no patch frames or no posed image,
/// a setting the step does not take, or an index in ``point_indexes`` past the
/// last point.
#[pyfunction]
#[pyo3(signature = (
    recon,
    images,
    *,
    point_indexes = None,
    working_resolution = None,
    spot_window = None,
    window_fractions = None,
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
    max_seed_offset_px = None,
    max_cache_bytes = None,
    sigma_px = None,
    depth_likelihood_ratio_threshold = None,
    progress = None,
))]
#[allow(clippy::too_many_arguments)]
fn refine_normals(
    py: Python<'_>,
    recon: &PySfmrReconstruction,
    images: &Bound<'_, PyAny>,
    point_indexes: Option<Vec<u32>>,
    working_resolution: Option<u32>,
    spot_window: Option<u32>,
    window_fractions: Option<Vec<f64>>,
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
    max_seed_offset_px: Option<f64>,
    max_cache_bytes: Option<usize>,
    sigma_px: Option<f64>,
    depth_likelihood_ratio_threshold: Option<f64>,
    progress: Option<ProgressCounter>,
) -> PyResult<(PySfmrReconstruction, Py<PyDict>)> {
    let normal = GrownPlaneKeywords {
        working_resolution,
        spot_window,
        window_fractions,
        zncc_tolerance,
        max_agreement_distance,
        min_held_zncc,
        low_confidence_sigma_deg,
        weak_turn_deg,
        min_turn_sigma_ratio,
        max_pairs,
        target_sigma_deg,
        max_selected_views,
        first_rung_deg,
        first_rung_min_deg,
        first_rung_max_deg,
        rung_ratio,
        min_coverage,
        max_iterations,
        initial_damping,
        min_improvement,
        max_seed_offset_px,
        max_cache_bytes,
        sigma_px,
        depth_likelihood_ratio_threshold,
    }
    .options();
    // Refuse the settings before the photographs are decoded.
    normal.check().map_err(refused)?;
    let options = RefineNormalsOptions {
        normal,
        point_indexes,
    };
    let posed = PosedViews::from_reconstruction(&recon.inner);
    let pyramids = resolve_pyramids(&posed, images)?;
    let views = views_of(&posed, &pyramids);
    // The run's only count is its points, reported from parallel workers, so
    // the counter keeps the largest seen.
    let counter = progress.as_ref().map(|p| p.handle());
    let sink = |event: Event<'_>| {
        if let (Event::Count { done, .. }, Some(counter)) = (event, &counter) {
            counter.fetch_max(done as usize, Ordering::Relaxed);
        }
    };
    let (next, report) = py
        .detach(|| core_refine_normals(&recon.inner, &views, &options, &Progress::to(&sink)))
        .map_err(refused)?;
    let d = report_dict(py, &report)?;
    Ok((PySfmrReconstruction { inner: next }, d.unbind()))
}

/// The dict form of a refinement's report: the counts, and one dict per point.
fn report_dict<'py>(py: Python<'py>, report: &RefineNormalsReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("read", report.read())?;
    d.set_item("turned", report.turned())?;
    d.set_item("unchanged", report.unchanged())?;
    d.set_item("low_confidence", report.low_confidence())?;
    d.set_item("none_joined", report.none_joined())?;
    d.set_item("weak_turn", report.weak_turn())?;
    d.set_item("kept", report.kept())?;
    d.set_item("at_infinity", report.at_infinity())?;
    d.set_item("not_asked", report.not_asked())?;
    d.set_item("median_turn_deg", report.median_turn_deg())?;
    d.set_item("mean_agreement", report.mean_agreement())?;
    let points = PyList::empty(py);
    for point in &report.points {
        let p = PyDict::new(py);
        p.set_item("index", point.index)?;
        match &point.outcome {
            RefineNormalsOutcome::Turned {
                turn_deg,
                joined,
                low_confidence,
                unchanged,
                agreement,
            } => {
                p.set_item("outcome", "turned")?;
                p.set_item("turn_deg", turn_deg)?;
                p.set_item("joined", joined)?;
                p.set_item("low_confidence", low_confidence)?;
                p.set_item("unchanged", unchanged)?;
                p.set_item("agreement", agreement)?;
            }
            RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::NoneJoined) => {
                p.set_item("outcome", "none_joined")?
            }
            RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::WeakTurn {
                turn_deg,
                agreement,
            }) => {
                p.set_item("outcome", "weak_turn")?;
                p.set_item("turn_deg", turn_deg)?;
                p.set_item("agreement", agreement)?;
            }
            RefineNormalsOutcome::NotRefined(e) => {
                p.set_item("outcome", "not_refined")?;
                p.set_item("error", normal_error_name(e))?;
                p.set_item("reason", e.to_string())?;
            }
            RefineNormalsOutcome::NoBitmap => p.set_item("outcome", "no_bitmap")?,
            RefineNormalsOutcome::AtInfinity => p.set_item("outcome", "at_infinity")?,
            RefineNormalsOutcome::NotAsked => p.set_item("outcome", "not_asked")?,
        }
        points.append(p)?;
    }
    d.set_item("points", points)?;
    Ok(d)
}

/// The name a refused point's report gives its [`NormalError`], so a script
/// can count the refusals of one kind.
fn normal_error_name(e: &NormalError) -> &'static str {
    match e {
        NormalError::ClusterStage => "cluster_stage",
        NormalError::NoPatchFrame => "no_patch_frame",
        NormalError::AtInfinity => "at_infinity",
        NormalError::TooFewObservations(_) => "too_few_observations",
        NormalError::BadPieces(_) => "bad_pieces",
        NormalError::BadOverlap(_) => "bad_overlap",
        NormalError::NotScored { .. } => "not_scored",
        NormalError::TooFewPieces { .. } => "too_few_pieces",
        NormalError::TooFewGridPieces { .. } => "too_few_grid_pieces",
        NormalError::BadGrownPlaneOption(_) => "bad_grown_plane_option",
        NormalError::Tilt(_) => "tilt",
        NormalError::Evaluate(_) => "evaluate",
        NormalError::Cancelled => "cancelled",
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(refine_normals, m)?)?;
    Ok(())
}
