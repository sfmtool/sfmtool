// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for refining the normal of every finite point of a
//! reconstruction, or of a list of points: the bench's *Grown Plane Normal*,
//! or the pairwise-flow normal, and *Commit* of each point, written back as
//! one new value.

use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Duration;

use pyo3::exceptions::{PyInterruptedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use sfmtool_core::bench::{
    refine_normals as core_refine_normals, NormalError, NormalLeftUnchanged, NormalTurned,
    RefineNormalsError, RefineNormalsOptions, RefineNormalsOutcome, RefineNormalsReport,
    RefineNormalsStep,
};
use sfmtool_core::progress::{Event, Progress};

use super::normal::GrownPlaneKeywords;
use super::pairwise_flow_normal::PairwiseFlowKeywords;
use super::{refused, views_of};
use crate::patches::views::{resolve_pyramids, PosedViews};
use crate::signals::SignalWatch;
use crate::{ProgressCounter, PySfmrReconstruction};

/// The `method` that runs the grown-plane normal, the default.
const GROWN_PLANE: &str = "grown_plane";
/// The `method` that runs the pairwise-flow normal.
const PAIRWISE_FLOW: &str = "pairwise_flow";
/// Every `method` [`refine_normals`] takes, the default first, exported as
/// ``REFINE_NORMALS_METHODS`` so a caller can refuse another before it
/// loads anything.
const METHODS: [&str; 2] = [GROWN_PLANE, PAIRWISE_FLOW];

/// Turn the patch of every finite point of `recon`, or of every finite point
/// of ``point_indexes``, to its grown-plane normal, or to its pairwise-flow
/// normal with ``method="pairwise_flow"``, and give back a new reconstruction
/// that holds the answers.
///
/// For each point this does what :func:`grown_plane_normal` (or
/// :func:`pairwise_flow_normal`) and :func:`commit` do to a track put on the
/// bench from that point with every observation ``in`` and pinned: the
/// step finds a normal from the photographs the point is seen in, the patch
/// is turned about its centre to face along it, its bitmap is rendered again
/// and each observation's readings are taken again. The point's centre does
/// not move and no keypoint moves. No index moves: the value returned has the
/// input's points at the input's indexes, and each track keeps its images.
/// `recon` itself is not changed. A caller holding an
/// :class:`~sfmtool.reconstruction.EditedReconstruction` folds it first with
/// its ``materialize``.
///
/// A point at infinity is skipped. Held and ranged points are refined and keep
/// their constraint. A point the step leaves unchanged or refuses, or renders
/// no bitmap for in a value that stores one per point, keeps every column it
/// had. ``point_indexes`` lists the points to refine (``None``, the default,
/// refines every finite point); a point not listed keeps every column it had,
/// and a point listed twice is refined once.
///
/// ``images`` is one decoded image per image of `recon`, or a prebuilt
/// :class:`ImagePyramidSet`, as :func:`fit` takes them. ``method`` names the
/// step: ``"grown_plane"`` (the default) or ``"pairwise_flow"``, the names
/// ``REFINE_NORMALS_METHODS`` lists. The other
/// keywords are :func:`grown_plane_normal`'s and :func:`pairwise_flow_normal`'s,
/// with their defaults; ``max_pairs``, ``resolution`` and :func:`fit`'s four
/// are shared and set the chosen step's, and a keyword only the other step
/// takes is refused. The pairwise-flow normal costs seconds of one thread per
/// point, against tens of milliseconds for the grown plane. ``progress``, a
/// :class:`~sfmtool.ProgressCounter`, climbs while the run goes toward the
/// number of points asked for: those listed, each counted once, or every
/// point, those at infinity included. The points are refined in parallel with
/// the GIL released, while the calling thread takes the GIL every 50 ms to
/// run Python's signal handlers; one that raises (``KeyboardInterrupt`` for
/// Ctrl+C) stops the run, and its exception is raised.
///
/// Returns ``(SfmrReconstruction, report)``. The report carries the counts
/// ``read`` (every point but those at infinity and those not asked for),
/// ``turned``, ``unchanged`` (turned points whose record agreed with the
/// stored one, so they were not rewritten), ``low_confidence`` (points the
/// grown plane turned whose plane was found with low confidence),
/// ``single_pair`` (points the pairwise-flow normal turned on its one pair),
/// ``none_joined`` and ``weak_turn`` (points whose normal the grown plane left
/// unchanged, because no view joined or because the turn was a large one on
/// weak evidence), ``undetermined`` and ``unstable`` (points whose normal the
/// pairwise-flow normal left unchanged), ``kept`` (points the step refused or
/// rendered no bitmap for), ``at_infinity`` and ``not_asked``;
/// ``median_turn_deg`` (over the turned points; ``nan`` where none was
/// turned); ``mean_agreement`` (over the points the grown plane turned with
/// an agreement; ``nan`` where there are none); and ``points``, a list with
/// one dict per point in index order. Each has ``index`` and ``outcome``:
/// ``"turned"``, ``"none_joined"``, ``"weak_turn"``, ``"undetermined"``,
/// ``"unstable"``, ``"not_refined"``, ``"no_bitmap"``, ``"at_infinity"`` or
/// ``"not_asked"``. A turned point adds ``turn_deg`` and ``unchanged``, and
/// from the grown plane ``joined`` (the views that joined the plane),
/// ``low_confidence`` and ``agreement`` (the mean ``grid_zncc`` at the plane
/// of the views that joined it or had their shift fitted at it, ``nan`` when
/// none has a score), or from the pairwise-flow normal ``single_pair``,
/// ``chosen_radius`` (patch half-extents) and ``sigma_deg``. A
/// ``"weak_turn"`` point adds ``turn_deg`` and ``agreement``, the turn not
/// made and the agreement at the plane not turned to. A point the step
/// refused adds ``error``, the refusal's name (``"too_few_observations"``,
/// ...), and ``reason``, its sentence.
///
/// Raises ``ValueError`` with the reason when the run as a whole is refused: a
/// ``sift_files`` reconstruction, one with no patch frames or no posed image,
/// an unknown ``method``, a keyword the chosen step does not take, a setting
/// the step does not take, or an index in ``point_indexes`` past the last
/// point. Raises the exception a signal handler raised while the run went,
/// or ``InterruptedError`` where the run was cancelled otherwise.
#[pyfunction]
#[pyo3(signature = (
    recon,
    images,
    *,
    point_indexes = None,
    method = None,
    resolution = None,
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
    turned_support = None,
    stage_turn_cap_deg = None,
    min_stage_turn_cap_deg = None,
    max_grid_rounds = None,
    rounds = None,
    max_view_angle_deg = None,
    max_normal_from_mean_view_deg = None,
    fallback_radius = None,
    min_samples_per_patch = None,
    max_samples_per_patch = None,
    crop_half_extents = None,
    min_crop_half_width_px = None,
    max_crop_samples = None,
    radii = None,
    max_grid_tilt_deg = None,
    grid_tilt_step_deg = None,
    grid_azimuth_step_deg = None,
    tukey_c = None,
    correlation_length_px = None,
    noise_false_alarm = None,
    flow_floor_px = None,
    max_sigma_deg = None,
    agreement_sigmas = None,
    agreement_margin_deg = None,
    max_mean_chi2 = None,
    single_pair_radius = None,
    flow_patch_size = None,
    flow_patch_stride = None,
    flow_grad_descent_iterations = None,
    flow_finest_scale = None,
    flow_coarsest_scale = None,
    flow_variational_refinement = None,
    flow_variational_jacobi_iterations = None,
    flow_variational_outer_iterations_base = None,
    flow_variational_alpha = None,
    flow_variational_gamma = None,
    flow_variational_delta = None,
    flow_normalize_patches = None,
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
    method: Option<&str>,
    resolution: Option<u32>,
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
    turned_support: Option<bool>,
    stage_turn_cap_deg: Option<f64>,
    min_stage_turn_cap_deg: Option<f64>,
    max_grid_rounds: Option<u32>,
    rounds: Option<u32>,
    max_view_angle_deg: Option<f64>,
    max_normal_from_mean_view_deg: Option<f64>,
    fallback_radius: Option<f64>,
    min_samples_per_patch: Option<u32>,
    max_samples_per_patch: Option<u32>,
    crop_half_extents: Option<f64>,
    min_crop_half_width_px: Option<f64>,
    max_crop_samples: Option<u32>,
    radii: Option<Vec<f64>>,
    max_grid_tilt_deg: Option<f64>,
    grid_tilt_step_deg: Option<f64>,
    grid_azimuth_step_deg: Option<f64>,
    tukey_c: Option<f64>,
    correlation_length_px: Option<f64>,
    noise_false_alarm: Option<f64>,
    flow_floor_px: Option<f64>,
    max_sigma_deg: Option<f64>,
    agreement_sigmas: Option<f64>,
    agreement_margin_deg: Option<f64>,
    max_mean_chi2: Option<f64>,
    single_pair_radius: Option<f64>,
    flow_patch_size: Option<u32>,
    flow_patch_stride: Option<u32>,
    flow_grad_descent_iterations: Option<u32>,
    flow_finest_scale: Option<u32>,
    flow_coarsest_scale: Option<u32>,
    flow_variational_refinement: Option<bool>,
    flow_variational_jacobi_iterations: Option<u32>,
    flow_variational_outer_iterations_base: Option<u32>,
    flow_variational_alpha: Option<f32>,
    flow_variational_gamma: Option<f32>,
    flow_variational_delta: Option<f32>,
    flow_normalize_patches: Option<bool>,
    max_seed_offset_px: Option<f64>,
    max_cache_bytes: Option<usize>,
    sigma_px: Option<f64>,
    depth_likelihood_ratio_threshold: Option<f64>,
    progress: Option<ProgressCounter>,
) -> PyResult<(PySfmrReconstruction, Py<PyDict>)> {
    let grown_plane = GrownPlaneKeywords {
        resolution,
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
        turned_support,
        stage_turn_cap_deg,
        min_stage_turn_cap_deg,
        max_grid_rounds,
        max_seed_offset_px,
        max_cache_bytes,
        sigma_px,
        depth_likelihood_ratio_threshold,
    };
    let pairwise_flow = PairwiseFlowKeywords {
        rounds,
        max_view_angle_deg,
        max_normal_from_mean_view_deg,
        fallback_radius,
        min_samples_per_patch,
        max_samples_per_patch,
        crop_half_extents,
        min_crop_half_width_px,
        max_crop_samples,
        radii,
        max_grid_tilt_deg,
        grid_tilt_step_deg,
        grid_azimuth_step_deg,
        tukey_c,
        correlation_length_px,
        noise_false_alarm,
        flow_floor_px,
        max_sigma_deg,
        agreement_sigmas,
        agreement_margin_deg,
        max_mean_chi2,
        single_pair_radius,
        flow_patch_size,
        flow_patch_stride,
        flow_grad_descent_iterations,
        flow_finest_scale,
        flow_coarsest_scale,
        flow_variational_refinement,
        flow_variational_jacobi_iterations,
        flow_variational_outer_iterations_base,
        flow_variational_alpha,
        flow_variational_gamma,
        flow_variational_delta,
        flow_normalize_patches,
        max_pairs,
        resolution,
        max_seed_offset_px,
        max_cache_bytes,
        sigma_px,
        depth_likelihood_ratio_threshold,
    };
    let other = |name: &str, step: &str, method: &str| {
        PyValueError::new_err(format!(
            "{name} is a keyword of the {step} normal, and method is {method:?}"
        ))
    };
    let step = match method.unwrap_or(GROWN_PLANE) {
        GROWN_PLANE => {
            if let Some(name) = pairwise_flow.first_own_given() {
                return Err(other(name, "pairwise flow", GROWN_PLANE));
            }
            RefineNormalsStep::GrownPlane(grown_plane.options())
        }
        PAIRWISE_FLOW => {
            if let Some(name) = grown_plane.first_own_given() {
                return Err(other(name, "grown plane", PAIRWISE_FLOW));
            }
            RefineNormalsStep::PairwiseFlow(pairwise_flow.options())
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method: {other:?} (expected {})",
                METHODS.join("|")
            )))
        }
    };
    // Refuse the settings before the photographs are decoded.
    step.check().map_err(refused)?;
    let options = RefineNormalsOptions {
        step,
        point_indexes,
    };
    let posed = PosedViews::from_reconstruction(recon.inner());
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
    // The points are refined on the rayon pool, whose workers cannot run
    // Python's signal handlers, so the run goes on a thread of its own while
    // the calling thread waits for it, taking the GIL every `PATIENCE` to run
    // them, as `pick_images` does from its own waits. One that raises stops
    // the run, and its exception is raised in place of the run's result.
    const PATIENCE: Duration = Duration::from_millis(50);
    let watch = SignalWatch::new();
    let recon = recon.inner();
    let result = py.detach(|| {
        std::thread::scope(|scope| {
            let (sender, receiver) = mpsc::channel();
            let (sink, cancel, views, options) = (&sink, watch.cancel_flag(), &views, &options);
            let run = scope.spawn(move || {
                let progress = Progress::to(sink).cancelled_by(cancel);
                let result = core_refine_normals(recon, views, options, &progress);
                // The calling thread receives until this arrives, so the send succeeds.
                let _ = sender.send(result);
            });
            loop {
                match receiver.recv_timeout(PATIENCE) {
                    Ok(result) => return result,
                    Err(mpsc::RecvTimeoutError::Timeout) => watch.check(),
                    // The run's thread ended without sending: it panicked.
                    Err(mpsc::RecvTimeoutError::Disconnected) => match run.join() {
                        Err(panic) => std::panic::resume_unwind(panic),
                        Ok(()) => unreachable!("the run sends its result before it ends"),
                    },
                }
            }
        })
    });
    let (next, report) = watch.finish(result)?.map_err(|e| match e {
        RefineNormalsError::Cancelled => PyInterruptedError::new_err(e.to_string()),
        other => refused(other),
    })?;
    let d = report_dict(py, &report)?;
    Ok((PySfmrReconstruction::new(next), d.unbind()))
}

/// The dict form of a refinement's report: the counts, and one dict per point.
fn report_dict<'py>(py: Python<'py>, report: &RefineNormalsReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("read", report.read())?;
    d.set_item("turned", report.turned())?;
    d.set_item("unchanged", report.unchanged())?;
    d.set_item("low_confidence", report.low_confidence())?;
    d.set_item("single_pair", report.single_pair())?;
    d.set_item("none_joined", report.none_joined())?;
    d.set_item("weak_turn", report.weak_turn())?;
    d.set_item("undetermined", report.undetermined())?;
    d.set_item("unstable", report.unstable())?;
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
                unchanged,
                by,
            } => {
                p.set_item("outcome", "turned")?;
                p.set_item("turn_deg", turn_deg)?;
                p.set_item("unchanged", unchanged)?;
                match by {
                    NormalTurned::GrownPlane {
                        joined,
                        low_confidence,
                        agreement,
                    } => {
                        p.set_item("joined", joined)?;
                        p.set_item("low_confidence", low_confidence)?;
                        p.set_item("agreement", agreement)?;
                    }
                    NormalTurned::PairwiseFlow {
                        single_pair,
                        chosen_radius,
                        sigma_deg,
                    } => {
                        p.set_item("single_pair", single_pair)?;
                        p.set_item("chosen_radius", chosen_radius)?;
                        p.set_item("sigma_deg", sigma_deg)?;
                    }
                }
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
            RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::Undetermined) => {
                p.set_item("outcome", "undetermined")?
            }
            RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::Unstable) => {
                p.set_item("outcome", "unstable")?
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
        NormalError::TooFewPieces { .. } => "too_few_pieces",
        NormalError::TooFewGridPieces { .. } => "too_few_grid_pieces",
        NormalError::BadGridPlaneOption(_) => "bad_grid_plane_option",
        NormalError::BadGrownPlaneOption(_) => "bad_grown_plane_option",
        NormalError::BadPairwiseFlowOption(_) => "bad_pairwise_flow_option",
        NormalError::Flow(_) => "flow",
        NormalError::Tilt(_) => "tilt",
        NormalError::Evaluate(_) => "evaluate",
        NormalError::Cancelled => "cancelled",
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(refine_normals, m)?)?;
    m.add("REFINE_NORMALS_METHODS", PyTuple::new(m.py(), METHODS)?)?;
    Ok(())
}
