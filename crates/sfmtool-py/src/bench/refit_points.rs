// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for refitting every point of a reconstruction: the bench's
//! *Fit* and *Commit* of each point, written back as one new value.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::bench::{
    refit_points as core_refit_points, FitError, RefitOutcome, RefitPointsOptions,
    RefitPointsReport, RefitRange, BENCH_MAX_SHIFT_PX,
};
use sfmtool_core::progress::Progress;

use super::{classification_dict, fit_options, refused, views_of};
use crate::patches::views::{resolve_pyramids, PosedViews};
use crate::PySfmrReconstruction;

/// Refit every point of `recon` to the render of its reference observation,
/// and give back a new reconstruction that holds the answers.
///
/// For each point this does what :func:`fit` and :func:`commit` do to a track
/// put on the bench from that point with every observation ``in`` and pinned:
/// the other observations are aligned to the render of the reference
/// observation and their keypoints rewritten, the point is re-triangulated (a
/// point at infinity has its direction estimated again, and the classification
/// decides afresh whether it is a position or a direction), the patch is placed
/// at the answer, its bitmap is rendered again, and each observation's readings
/// are taken again against that bitmap. No index moves: the value returned has
/// the input's points at the input's indexes, and each track keeps its images.
/// `recon` itself is not changed. A caller holding an
/// :class:`~sfmtool.reconstruction.EditedReconstruction` folds it first with
/// its ``materialize``.
///
/// A held point is left out. A ranged point is read at its distance and stays
/// ranged; one ranged at a finite distance from an image with no pose is fitted
/// free, and released only where the fit makes it a direction. A point the fit
/// refuses, or renders no bitmap for in a value that stores one per point,
/// keeps every column it had.
///
/// ``images`` is one decoded image per image of `recon`, or a prebuilt
/// :class:`ImagePyramidSet`, as :func:`fit` takes them.
/// ``max_seed_offset_px``, ``max_cache_bytes``, ``sigma_px`` and
/// ``depth_likelihood_ratio_threshold`` are :func:`fit`'s; with no ``sigma_px``
/// the reconstruction's reprojection noise is measured once for the whole run.
/// ``max_shift_px`` is how far, in grid px, a fit may move a sighting before
/// it keeps its keypoint instead (6, the bound every track put on the bench
/// starts with). The points are fitted in parallel with the GIL released.
///
/// Returns ``(SfmrReconstruction, report)``. The report carries the counts
/// ``read`` (every point but the held ones), ``refit``, ``unchanged`` (refit
/// points whose record agreed with the stored one, so they were not
/// rewritten), ``crossed`` (refit points that went from a position to a
/// direction or back), ``kept`` (points left as they were), ``held``,
/// ``ranged``, ``released`` and ``kept_at_seed`` (sightings, summed over the
/// refit points), ``median_shift`` (in the reconstruction's units, over the
/// points that were positions before and after; ``nan`` where there are none)
/// and ``points``, a list with one dict per point in index order. Each has
/// ``index`` and ``outcome``: ``"refit"``, ``"held"``, ``"not_fitted"`` or
/// ``"no_bitmap"``. A refit point adds ``placed``, ``kept_at_seed``,
/// ``crossed``, ``shift`` (``None`` for a direction or a crossing),
/// ``classification`` (as :func:`fit` reports it; ``None`` where the distance
/// rule decided the coordinate), ``range`` (``"free"``, ``"applied"`` or
/// ``"unresolved"``), ``released`` and ``unchanged``. A point the fit refused
/// adds ``error``, the refusal's name (``"too_few_observations"``,
/// ``"too_few_rays"``, ...), and ``reason``, its sentence.
///
/// Raises ``ValueError`` with the reason when the run as a whole is refused: a
/// ``sift_files`` reconstruction, one with no patch frames or no posed image,
/// a track naming an image the reconstruction does not hold, or no noise level
/// to weight the rays by.
#[pyfunction]
#[pyo3(signature = (
    recon,
    images,
    *,
    max_seed_offset_px = None,
    max_cache_bytes = None,
    sigma_px = None,
    depth_likelihood_ratio_threshold = None,
    max_shift_px = BENCH_MAX_SHIFT_PX,
))]
#[allow(clippy::too_many_arguments)]
fn refit_points(
    py: Python<'_>,
    recon: &PySfmrReconstruction,
    images: &Bound<'_, PyAny>,
    max_seed_offset_px: Option<f64>,
    max_cache_bytes: Option<usize>,
    sigma_px: Option<f64>,
    depth_likelihood_ratio_threshold: Option<f64>,
    max_shift_px: f64,
) -> PyResult<(PySfmrReconstruction, Py<PyDict>)> {
    let posed = PosedViews::from_reconstruction(recon.inner());
    let pyramids = resolve_pyramids(&posed, images)?;
    let views = views_of(&posed, &pyramids);
    let options = RefitPointsOptions {
        fit: fit_options(
            max_seed_offset_px,
            max_cache_bytes,
            sigma_px,
            depth_likelihood_ratio_threshold,
        ),
        max_shift_px,
    };
    let (next, report) = py
        .detach(|| core_refit_points(recon.inner(), &views, &options, &Progress::none()))
        .map_err(refused)?;
    let d = report_dict(py, &report)?;
    Ok((PySfmrReconstruction::new(next), d.unbind()))
}

/// The dict form of a refit's report: the counts, and one dict per point.
fn report_dict<'py>(py: Python<'py>, report: &RefitPointsReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("read", report.read())?;
    d.set_item("refit", report.refit())?;
    d.set_item("unchanged", report.unchanged())?;
    d.set_item("crossed", report.crossed())?;
    d.set_item("kept", report.kept())?;
    d.set_item("held", report.held())?;
    d.set_item("ranged", report.ranged())?;
    d.set_item("released", report.released())?;
    d.set_item("kept_at_seed", report.kept_at_seed())?;
    d.set_item("median_shift", report.median_shift())?;
    let points = PyList::empty(py);
    for point in &report.points {
        let p = PyDict::new(py);
        p.set_item("index", point.index)?;
        match &point.outcome {
            RefitOutcome::Refit {
                placed,
                kept_at_seed,
                crossed,
                shift,
                classification,
                range,
                unchanged,
            } => {
                p.set_item("outcome", "refit")?;
                p.set_item("placed", placed)?;
                p.set_item("kept_at_seed", kept_at_seed)?;
                p.set_item("crossed", crossed)?;
                p.set_item("shift", shift)?;
                match classification {
                    Some(call) => p.set_item("classification", classification_dict(py, call)?)?,
                    None => p.set_item("classification", py.None())?,
                }
                let (word, released) = match range {
                    RefitRange::Free => ("free", false),
                    RefitRange::Applied => ("applied", false),
                    RefitRange::Unresolved { released } => ("unresolved", *released),
                };
                p.set_item("range", word)?;
                p.set_item("released", released)?;
                p.set_item("unchanged", unchanged)?;
            }
            RefitOutcome::Held => p.set_item("outcome", "held")?,
            RefitOutcome::NotFitted(e) => {
                p.set_item("outcome", "not_fitted")?;
                p.set_item("error", fit_error_name(e))?;
                p.set_item("reason", e.to_string())?;
            }
            RefitOutcome::NoBitmap => p.set_item("outcome", "no_bitmap")?,
        }
        points.append(p)?;
    }
    d.set_item("points", points)?;
    Ok(d)
}

/// The name a refused point's report gives its [`FitError`], so a script can
/// count the refusals of one kind.
fn fit_error_name(e: &FitError) -> &'static str {
    match e {
        FitError::ViewsMissing { .. } => "views_missing",
        FitError::NoView { .. } => "no_view",
        FitError::TooFewObservations(_) => "too_few_observations",
        FitError::NoPatchFrame => "no_patch_frame",
        FitError::Triangulation => "triangulation",
        FitError::TooLarge { .. } => "too_large",
        FitError::OutOfMemory { .. } => "out_of_memory",
        FitError::TooFewRays { .. } => "too_few_rays",
        FitError::NoNoiseLevel(_) => "no_noise_level",
        FitError::Cancelled => "cancelled",
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(refit_points, m)?)?;
    Ok(())
}
