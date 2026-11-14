// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the pairwise-flow normal (`pairwise_flow_normal`), and
//! the dict form of its report's rounds.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::bench::{
    pairwise_flow_normal as core_pairwise_flow_normal, PairwiseFlowCrop, PairwiseFlowFit,
    PairwiseFlowNormalOptions, PairwiseFlowOutcome, PairwiseFlowPair, PairwiseFlowRadius,
    PairwiseFlowRound, PairwiseFlowRounds,
};
use sfmtool_core::progress::{Event, Progress};

use super::normal::{maybe_vector, normal_report_dict};
use super::{fit_options, refused, views_of, PyEditableTrack};
use crate::flow::optical::FlowParamOverrides;
use crate::patches::views::{resolve_pyramids, PosedViews};
use crate::phase_record::PhaseRecord;
use crate::reconstruction::edited::PyEditedReconstruction;

/// The word for how a round of the pairwise-flow normal, or the step, ended.
fn outcome_name(outcome: PairwiseFlowOutcome) -> &'static str {
    match outcome {
        PairwiseFlowOutcome::Turned => "turned",
        PairwiseFlowOutcome::SinglePair => "single_pair",
        PairwiseFlowOutcome::Undetermined => "undetermined",
        PairwiseFlowOutcome::Unstable => "unstable",
    }
}

/// The dict form of one observation's crop in a round, `kind` naming which.
fn crop_dict<'py>(py: Python<'py>, crop: &PairwiseFlowCrop) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match crop {
        &PairwiseFlowCrop::Rendered {
            image,
            angle_deg,
            samples_per_pixel,
            coverage,
        } => {
            d.set_item("kind", "rendered")?;
            d.set_item("image", image)?;
            d.set_item("angle_deg", angle_deg)?;
            d.set_item("samples_per_pixel", samples_per_pixel)?;
            d.set_item("coverage", coverage)?;
        }
        &PairwiseFlowCrop::TooOblique { image, angle_deg } => {
            d.set_item("kind", "too_oblique")?;
            d.set_item("image", image)?;
            d.set_item("angle_deg", angle_deg)?;
        }
        PairwiseFlowCrop::NotAttempted => d.set_item("kind", "not_attempted")?,
    }
    Ok(d)
}

/// The dict form of one pair's plane at one radius.
fn fit_dict<'py>(py: Python<'py>, fit: &PairwiseFlowFit) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("normal", maybe_vector(py, Some(fit.normal)))?;
    d.set_item("sigma_deg", fit.sigma_deg)?;
    d.set_item("sigma_texture_deg", fit.sigma_texture_deg)?;
    d.set_item("sigma_residual_deg", fit.sigma_residual_deg)?;
    d.set_item("offset", maybe_vector(py, Some(fit.offset)))?;
    d.set_item("rms_residual", fit.rms_residual)?;
    d.set_item("residual_variance", fit.residual_variance)?;
    d.set_item("determined", fit.determined)?;
    d.set_item("combined", fit.combined)?;
    Ok(d)
}

/// The dict form of one pair of rendered crops in a round.
fn pair_dict<'py>(py: Python<'py>, pair: &PairwiseFlowPair) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("observations", pair.observations.to_vec())?;
    d.set_item("images", pair.images.to_vec())?;
    d.set_item("angle_deg", pair.angle_deg)?;
    d.set_item("level", pair.level)?;
    d.set_item("half_width", pair.half_width)?;
    let fits = PyList::empty(py);
    for fit in &pair.fits {
        match fit {
            Some(fit) => fits.append(fit_dict(py, fit)?)?,
            None => fits.append(py.None())?,
        }
    }
    d.set_item("fits", fits)?;
    Ok(d)
}

/// The dict form of the pairs combined at one radius.
fn radius_dict<'py>(py: Python<'py>, row: &PairwiseFlowRadius) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("radius", row.radius)?;
    d.set_item("normal", maybe_vector(py, row.normal))?;
    d.set_item("sigma_deg", row.sigma_deg)?;
    d.set_item("mean_chi2", row.mean_chi2)?;
    d.set_item("pairs_fitted", row.pairs_fitted)?;
    d.set_item("pairs_determined", row.pairs_determined)?;
    d.set_item("pairs_combined", row.pairs_combined)?;
    Ok(d)
}

/// The dict form of one round.
fn round_dict<'py>(py: Python<'py>, round: &PairwiseFlowRound) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("plane_normal", maybe_vector(py, Some(round.plane_normal)))?;
    d.set_item("sample_spacing", round.sample_spacing)?;
    d.set_item("samples_per_patch", round.samples_per_patch)?;
    d.set_item("crop_half_width", round.crop_half_width)?;
    let crops = PyList::empty(py);
    for crop in &round.crops {
        crops.append(crop_dict(py, crop)?)?;
    }
    d.set_item("crops", crops)?;
    let pairs = PyList::empty(py);
    for pair in &round.pairs {
        pairs.append(pair_dict(py, pair)?)?;
    }
    d.set_item("pairs", pairs)?;
    let radii = PyList::empty(py);
    for row in &round.radii {
        radii.append(radius_dict(py, row)?)?;
    }
    d.set_item("radii", radii)?;
    d.set_item("outcome", outcome_name(round.outcome))?;
    d.set_item("chosen_radius", round.chosen_radius)?;
    d.set_item("normal", maybe_vector(py, round.normal))?;
    d.set_item("sigma_deg", round.sigma_deg)?;
    d.set_item("from_mean_view_deg", round.from_mean_view_deg)?;
    d.set_item("next_plane", maybe_vector(py, round.next_plane))?;
    Ok(d)
}

/// The dict form of what the pairwise-flow normal did: how it ended, the
/// radius, normal, σ and turn of the answer it turned the patch to, and every
/// round.
pub(super) fn rounds_dict<'py>(
    py: Python<'py>,
    rounds: &PairwiseFlowRounds,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", "pairwise_flow")?;
    d.set_item("outcome", outcome_name(rounds.outcome))?;
    d.set_item("chosen_radius", rounds.chosen_radius)?;
    d.set_item("normal", maybe_vector(py, rounds.normal))?;
    d.set_item("sigma_deg", rounds.sigma_deg)?;
    d.set_item("turn_deg", rounds.turn_deg)?;
    let list = PyList::empty(py);
    for round in &rounds.rounds {
        list.append(round_dict(py, round)?)?;
    }
    d.set_item("rounds", list)?;
    Ok(d)
}

/// Turn a track-stage `track`'s patch to the plane on which dense optical
/// flow between pairs of its observations agrees, at the largest support where
/// it agrees: the pairwise-flow normal.
///
/// In each round every ``in`` observation's photograph is rendered onto one
/// plane through the patch's centre, on one grid all the views share; a view
/// more than ``max_view_angle_deg`` (default 80.0) from the plane's normal is
/// left out. For every pair of rendered crops, or ``max_pairs`` (default 10)
/// of them chosen by a shuffle with a fixed seed, the DIS flow from one crop to
/// the other is computed at the pyramid level of the pair's coarser view, and
/// the plane through the point with a free 2D offset is fitted to that flow
/// over a disc at each of ``radii`` (patch half-extents, ascending; default
/// ``[0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12]``). At each radius the pairs whose
/// σ is at most ``max_sigma_deg`` (default 10.0) are combined, and the round's
/// answer is the combined normal at the largest radius with at least two such
/// pairs whose mean χ² is at most ``max_mean_chi2`` (default 2.0). The step runs
/// ``rounds`` (default 3) rounds, the first on the track's normal and each
/// later one on the round before's answer, or its combination at
/// ``fallback_radius`` (default 1.0) where it had none; a point with exactly
/// two observations runs one round, read at ``single_pair_radius`` (default
/// 1.0). An answer more than ``max_normal_from_mean_view_deg`` (default 80.0)
/// from the mean viewing direction is no answer.
///
/// The grid takes ``min_samples_per_patch`` (default 24) to
/// ``max_samples_per_patch`` (default 128) samples along one side of the
/// patch; the crops' half-width is ``crop_half_extents`` (default 3.0) patch
/// half-extents and at least ``min_crop_half_width_px`` (default 48.0)
/// photograph px of the finest view, with at most ``max_crop_samples``
/// (default 512) samples a side. Each pair's fit starts from a grid of normals
/// up to ``max_grid_tilt_deg`` (default 85.0) in steps of
/// ``grid_tilt_step_deg`` (default 10.0) and ``grid_azimuth_step_deg``
/// (default 30.0), weighs its residuals with Tukey's constant ``tukey_c``
/// (default 4.685), and has a σ from the crops' texture, thresholded against
/// noise at ``noise_false_alarm`` (default 0.001) over squares of
/// ``correlation_length_px`` (default 8.0), with a floor of ``flow_floor_px``
/// (default 0.03) on the flow's precision. A pair joins a radius's combined
/// normal when it lies within ``agreement_sigmas`` (default 3.0) times its σ
/// plus ``agreement_margin_deg`` (default 5.0) of the pairs' medoid.
///
/// The ``flow_`` keywords set the fields of the DIS flow's settings of the
/// same name without the prefix, as :func:`sfmtool.flow.compute_optical_flow`
/// takes them: ``flow_patch_size``, ``flow_patch_stride``,
/// ``flow_grad_descent_iterations``, ``flow_finest_scale``,
/// ``flow_coarsest_scale``, ``flow_variational_refinement``,
/// ``flow_variational_jacobi_iterations``,
/// ``flow_variational_outer_iterations_base``, ``flow_variational_alpha``,
/// ``flow_variational_gamma``, ``flow_variational_delta`` and
/// ``flow_normalize_patches``. Their defaults are the ``"high_quality"``
/// preset's with ``finest_scale`` 0 and ``patch_stride`` 1. The flow runs on
/// the CPU, so ``gpu_min_pixels`` is not taken. A pair whose crops at its
/// level are narrower than ``flow_patch_size`` has no flow and no fits.
/// Every default is the Rust step's, and a setting outside what the step
/// takes is refused before the photographs are decoded.
///
/// Only the outcomes ``"turned"`` and ``"single_pair"`` turn the patch; under
/// ``"undetermined"`` and ``"unstable"`` the track is read back at the normal
/// it was given. The centre does not move and no keypoint is written.
/// ``images`` and the four keywords after ``resolution`` are :func:`fit`'s,
/// and ``resolution`` (at least 2) is the patch resolution of the reading the
/// step ends with where the reconstruction stores no patch bitmaps; together
/// they set that reading.
///
/// Returns ``(EditableTrack, report)``, the report described under
/// :func:`grid_plane_normal`. Here ``estimate`` has ``kind``
/// ``"pairwise_flow"`` and carries ``outcome`` (the last round's:
/// ``"turned"``, ``"single_pair"``, ``"undetermined"`` or ``"unstable"``),
/// ``chosen_radius``, ``normal`` (facing as the given normal faces),
/// ``sigma_deg`` and ``turn_deg`` (each ``None`` where the patch was not
/// turned) and ``rounds``, one dict per round in the order run, with
/// ``plane_normal`` (the plane the crops were rendered on, facing the views),
/// ``sample_spacing`` (world units per grid sample) and ``samples_per_patch``
/// (both ``NaN`` where no view saw the plane), ``crop_half_width`` (grid
/// samples), ``crops``, ``pairs``, ``radii``, ``outcome`` (that round's,
/// on that round alone), ``chosen_radius``, ``normal`` (the round's answer,
/// facing the views, or ``None``), ``sigma_deg``, ``from_mean_view_deg`` and
/// ``next_plane`` (``None`` where no round follows).
///
/// ``crops`` has one dict per observation, ``kind`` naming what the round
/// made of it: ``"rendered"`` with ``image``, ``angle_deg`` (between the
/// view's line of sight and the plane's normal), ``samples_per_pixel`` and
/// ``coverage``; ``"too_oblique"`` with ``image`` and ``angle_deg``; or
/// ``"not_attempted"``. ``pairs`` has one dict per pair read, in observation
/// order: ``observations``, ``images``, ``angle_deg`` (between the two
/// views), ``level`` (the pyramid level read), ``half_width`` (of the square
/// read, samples at that level) and ``fits``, one per radius, ``None`` where
/// the pair has no fit there, else a dict with ``normal``, ``sigma_deg``
/// (the larger of ``sigma_texture_deg`` and ``sigma_residual_deg``),
/// ``sigma_texture_deg``, ``sigma_residual_deg``, ``offset`` (``(2,)``, grid
/// samples), ``rms_residual``, ``residual_variance``, ``determined`` and
/// ``combined``. ``radii`` has one dict per radius: ``radius``, ``normal``
/// (``None`` with no determined pair), ``sigma_deg`` and ``mean_chi2``
/// (``NaN`` where there is none), ``pairs_fitted``, ``pairs_determined`` and
/// ``pairs_combined``. Vectors are float64 arrays.
///
/// With ``record_phases=True`` (default ``False``; the result is the same
/// either way) the report also carries ``phases``, as
/// :func:`grown_plane_normal`'s does. The step runs with the GIL released.
///
/// Raises ``ValueError`` with the reason when the step is refused, including a
/// setting it does not take.
#[pyfunction]
#[pyo3(signature = (
    track,
    edited,
    images,
    *,
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
    max_pairs = None,
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
    resolution = None,
    max_seed_offset_px = None,
    max_cache_bytes = None,
    sigma_px = None,
    depth_likelihood_ratio_threshold = None,
    record_phases = false,
))]
#[allow(clippy::too_many_arguments)]
fn pairwise_flow_normal(
    py: Python<'_>,
    track: &PyEditableTrack,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
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
    max_pairs: Option<usize>,
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
    resolution: Option<u32>,
    max_seed_offset_px: Option<f64>,
    max_cache_bytes: Option<usize>,
    sigma_px: Option<f64>,
    depth_likelihood_ratio_threshold: Option<f64>,
    record_phases: bool,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let defaults = PairwiseFlowNormalOptions::default();
    let flow = FlowParamOverrides {
        patch_size: flow_patch_size,
        patch_stride: flow_patch_stride,
        grad_descent_iterations: flow_grad_descent_iterations,
        finest_scale: flow_finest_scale,
        coarsest_scale: flow_coarsest_scale,
        variational_refinement: flow_variational_refinement,
        variational_jacobi_iterations: flow_variational_jacobi_iterations,
        variational_outer_iterations_base: flow_variational_outer_iterations_base,
        variational_alpha: flow_variational_alpha,
        variational_gamma: flow_variational_gamma,
        variational_delta: flow_variational_delta,
        normalize_patches: flow_normalize_patches,
        gpu_min_pixels: None,
    }
    .apply(defaults.flow.clone());
    let fit = fit_options(
        max_seed_offset_px,
        max_cache_bytes,
        sigma_px,
        depth_likelihood_ratio_threshold,
    );
    let options = PairwiseFlowNormalOptions {
        rounds: rounds.unwrap_or(defaults.rounds),
        max_view_angle_deg: max_view_angle_deg.unwrap_or(defaults.max_view_angle_deg),
        max_normal_from_mean_view_deg: max_normal_from_mean_view_deg
            .unwrap_or(defaults.max_normal_from_mean_view_deg),
        fallback_radius: fallback_radius.unwrap_or(defaults.fallback_radius),
        min_samples_per_patch: min_samples_per_patch.unwrap_or(defaults.min_samples_per_patch),
        max_samples_per_patch: max_samples_per_patch.unwrap_or(defaults.max_samples_per_patch),
        crop_half_extents: crop_half_extents.unwrap_or(defaults.crop_half_extents),
        min_crop_half_width_px: min_crop_half_width_px.unwrap_or(defaults.min_crop_half_width_px),
        max_crop_samples: max_crop_samples.unwrap_or(defaults.max_crop_samples),
        radii: radii.unwrap_or_else(|| defaults.radii.clone()),
        max_pairs: max_pairs.unwrap_or(defaults.max_pairs),
        flow,
        max_grid_tilt_deg: max_grid_tilt_deg.unwrap_or(defaults.max_grid_tilt_deg),
        grid_tilt_step_deg: grid_tilt_step_deg.unwrap_or(defaults.grid_tilt_step_deg),
        grid_azimuth_step_deg: grid_azimuth_step_deg.unwrap_or(defaults.grid_azimuth_step_deg),
        tukey_c: tukey_c.unwrap_or(defaults.tukey_c),
        correlation_length_px: correlation_length_px.unwrap_or(defaults.correlation_length_px),
        noise_false_alarm: noise_false_alarm.unwrap_or(defaults.noise_false_alarm),
        flow_floor_px: flow_floor_px.unwrap_or(defaults.flow_floor_px),
        max_sigma_deg: max_sigma_deg.unwrap_or(defaults.max_sigma_deg),
        agreement_sigmas: agreement_sigmas.unwrap_or(defaults.agreement_sigmas),
        agreement_margin_deg: agreement_margin_deg.unwrap_or(defaults.agreement_margin_deg),
        max_mean_chi2: max_mean_chi2.unwrap_or(defaults.max_mean_chi2),
        single_pair_radius: single_pair_radius.unwrap_or(defaults.single_pair_radius),
        fit: match resolution {
            Some(resolution) => fit.at_resolution(resolution),
            None => fit,
        },
    };
    // Refuse the settings before the photographs are decoded.
    options.check().map_err(refused)?;
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let views = views_of(&posed, &pyramids);
    let record = PhaseRecord::new();
    let sink = |event: Event<'_>| record.sink(event);
    let (next, report) = py
        .detach(|| {
            let progress = if record_phases {
                Progress::to(&sink).detailed(true)
            } else {
                Progress::none()
            };
            core_pairwise_flow_normal(&track.inner, &edited.inner, &views, &options, &progress)
        })
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

/// Register the pairwise-flow normal on the `sfmtool.bench` submodule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pairwise_flow_normal, m)?)?;
    Ok(())
}
