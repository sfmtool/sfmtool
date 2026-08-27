// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Binding for the held-out resection of an image set
//! (``sfmtool._sfmtool.geometry.resect_images``; see
//! ``specs/gui/gui-resect-image.md``).
//!
//! The GUI reaches the same core primitive through
//! `crates/sfm-explorer/src/resect.rs`, on a one-element set; this is the
//! offline caller's door to it. Targets are named rather than indexed, because
//! a name is what a reconstruction stores and what a script has in hand.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use matches_format::MatchesData;
use sfmtool_core::geometry::batch_resection::ResectOptions;
use sfmtool_core::geometry::resect_images::{
    resect_images as core_resect_images, ResectImageError, ResectImageOptions, ResectImageReport,
    ResectSource, ResectTotals,
};

use crate::PySfmrReconstruction;

/// Map a core resection error onto the Python exception the caller sees.
///
/// Everything that stops the call from being *attempted* raises; a refused
/// estimate is one target's outcome and comes back in that target's report.
fn err_to_py(e: ResectImageError) -> PyErr {
    match e {
        ResectImageError::Observations(_) => pyo3::exceptions::PyIOError::new_err(e.to_string()),
        _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
    }
}

/// One target's report: every field of `ResectImageReport`, plus the `refused`
/// convenience negation of `accepted`.
fn report_to_py<'py>(py: Python<'py>, r: &ResectImageReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("image_index", r.image_index)?;
    d.set_item("image_name", &r.image_name)?;
    d.set_item("source", r.source)?;
    d.set_item("rotation_only", r.rotation_only)?;
    d.set_item("correspondences", r.correspondences)?;
    d.set_item("inliers", r.inliers)?;
    d.set_item("inlier_fraction", r.inlier_fraction)?;
    d.set_item("accepted", r.accepted)?;
    d.set_item("refused", !r.accepted)?;
    d.set_item("refusal", r.refusal.clone())?;
    d.set_item("rotation_deg", r.rotation_deg)?;
    d.set_item("translation", r.translation)?;
    d.set_item("translation_scene", r.translation_scene)?;
    d.set_item("scene_scale", r.scene_scale)?;
    d.set_item("held_out_points", r.held_out_points)?;
    d.set_item("retriangulated", r.retriangulated)?;
    d.set_item("removed_points", r.removed_points)?;
    Ok(d)
}

/// The set's totals, with the per-image reports under `"images"`.
fn totals_to_py<'py>(
    py: Python<'py>,
    reports: &[ResectImageReport],
    t: &ResectTotals,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let images: Vec<Bound<'py, PyDict>> = reports
        .iter()
        .map(|r| report_to_py(py, r))
        .collect::<PyResult<_>>()?;
    d.set_item("images", images)?;
    d.set_item("targets", t.targets)?;
    d.set_item("accepted", t.accepted)?;
    d.set_item("refused", t.refused)?;
    d.set_item("correspondences", t.correspondences)?;
    d.set_item("inliers", t.inliers)?;
    d.set_item("inlier_fraction", t.inlier_fraction)?;
    d.set_item("held_out_points", t.held_out_points)?;
    d.set_item("retriangulated", t.retriangulated)?;
    d.set_item("removed_points", t.removed_points)?;
    d.set_item("scene_scale", t.scene_scale)?;
    Ok(d)
}

/// Re-estimate a set of images' poses against structure held out from all of
/// them (see ``specs/gui/gui-resect-image.md``).
///
/// A stored pose was fit jointly with the points it observes, so it always
/// agrees with them. This removes the whole target set's contribution first —
/// every finite point any target observes that keeps at least two *non-target*
/// observations is re-triangulated from those alone, at the non-target images'
/// stored poses, and every direction a target observes is re-derived from the
/// non-target rotations — then re-estimates each target's pose against what is
/// left, then re-triangulates the points the accepted targets observe at their
/// new poses. No bundle adjustment runs. A point two targets share is
/// re-triangulated from neither, so holding a set out together questions the
/// group rather than its members one at a time.
///
/// A target with at least ``min_obs`` held-out finite correspondences takes the
/// finite path: RANSAC P3P polished by trimmed pose-only refinement through the
/// image's own camera model, scored by the all-observation inlier fraction at
/// the 3 px bound, run as one batch over every target sharing a camera. A
/// target below that floor takes the rotation-only path — its rotation is fit
/// in closed form to the held-out bearings it observes (trimmed and iterated)
/// and its translation is left at its stored value.
///
/// The input reconstruction is never modified; the answer is a new one. A
/// target whose estimate misses ``accept_gate``, or that has no support on
/// either path, is **refused** rather than raising: it keeps its stored pose,
/// the other targets proceed, and its report says ``refused`` with a reason.
/// Only a property of the call itself raises — an empty or duplicated target
/// list, an unknown image name, an unposed target, fewer than three non-target
/// posed images, an unjoinable ``.matches`` file (``ValueError``), or
/// unreadable ``.sift`` observations (``OSError``).
///
/// Args:
///     reconstruction: The source ``SfmrReconstruction``. Left untouched.
///     image_names: The targets' workspace-relative names as the
///         reconstruction stores them (e.g. ``["frames/000123.jpg"]``).
///         ``ValueError`` when a name is not one of its images.
///     matches_path: Optional ``.matches`` file. Without it the 2D-3D pairs
///         are each target's own stored observations; with it they come from
///         the file's match graph — the target's keypoints, to matched
///         keypoints in the non-target posed images, to those observations'
///         held-out positions. That admits points the reconstruction never
///         assigned to the target, and requires a ``sift_files``
///         reconstruction (match rows join through feature indexes).
///     min_obs: Held-out finite correspondences below which a target takes the
///         rotation-only path (default 8).
///     accept_gate: Accept an estimate at or above this inlier fraction
///         (default 0.30).
///     seed: RANSAC seed; the same inputs and seed give a bit-identical
///         answer (default 0).
///
/// Returns:
///     ``(reconstruction, report)``. The derived ``SfmrReconstruction``
///     differs from the source only in the accepted targets' poses and in the
///     points the set observes, and records the operation, the targets, the
///     correspondence source and the inlier fractions in its metadata. The
///     report dict carries ``images`` — one per-target dict, in the order the
///     names were given — plus the set's totals: ``targets``, ``accepted``,
///     ``refused``, ``correspondences``, ``inliers``, ``inlier_fraction``,
///     ``held_out_points``, ``retriangulated``, ``removed_points`` (each point
///     counted once however many targets observe it) and ``scene_scale``.
///     Each per-target dict carries ``image_index``, ``image_name``,
///     ``source`` (``"observations"`` or ``"matches"``), ``rotation_only``,
///     ``correspondences``, ``inliers``, ``inlier_fraction``, ``accepted``,
///     ``refused``, ``refusal`` (the reason or ``None``), ``rotation_deg`` and
///     ``translation`` (the move away from that image's stored pose),
///     ``translation_scene`` and ``scene_scale`` (the translation in units of
///     the source's median camera-to-structure distance, and that distance;
///     both ``None`` when it is undefined), and that target's share of
///     ``held_out_points``, ``retriangulated`` and ``removed_points``.
#[pyfunction]
#[pyo3(signature = (reconstruction, image_names, *, matches_path=None, min_obs=8, accept_gate=0.30, seed=0))]
pub fn resect_images<'py>(
    py: Python<'py>,
    reconstruction: &PySfmrReconstruction,
    image_names: Vec<String>,
    matches_path: Option<PathBuf>,
    min_obs: usize,
    accept_gate: f64,
    seed: u64,
) -> PyResult<(PySfmrReconstruction, Bound<'py, PyDict>)> {
    let image_indexes: Vec<usize> = image_names
        .iter()
        .map(|name| {
            reconstruction
                .inner
                .images
                .iter()
                .position(|img| &img.name == name)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "no image named {name:?} in this reconstruction ({} images)",
                        reconstruction.inner.images.len()
                    ))
                })
        })
        .collect::<PyResult<_>>()?;

    let matches: Option<MatchesData> = match &matches_path {
        Some(path) => Some(
            py.detach(|| matches_format::read_matches(path))
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?,
        ),
        None => None,
    };

    let options = ResectImageOptions {
        resect: ResectOptions {
            min_obs,
            accept_gate,
            seed,
        },
    };

    let out = py
        .detach(|| {
            let source = match &matches {
                Some(data) => ResectSource::Matches(data),
                None => ResectSource::StoredObservations,
            };
            core_resect_images(&reconstruction.inner, &image_indexes, source, &options)
        })
        .map_err(err_to_py)?;

    let report = totals_to_py(py, &out.reports, &out.totals)?;
    Ok((
        PySfmrReconstruction {
            inner: out.reconstruction,
        },
        report,
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(resect_images, m)?)?;
    Ok(())
}
