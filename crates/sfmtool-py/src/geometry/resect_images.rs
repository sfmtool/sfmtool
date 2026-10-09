// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Binding for the held-out resection of an image set
//! (``sfmtool._sfmtool.geometry.resect_images``; see
//! ``specs/gui/edits/resect-image.md``).
//!
//! The GUI reaches the same core primitive through
//! `crates/sfm-explorer/src/resect.rs`, on a one-element set; this is the
//! offline caller's door to it. Targets are named rather than indexed, because
//! a name is what a reconstruction stores and what a script has in hand.

use std::path::{Path, PathBuf};

use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::geometry::batch_resection::ResectOptions;
use sfmtool_core::geometry::resect_images::{
    resect_images as core_resect_images, ResectImageError, ResectImageOptions, ResectImageReport,
    ResectSource, ResectTotals, DEFAULT_MAX_CLUSTER_RESIDUAL_PX,
};
use sfmtool_matches_format::MatchesData;

use crate::helpers::{os_err, value_err};
use crate::PySfmrReconstruction;

/// Map a core resection error onto the Python exception the caller sees.
///
/// Everything that stops the call from being *attempted* raises; a refused
/// estimate is one target's outcome and comes back in that target's report.
pub(crate) fn err_to_py(e: ResectImageError) -> PyErr {
    match e {
        ResectImageError::Observations(_) => os_err(e),
        _ => value_err(e),
    }
}

/// One target's report: every field of `ResectImageReport`, plus the `refused`
/// convenience negation of `accepted`.
pub(crate) fn report_to_py<'py>(
    py: Python<'py>,
    r: &ResectImageReport,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("image_index", r.image_index)?;
    d.set_item("image_name", &r.image_name)?;
    d.set_item("source", r.source)?;
    d.set_item("rotation_only", r.rotation_only)?;
    d.set_item("correspondences", r.correspondences)?;
    d.set_item("track_correspondences", r.track_correspondences)?;
    d.set_item("bearing_correspondences", r.bearing_correspondences)?;
    d.set_item("cluster_correspondences", r.cluster_correspondences)?;
    d.set_item("inliers", r.inliers)?;
    d.set_item("track_inliers", r.track_inliers)?;
    d.set_item("bearing_inliers", r.bearing_inliers)?;
    d.set_item("cluster_inliers", r.cluster_inliers)?;
    d.set_item("clusters_considered", r.clusters_considered)?;
    d.set_item("clusters_skipped", r.clusters_skipped)?;
    d.set_item("clusters_untracked", r.clusters_untracked)?;
    d.set_item("clusters_failed", r.clusters_failed)?;
    d.set_item("clusters_inconsistent", r.clusters_inconsistent)?;
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
    d.set_item("track_correspondences", t.track_correspondences)?;
    d.set_item("bearing_correspondences", t.bearing_correspondences)?;
    d.set_item("cluster_correspondences", t.cluster_correspondences)?;
    d.set_item("inliers", t.inliers)?;
    d.set_item("track_inliers", t.track_inliers)?;
    d.set_item("bearing_inliers", t.bearing_inliers)?;
    d.set_item("cluster_inliers", t.cluster_inliers)?;
    d.set_item("clusters_untracked", t.clusters_untracked)?;
    d.set_item("clusters_inconsistent", t.clusters_inconsistent)?;
    d.set_item("inlier_fraction", t.inlier_fraction)?;
    d.set_item("held_out_points", t.held_out_points)?;
    d.set_item("retriangulated", t.retriangulated)?;
    d.set_item("removed_points", t.removed_points)?;
    d.set_item("scene_scale", t.scene_scale)?;
    Ok(d)
}

/// Re-estimate a set of images' poses against structure held out from all of
/// them.
///
/// The points and directions the targets observe are re-derived from the
/// non-target images alone, each target's pose is re-estimated against them
/// (RANSAC P3P with refinement, or a rotation-only fit for a target with fewer
/// than ``min_obs`` finite pairs), and the points the accepted targets observe
/// are re-triangulated at their new poses. No bundle adjustment runs. The
/// mechanism, the cluster-pair rules and the report fields are described in
/// ``specs/gui/edits/resect-image.md``.
///
/// The input reconstruction is never modified. A target whose estimate misses
/// ``accept_gate``, or that has no support, is **refused** rather than
/// raising: it keeps its stored pose and its report says why.
///
/// Args:
///     reconstruction: The source ``SfmrReconstruction``. Left untouched.
///     image_names: The targets' workspace-relative names as the
///         reconstruction stores them (e.g. ``["frames/000123.jpg"]``).
///     cluster_patches_path: Optional ``.matches`` file with both the clusters
///         and the cluster-patches sections. With it, clusters add pairs
///         beside the tracks' (pose estimate only; they create no points).
///     min_obs: Held-out finite correspondences below which a target takes the
///         rotation-only path (default 8).
///     accept_gate: Accept an estimate at or above this inlier fraction
///         (default 0.30).
///     seed: RANSAC seed; the same inputs and seed give a bit-identical
///         answer (default 0).
///     max_cluster_residual_px: The farthest, in pixels, a cluster's
///         non-target member may lie from the reprojection of the cluster's
///         triangulated position (default 1.5). Pass ``float("inf")`` to keep
///         every cluster that triangulates.
///
/// Returns:
///     ``(reconstruction, report)``. The derived ``SfmrReconstruction``
///     differs from the source only in the accepted targets' poses and in the
///     points the set observes. The report dict carries ``images`` (one
///     per-target dict, in the order the names were given) plus the set's
///     totals: ``targets``, ``accepted``, ``refused``, ``correspondences``,
///     ``track_correspondences``, ``bearing_correspondences``,
///     ``cluster_correspondences``, ``inliers``, ``track_inliers``,
///     ``bearing_inliers``, ``cluster_inliers``, ``clusters_untracked``,
///     ``clusters_inconsistent``, ``inlier_fraction``, ``held_out_points``,
///     ``retriangulated``, ``removed_points`` and ``scene_scale``. Each
///     per-target dict carries ``image_index``, ``image_name``, ``source``
///     (``"tracks"`` or ``"tracks_and_clusters"``), ``rotation_only``, the
///     same correspondence and inlier counts, ``clusters_considered``,
///     ``clusters_skipped``, ``clusters_untracked``, ``clusters_failed``,
///     ``clusters_inconsistent``, ``inlier_fraction``, ``accepted``,
///     ``refused``, ``refusal`` (the reason or ``None``), ``rotation_deg``,
///     ``translation``, ``translation_scene`` and ``scene_scale`` (both
///     ``None`` when undefined), ``held_out_points``, ``retriangulated`` and
///     ``removed_points``.
///
/// Raises:
///     ValueError: An empty or duplicated target list, an unknown image name,
///         an unposed target, fewer than three non-target posed images, or a
///         ``.matches`` file without the clusters and cluster-patches sections.
///     OSError: An unreadable ``.matches`` file or ``.sift`` observations.
#[pyfunction]
#[pyo3(signature = (reconstruction, image_names, *, cluster_patches_path=None, min_obs=8, accept_gate=0.30, seed=0, max_cluster_residual_px=DEFAULT_MAX_CLUSTER_RESIDUAL_PX))]
#[allow(clippy::too_many_arguments)]
pub fn resect_images<'py>(
    py: Python<'py>,
    reconstruction: &PySfmrReconstruction,
    image_names: Vec<String>,
    cluster_patches_path: Option<PathBuf>,
    min_obs: usize,
    accept_gate: f64,
    seed: u64,
    max_cluster_residual_px: f64,
) -> PyResult<(PySfmrReconstruction, Bound<'py, PyDict>)> {
    let image_indexes: Vec<usize> = image_names
        .iter()
        .map(|name| {
            reconstruction
                .inner
                .image_table
                .images
                .iter()
                .position(|img| &img.name == name)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "no image named {name:?} in this reconstruction ({} images)",
                        reconstruction.inner.image_table.images.len()
                    ))
                })
        })
        .collect::<PyResult<_>>()?;

    let clusters = read_cluster_patches(py, cluster_patches_path.as_deref())?;

    let options = ResectImageOptions {
        resect: ResectOptions {
            min_obs,
            accept_gate,
            seed,
        },
        max_cluster_residual_px,
    };

    let out = py
        .detach(|| {
            core_resect_images(
                &reconstruction.inner,
                &image_indexes,
                source(clusters.as_ref()),
                &options,
            )
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

/// Read the optional cluster-patches file a resection is given, raising
/// ``OSError`` when it cannot be read.
pub(crate) fn read_cluster_patches(
    py: Python<'_>,
    path: Option<&Path>,
) -> PyResult<Option<MatchesData>> {
    path.map(|path| {
        py.detach(|| sfmtool_matches_format::read_matches(path))
            .map_err(os_err)
    })
    .transpose()
}

/// The correspondence source for an optional cluster-patches file.
pub(crate) fn source(clusters: Option<&MatchesData>) -> ResectSource<'_> {
    match clusters {
        Some(data) => ResectSource::TracksAndClusters(data),
        None => ResectSource::Tracks,
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(resect_images, m)?)?;
    Ok(())
}
