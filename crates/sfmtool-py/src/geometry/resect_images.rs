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

use crate::PySfmrReconstruction;

/// Map a core resection error onto the Python exception the caller sees.
///
/// Everything that stops the call from being *attempted* raises; a refused
/// estimate is one target's outcome and comes back in that target's report.
pub(crate) fn err_to_py(e: ResectImageError) -> PyErr {
    match e {
        ResectImageError::Observations(_) => pyo3::exceptions::PyIOError::new_err(e.to_string()),
        _ => pyo3::exceptions::PyValueError::new_err(e.to_string()),
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
/// them (see ``specs/gui/edits/resect-image.md``).
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
/// A target's track pairs are its observations of points with a held-out
/// position and, as bearings, of points at infinity with a held-out direction.
/// When ``cluster_patches_path`` is given, its cluster pairs stand beside them.
/// A target with at least ``min_obs`` finite pairs (tracks and clusters) takes
/// the finite path: RANSAC P3P through the image's own camera model, whose
/// minimal samples are the finite track pairs when there are at least three
/// (the tracks lead and the clusters only support) and every finite pair
/// otherwise, scored over every pair at the 3 px bound, then a trimmed
/// refinement. A bearing's residual is its angle times the camera's focal
/// length, so it constrains the rotation only. A target below that floor takes
/// the rotation-only path: its rotation is fit in closed form to the bearings
/// (trimmed and iterated) and its translation is left at its stored value.
///
/// The input reconstruction is never modified; the answer is a new one. A
/// target whose estimate misses ``accept_gate``, or that has no support on
/// either path, is **refused** rather than raising: it keeps its stored pose,
/// the other targets proceed, and its report says ``refused`` with a reason.
/// Only a property of the call itself raises — an empty or duplicated target
/// list, an unknown image name, an unposed target, fewer than three non-target
/// posed images, a ``.matches`` file without the clusters and cluster-patches
/// sections (``ValueError``), or an unreadable ``.matches`` file or ``.sift``
/// observations (``OSError``).
///
/// Args:
///     reconstruction: The source ``SfmrReconstruction``. Left untouched.
///     image_names: The targets' workspace-relative names as the
///         reconstruction stores them (e.g. ``["frames/000123.jpg"]``).
///         ``ValueError`` when a name is not one of its images.
///     cluster_patches_path: Optional cluster-patches ``.matches`` file (one
///         with both the clusters and the cluster-patches sections). Without
///         it the pairs are the tracks' alone. With it each cluster is also
///         used as a track of its own: a cluster with exactly one kept member
///         in the target and kept members in at least two non-target posed
///         images that have tracks is triangulated from those members at
///         their stored poses, and paired with the target member's refined
///         position when every one of those members lies within
///         ``max_cluster_residual_px`` of the triangulated point's
///         reprojection in its own image. A member in an image with no track
///         observation does not count. Clusters feed the pose estimate only;
///         they create no points. Works the same on ``sift_files`` and
///         ``embedded_patches`` reconstructions.
///     min_obs: Held-out finite correspondences below which a target takes the
///         rotation-only path (default 8).
///     accept_gate: Accept an estimate at or above this inlier fraction
///         (default 0.30).
///     seed: RANSAC seed; the same inputs and seed give a bit-identical
///         answer (default 0).
///     max_cluster_residual_px: The farthest, in pixels, a cluster's
///         non-target member may lie from the reprojection of the cluster's
///         triangulated position into its image; a cluster with a member
///         farther away, behind its camera or outside its frame gives no pair
///         (default 1.5). Pass ``float("inf")`` to keep every cluster that
///         triangulates.
///
/// Returns:
///     ``(reconstruction, report)``. The derived ``SfmrReconstruction``
///     differs from the source only in the accepted targets' poses and in the
///     points the set observes, and records the operation, the targets, the
///     correspondence source and the inlier fractions in its metadata. The
///     report dict carries ``images`` (one per-target dict, in the order the
///     names were given) plus the set's totals: ``targets``, ``accepted``,
///     ``refused``, ``correspondences``, ``track_correspondences``,
///     ``bearing_correspondences``, ``cluster_correspondences``, ``inliers``,
///     ``track_inliers``, ``bearing_inliers``, ``cluster_inliers``,
///     ``clusters_untracked``, ``clusters_inconsistent`` (summed over the
///     targets), ``inlier_fraction``, ``held_out_points``,
///     ``retriangulated``, ``removed_points`` (each point counted once however
///     many targets observe it) and ``scene_scale``. Each per-target dict
///     carries ``image_index``, ``image_name``, ``source`` (``"tracks"`` or
///     ``"tracks_and_clusters"``), ``rotation_only``, ``correspondences`` and
///     its split ``track_correspondences`` / ``cluster_correspondences``, with
///     ``bearing_correspondences`` the track pairs at infinity, ``inliers``
///     and its split ``track_inliers`` / ``cluster_inliers``, with
///     ``bearing_inliers`` the track inliers at infinity,
///     ``clusters_considered`` (clusters with a kept member in the image),
///     ``clusters_skipped`` (set aside by the member rules),
///     ``clusters_untracked`` (kept members in two or more non-target posed
///     images, but in fewer than two with tracks), ``clusters_failed`` (did
///     not triangulate), ``clusters_inconsistent`` (triangulated, but a member
///     lies farther than ``max_cluster_residual_px`` from the point),
///     ``inlier_fraction``, ``accepted``,
///     ``refused``, ``refusal`` (the reason or ``None``), ``rotation_deg`` and
///     ``translation`` (the move away from that image's stored pose),
///     ``translation_scene`` and ``scene_scale`` (the translation in units of
///     the source's median camera-to-structure distance, and that distance;
///     both ``None`` when it is undefined), and that target's share of
///     ``held_out_points``, ``retriangulated`` and ``removed_points``.
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
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
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
