// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for candidate track spawning.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::patch::normal_refine::ProjectedImage;
use sfmtool_core::patch::spawn::{spawn_candidate_tracks as core_spawn, SpawnParams};

use super::cloud::PyPatchCloud;
use super::views::{resolve_pyramids, resolve_scene};

/// Congeal a new candidate track at an in-plane offset from each parent patch.
///
/// Each request places a synthetic patch — the parent's frame translated to
/// ``X_p + du * hu_p + dv * hv_p``, so the offsets speak the parent's own scale
/// and stay in its plane — finds it photometrically in the given views, and
/// triangulates what was found, exactly the way a real track is congealed and
/// with the same acceptance gates. Callers choose the offsets and assemble the
/// survivors; this turns ``(parent, offset)`` requests into vetted
/// ``(position, views, keypoints)`` results, batch.
///
/// A candidate that fails a gate is reported with the stage that killed it
/// rather than dropped, so a caller can budget and diagnose on the counts.
///
/// Args:
///     views: The scene the parents live in — a :class:`CameraViews` or an
///         :class:`SfmrReconstruction` (only its cameras and poses are read;
///         the per-candidate view sets are always explicit).
///     images: An :class:`ImagePyramidSet` built from that scene, or one source
///         image (HxWxC uint8 numpy array) per view.
///     cloud: The :class:`PatchCloud` holding the parents.
///     parents: Per candidate, shape ``(n,)`` uint32 — an index into ``cloud``'s
///         patches. May repeat.
///     offsets_uv: Per candidate, shape ``(n, 2)`` float64 ``(du, dv)`` in units
///         of the parent's half-extent vectors.
///     view_sets: Per candidate, a sequence of image indices to search in
///         (typically the parent's views). An empty entry is allowed — that
///         candidate simply dies at ``too_few_views``.
///     resolution: The R×R sampling grid every stage scores on.
///     search: Localizer search half-width, in patch-grid px.
///     max_shift_px: Localizer shift gate, in source-image px.
///     subpixel_sweeps: Sub-pixel refinement outer sweeps; ``0`` skips
///         refinement and triangulates the discrete keypoints.
///     min_views: Surviving-view floor.
///     max_reproj_rms_px: Acceptance gate on the RMS reprojection error of the
///         triangulated position against the refined keypoints.
///
/// Returns:
///     A dict of per-candidate arrays plus the surviving observations in CSR
///     layout: ``status`` ``(n,)`` uint8 (0 ``spawned``, 1 ``too_few_views``,
///     2 ``bad_triangulation``, 3 ``high_reproj``), ``positions`` ``(n, 3)``
///     float64 (NaN rows for non-``spawned``), ``requested_centers``
///     ``(n, 3)`` float64 (always filled), ``reproj_rms_px`` ``(n,)`` float64,
///     ``n_views`` ``(n,)`` uint32, ``obs_offsets`` ``(n + 1,)`` uint32,
///     ``obs_view_indexes`` ``(n_obs,)`` uint32 and ``obs_keypoints_xy``
///     ``(n_obs, 2)`` float64. Observations are reported for every candidate
///     that reached triangulation, ascending by view index.
///
/// See ``specs/core/patch/candidate-track-spawning.md``.
#[pyfunction]
#[pyo3(signature = (
    views,
    images,
    cloud,
    parents,
    offsets_uv,
    view_sets,
    *,
    resolution = 24,
    search = 6.0,
    max_shift_px = 8.0,
    subpixel_sweeps = 1,
    min_views = 3,
    max_reproj_rms_px = 2.0,
))]
#[allow(clippy::too_many_arguments)]
pub fn spawn_candidate_tracks(
    py: Python<'_>,
    views: &Bound<'_, PyAny>,
    images: &Bound<'_, PyAny>,
    cloud: &PyPatchCloud,
    parents: PyReadonlyArray1<u32>,
    offsets_uv: PyReadonlyArray2<f64>,
    view_sets: Vec<Vec<u32>>,
    resolution: u32,
    search: f64,
    max_shift_px: f64,
    subpixel_sweeps: u32,
    min_views: u32,
    max_reproj_rms_px: f64,
) -> PyResult<Py<PyAny>> {
    let n = parents.shape()[0];
    if offsets_uv.shape()[1] != 2 {
        return Err(PyValueError::new_err("offsets_uv must have shape (n, 2)"));
    }
    if offsets_uv.shape()[0] != n {
        return Err(PyValueError::new_err(format!(
            "offsets_uv has {} rows but parents has {n} entries",
            offsets_uv.shape()[0]
        )));
    }
    if view_sets.len() != n {
        return Err(PyValueError::new_err(format!(
            "view_sets has {} entries but parents has {n}",
            view_sets.len()
        )));
    }

    let parent_data = to_contiguous!(parents);
    let n_patches = cloud.inner.len();
    if let Some(&bad) = parent_data.iter().find(|&&p| p as usize >= n_patches) {
        return Err(PyValueError::new_err(format!(
            "parents contains {bad}, out of range for the cloud's {n_patches} patches"
        )));
    }
    // The offsets displace the frame in world units and the output is a
    // triangulated 3D position; a point at infinity has neither, so an infinity
    // parent is a caller mistake rather than a candidate that could fail a gate.
    if let Some(&bad) = parent_data
        .iter()
        .find(|&&p| cloud.inner.patch(p as usize).w == 0.0)
    {
        return Err(PyValueError::new_err(format!(
            "parents contains {bad}, whose patch is a point at infinity; \
             candidate spawning displaces a frame in world units and triangulates \
             a finite position, so only finite parents can be spawned from"
        )));
    }

    let (posed, _recon) = resolve_scene(views)?;
    let n_images = posed.len() as u32;
    for (i, vs) in view_sets.iter().enumerate() {
        if let Some(&bad) = vs.iter().find(|&&v| v >= n_images) {
            return Err(PyValueError::new_err(format!(
                "view_sets[{i}] contains image index {bad} out of range for this \
                 scene's {n_images} views"
            )));
        }
    }

    let offsets: Vec<[f64; 2]> = to_contiguous!(offsets_uv)
        .chunks_exact(2)
        .map(|c| [c[0], c[1]])
        .collect();
    let params = SpawnParams {
        resolution,
        search,
        max_shift_px,
        subpixel_sweeps,
        min_views,
        max_reproj_rms_px,
    };

    let pyramid_set = resolve_pyramids(&posed, images)?;
    let pyramids = pyramid_set.as_slice();
    let projected: Vec<ProjectedImage<'_>> = (0..posed.len())
        .map(|i| ProjectedImage {
            camera: &posed.cameras[i],
            cam_from_world: &posed.poses[i],
            pyramid: &pyramids[i],
        })
        .collect();

    let out = py.detach(|| {
        core_spawn(
            &projected,
            &cloud.inner,
            &parent_data,
            &offsets,
            &view_sets,
            &params,
        )
    });

    // Flat + reshape so an empty batch still yields clean `(0, 3)` / `(0, 2)`.
    let n_obs = out.obs_view_indexes.len();
    let positions: Vec<f64> = out.positions.iter().flatten().copied().collect();
    let centers: Vec<f64> = out.requested_centers.iter().flatten().copied().collect();
    let keypoints: Vec<f64> = out.obs_keypoints_xy.iter().flatten().copied().collect();

    let dict = PyDict::new(py);
    dict.set_item("status", PyArray1::from_vec(py, out.status))?;
    dict.set_item(
        "positions",
        PyArray1::from_vec(py, positions).reshape([n, 3])?,
    )?;
    dict.set_item(
        "requested_centers",
        PyArray1::from_vec(py, centers).reshape([n, 3])?,
    )?;
    dict.set_item("reproj_rms_px", PyArray1::from_vec(py, out.reproj_rms_px))?;
    dict.set_item("n_views", PyArray1::from_vec(py, out.n_views))?;
    dict.set_item("obs_offsets", PyArray1::from_vec(py, out.obs_offsets))?;
    dict.set_item(
        "obs_view_indexes",
        PyArray1::from_vec(py, out.obs_view_indexes),
    )?;
    dict.set_item(
        "obs_keypoints_xy",
        PyArray1::from_vec(py, keypoints).reshape([n_obs, 2])?,
    )?;
    Ok(dict.into_any().unbind())
}
