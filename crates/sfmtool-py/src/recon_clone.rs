// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Implementation of `SfmrReconstruction.clone_with_changes`.
//!
//! Pulled out of the `#[pymethods]` block in `py_sfmr_reconstruction.rs`: this
//! is the one large, self-contained kwargs-driven editor that hand-extracts
//! every mutable reconstruction field and rebuilds the derived caches. Its
//! Python signature lives on the thin wrapper method there; everything else is
//! here.

use nalgebra::{UnitQuaternion, Vector3};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::SfmrReconstruction;

use crate::helpers::{extract_cameras_as_sfmr, extract_rig_frame_data, py_to_u128_bytes};

/// Extract a typed numpy array from a Python value, producing a descriptive
/// `TypeError` (reporting the actual Python type and dtype) on mismatch.
///
/// `$arr_ty` is the target `numpy::PyReadonlyArrayN<$elem>`; `$shape` describes
/// the expected array shape and `$dtype` the expected dtype, both for the error
/// message. `extract_array1!`/`extract_array2!` below are the ergonomic
/// 1D/2D wrappers.
macro_rules! extract_ndarray {
    ($value:expr, $param:expr, $arr_ty:ty, $shape:expr, $dtype:expr) => {
        $value.extract::<$arr_ty>().map_err(|_| {
            let actual_type = $value
                .get_type()
                .qualname()
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            let actual_dtype = $value
                .getattr("dtype")
                .ok()
                .and_then(|d| d.str().ok())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            pyo3::exceptions::PyTypeError::new_err(format!(
                "clone_with_changes(): '{}' must be {} with dtype {}, got {} with dtype {}",
                $param, $shape, $dtype, actual_type, actual_dtype
            ))
        })
    };
}

macro_rules! extract_array1 {
    ($value:expr, $param:expr, $ty:ty) => {
        extract_ndarray!(
            $value,
            $param,
            PyReadonlyArray1<$ty>,
            "a contiguous ndarray",
            stringify!($ty)
        )
    };
}

macro_rules! extract_array2 {
    ($value:expr, $param:expr, $ty:ty) => {
        extract_ndarray!(
            $value,
            $param,
            PyReadonlyArray2<$ty>,
            "a 2D contiguous ndarray",
            stringify!($ty)
        )
    };
}

/// Create a copy of `inner` with some fields replaced from `kwargs`.
///
/// See `PySfmrReconstruction::clone_with_changes` for the public docstring and
/// the list of supported fields.
pub(crate) fn clone_with_changes(
    inner: &SfmrReconstruction,
    py: Python<'_>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<SfmrReconstruction> {
    let mut recon = inner.clone();

    let Some(kw) = kwargs else {
        return Ok(recon);
    };

    // Track whether we need to rebuild images from scratch
    let mut new_image_names: Option<Vec<String>> = None;
    let mut new_camera_indexes: Option<Vec<u32>> = None;
    // Observation-source columns collected here and recombined into the
    // `ObservationSource` enum after the image count is settled.
    let mut new_feature_tool_hashes: Option<Vec<[u8; 16]>> = None;
    let mut new_sift_content_hashes: Option<Vec<[u8; 16]>> = None;
    let mut new_image_file_hashes: Option<Vec<[u8; 16]>> = None;
    let mut new_keypoints_xy: Option<ndarray::Array2<f32>> = None;
    let mut new_feature_source: Option<String> = None;

    for (key, value) in kw.iter() {
        let key_str: String = key.extract()?;
        match key_str.as_str() {
            "positions" => {
                let arr = extract_array2!(value, "positions", f64)?;
                let s = arr.as_slice().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'positions' must be C-contiguous: {e}"
                    ))
                })?;
                let cols = arr.shape()[1];
                if cols != 3 && cols != 4 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'positions' must have shape (N, 3) \
                         [Euclidean] or (N, 4) [homogeneous], got shape ({}, {})",
                        arr.shape()[0],
                        cols
                    )));
                }
                // Allow changing number of points
                let n = arr.shape()[0];
                recon.points.resize(
                    n,
                    sfmtool_core::Point3D {
                        position: nalgebra::Point3::origin(),
                        w: 1.0,
                        color: [0, 0, 0],
                        error: 0.0,
                        normal: Vector3::zeros(),
                    },
                );
                // (N, 3) input is Euclidean (w = 1). (N, 4) input is
                // homogeneous; normalise into the ergonomic form — a finite
                // point stores its Euclidean position with w = 1, a point
                // at infinity stores a unit-length direction with w = 0.
                for (i, pt) in recon.points.iter_mut().enumerate() {
                    let off = i * cols;
                    let (x, y, z) = (s[off], s[off + 1], s[off + 2]);
                    let w = if cols == 4 { s[off + 3] } else { 1.0 };
                    if w != 0.0 {
                        pt.position = nalgebra::Point3::new(x / w, y / w, z / w);
                        pt.w = 1.0;
                    } else {
                        let dir = Vector3::new(x, y, z);
                        let norm = dir.norm();
                        if norm == 0.0 {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "clone_with_changes(): 'positions' row {i} is the \
                                 all-zero homogeneous coordinate (0, 0, 0, 0), which \
                                 denotes no point; a point at infinity (w = 0) needs \
                                 a non-zero direction"
                            )));
                        }
                        pt.position = nalgebra::Point3::from(dir / norm);
                        pt.w = 0.0;
                    }
                }
            }
            "colors" => {
                let arr = extract_array2!(value, "colors", u8)?;
                let s = arr.as_slice().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'colors' must be C-contiguous: {e}"
                    ))
                })?;
                if arr.shape()[1] != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'colors' must have shape (N, 3), \
                         got shape ({}, {})",
                        arr.shape()[0],
                        arr.shape()[1]
                    )));
                }
                if arr.shape()[0] != recon.points.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'colors' length ({}) must match point count ({}). \
                         Hint: pass 'positions' first if changing point count.",
                        arr.shape()[0],
                        recon.points.len()
                    )));
                }
                for (i, pt) in recon.points.iter_mut().enumerate() {
                    let off = i * 3;
                    pt.color = [s[off], s[off + 1], s[off + 2]];
                }
            }
            "errors" => {
                let arr = extract_array1!(value, "errors", f32)?;
                let s = arr.as_slice().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'errors' must be C-contiguous: {e}"
                    ))
                })?;
                if s.len() != recon.points.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'errors' length ({}) must match point count ({}). \
                         Hint: pass 'positions' first if changing point count.",
                        s.len(),
                        recon.points.len()
                    )));
                }
                for (i, pt) in recon.points.iter_mut().enumerate() {
                    pt.error = s[i];
                }
            }
            "normals" => {
                if value.is_none() {
                    // Opt out of normals entirely (no normals_xyz written).
                    recon.has_normals = false;
                    for pt in recon.points.iter_mut() {
                        pt.normal = Vector3::zeros();
                    }
                } else {
                    let arr = extract_array2!(value, "normals", f32)?;
                    let s = arr.as_slice().map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'normals' must be C-contiguous: {e}"
                        ))
                    })?;
                    if arr.shape()[1] != 3 {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'normals' must have shape (N, 3), \
                             got shape ({}, {})",
                            arr.shape()[0],
                            arr.shape()[1]
                        )));
                    }
                    if arr.shape()[0] != recon.points.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'normals' length ({}) must match point count ({})",
                            arr.shape()[0],
                            recon.points.len()
                        )));
                    }
                    recon.has_normals = true;
                    for (i, pt) in recon.points.iter_mut().enumerate() {
                        let off = i * 3;
                        pt.normal = Vector3::new(s[off], s[off + 1], s[off + 2]);
                    }
                }
            }
            "patches" => {
                if value.is_none() {
                    recon.patch_u_halfvec_xyz = None;
                    recon.patch_v_halfvec_xyz = None;
                    recon.patch_bitmaps_y_x_rgba = None;
                } else {
                    let cloud: PyRef<crate::py_patch_cloud::PyPatchCloud> =
                        value.extract().map_err(|_| {
                            pyo3::exceptions::PyTypeError::new_err(
                                "clone_with_changes(): 'patches' must be a PatchCloud or None",
                            )
                        })?;
                    let (u, v) = cloud.inner.to_halfvec_arrays(recon.points.len());
                    recon.patch_u_halfvec_xyz = Some(u);
                    recon.patch_v_halfvec_xyz = Some(v);
                    // The cloud carries geometry only; clear any stale bitmaps.
                    recon.patch_bitmaps_y_x_rgba = None;
                }
            }
            "quaternions_wxyz" => {
                let arr = extract_array2!(value, "quaternions_wxyz", f64)?;
                let s = arr.as_slice().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'quaternions_wxyz' must be C-contiguous: {e}"
                    ))
                })?;
                if arr.shape()[1] != 4 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'quaternions_wxyz' must have shape (N, 4), \
                         got shape ({}, {})",
                        arr.shape()[0],
                        arr.shape()[1]
                    )));
                }
                let n = arr.shape()[0];
                // Resize images if needed (when combined with image_names)
                while recon.images.len() < n {
                    recon.images.push(sfmtool_core::SfmrImage {
                        name: String::new(),
                        camera_index: 0,
                        quaternion_wxyz: UnitQuaternion::identity(),
                        translation_xyz: Vector3::zeros(),
                    });
                }
                recon.images.truncate(n);
                for (i, im) in recon.images.iter_mut().enumerate() {
                    let off = i * 4;
                    im.quaternion_wxyz = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
                        s[off],
                        s[off + 1],
                        s[off + 2],
                        s[off + 3],
                    ));
                }
            }
            "translations" => {
                let arr = extract_array2!(value, "translations", f64)?;
                let s = arr.as_slice().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'translations' must be C-contiguous: {e}"
                    ))
                })?;
                if arr.shape()[1] != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'translations' must have shape (N, 3), \
                         got shape ({}, {})",
                        arr.shape()[0],
                        arr.shape()[1]
                    )));
                }
                if arr.shape()[0] != recon.images.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'translations' length ({}) must match image count ({}). \
                         Hint: pass 'quaternions_wxyz' or 'image_names' first to resize.",
                        arr.shape()[0],
                        recon.images.len()
                    )));
                }
                for (i, im) in recon.images.iter_mut().enumerate() {
                    let off = i * 3;
                    im.translation_xyz = Vector3::new(s[off], s[off + 1], s[off + 2]);
                }
            }
            "track_image_indexes" | "track_feature_indexes" | "track_point_indexes" => {
                // These must all be set together to rebuild tracks
                // Defer to after the loop
            }
            "patch_bitmaps" => {
                // Deferred to after the loop so it always runs *after* 'patches'
                // (which clears any bitmaps), regardless of kwargs order.
            }
            "observation_counts" => {
                let arr = extract_array1!(value, "observation_counts", u32)?;
                recon.observation_counts = arr
                    .as_slice()
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "clone_with_changes(): 'observation_counts' must be C-contiguous: {e}"
                        ))
                    })?
                    .to_vec();
            }
            "image_names" => {
                new_image_names = Some(value.extract()?);
            }
            "camera_indexes" => {
                let arr = extract_array1!(value, "camera_indexes", u32)?;
                new_camera_indexes = Some(
                    arr.as_slice()
                        .map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "clone_with_changes(): 'camera_indexes' must be C-contiguous: {e}"
                            ))
                        })?
                        .to_vec(),
                );
            }
            "cameras" => {
                use sfmtool_core::CameraIntrinsics;
                let sfmr_cameras = extract_cameras_as_sfmr(&value)?;
                recon.cameras = sfmr_cameras
                    .iter()
                    .map(|sc| {
                        CameraIntrinsics::try_from(sc).map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "clone_with_changes(): failed to convert camera: {e}"
                            ))
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
            }
            "feature_tool_hashes" => {
                new_feature_tool_hashes = Some(py_to_u128_bytes(&value)?);
            }
            "sift_content_hashes" => {
                new_sift_content_hashes = Some(py_to_u128_bytes(&value)?);
            }
            "feature_source" => {
                new_feature_source = Some(value.extract()?);
            }
            "keypoints_xy" => {
                let arr = extract_array2!(value, "keypoints_xy", f32)?;
                if arr.shape()[1] != 2 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'keypoints_xy' must have shape (K, 2), \
                         got shape {:?}",
                        arr.shape()
                    )));
                }
                // Validate the row count eagerly only when the tracks are not
                // also being replaced in this call (in which case the count is
                // fixed). When tracks change too, the observation count isn't
                // known until they're rebuilt, so defer to the final
                // `validate_observation_columns`.
                let replacing_tracks = kw.contains("track_image_indexes")?
                    || kw.contains("track_feature_indexes")?
                    || kw.contains("track_point_indexes")?;
                if !replacing_tracks && arr.shape()[0] != recon.tracks.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'keypoints_xy' must have shape (K, 2) with \
                         K = observation count ({}), got shape {:?}",
                        recon.tracks.len(),
                        arr.shape()
                    )));
                }
                new_keypoints_xy = Some(arr.as_array().to_owned());
            }
            "image_file_hashes" => {
                if !value.is_none() {
                    new_image_file_hashes = Some(py_to_u128_bytes(&value)?);
                }
            }
            "thumbnails_y_x_rgb" => {
                // The `$dtype` slot also carries the shape suffix here so the
                // rendered message reproduces the legacy thumbnails wording.
                let arr = extract_ndarray!(
                    value,
                    "thumbnails_y_x_rgb",
                    numpy::PyReadonlyArray4<u8>,
                    "a 4D contiguous ndarray",
                    "uint8 and shape (N, 128, 128, 3)"
                )?;
                recon.thumbnails_y_x_rgb = arr.as_array().to_owned();
            }
            "rig_frame_data" => {
                if value.is_none() {
                    recon.rig_frame_data = None;
                } else {
                    // Wrap in a temporary dict for extract_rig_frame_data
                    let tmp = PyDict::new(py);
                    tmp.set_item("rig_frame_data", &value)?;
                    recon.rig_frame_data = extract_rig_frame_data(py, &tmp)?;
                }
            }
            "world_space_unit" => {
                if value.is_none() {
                    recon.metadata.world_space_unit = None;
                } else {
                    recon.metadata.world_space_unit = Some(value.extract()?);
                }
            }
            other => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "clone_with_changes() got unexpected keyword argument: '{other}'"
                )));
            }
        }
    }

    // Apply image-level field updates that may change the image count
    if let Some(names) = new_image_names {
        let n = names.len();
        // Resize images vec to match
        while recon.images.len() < n {
            recon.images.push(sfmtool_core::SfmrImage {
                name: String::new(),
                camera_index: 0,
                quaternion_wxyz: UnitQuaternion::identity(),
                translation_xyz: Vector3::zeros(),
            });
        }
        recon.images.truncate(n);
        for (i, im) in recon.images.iter_mut().enumerate() {
            im.name.clone_from(&names[i]);
        }
    }
    if let Some(ref indexes) = new_camera_indexes {
        if indexes.len() != recon.images.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): 'camera_indexes' length ({}) must match image count ({})",
                indexes.len(),
                recon.images.len()
            )));
        }
        for (i, im) in recon.images.iter_mut().enumerate() {
            im.camera_index = indexes[i];
        }
    }
    // Recombine the observation-source columns into the enum once the image
    // count is settled. Any column not supplied falls back to the current value.
    rebuild_observation_source(
        &mut recon,
        new_feature_source,
        new_feature_tool_hashes,
        new_sift_content_hashes,
        new_image_file_hashes,
        new_keypoints_xy,
    )?;

    // Resize depth_histogram_counts to match the (possibly new) image count.
    // When the image count changes, histogram data becomes stale so we reset it.
    if recon.depth_histogram_counts.len() != recon.images.len() {
        let num_buckets = recon.depth_statistics.num_histogram_buckets as usize;
        recon.depth_histogram_counts = vec![vec![0u32; num_buckets]; recon.images.len()];
    }

    // Per-point patch bitmaps (deferred so this wins over the 'patches' clear,
    // regardless of kwargs order). Requires the patch frame to be present —
    // either already on the reconstruction or attached via 'patches' in the same
    // call (which is processed in the loop above).
    if let Some(value) = kw.get_item("patch_bitmaps")? {
        if value.is_none() {
            recon.patch_bitmaps_y_x_rgba = None;
        } else {
            let arr = extract_ndarray!(
                value,
                "patch_bitmaps",
                numpy::PyReadonlyArray4<u8>,
                "a 4D contiguous ndarray",
                "uint8 and shape (N, R, R, 4)"
            )?;
            let shape = arr.shape();
            let npoints = recon.points.len();
            if shape[0] != npoints || shape[1] != shape[2] || shape[3] != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'patch_bitmaps' must have shape (N, R, R, 4) with \
                     N = point count ({npoints}), got shape {shape:?}"
                )));
            }
            if recon.patch_u_halfvec_xyz.is_none() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "clone_with_changes(): 'patch_bitmaps' requires the patch frame; pass \
                     'patches=<cloud>' in the same call (or on a reconstruction that already \
                     carries one)",
                ));
            }
            recon.patch_bitmaps_y_x_rgba = Some(arr.as_array().to_owned());
        }
    }

    // Rebuild tracks if any track arrays were provided
    let has_tracks = kw.contains("track_image_indexes")?
        || kw.contains("track_feature_indexes")?
        || kw.contains("track_point_indexes")?;

    if has_tracks {
        let img_idx = kw.get_item("track_image_indexes")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "clone_with_changes(): track_image_indexes, track_feature_indexes, and \
                 track_point_indexes must all be provided together",
            )
        })?;
        let img_idx: PyReadonlyArray1<u32> = extract_array1!(img_idx, "track_image_indexes", u32)?;

        let feat_idx = kw.get_item("track_feature_indexes")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "clone_with_changes(): track_image_indexes, track_feature_indexes, and \
                 track_point_indexes must all be provided together",
            )
        })?;
        let feat_idx: PyReadonlyArray1<u32> =
            extract_array1!(feat_idx, "track_feature_indexes", u32)?;

        let pt_idx = kw.get_item("track_point_indexes")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "clone_with_changes(): track_image_indexes, track_feature_indexes, and \
                 track_point_indexes must all be provided together",
            )
        })?;
        let pt_idx: PyReadonlyArray1<u32> = extract_array1!(pt_idx, "track_point_indexes", u32)?;

        let img_s = img_idx.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): 'track_image_indexes' must be C-contiguous: {e}"
            ))
        })?;
        let feat_s = feat_idx.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): 'track_feature_indexes' must be C-contiguous: {e}"
            ))
        })?;
        let pt_s = pt_idx.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): 'track_point_indexes' must be C-contiguous: {e}"
            ))
        })?;

        if img_s.len() != feat_s.len() || img_s.len() != pt_s.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): track arrays must all have the same length, \
                 got track_image_indexes={}, track_feature_indexes={}, track_point_indexes={}",
                img_s.len(),
                feat_s.len(),
                pt_s.len()
            )));
        }

        recon.tracks = (0..img_s.len())
            .map(|i| sfmtool_core::TrackObservation {
                image_index: img_s[i],
                point_index: pt_s[i],
            })
            .collect();
        // Per-observation feature indices live in the observation source for
        // sift_files reconstructions; keep them in step with the new tracks.
        // (embedded_patches has no feature indices — its per-observation data
        // is keypoints_xy, updated via the 'keypoints_xy' kwarg.)
        if let sfmtool_core::ObservationSource::SiftFiles {
            feature_indexes, ..
        } = &mut recon.observations
        {
            *feature_indexes = feat_s.to_vec();
        }

        // Derive observation_counts from the new tracks (which are grouped by
        // point) so the per-point counts/offsets don't go stale relative to the
        // replaced tracks. This overrides any 'observation_counts' kwarg, since
        // the tracks are authoritative.
        let point_count = recon.points.len();
        let mut new_counts = vec![0u32; point_count];
        for t in &recon.tracks {
            let p = t.point_index as usize;
            if p >= point_count {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'track_point_indexes' contains point index {p} \
                     out of range (point count {point_count})"
                )));
            }
            new_counts[p] += 1;
        }
        recon.observation_counts = new_counts;
    }

    // Recompute derived fields
    recon.rebuild_derived_fields();

    // The track arrays and the observation-source columns can be supplied in the
    // same call (and are applied in separate passes), so guard against leaving a
    // per-observation column out of step with the new track count — e.g.
    // replacing the tracks of an embedded_patches recon without also passing a
    // matching `keypoints_xy`.
    recon.validate_observation_columns().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("clone_with_changes(): {e}"))
    })?;

    Ok(recon)
}

/// Recombine the (optionally updated) observation-source columns into the
/// `ObservationSource` enum, falling back to the reconstruction's current values
/// for any column not supplied. The target mode is `feature_source` if given,
/// else the current one.
fn rebuild_observation_source(
    recon: &mut SfmrReconstruction,
    new_feature_source: Option<String>,
    new_feature_tool_hashes: Option<Vec<[u8; 16]>>,
    new_sift_content_hashes: Option<Vec<[u8; 16]>>,
    new_image_file_hashes: Option<Vec<[u8; 16]>>,
    new_keypoints_xy: Option<ndarray::Array2<f32>>,
) -> PyResult<()> {
    use sfmtool_core::ObservationSource;

    // Nothing to do when no observation-source kwarg was passed.
    if new_feature_source.is_none()
        && new_feature_tool_hashes.is_none()
        && new_sift_content_hashes.is_none()
        && new_image_file_hashes.is_none()
        && new_keypoints_xy.is_none()
    {
        return Ok(());
    }

    let n_img = recon.images.len();
    let require_img_len = |name: &str, len: usize| -> PyResult<()> {
        if len != n_img {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): '{name}' length ({len}) must match image count ({n_img})"
            )));
        }
        Ok(())
    };

    let target = new_feature_source.unwrap_or_else(|| recon.feature_source().to_string());

    let observations = match target.as_str() {
        "sift_files" => {
            let feature_indexes = recon.feature_indexes().map(|f| f.to_vec()).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "clone_with_changes(): converting to sift_files is not supported \
                     (feature indices cannot be supplied)",
                )
            })?;
            let feature_tool_hashes = new_feature_tool_hashes
                .or_else(|| recon.feature_tool_hashes().map(|h| h.to_vec()))
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "clone_with_changes(): sift_files requires feature_tool_hashes",
                    )
                })?;
            require_img_len("feature_tool_hashes", feature_tool_hashes.len())?;
            let sift_content_hashes = new_sift_content_hashes
                .or_else(|| recon.sift_content_hashes().map(|h| h.to_vec()))
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "clone_with_changes(): sift_files requires sift_content_hashes",
                    )
                })?;
            require_img_len("sift_content_hashes", sift_content_hashes.len())?;
            ObservationSource::SiftFiles {
                feature_indexes,
                feature_tool_hashes,
                sift_content_hashes,
            }
        }
        "embedded_patches" => {
            let keypoints_xy = new_keypoints_xy
                .or_else(|| recon.keypoints_xy().cloned())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "clone_with_changes(): embedded_patches requires keypoints_xy",
                    )
                })?;
            let image_file_hashes = new_image_file_hashes
                .or_else(|| recon.image_file_hashes().map(|h| h.to_vec()))
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "clone_with_changes(): embedded_patches requires image_file_hashes",
                    )
                })?;
            require_img_len("image_file_hashes", image_file_hashes.len())?;
            ObservationSource::EmbeddedPatches {
                keypoints_xy,
                image_file_hashes,
            }
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): unknown feature_source {other:?}"
            )));
        }
    };

    recon.metadata.feature_source = target;
    recon.observations = observations;
    Ok(())
}
