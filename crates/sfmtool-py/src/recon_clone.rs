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
    let mut new_feature_tool_hashes: Option<Vec<[u8; 16]>> = None;
    let mut new_sift_content_hashes: Option<Vec<[u8; 16]>> = None;

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
                        feature_tool_hash: [0u8; 16],
                        sift_content_hash: [0u8; 16],
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
            "track_image_indexes" | "track_feature_indexes" | "track_point_ids" => {
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
                feature_tool_hash: [0u8; 16],
                sift_content_hash: [0u8; 16],
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
    if let Some(ref hashes) = new_feature_tool_hashes {
        if hashes.len() != recon.images.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): 'feature_tool_hashes' length ({}) must match image count ({})",
                hashes.len(),
                recon.images.len()
            )));
        }
        for (i, im) in recon.images.iter_mut().enumerate() {
            im.feature_tool_hash = hashes[i];
        }
    }
    if let Some(ref hashes) = new_sift_content_hashes {
        if hashes.len() != recon.images.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): 'sift_content_hashes' length ({}) must match image count ({})",
                hashes.len(),
                recon.images.len()
            )));
        }
        for (i, im) in recon.images.iter_mut().enumerate() {
            im.sift_content_hash = hashes[i];
        }
    }

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
        || kw.contains("track_point_ids")?;

    if has_tracks {
        let img_idx = kw.get_item("track_image_indexes")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "clone_with_changes(): track_image_indexes, track_feature_indexes, and \
                 track_point_ids must all be provided together",
            )
        })?;
        let img_idx: PyReadonlyArray1<u32> = extract_array1!(img_idx, "track_image_indexes", u32)?;

        let feat_idx = kw.get_item("track_feature_indexes")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "clone_with_changes(): track_image_indexes, track_feature_indexes, and \
                 track_point_ids must all be provided together",
            )
        })?;
        let feat_idx: PyReadonlyArray1<u32> =
            extract_array1!(feat_idx, "track_feature_indexes", u32)?;

        let pt_idx = kw.get_item("track_point_ids")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "clone_with_changes(): track_image_indexes, track_feature_indexes, and \
                 track_point_ids must all be provided together",
            )
        })?;
        let pt_idx: PyReadonlyArray1<u32> = extract_array1!(pt_idx, "track_point_ids", u32)?;

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
                "clone_with_changes(): 'track_point_ids' must be C-contiguous: {e}"
            ))
        })?;

        if img_s.len() != feat_s.len() || img_s.len() != pt_s.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): track arrays must all have the same length, \
                 got track_image_indexes={}, track_feature_indexes={}, track_point_ids={}",
                img_s.len(),
                feat_s.len(),
                pt_s.len()
            )));
        }

        recon.tracks = (0..img_s.len())
            .map(|i| sfmtool_core::TrackObservation {
                image_index: img_s[i],
                feature_index: feat_s[i],
                point_index: pt_s[i],
            })
            .collect();
    }

    // Recompute derived fields
    recon.rebuild_derived_fields();

    Ok(recon)
}
