// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Implementation of `SfmrReconstruction.clone_with_changes`.
//!
//! Pulled out of the `#[pymethods]` block in `sfmr_reconstruction.rs`: this
//! is the one large, self-contained kwargs-driven editor that hand-extracts
//! every mutable reconstruction field and rebuilds the derived caches. Its
//! Python signature lives on the thin wrapper method there; everything else is
//! here.

use std::collections::HashMap;
use std::sync::Arc;

use nalgebra::{UnitQuaternion, Vector3};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::progress::Progress;
use sfmtool_core::{SfmrReconstruction, SiftKeypointFill};

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
            let actual_dtype =
                $crate::helpers::dtype_name(&$value).unwrap_or_else(|_| "unknown".to_string());
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
///
/// The work runs in two passes. The first visits the keyword arguments in the
/// order the caller passed them and hands each key to the applicator for its
/// data family: [`apply_point_field`], [`apply_image_field`] or
/// [`apply_observation_field`]. A key whose value can be applied on its own is
/// applied there; a key that depends on another key, or on a count that a later
/// key may still change, is recorded in [`DeferredChanges`]. The second pass,
/// [`finalize`], applies the recorded values in a fixed order. Because the
/// first pass follows the caller's order, a check that compares against a
/// count (for example `colors` against the point count) sees the count as the
/// keys before it left it, and when several keys fail a first-pass check the
/// first one passed raises the error. A first-pass error comes before any
/// error from [`finalize`], whose checks run in its own fixed order.
pub(crate) fn clone_with_changes(
    inner: &SfmrReconstruction,
    py: Python<'_>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<SfmrReconstruction> {
    let mut recon = inner.clone();

    let Some(kw) = kwargs else {
        return Ok(recon);
    };

    let old_point_count = recon.point_set.points.len();
    // Any one of the three track arrays marks the tracks as replaced in this
    // call; `rebuild_tracks` refuses a call that passes only some of them.
    let replacing_tracks = kw.contains("track_image_indexes")?
        || kw.contains("track_feature_indexes")?
        || kw.contains("track_point_indexes")?;
    let mut deferred = DeferredChanges::default();

    for (key, value) in kw.iter() {
        let key_str: String = key.extract()?;
        let key = key_str.as_str();
        let handled = apply_point_field(&mut recon, &mut deferred, key, &value)?
            || apply_image_field(&mut recon, &mut deferred, py, key, &value)?
            || apply_observation_field(&mut recon, &mut deferred, replacing_tracks, key, &value)?;
        if !handled {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "clone_with_changes() got unexpected keyword argument: '{key}'"
            )));
        }
    }

    finalize(
        inner,
        recon,
        kw,
        deferred,
        old_point_count,
        replacing_tracks,
    )
}

/// Values read in the first pass of `clone_with_changes` whose application
/// waits for [`finalize`], because they depend on a count or on another key
/// that a later keyword argument may still change.
#[derive(Default)]
struct DeferredChanges {
    /// `image_names`, applied first in [`finalize`]; it can change the image
    /// count.
    image_names: Option<Vec<String>>,
    /// `camera_indexes`, checked against the image count once `image_names`
    /// has settled it.
    camera_indexes: Option<Vec<u32>>,
    // Observation-source columns, recombined into the `ObservationSource`
    // enum after the image count is settled.
    feature_tool_hashes: Option<Vec<[u8; 16]>>,
    sift_content_hashes: Option<Vec<[u8; 16]>>,
    image_file_hashes: Option<Vec<[u8; 16]>>,
    keypoints_xy: Option<ndarray::Array2<f32>>,
    feature_source: Option<String>,
    /// Whether `keypoints_xy` was passed with an array.
    keypoints_given: bool,
    /// `keypoints_xy=None` on a `sift_files` value: the result carries no
    /// inline keypoint column, even when its tracks are replaced.
    drop_keypoints: bool,
    // The constraint triple, applied once the point count is settled. The
    // outer `Option` is "was the kwarg passed", the inner one "with an array,
    // or with `None` to drop the set".
    point_constraints: Option<Option<Vec<u8>>>,
    constraint_distances: Option<Option<Vec<f64>>>,
    constraint_reference_images: Option<Option<Vec<u32>>>,
    /// `reference_observations`, settled after the tracks: the outer `Option`
    /// is "was the kwarg passed", the inner one "with an array, or with `None`
    /// to drop the column".
    reference_observations: Option<Option<Vec<i32>>>,
    /// `observation_readings`, settled after the tracks and the references:
    /// the outer `Option` is "was the kwarg passed", the inner one "with a
    /// dict, or with `None` to drop the columns".
    observation_readings: Option<Option<sfmtool_core::reconstruction::ObservationReadings>>,
}

/// Apply one per-point keyword argument (positions, colors, errors, normals,
/// their confidence, the constraint triple and the patch columns).
///
/// Returns `Ok(false)` when `key` is not a per-point key.
fn apply_point_field(
    recon: &mut SfmrReconstruction,
    deferred: &mut DeferredChanges,
    key: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    match key {
        "positions" => {
            let arr = extract_array2!(value, "positions", f64)?;
            let s = to_contiguous!(arr);
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
            recon.point_set.points.resize(
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
            for (i, pt) in recon.point_set.points.iter_mut().enumerate() {
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
            let s = to_contiguous!(arr);
            if arr.shape()[1] != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'colors' must have shape (N, 3), \
                     got shape ({}, {})",
                    arr.shape()[0],
                    arr.shape()[1]
                )));
            }
            if arr.shape()[0] != recon.point_set.points.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'colors' length ({}) must match point count ({}). \
                     Hint: pass 'positions' first if changing point count.",
                    arr.shape()[0],
                    recon.point_set.points.len()
                )));
            }
            for (i, pt) in recon.point_set.points.iter_mut().enumerate() {
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
            if s.len() != recon.point_set.points.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'errors' length ({}) must match point count ({}). \
                     Hint: pass 'positions' first if changing point count.",
                    s.len(),
                    recon.point_set.points.len()
                )));
            }
            for (i, pt) in recon.point_set.points.iter_mut().enumerate() {
                pt.error = s[i];
            }
        }
        "normals" => {
            if value.is_none() {
                // Opt out of normals entirely (no normals_xyz written).
                recon.point_set.has_normals = false;
                for pt in recon.point_set.points.iter_mut() {
                    pt.normal = Vector3::zeros();
                }
            } else {
                let arr = extract_array2!(value, "normals", f32)?;
                let s = to_contiguous!(arr);
                if arr.shape()[1] != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'normals' must have shape (N, 3), \
                         got shape ({}, {})",
                        arr.shape()[0],
                        arr.shape()[1]
                    )));
                }
                if arr.shape()[0] != recon.point_set.points.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'normals' length ({}) must match point count ({})",
                        arr.shape()[0],
                        recon.point_set.points.len()
                    )));
                }
                recon.point_set.has_normals = true;
                for (i, pt) in recon.point_set.points.iter_mut().enumerate() {
                    let off = i * 3;
                    pt.normal = Vector3::new(s[off], s[off + 1], s[off + 2]);
                }
            }
        }
        "normal_confidence" => {
            // Matches the `normals` convention above: `None` clears the
            // column outright (nothing is written), an array replaces it,
            // and omitting the kwarg preserves whatever the source carried.
            if value.is_none() {
                recon.point_set.normal_confidence = None;
            } else {
                let arr = extract_array1!(value, "normal_confidence", u8)?;
                let s = arr.as_slice().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'normal_confidence' must be C-contiguous: {e}"
                    ))
                })?;
                if s.len() != recon.point_set.points.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'normal_confidence' length ({}) must match \
                         point count ({})",
                        s.len(),
                        recon.point_set.points.len()
                    )));
                }
                recon.point_set.normal_confidence = Some(s.to_vec());
            }
        }
        // The constraint triple. Each is recorded rather than applied here:
        // the three columns are one statement and the point count may still
        // be changing in this same call, so they are settled together after
        // the loop.
        "point_constraints" => {
            deferred.point_constraints = Some(if value.is_none() {
                None
            } else {
                let arr = extract_array1!(value, "point_constraints", u8)?;
                Some(to_contiguous!(arr).into_owned())
            });
        }
        "constraint_distances" => {
            deferred.constraint_distances = Some(if value.is_none() {
                None
            } else {
                let arr = extract_array1!(value, "constraint_distances", f64)?;
                Some(to_contiguous!(arr).into_owned())
            });
        }
        "constraint_reference_images" => {
            deferred.constraint_reference_images = Some(if value.is_none() {
                None
            } else {
                let arr = extract_array1!(value, "constraint_reference_images", u32)?;
                Some(to_contiguous!(arr).into_owned())
            });
        }
        "patches" | "patch_bitmaps" => {
            // Deferred to after the loop: `patches` sizes the frame by the
            // point count, which `positions` may still change in this call,
            // and `patch_bitmaps` runs after `patches`, which clears any
            // bitmaps, whatever order the two were passed in.
        }
        "reference_observations" => {
            deferred.reference_observations = Some(if value.is_none() {
                None
            } else {
                let arr = extract_array1!(value, "reference_observations", i32)?;
                Some(to_contiguous!(arr).into_owned())
            });
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Apply one per-image keyword argument (poses, names, camera assignments,
/// cameras, thumbnails and rig frames), or the reconstruction-level
/// `world_space_unit`.
///
/// Returns `Ok(false)` when `key` is not one of these keys.
fn apply_image_field(
    recon: &mut SfmrReconstruction,
    deferred: &mut DeferredChanges,
    py: Python<'_>,
    key: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    match key {
        "quaternions_wxyz" => {
            let arr = extract_array2!(value, "quaternions_wxyz", f64)?;
            let s = to_contiguous!(arr);
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
            while recon.image_table.images.len() < n {
                recon.image_table.images.push(sfmtool_core::SfmrImage {
                    name: String::new(),
                    camera_index: 0,
                    quaternion_wxyz: UnitQuaternion::identity(),
                    translation_xyz: Vector3::zeros(),
                });
            }
            recon.image_table.images.truncate(n);
            for (i, im) in recon.image_table.images.iter_mut().enumerate() {
                let off = i * 4;
                // Bit-preserving for already-unit inputs, so cloning a
                // reconstruction with its own accessor arrays round-trips
                // the poses exactly (see the helper's docs).
                im.quaternion_wxyz = sfmtool_core::reconstruction::unit_quaternion_preserving(
                    s[off],
                    s[off + 1],
                    s[off + 2],
                    s[off + 3],
                );
            }
        }
        "translations" => {
            let arr = extract_array2!(value, "translations", f64)?;
            let s = to_contiguous!(arr);
            if arr.shape()[1] != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'translations' must have shape (N, 3), \
                     got shape ({}, {})",
                    arr.shape()[0],
                    arr.shape()[1]
                )));
            }
            if arr.shape()[0] != recon.image_table.images.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'translations' length ({}) must match image count ({}). \
                     Hint: pass 'quaternions_wxyz' or 'image_names' first to resize.",
                    arr.shape()[0],
                    recon.image_table.images.len()
                )));
            }
            for (i, im) in recon.image_table.images.iter_mut().enumerate() {
                let off = i * 3;
                im.translation_xyz = Vector3::new(s[off], s[off + 1], s[off + 2]);
            }
        }
        "image_names" => {
            deferred.image_names = Some(value.extract()?);
        }
        "camera_indexes" => {
            let arr = extract_array1!(value, "camera_indexes", u32)?;
            deferred.camera_indexes = Some(
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
            let sfmr_cameras = extract_cameras_as_sfmr(value)?;
            recon.image_table.cameras = sfmr_cameras
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
        "thumbnails_y_x_rgb" if value.is_none() => {
            // Drop the column: every row of everything else is kept.
            recon.image_table.thumbnails_y_x_rgb = None;
        }
        "thumbnails_y_x_rgb" => {
            // The `$dtype` slot also carries the shape suffix here so the
            // rendered message reproduces the legacy thumbnails wording.
            let s = sfmtool_core::THUMBNAIL_SIZE;
            let arr = extract_ndarray!(
                value,
                "thumbnails_y_x_rgb",
                numpy::PyReadonlyArray4<u8>,
                "a 4D contiguous ndarray",
                format!("uint8 and shape (N, {s}, {s}, 3)")
            )?;
            let shape = arr.shape();
            if shape[1] != s || shape[2] != s || shape[3] != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'thumbnails_y_x_rgb' must have shape \
                     (N, {s}, {s}, 3), got shape {shape:?}"
                )));
            }
            recon.image_table.thumbnails_y_x_rgb =
                Some(Arc::new(arr.as_array().as_standard_layout().into_owned()));
        }
        "rig_frame_data" => {
            if value.is_none() {
                recon.image_table.rig_frame_data = None;
            } else {
                // Wrap in a temporary dict for extract_rig_frame_data
                let tmp = PyDict::new(py);
                tmp.set_item("rig_frame_data", value)?;
                recon.image_table.rig_frame_data = extract_rig_frame_data(py, &tmp)?;
            }
        }
        "world_space_unit" => {
            if value.is_none() {
                recon.metadata.world_space_unit = None;
            } else {
                recon.metadata.world_space_unit = Some(value.extract()?);
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Apply one keyword argument of the tracks and the observation source (the
/// track arrays, per-point observation counts, per-observation columns, and
/// the per-image hashes that name where the observations come from).
///
/// `replacing_tracks` says whether the call also replaces the tracks, in which
/// case a per-observation column's row count is checked only at the end.
/// Returns `Ok(false)` when `key` is not one of these keys.
fn apply_observation_field(
    recon: &mut SfmrReconstruction,
    deferred: &mut DeferredChanges,
    replacing_tracks: bool,
    key: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    match key {
        "track_image_indexes" | "track_feature_indexes" | "track_point_indexes" => {
            // These must all be set together to rebuild tracks
            // Defer to after the loop
        }
        "observation_counts" => {
            let arr = extract_array1!(value, "observation_counts", u32)?;
            recon.point_set.observation_counts = arr
                .as_slice()
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'observation_counts' must be C-contiguous: {e}"
                    ))
                })?
                .to_vec();
        }
        "feature_tool_hashes" => {
            deferred.feature_tool_hashes = Some(py_to_u128_bytes(value)?);
        }
        "sift_content_hashes" => {
            deferred.sift_content_hashes = Some(py_to_u128_bytes(value)?);
        }
        "feature_source" => {
            deferred.feature_source = Some(value.extract()?);
        }
        "observation_confidence" => {
            // Matches the `normal_confidence` convention: `None` clears the
            // column outright, an array replaces it, and omitting the kwarg
            // preserves whatever the source carried. The row count is checked
            // eagerly only when the tracks are not also being replaced in
            // this call -- when they are, the observation count is not known
            // until they are rebuilt, so the final
            // `validate_observation_columns` is what catches a desync.
            if value.is_none() {
                recon.point_set.observation_confidence = None;
            } else {
                let arr = extract_array1!(value, "observation_confidence", u8)?;
                let s = arr.as_slice().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'observation_confidence' must be \
                         C-contiguous: {e}"
                    ))
                })?;
                if !replacing_tracks && s.len() != recon.point_set.tracks.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clone_with_changes(): 'observation_confidence' length ({}) \
                         must match observation count ({})",
                        s.len(),
                        recon.point_set.tracks.len()
                    )));
                }
                recon.point_set.observation_confidence = Some(s.to_vec());
            }
        }
        "observation_readings" => {
            // `None` drops the columns, a dict replaces them, and omitting the
            // kwarg carries the rows ([`settle_observation_readings`]). The row
            // count is checked once the tracks are settled.
            deferred.observation_readings = Some(crate::readings::readings_from_py(value, None)?);
        }
        "keypoints_xy" if value.is_none() => {
            // Drop the optional inline copy a `sift_files` value carries; an
            // `embedded_patches` value's keypoints are its observations.
            match &mut recon.point_set.observations {
                sfmtool_core::ObservationSource::SiftFiles { keypoints_xy, .. } => {
                    *keypoints_xy = None;
                    deferred.drop_keypoints = true;
                }
                sfmtool_core::ObservationSource::EmbeddedPatches { .. } => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "clone_with_changes(): an embedded_patches reconstruction \
                         cannot drop keypoints_xy",
                    ));
                }
            }
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
            if !replacing_tracks && arr.shape()[0] != recon.point_set.tracks.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'keypoints_xy' must have shape (K, 2) with \
                     K = observation count ({}), got shape {:?}",
                    recon.point_set.tracks.len(),
                    arr.shape()
                )));
            }
            deferred.keypoints_xy = Some(arr.as_array().as_standard_layout().into_owned());
            deferred.keypoints_given = true;
        }
        "image_file_hashes" => {
            if !value.is_none() {
                deferred.image_file_hashes = Some(py_to_u128_bytes(value)?);
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Apply the deferred changes and rebuild the derived fields, in this order:
///
/// 1. `image_names`, then `camera_indexes`: the names can change the image
///    count, and the camera indexes are checked against the result.
/// 2. The observation source, whose per-image hash columns are checked against
///    the settled image count.
/// 3. The depth histogram, reset when the image count changed.
/// 4. `patches`, at the point count the first pass settled, then
///    `patch_bitmaps`, which wins over the clear that `patches` does, whatever
///    order the two were passed in.
/// 5. The tracks, which also recompute `observation_counts` and so override an
///    `observation_counts` value passed in the same call.
/// 6. The constraint triple, checked against the settled point count.
/// 7. `rebuild_derived_fields`.
/// 8. The inline keypoint column of a `sift_files` value whose tracks were
///    replaced, which reads image names and the rebuilt tracks.
/// 9. The patch frames' and bitmaps' row counts, checked against the settled
///    point count ([`check_patch_rows`]) before the references, so a call that
///    changed the point count and kept the old frames is refused for that.
/// 10. The reference observations, which read the rebuilt tracks
///     ([`settle_reference_observations`]).
/// 11. The checks that every per-observation and per-point column matches its
///     count.
fn finalize(
    inner: &SfmrReconstruction,
    mut recon: SfmrReconstruction,
    kw: &Bound<'_, PyDict>,
    deferred: DeferredChanges,
    old_point_count: usize,
    replacing_tracks: bool,
) -> PyResult<SfmrReconstruction> {
    let DeferredChanges {
        image_names,
        camera_indexes,
        feature_tool_hashes,
        sift_content_hashes,
        image_file_hashes,
        keypoints_xy,
        feature_source,
        keypoints_given,
        drop_keypoints,
        point_constraints,
        constraint_distances,
        constraint_reference_images,
        reference_observations,
        observation_readings,
    } = deferred;

    apply_image_count_changes(&mut recon, image_names, camera_indexes)?;

    // Recombine the observation-source columns into the enum once the image
    // count is settled. Any column not supplied falls back to the current value.
    rebuild_observation_source(
        &mut recon,
        feature_source,
        feature_tool_hashes,
        sift_content_hashes,
        image_file_hashes,
        keypoints_xy,
    )?;

    // Resize depth_histogram_counts to match the (possibly new) image count.
    // When the image count changes, histogram data becomes stale so we reset it.
    if recon.image_table.depth_histogram_counts.len() != recon.image_table.images.len() {
        let num_buckets = recon.image_table.depth_statistics.num_histogram_buckets as usize;
        recon.image_table.depth_histogram_counts =
            vec![vec![0u32; num_buckets]; recon.image_table.images.len()];
    }

    apply_patches(&mut recon, kw)?;
    apply_patch_bitmaps(&mut recon, kw)?;

    if replacing_tracks {
        rebuild_tracks(&mut recon, kw)?;
    }

    apply_point_constraints(
        &mut recon,
        old_point_count,
        point_constraints,
        constraint_distances,
        constraint_reference_images,
    )?;

    // Recompute derived fields
    recon.rebuild_derived_fields();

    // A `sift_files` value whose tracks were replaced without a keypoint
    // column of their own gets one rebuilt for the new tracks.
    if replacing_tracks && !keypoints_given && !drop_keypoints {
        carry_sift_keypoints(inner, &mut recon);
    }

    check_patch_rows(&recon)?;
    settle_reference_observations(
        inner,
        &mut recon,
        old_point_count,
        replacing_tracks,
        reference_observations,
    )?;
    settle_observation_readings(
        inner,
        &mut recon,
        old_point_count,
        replacing_tracks,
        observation_readings,
    );

    // The track arrays and the observation-source columns can be supplied in the
    // same call (and are applied in separate passes), so guard against leaving a
    // per-observation column out of step with the new track count — e.g.
    // replacing the tracks of an embedded_patches recon without also passing a
    // matching `keypoints_xy`.
    recon.validate_observation_columns().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("clone_with_changes(): {e}"))
    })?;
    // The same guard on the point axis: the constraint columns can be replaced
    // in the same call that replaces the images a distance references.
    recon.validate_point_columns().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("clone_with_changes(): {e}"))
    })?;

    Ok(recon)
}

/// Apply `image_names`, which may change the image count, then check
/// `camera_indexes` against the resulting count and apply it.
fn apply_image_count_changes(
    recon: &mut SfmrReconstruction,
    image_names: Option<Vec<String>>,
    camera_indexes: Option<Vec<u32>>,
) -> PyResult<()> {
    if let Some(names) = image_names {
        let n = names.len();
        // Resize images vec to match
        while recon.image_table.images.len() < n {
            recon.image_table.images.push(sfmtool_core::SfmrImage {
                name: String::new(),
                camera_index: 0,
                quaternion_wxyz: UnitQuaternion::identity(),
                translation_xyz: Vector3::zeros(),
            });
        }
        recon.image_table.images.truncate(n);
        for (i, im) in recon.image_table.images.iter_mut().enumerate() {
            im.name.clone_from(&names[i]);
        }
    }
    if let Some(ref indexes) = camera_indexes {
        if indexes.len() != recon.image_table.images.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): 'camera_indexes' length ({}) must match image count ({})",
                indexes.len(),
                recon.image_table.images.len()
            )));
        }
        for (i, im) in recon.image_table.images.iter_mut().enumerate() {
            im.camera_index = indexes[i];
        }
    }
    Ok(())
}

/// Apply a `patches` keyword argument. It runs after the first pass so the
/// frame is sized by the point count that `positions` settled, whatever order
/// the two were passed in.
fn apply_patches(recon: &mut SfmrReconstruction, kw: &Bound<'_, PyDict>) -> PyResult<()> {
    let Some(value) = kw.get_item("patches")? else {
        return Ok(());
    };
    if value.is_none() {
        recon.point_set.patch_u_halfvec_xyz = None;
        recon.point_set.patch_v_halfvec_xyz = None;
        recon.point_set.drop_patch_bitmaps();
    } else {
        let cloud: PyRef<crate::PyPatchCloud> = value.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "clone_with_changes(): 'patches' must be a PatchCloud or None",
            )
        })?;
        let (u, v) = cloud.inner.to_halfvec_arrays(recon.point_set.points.len());
        recon.point_set.patch_u_halfvec_xyz = Some(u);
        recon.point_set.patch_v_halfvec_xyz = Some(v);
        // The cloud carries geometry only; clear any stale bitmaps, keeping
        // the references they are to be rendered from again.
        recon.point_set.drop_patch_bitmaps();
    }
    Ok(())
}

/// Apply a `patch_bitmaps` keyword argument. It runs after [`apply_patches`]
/// so that it wins over the clear that `patches` does, and so that a patch frame
/// attached by `patches` in the same call is present when it is checked.
fn apply_patch_bitmaps(recon: &mut SfmrReconstruction, kw: &Bound<'_, PyDict>) -> PyResult<()> {
    if let Some(value) = kw.get_item("patch_bitmaps")? {
        // Either way the old column goes, and with it any pick only a display
        // render made; the references stay.
        recon.point_set.drop_patch_bitmaps();
        if !value.is_none() {
            let arr = extract_ndarray!(
                value,
                "patch_bitmaps",
                numpy::PyReadonlyArray4<u8>,
                "a 4D contiguous ndarray",
                "uint8 and shape (N, R, R, 4)"
            )?;
            let shape = arr.shape();
            let npoints = recon.point_set.points.len();
            if shape[0] != npoints || shape[1] != shape[2] || shape[3] != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clone_with_changes(): 'patch_bitmaps' must have shape (N, R, R, 4) with \
                     N = point count ({npoints}), got shape {shape:?}"
                )));
            }
            if recon.point_set.patch_u_halfvec_xyz.is_none() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "clone_with_changes(): 'patch_bitmaps' requires the patch frame; pass \
                     'patches=<cloud>' in the same call (or on a reconstruction that already \
                     carries one)",
                ));
            }
            // A column handed in is the reconstruction's own.
            recon.point_set.patch_bitmaps_y_x_rgba =
                Some(Arc::new(arr.as_array().as_standard_layout().into_owned()));
        }
    }
    Ok(())
}

/// Refuse patch frames or patch bitmaps whose row count is not `recon`'s point
/// count: a call that changes the point count and passes no `patches` keeps the
/// input's frames, which no longer describe the points.
fn check_patch_rows(recon: &SfmrReconstruction) -> PyResult<()> {
    let set = &recon.point_set;
    let point_count = set.points.len();
    let rows = [
        (
            "patch frames",
            set.patch_u_halfvec_xyz.as_ref().map(|a| a.nrows()),
        ),
        (
            "patch frames",
            set.patch_v_halfvec_xyz.as_ref().map(|a| a.nrows()),
        ),
        (
            "patch bitmaps",
            set.patch_bitmaps_y_x_rgba.as_ref().map(|a| a.shape()[0]),
        ),
    ];
    for (what, n) in rows {
        if let Some(n) = n.filter(|&n| n != point_count) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): the {what} have {n} rows but the result has \
                 {point_count} points; pass patches (and patch_bitmaps) for the new \
                 points, or patches=None to drop them"
            )));
        }
    }
    Ok(())
}

/// Settle `recon`'s reference observations once its points and tracks are
/// final, `given` the `reference_observations` keyword argument (outer `None`
/// when it was not passed).
///
/// A value passed in is taken as it is (and checked with the other point
/// columns); `None` drops the column. Either is refused where it would leave
/// the column present without patch frames or absent with them, which a save
/// would otherwise drop or fill with `-1` without a word. Otherwise the column
/// follows the patch frame: a value with no frame carries none, and one with a
/// frame carries one.
///
/// A reference names the observation the point's bitmap is, or is to be,
/// rendered from, so the column `inner` carried comes across whether or not
/// the bitmaps do -- through new bitmaps passed without references, a new
/// patch frame for the same points, and dropped bitmaps alike -- wherever the
/// points are the same points: unchanged where the tracks are, and where the
/// tracks were replaced, each point's reference moves to the observation of
/// the same image in its new track, `-1` where there is none. The same image
/// is the one at the same index when the image table is unchanged (the same
/// names in the same order), and otherwise the one with the same name, so
/// images appended or reordered around it carry it. Every row is `-1` for a
/// frame that is new.
///
/// Two changes leave no way to carry a reference, and are refused when the
/// input names any reference (a row `>= 0`) and the result has patch frames
/// without a `reference_observations` value: a changed point count, since
/// there is no mapping from the old points to the new, and replaced tracks,
/// with a changed image table, where an image a reference is in no longer has
/// its name on exactly one image of the result, since a reference is then
/// carried by its image's name. Resetting every row to `-1` there would cut
/// each stored bitmap from its observation without a word, so the caller
/// passes the remapped column (or `-1` rows) instead.
fn settle_reference_observations(
    inner: &SfmrReconstruction,
    recon: &mut SfmrReconstruction,
    old_point_count: usize,
    replacing_tracks: bool,
    given: Option<Option<Vec<i32>>>,
) -> PyResult<()> {
    use sfmtool_sfmr_format::NO_REFERENCE_OBSERVATION;

    let framed = recon.point_set.patch_u_halfvec_xyz.is_some();
    if let Some(given) = given {
        if given.is_some() != framed {
            return Err(pyo3::exceptions::PyValueError::new_err(if framed {
                "clone_with_changes(): a reconstruction with patch frames carries \
                 reference_observations; pass -1 for a point with none rather than None"
            } else {
                "clone_with_changes(): reference_observations requires patch frames, \
                 and this reconstruction has none"
            }));
        }
        recon.point_set.reference_observations = given;
        return Ok(());
    }
    let point_count = recon.point_set.points.len();
    if !framed {
        recon.point_set.reference_observations = None;
        return Ok(());
    }
    let carried = inner.point_set.reference_observations.as_ref();
    let names_any = carried.is_some_and(|old| old.iter().any(|&r| r >= 0));
    if names_any && point_count != old_point_count {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "clone_with_changes(): the point count changed and the result has patch \
             frames, but reference_observations was not passed; the input's references \
             cannot be mapped to the new points, so pass the remapped column (-1 for a \
             point with no reference observation)",
        ));
    }
    // Across replaced tracks a reference stays on its image. Where the image
    // table is unchanged (the same names in the same order) that is the same
    // index, whatever the names are. Otherwise it follows its image by name,
    // so images appended, reordered or renamed around it carry it; an image
    // name the result no longer holds once (gone, or now on two images)
    // carries nothing.
    let same_images = inner.image_table.images.len() == recon.image_table.images.len()
        && inner
            .image_table
            .images
            .iter()
            .zip(&recon.image_table.images)
            .all(|(a, b)| a.name == b.name);
    let mut new_index: HashMap<&str, Option<u32>> = HashMap::new();
    if !same_images {
        for (i, image) in recon.image_table.images.iter().enumerate() {
            new_index
                .entry(image.name.as_str())
                .and_modify(|e| *e = None)
                .or_insert(Some(i as u32));
        }
    }
    let referenced_image = |p: usize| -> Option<Option<u32>> {
        let row = inner.point_set.reference_observation_row(p)?;
        let image = inner.point_set.tracks[row].image_index;
        if same_images {
            return Some(Some(image));
        }
        let name = &inner.image_table.images[image as usize].name;
        Some(new_index.get(name.as_str()).copied().flatten())
    };
    if names_any && replacing_tracks && (0..point_count).any(|p| referenced_image(p) == Some(None))
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "clone_with_changes(): the tracks were replaced and the result has patch \
             frames, but reference_observations was not passed, and an image a reference \
             observation is in is no longer one image of image_names; the input's \
             references are carried across new tracks by image name, so pass the remapped \
             column (-1 for a point with no reference observation)",
        ));
    }
    let carried = carried.filter(|_| point_count == old_point_count);
    recon.point_set.reference_observations = Some(match carried {
        None => vec![NO_REFERENCE_OBSERVATION; point_count],
        Some(old) if !replacing_tracks => old.clone(),
        Some(_) => (0..point_count)
            .map(|p| {
                let Some(Some(image)) = referenced_image(p) else {
                    return NO_REFERENCE_OBSERVATION;
                };
                recon
                    .point_set
                    .observations_for_point(p)
                    .iter()
                    .position(|o| o.image_index == image)
                    .map_or(NO_REFERENCE_OBSERVATION, |k| k as i32)
            })
            .collect(),
    });
    // A point left with a zero frame keeps its reference: the frame is the
    // point's geometry, and the reference observation is still in its track,
    // so a later render that gives the point a frame renders from it.
    Ok(())
}

/// Settle the observation readings once the tracks and the references are.
///
/// A passed value replaces them (`None` drops them); its row count is checked
/// with the other per-observation columns. Otherwise the rows travel with
/// their observations. Where the tracks are replaced and the point count is
/// unchanged, each new observation takes the row of the observation of
/// `inner` with the same point index and the same image name, and an
/// observation `inner` does not have is not measured; where the point count
/// changed, no observation can be matched, and the columns are dropped. Where
/// the tracks are kept, every row is kept, except that a point whose reference
/// observation changed has its scores cleared, since they were read against
/// the bitmap of another reference.
fn settle_observation_readings(
    inner: &SfmrReconstruction,
    recon: &mut SfmrReconstruction,
    old_point_count: usize,
    replacing_tracks: bool,
    passed: Option<Option<sfmtool_core::reconstruction::ObservationReadings>>,
) {
    use sfmtool_core::reconstruction::{ObservationReading, ObservationReadings};
    if let Some(passed) = passed {
        recon.point_set.observation_readings = passed;
        return;
    }
    let Some(old) = inner.point_set.observation_readings.as_ref() else {
        return;
    };
    if replacing_tracks {
        if recon.point_set.points.len() != old_point_count {
            recon.point_set.observation_readings = None;
            return;
        }
        let name = |r: &SfmrReconstruction, image: u32| -> String {
            r.image_table.images[image as usize].name.clone()
        };
        let by_key: HashMap<(u32, String), ObservationReading> = inner
            .point_set
            .tracks
            .iter()
            .zip(&old.rows)
            .map(|(t, &row)| ((t.point_index, name(inner, t.image_index)), row))
            .collect();
        let rows = recon
            .point_set
            .tracks
            .iter()
            .map(|t| {
                by_key
                    .get(&(t.point_index, name(recon, t.image_index)))
                    .copied()
                    .unwrap_or(ObservationReading::NOT_MEASURED)
            })
            .collect();
        recon.point_set.observation_readings = Some(ObservationReadings {
            rows,
            options: old.options,
        });
        return;
    }
    if let (Some(before), Some(after)) = (
        inner.point_set.reference_observations.as_ref(),
        recon.point_set.reference_observations.clone(),
    ) {
        if before.len() == after.len() {
            for (p, (a, b)) in before.iter().zip(&after).enumerate() {
                if a != b {
                    recon.point_set.clear_observation_scores(p);
                }
            }
        }
    }
}

/// Replace the tracks from the three track arrays, which must all be passed,
/// and recompute `observation_counts` from them.
fn rebuild_tracks(recon: &mut SfmrReconstruction, kw: &Bound<'_, PyDict>) -> PyResult<()> {
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
    let feat_idx: PyReadonlyArray1<u32> = extract_array1!(feat_idx, "track_feature_indexes", u32)?;

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

    recon.point_set.tracks = (0..img_s.len())
        .map(|i| sfmtool_core::TrackObservation {
            image_index: img_s[i],
            point_index: pt_s[i],
        })
        .collect();
    // Per-observation feature indices live in the observation source for
    // sift_files reconstructions; keep them in step with the new tracks.
    // (embedded_patches has no feature indices — its per-observation data
    // is keypoints_xy, updated via the 'keypoints_xy' kwarg, which is also
    // how a sift_files recon carrying the optional inline column keeps that
    // column in step.)
    if let sfmtool_core::ObservationSource::SiftFiles {
        feature_indexes, ..
    } = &mut recon.point_set.observations
    {
        *feature_indexes = feat_s.to_vec();
    }

    // Derive observation_counts from the new tracks (which are grouped by
    // point) so the per-point counts/offsets don't go stale relative to the
    // replaced tracks. This overrides any 'observation_counts' kwarg, since
    // the tracks are authoritative.
    let point_count = recon.point_set.points.len();
    let mut new_counts = vec![0u32; point_count];
    for t in &recon.point_set.tracks {
        let p = t.point_index as usize;
        if p >= point_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): 'track_point_indexes' contains point index {p} \
                 out of range (point count {point_count})"
            )));
        }
        new_counts[p] += 1;
    }
    recon.point_set.observation_counts = new_counts;
    Ok(())
}

/// Rebuild the inline keypoint column of a `sift_files` `recon` whose tracks
/// replaced those of `source`.
///
/// An observation is its image and feature, so each new row whose image name
/// and feature index `source` also observes takes `source`'s pixel for it,
/// which keeps a coordinate a producer refined past the detection. Rows
/// `source` did not have are read from the `.sift` files
/// ([`SfmrReconstruction::fill_keypoints_from_sift`]). When those files cannot
/// supply them the column is dropped, as it is for a file that never had one.
fn carry_sift_keypoints(source: &SfmrReconstruction, recon: &mut SfmrReconstruction) {
    use sfmtool_core::ObservationSource;

    let ObservationSource::SiftFiles {
        feature_indexes,
        keypoints_xy,
        ..
    } = &mut recon.point_set.observations
    else {
        return;
    };
    *keypoints_xy = None;

    let mut known = std::collections::HashMap::new();
    if let (Some(old_xy), Some(old_features)) = (source.keypoints_xy(), source.feature_indexes()) {
        for (row, (obs, &feature)) in source.point_set.tracks.iter().zip(old_features).enumerate() {
            let name = source.image_table.images[obs.image_index as usize]
                .name
                .as_str();
            known.insert((name, feature), [old_xy[[row, 0]], old_xy[[row, 1]]]);
        }
    }
    let carried: Vec<Option<[f32; 2]>> = recon
        .point_set
        .tracks
        .iter()
        .zip(feature_indexes.iter())
        .map(|(obs, &feature)| {
            let name = recon.image_table.images[obs.image_index as usize]
                .name
                .as_str();
            known.get(&(name, feature)).copied()
        })
        .collect();

    if carried.iter().any(Option::is_none)
        && recon.fill_keypoints_from_sift(&Progress::none()) != SiftKeypointFill::Filled
    {
        return;
    }
    let mut column = recon
        .keypoints_xy()
        .cloned()
        .unwrap_or_else(|| ndarray::Array2::<f32>::zeros((carried.len(), 2)));
    for (row, xy) in carried.iter().enumerate() {
        if let Some([x, y]) = xy {
            column[[row, 0]] = *x;
            column[[row, 1]] = *y;
        }
    }
    if let ObservationSource::SiftFiles { keypoints_xy, .. } = &mut recon.point_set.observations {
        *keypoints_xy = Some(column);
    }
}

/// Settle the per-point constraint triple once the point count is final.
///
/// The three columns are one statement, so they are replaced together or
/// dropped together; passing one of them alone is refused rather than merged
/// into whatever the source carried, which would leave a constraint describing a
/// distance the caller never wrote.
///
/// A call that changes the point count without supplying new constraints drops
/// whatever the source carried: those rows described points this reconstruction
/// no longer has, and there is no mapping from the old point set to the new one
/// for the columns to follow. Dropping them is every point free, which is what a
/// freshly rebuilt point set is.
fn apply_point_constraints(
    recon: &mut SfmrReconstruction,
    old_point_count: usize,
    point_constraints: Option<Option<Vec<u8>>>,
    constraint_distances: Option<Option<Vec<f64>>>,
    constraint_reference_images: Option<Option<Vec<u32>>>,
) -> PyResult<()> {
    let given = [
        point_constraints.is_some(),
        constraint_distances.is_some(),
        constraint_reference_images.is_some(),
    ];
    if given.iter().all(|&g| !g) {
        if recon.point_set.points.len() != old_point_count {
            recon.point_set.point_constraints = None;
        }
        return Ok(());
    }
    if !given.iter().all(|&g| g) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "clone_with_changes(): 'point_constraints', 'constraint_distances' and \
             'constraint_reference_images' are one statement and must be passed together",
        ));
    }
    let (point_constraints, constraint_distances, constraint_reference_images) = (
        point_constraints.unwrap(),
        constraint_distances.unwrap(),
        constraint_reference_images.unwrap(),
    );
    let (Some(point_constraints), Some(constraint_distances), Some(constraint_reference_images)) = (
        point_constraints,
        constraint_distances,
        constraint_reference_images,
    ) else {
        recon.point_set.point_constraints = None;
        return Ok(());
    };
    let n_pt = recon.point_set.points.len();
    for (name, len) in [
        ("point_constraints", point_constraints.len()),
        ("constraint_distances", constraint_distances.len()),
        (
            "constraint_reference_images",
            constraint_reference_images.len(),
        ),
    ] {
        if len != n_pt {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "clone_with_changes(): '{name}' length ({len}) must match point count ({n_pt})"
            )));
        }
    }
    recon.point_set.point_constraints = Some(sfmtool_core::PointConstraintColumns {
        point_constraints,
        constraint_distances,
        constraint_reference_images,
    });
    Ok(())
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

    let n_img = recon.image_table.images.len();
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
            // The inline keypoint column is optional here: a supplied one
            // replaces it, and otherwise whatever the source carried rides
            // along. Its row count is checked by the caller's final
            // `validate_observation_columns`.
            let keypoints_xy = new_keypoints_xy.or_else(|| recon.keypoints_xy().cloned());
            ObservationSource::SiftFiles {
                feature_indexes,
                keypoints_xy,
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
    recon.point_set.observations = observations;
    Ok(())
}
