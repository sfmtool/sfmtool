// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for [`EditedReconstruction`]: a shared immutable base plus
//! the point edits made on it, the accessor that reads through the overlay, and
//! the materialisation with its row map.
//!
//! A point record crosses the boundary as a dict, one key per column, with a
//! key present exactly when the base carries that column. The same shape is
//! read back by [`PyEditedReconstruction::point`], so a caller reads a point,
//! changes one key, and hands it to `replace_point`.

use std::sync::Arc;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use sfmtool_core::patch::normal_refine::ProjectedImage;
use sfmtool_core::reconstruction::edited::{
    EditedReconstruction, PointRecord, RecordObservation, RowMap,
};
use sfmtool_core::{
    add_observation, create_point, AddObservationOptions, CreatePointOptions, Point3D,
};

use crate::patches::views::{resolve_pyramids, PosedViews};

use super::sfmr_reconstruction::PySfmrReconstruction;

/// Turn a core edit refusal into a Python `ValueError`.
fn edit_err(e: sfmtool_core::EditError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The value at `key`, or `None` when the dict does not carry the key (or
/// carries it as `None`).
fn get<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(d.get_item(key)?.filter(|v| !v.is_none()))
}

/// The value at `key`, or a `ValueError` naming the missing key.
fn require<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    get(d, key)?.ok_or_else(|| PyValueError::new_err(format!("point record needs a '{key}'")))
}

/// A fixed-length vector out of a Python sequence, refusing the wrong length
/// rather than silently taking a prefix.
fn fixed<const N: usize, T: Copy + Default>(v: Vec<T>, key: &str) -> PyResult<[T; N]> {
    if v.len() != N {
        return Err(PyValueError::new_err(format!(
            "'{key}' must hold {N} values, and holds {}",
            v.len()
        )));
    }
    let mut out = [T::default(); N];
    out.copy_from_slice(&v);
    Ok(out)
}

/// Build a [`PointRecord`] from the dict form.
///
/// Nothing here checks a record against the base -- the core's own validation
/// does that, once, at the edit -- so this only turns Python values into Rust
/// ones and refuses what is not a record at all.
fn record_from_dict(d: &Bound<'_, PyDict>) -> PyResult<PointRecord> {
    let position: [f64; 3] = fixed(require(d, "position")?.extract()?, "position")?;
    let color: [u8; 3] = match get(d, "color")? {
        Some(v) => fixed(v.extract()?, "color")?,
        None => [0, 0, 0],
    };
    let normal: [f32; 3] = match get(d, "normal")? {
        Some(v) => fixed(v.extract()?, "normal")?,
        None => [0.0, 0.0, 0.0],
    };
    let point = Point3D {
        position: nalgebra::Point3::new(position[0], position[1], position[2]),
        w: get(d, "w")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(1.0),
        color,
        error: get(d, "error")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0.0),
        normal: nalgebra::Vector3::new(normal[0], normal[1], normal[2]),
    };

    let image_indexes: Vec<u32> = require(d, "image_indexes")?.extract()?;
    let feature_indexes: Option<Vec<u32>> = get(d, "feature_indexes")?
        .map(|v| v.extract())
        .transpose()?;
    let keypoints: Option<Vec<Vec<f32>>> =
        get(d, "keypoints_xy")?.map(|v| v.extract()).transpose()?;
    let confidence: Option<Vec<u8>> = get(d, "observation_confidence")?
        .map(|v| v.extract())
        .transpose()?;
    let k = image_indexes.len();
    for (name, len) in [
        ("feature_indexes", feature_indexes.as_ref().map(|v| v.len())),
        ("keypoints_xy", keypoints.as_ref().map(|v| v.len())),
        (
            "observation_confidence",
            confidence.as_ref().map(|v| v.len()),
        ),
    ] {
        if let Some(len) = len {
            if len != k {
                return Err(PyValueError::new_err(format!(
                    "'{name}' holds {len} rows and 'image_indexes' holds {k}"
                )));
            }
        }
    }
    let observations = (0..k)
        .map(|i| {
            Ok(RecordObservation {
                image_index: image_indexes[i],
                feature_index: feature_indexes.as_ref().map(|f| f[i]),
                keypoint_xy: keypoints
                    .as_ref()
                    .map(|kp| fixed::<2, f32>(kp[i].clone(), "keypoints_xy"))
                    .transpose()?,
                confidence: confidence.as_ref().map(|c| c[i]),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let patch_bitmap = match get(d, "patch_bitmap")? {
        Some(b) => Some(b.extract::<PyReadonlyArray3<u8>>()?.as_array().to_owned()),
        None => None,
    };
    let constraint: Option<(u8, f64, u32)> =
        get(d, "constraint")?.map(|v| v.extract()).transpose()?;
    let halfvec = |key: &'static str| -> PyResult<Option<[f32; 3]>> {
        match get(d, key)? {
            Some(v) => Ok(Some(fixed(v.extract()?, key)?)),
            None => Ok(None),
        }
    };

    Ok(PointRecord {
        point,
        observations,
        patch_u_halfvec: halfvec("patch_u_halfvec")?,
        patch_v_halfvec: halfvec("patch_v_halfvec")?,
        patch_bitmap,
        normal_confidence: get(d, "normal_confidence")?
            .map(|v| v.extract())
            .transpose()?,
        constraint,
    })
}

/// The dict form of a record: every column the base carries, and no key for a
/// column it does not.
fn record_to_dict<'py>(py: Python<'py>, r: &PointRecord) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let p = &r.point;
    d.set_item(
        "position",
        PyArray1::from_vec(py, vec![p.position.x, p.position.y, p.position.z]),
    )?;
    d.set_item("w", p.w)?;
    d.set_item("color", PyArray1::from_vec(py, p.color.to_vec()))?;
    d.set_item("error", p.error)?;
    d.set_item(
        "normal",
        PyArray1::from_vec(py, vec![p.normal.x, p.normal.y, p.normal.z]),
    )?;
    d.set_item(
        "image_indexes",
        PyArray1::from_vec(py, r.observations.iter().map(|o| o.image_index).collect()),
    )?;
    if r.observations
        .first()
        .is_some_and(|o| o.feature_index.is_some())
    {
        let v: Vec<u32> = r
            .observations
            .iter()
            .map(|o| o.feature_index.unwrap_or_default())
            .collect();
        d.set_item("feature_indexes", PyArray1::from_vec(py, v))?;
    }
    if r.observations
        .first()
        .is_some_and(|o| o.keypoint_xy.is_some())
    {
        let mut kp = Array2::<f32>::zeros((r.observations.len(), 2));
        for (i, o) in r.observations.iter().enumerate() {
            let [x, y] = o.keypoint_xy.unwrap_or_default();
            kp[[i, 0]] = x;
            kp[[i, 1]] = y;
        }
        d.set_item("keypoints_xy", kp.into_pyarray(py))?;
    }
    if r.observations
        .first()
        .is_some_and(|o| o.confidence.is_some())
    {
        let v: Vec<u8> = r
            .observations
            .iter()
            .map(|o| o.confidence.unwrap_or_default())
            .collect();
        d.set_item("observation_confidence", PyArray1::from_vec(py, v))?;
    }
    if let Some(u) = r.patch_u_halfvec {
        d.set_item("patch_u_halfvec", PyArray1::from_vec(py, u.to_vec()))?;
    }
    if let Some(v) = r.patch_v_halfvec {
        d.set_item("patch_v_halfvec", PyArray1::from_vec(py, v.to_vec()))?;
    }
    if let Some(b) = &r.patch_bitmap {
        d.set_item("patch_bitmap", b.clone().into_pyarray(py))?;
    }
    if let Some(c) = r.normal_confidence {
        d.set_item("normal_confidence", c)?;
    }
    if let Some(c) = r.constraint {
        d.set_item("constraint", c)?;
    }
    Ok(d)
}

/// A reconstruction that is a shared immutable base plus the point edits made
/// on it.
///
/// Build one from a :class:`SfmrReconstruction`, delete, replace and add
/// points, read a point back through the overlay, and materialise the plain
/// reconstruction with the map from these indexes to its rows. The base is
/// never written: every edit is a change to the overlay, and an index that
/// resolved before an edit resolves to the same point after it.
#[pyclass(name = "EditedReconstruction", module = "sfmtool.reconstruction")]
pub struct PyEditedReconstruction {
    inner: EditedReconstruction,
}

#[pymethods]
impl PyEditedReconstruction {
    /// Wrap `recon` as the base of a version with no edits.
    ///
    /// The base is copied out of the Python object, so later edits to that
    /// object do not reach this one.
    #[new]
    fn new(recon: &PySfmrReconstruction) -> Self {
        Self {
            inner: EditedReconstruction::new(Arc::new(recon.inner.clone())),
        }
    }

    /// The points this version holds: the base's, less the deletions, plus the
    /// additions.
    #[getter]
    fn point_count(&self) -> usize {
        self.inner.point_count()
    }

    /// The images, which are the base's.
    #[getter]
    fn image_count(&self) -> usize {
        self.inner.image_count()
    }

    /// The base's point count, where the addition indexes start.
    #[getter]
    fn base_point_count(&self) -> usize {
        self.inner.base_point_count()
    }

    /// One past the largest index handed out; no index below it is reused.
    #[getter]
    fn index_bound(&self) -> u32 {
        self.inner.index_bound()
    }

    /// How many indexes have been deleted or replaced.
    #[getter]
    fn deleted_count(&self) -> usize {
        self.inner.deleted_points.len()
    }

    /// The `feature_source` discriminator, the base's.
    #[getter]
    fn feature_source(&self) -> &str {
        self.inner.feature_source()
    }

    /// Which optional columns the base carries, and so which keys a record must
    /// have, as a dict of flags.
    #[getter]
    fn columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("feature_indexes", self.inner.has_feature_indexes())?;
        d.set_item("keypoints_xy", self.inner.has_keypoints())?;
        d.set_item(
            "observation_confidence",
            self.inner.has_observation_confidence(),
        )?;
        d.set_item("patch_frames", self.inner.has_patch_frames())?;
        d.set_item("patch_bitmaps", self.inner.has_patch_bitmaps())?;
        d.set_item("normal_confidence", self.inner.has_normal_confidence())?;
        d.set_item("point_constraints", self.inner.has_point_constraints())?;
        Ok(d)
    }

    /// Whether `index` named a point that has since been deleted or replaced.
    fn is_deleted(&self, index: u32) -> bool {
        self.inner.is_deleted(index)
    }

    /// Every live index, ascending.
    fn live_indexes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_vec(py, self.inner.live_indexes().collect())
    }

    /// The point at `index` as a record dict, or ``None`` when that index names
    /// no live point. Read through the overlay: nothing is materialised.
    ///
    /// Every array in the dict is this call's own copy.
    fn point<'py>(&self, py: Python<'py>, index: u32) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.point(index) {
            None => Ok(None),
            Some(view) => Ok(Some(record_to_dict(py, &view.to_record())?)),
        }
    }

    /// Delete the point at `index`.
    fn delete_point(&mut self, index: u32) -> PyResult<()> {
        self.inner.delete_point(index).map_err(edit_err)
    }

    /// Replace the point at `index` with `record`, and give back the index the
    /// replacement took. The base index it occupies is remembered, so a
    /// materialisation puts it back in its place.
    fn replace_point(&mut self, index: u32, record: &Bound<'_, PyDict>) -> PyResult<u32> {
        let record = record_from_dict(record)?;
        self.inner.replace_point(index, record).map_err(edit_err)
    }

    /// Add a point the base does not hold, and give back its index.
    fn add_point(&mut self, record: &Bound<'_, PyDict>) -> PyResult<u32> {
        let record = record_from_dict(record)?;
        self.inner.add_point(record).map_err(edit_err)
    }

    /// Add an observation of `point` in `image`, at `pixel`, to this version.
    ///
    /// The clicked pixel is a seed: the point's stored patch is registered into
    /// `image` by the same two kernels, at the same parameters, that
    /// ``sfm embed-patches`` places every observation with, and the track is
    /// then re-triangulated with the new sighting in it. Returns
    /// ``(EditedReconstruction, report)``; this object is not changed, and the
    /// returned value shares its base.
    ///
    /// `images` is what every patch kernel takes -- a list of ``HxW[xC]``
    /// ``uint8`` arrays, one per image of the base, or a prebuilt
    /// :class:`ImagePyramidSet` -- because the photometric fit needs pixels and
    /// a reconstruction carries poses and lenses rather than photographs.
    ///
    /// `min_zncc` overrides the acceptance bar; the default is the localizer's
    /// own absolute floor, which is the bar the embed pass keeps an observation
    /// on. Raises ``ValueError`` with the reason when the edit is refused.
    #[pyo3(signature = (point, image, pixel, images, min_zncc = None))]
    fn add_observation(
        &self,
        py: Python<'_>,
        point: u32,
        image: u32,
        pixel: [f32; 2],
        images: &Bound<'_, PyAny>,
        min_zncc: Option<f64>,
    ) -> PyResult<(PyEditedReconstruction, Py<PyDict>)> {
        let posed = PosedViews::from_reconstruction(&self.inner.base);
        let pyramids = resolve_pyramids(&posed, images)?;
        let views: Vec<ProjectedImage<'_>> = posed
            .cameras
            .iter()
            .zip(&posed.poses)
            .zip(pyramids.as_slice())
            .map(|((camera, cam_from_world), pyramid)| ProjectedImage {
                camera,
                cam_from_world,
                pyramid,
            })
            .collect();
        let mut options = AddObservationOptions::default();
        if let Some(bar) = min_zncc {
            options.min_zncc = bar;
        }
        let (next, report) = add_observation(&self.inner, point, image, pixel, &views, &options)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let d = PyDict::new(py);
        d.set_item("point", report.point)?;
        d.set_item("replaced", report.replaced)?;
        d.set_item("image", report.image)?;
        d.set_item(
            "clicked_pixel",
            PyArray1::from_vec(py, report.clicked_pixel.to_vec()),
        )?;
        d.set_item("keypoint", PyArray1::from_vec(py, report.keypoint.to_vec()))?;
        d.set_item("shift_px", report.shift_px)?;
        d.set_item("zncc", report.zncc)?;
        d.set_item("observation_count", report.observation_count)?;
        d.set_item("position_shift", report.position_shift)?;
        d.set_item("from_infinity", report.from_infinity)?;
        d.set_item("condition_number", report.condition_number)?;
        Ok((PyEditedReconstruction { inner: next }, d.unbind()))
    }

    /// Create a point at infinity along `pixel`'s ray in `image`, with one
    /// observation there.
    ///
    /// One sighting fixes a bearing and no distance, so the point is stored as
    /// the format stores a bearing: ``w = 0``, with the pixel's unit world-space
    /// ray as its coordinate. Adding a second observation to it re-triangulates
    /// it to a finite position. Returns ``(EditedReconstruction, report)``; this
    /// object is not changed, and the returned value shares its base.
    ///
    /// `radius_px` is the patch's half-extent in this image's pixels: nothing in
    /// a pixel says how large the point's patch is, so the caller names it, and
    /// the stored frame is the angle that many pixels subtend through the camera
    /// model, distortion included.
    ///
    /// `images` is what every patch kernel takes -- a list of ``HxW[xC]``
    /// ``uint8`` arrays, one per image of the base, or a prebuilt
    /// :class:`ImagePyramidSet` -- because the colour and the patch bitmap are
    /// read out of the photograph. Raises ``ValueError`` with the reason when
    /// the edit is refused.
    #[pyo3(signature = (image, pixel, radius_px, images))]
    fn create_point(
        &self,
        py: Python<'_>,
        image: u32,
        pixel: [f32; 2],
        radius_px: f32,
        images: &Bound<'_, PyAny>,
    ) -> PyResult<(PyEditedReconstruction, Py<PyDict>)> {
        let posed = PosedViews::from_reconstruction(&self.inner.base);
        let pyramids = resolve_pyramids(&posed, images)?;
        let views: Vec<ProjectedImage<'_>> = posed
            .cameras
            .iter()
            .zip(&posed.poses)
            .zip(pyramids.as_slice())
            .map(|((camera, cam_from_world), pyramid)| ProjectedImage {
                camera,
                cam_from_world,
                pyramid,
            })
            .collect();
        let (next, report) = create_point(
            &self.inner,
            image,
            pixel,
            radius_px,
            &views,
            &CreatePointOptions::default(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let d = PyDict::new(py);
        d.set_item("point", report.point)?;
        d.set_item("image", report.image)?;
        d.set_item("pixel", PyArray1::from_vec(py, report.pixel.to_vec()))?;
        d.set_item(
            "direction",
            PyArray1::from_vec(py, report.direction.to_vec()),
        )?;
        d.set_item("radius_px", report.radius_px)?;
        d.set_item("half_extent", report.half_extent)?;
        d.set_item("color", PyArray1::from_vec(py, report.color.to_vec()))?;
        Ok((PyEditedReconstruction { inner: next }, d.unbind()))
    }

    /// The plain reconstruction this version is, with every point in its place,
    /// and the row map both ways.
    ///
    /// Returns ``(reconstruction, forward, inverse)``. ``forward`` is indexed by
    /// this version's indexes over ``0..index_bound`` and holds the new row, or
    /// ``-1`` where the index names no live point; ``inverse`` is indexed by the
    /// new rows and holds the index each came from. Both are ``int64`` copies.
    fn materialize<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        PySfmrReconstruction,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
    ) {
        let (recon, map) = self.inner.materialize();
        let point_count = recon.point_count() as u32;
        let forward = row_map_forward(&map, self.inner.index_bound());
        let inverse: Vec<i64> = map
            .inverse_dense(point_count)
            .into_iter()
            .map(|v| v as i64)
            .collect();
        (
            PySfmrReconstruction { inner: recon },
            PyArray1::from_vec(py, forward),
            PyArray1::from_vec(py, inverse),
        )
    }

    /// The base's ``content_xxh128``, computed from the value rather than from
    /// a file, and kept after the first call.
    ///
    /// It equals the hash a save of the base writes, as long as nothing stamps
    /// new metadata onto the value in between.
    fn base_content_hash(&self) -> PyResult<String> {
        Ok(self
            .inner
            .base_content_hash()
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .content_xxh128
            .clone())
    }

    /// The content hash of a point edit that would create `records` on this
    /// base: a function of the base, the images and pixels the observations
    /// name, and the points they triangulate to.
    fn point_edit_hash(&self, records: Vec<Bound<'_, PyDict>>) -> PyResult<String> {
        let records = records
            .iter()
            .map(record_from_dict)
            .collect::<PyResult<Vec<_>>>()?;
        self.inner
            .point_edit_hash(&records)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "EditedReconstruction(points={}, base_points={}, deleted={}, added={})",
            self.inner.point_count(),
            self.inner.base_point_count(),
            self.inner.deleted_points.len(),
            self.inner.replaces.len()
        )
    }
}

/// The forward row map as an `int64` array over `0..index_bound`, `-1` where
/// the index names no live point.
fn row_map_forward(map: &RowMap, index_bound: u32) -> Vec<i64> {
    map.forward_dense(index_bound)
        .into_iter()
        .map(|v| v.map_or(-1, |n| n as i64))
        .collect()
}
