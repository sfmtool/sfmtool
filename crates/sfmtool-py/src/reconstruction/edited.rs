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

use sfmtool_core::geometry::batch_resection::ResectOptions;
use sfmtool_core::geometry::{
    resect_image_in_place, BaSchedule, ResectImageOptions, ResectInPlaceError, ResectSource,
};
use sfmtool_core::patch::normal_refine::ProjectedImage;
use sfmtool_core::reconstruction::bundle_adjust::{
    bundle_adjust as core_bundle_adjust, BundleAdjustOptions,
};
use sfmtool_core::reconstruction::edited::{
    EditedReconstruction, PointRecord, RecordObservation, RowMap,
};
use sfmtool_core::{
    add_observation, create_point, remove_observation, AddObservationOptions, CreatePointOptions,
    Point3D, SfmrReconstruction,
};

use crate::patches::views::{resolve_pyramids, PosedViews};

use super::sfmr_reconstruction::PySfmrReconstruction;

/// Turn a core edit refusal into a Python `ValueError`.
fn edit_err(e: sfmtool_core::EditError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The plain reconstruction a version is, for a **bulk** edit to run over.
///
/// Borrowed when the overlay is empty, because an empty one materialises to its
/// own base and the copy would buy nothing; owned when it is not. This is the
/// same rule the viewer applies before a bulk edit, so the two reach the core
/// function with the same value.
fn materialised(edited: &EditedReconstruction) -> std::borrow::Cow<'_, SfmrReconstruction> {
    if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
        std::borrow::Cow::Borrowed(&edited.base)
    } else {
        std::borrow::Cow::Owned(edited.materialize().0)
    }
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

    /// Remove the observation of `point` in `image` from this version.
    ///
    /// The rows that remain keep their keypoints, feature indexes and
    /// confidences, and so do the point's colour, patch frame, patch bitmap and
    /// constraint. What the removal changes is the geometry: with two or more
    /// sightings left a finite point is re-triangulated from them; with one left
    /// it becomes a bearing at infinity along that sighting's ray, its frame
    /// divided by the depth it stood at so the patch keeps its angular size;
    /// with none left the point is deleted and the report's ``deleted`` says so.
    ///
    /// No images are needed: the rays come from the poses and lenses the
    /// reconstruction already carries. Returns
    /// ``(EditedReconstruction, report)``; this object is not changed, and the
    /// returned value shares its base. Raises ``ValueError`` with the reason
    /// when the edit is refused.
    fn remove_observation(
        &self,
        py: Python<'_>,
        point: u32,
        image: u32,
    ) -> PyResult<(PyEditedReconstruction, Py<PyDict>)> {
        let (next, report) = remove_observation(&self.inner, point, image)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let d = PyDict::new(py);
        d.set_item("point", report.point)?;
        d.set_item("replaced", report.replaced)?;
        d.set_item("image", report.image)?;
        d.set_item("observation_count", report.observation_count)?;
        d.set_item("deleted", report.deleted)?;
        d.set_item("to_infinity", report.to_infinity)?;
        d.set_item("retriangulated", report.retriangulated)?;
        d.set_item("position", PyArray1::from_vec(py, report.position.to_vec()))?;
        d.set_item("position_shift", report.position_shift)?;
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

    /// Re-estimate `image`'s pose against structure held out from it, and give
    /// back the answer as this version's successor.
    ///
    /// The estimate is ``geometry.resect_images`` on the one-element target set
    /// (see ``specs/gui/resect-image.md``): the points the image observes are
    /// re-triangulated without it, its pose is fit to what is left, and the
    /// points it observes are re-triangulated again at the new pose. A **bulk**
    /// edit, so the value that comes back is a whole new base with an empty
    /// overlay, and this object is not changed.
    ///
    /// Unlike ``geometry.resect_images``, a refused estimate raises rather than
    /// coming back as a reconstruction with the stored pose in it: installed as
    /// this version's successor, that answer would be a version that moved the
    /// points and left the pose alone.
    ///
    /// Args:
    ///     image: The image's index in this version's image table.
    ///     matches_path: Optional ``.matches`` file. Without it the 2D-3D pairs
    ///         are the image's own stored observations; with it they come from
    ///         the file's match graph, which requires a ``sift_files``
    ///         reconstruction.
    ///     min_obs: Held-out finite correspondences below which the estimate
    ///         takes the rotation-only path (default 8).
    ///     accept_gate: Accept the estimate at or above this inlier fraction
    ///         (default 0.30).
    ///     seed: RANSAC seed; the same inputs and seed give a bit-identical
    ///         answer (default 0).
    ///
    /// Returns:
    ///     ``(EditedReconstruction, report)``, the report being the one target's
    ///     dict of ``geometry.resect_images``. Raises ``ValueError`` with the
    ///     reason when the estimate is refused or the call cannot be attempted,
    ///     and ``OSError`` when the observations cannot be read.
    #[pyo3(signature = (image, *, matches_path=None, min_obs=8, accept_gate=0.30, seed=0))]
    fn resect_image_in_place(
        &self,
        py: Python<'_>,
        image: usize,
        matches_path: Option<std::path::PathBuf>,
        min_obs: usize,
        accept_gate: f64,
        seed: u64,
    ) -> PyResult<(PyEditedReconstruction, Py<PyDict>)> {
        let matches: Option<matches_format::MatchesData> = match &matches_path {
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
        let value = materialised(&self.inner);
        let (next, report) = py
            .detach(|| {
                let source = match &matches {
                    Some(data) => ResectSource::Matches(data),
                    None => ResectSource::StoredObservations,
                };
                resect_image_in_place(&value, image, source, &options)
            })
            .map_err(|e| match e {
                ResectInPlaceError::Resect(e) => crate::geometry::resect_images::err_to_py(e),
                ResectInPlaceError::Refused(reason) => PyValueError::new_err(reason),
            })?;
        let d = crate::geometry::resect_images::report_to_py(py, &report)?;
        Ok((
            PyEditedReconstruction {
                inner: EditedReconstruction::new(Arc::new(next)),
            },
            d.unbind(),
        ))
    }

    /// Bundle-adjust this version, and give back the answer as its successor.
    ///
    /// Every posed image's pose, every point's position and, under ``opt_f``,
    /// the shared focal are refined together against every observation that
    /// carries a pixel (see
    /// ``specs/core/reconstruction/bundle-adjust.md``). A point at infinity goes
    /// in as the direction it is and comes back as one, a held point comes back
    /// exactly as it went in, and a point the solve leaves unsupported is
    /// deleted from the value that comes back. A **bulk** edit, so that value is
    /// a whole new base with an empty overlay, and this object is not changed.
    ///
    /// Args:
    ///     opt_f: Release the shared focal length (default ``False``). Raises
    ///         on a camera model whose focal the adjustment cannot solve.
    ///     schedule: ``[(trim_px, loss_scale), ...]`` staged rounds (default
    ///         ``[(50, 5), (12, 2), (4, 1)]``).
    ///     max_iters: LM iteration budget per round (default 60).
    ///     min_track: Trim survivors a point needs to stay in a round's solve
    ///         (default 2). A point that falls below it is deleted.
    ///     min_obs: Below this many trim survivors the round exits degenerate,
    ///         which this call raises on rather than handing back an unsolved
    ///         value (default 12).
    ///
    /// Returns:
    ///     ``(EditedReconstruction, report)``. The report carries ``images``,
    ///     ``points``, ``observations``, ``points_deleted``,
    ///     ``median_residual_before``, ``median_residual_after``,
    ///     ``focal_before``, ``focal_after`` and ``focal_released``. Raises
    ///     ``ValueError`` with the reason when the adjustment is refused.
    #[pyo3(signature = (*, opt_f=false, schedule=None, max_iters=60, min_track=2, min_obs=12))]
    fn bundle_adjust(
        &self,
        py: Python<'_>,
        opt_f: bool,
        schedule: Option<Vec<(f64, f64)>>,
        max_iters: usize,
        min_track: usize,
        min_obs: usize,
    ) -> PyResult<(PyEditedReconstruction, Py<PyDict>)> {
        let options = BundleAdjustOptions {
            opt_f,
            schedule: match schedule {
                Some(rounds) => rounds
                    .into_iter()
                    .map(|(trim_px, loss_scale)| BaSchedule {
                        trim_px,
                        loss_scale,
                    })
                    .collect(),
                None => BundleAdjustOptions::default().schedule,
            },
            max_iters,
            min_track,
            min_obs,
        };
        let value = materialised(&self.inner);
        let (next, report) = py
            .detach(|| core_bundle_adjust(&value, &options))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let d = PyDict::new(py);
        d.set_item("images", report.images)?;
        d.set_item("points", report.points)?;
        d.set_item("observations", report.observations)?;
        d.set_item("points_deleted", report.points_deleted)?;
        d.set_item("median_residual_before", report.median_residual_before)?;
        d.set_item("median_residual_after", report.median_residual_after)?;
        d.set_item("focal_before", report.focal_before)?;
        d.set_item("focal_after", report.focal_after)?;
        d.set_item("focal_released", report.focal_released)?;
        Ok((
            PyEditedReconstruction {
                inner: EditedReconstruction::new(Arc::new(next)),
            },
            d.unbind(),
        ))
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
