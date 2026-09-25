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

use nalgebra::Vector3;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList};

use sfmtool_core::geometry::batch_resection::ResectOptions;
use sfmtool_core::geometry::{
    resect_image_in_place, BaSchedule, ResectImageOptions, ResectInPlaceError,
};
use sfmtool_core::progress::Progress;
use sfmtool_core::reconstruction::bundle_adjust::{
    bundle_adjust as core_bundle_adjust, BundleAdjustOptions,
};
use sfmtool_core::reconstruction::edited::{
    EditedReconstruction, PointMap, PointRecord, RecordObservation, RowMap,
};
use sfmtool_core::reconstruction::move_camera::move_camera as core_move_camera;
use sfmtool_core::reconstruction::prune_covered::{
    prune_covered_observations as core_prune_covered, PruneCoveredOptions,
};
use sfmtool_core::{Point3D, RotQuaternion, Se3Transform, SfmrReconstruction};

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

/// What one edit did to point indexes.
///
/// It is what a caller holding a point index across an edit follows: a
/// selection, a stored id, a row of its own bookkeeping. Every edit that can
/// change an index answers in this one vocabulary, so a caller reads
/// :meth:`forward` and :meth:`inverse` without knowing which edit made the map.
///
/// :attr:`kind` says which case it is, and :attr:`payload` is that case's
/// content:
///
/// - ``"removed"``: the indexes that stopped resolving, as a list of ints.
///   Every other index is unchanged.
/// - ``"replaced"``: ``(before, after)`` pairs, as a list of tuples. Every
///   index not named is unchanged.
/// - ``"created"``: the indexes the created points took, as a list of ints.
///   Forward is the identity; the inverse has no answer for one of these.
/// - ``"rows"``: a whole-value edit's row map. It renumbers everything, so it
///   has no short payload and ``payload`` is ``None``; read it through
///   :meth:`forward` and :meth:`inverse`.
/// - ``"chain"``: the steps one edit took, as a list of ``PointMap``, applied
///   in order.
#[pyclass(
    name = "PointMap",
    module = "sfmtool.reconstruction",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyPointMap {
    inner: PointMap,
}

impl PyPointMap {
    /// The wrapper an edit's report hands back.
    pub fn wrap(inner: PointMap) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPointMap {
    /// Which case this map is: ``"removed"``, ``"replaced"``, ``"created"``,
    /// ``"rows"`` or ``"chain"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            PointMap::Removed(_) => "removed",
            PointMap::Replaced(_) => "replaced",
            PointMap::Created(_) => "created",
            PointMap::Rows(_) => "rows",
            PointMap::Chain(_) => "chain",
        }
    }

    /// The case's content, as the class docstring describes it.
    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            PointMap::Removed(indexes) | PointMap::Created(indexes) => {
                Ok(PyList::new(py, indexes)?.into_any())
            }
            PointMap::Replaced(pairs) => {
                Ok(PyList::new(py, pairs.iter().map(|&(from, to)| (from, to)))?.into_any())
            }
            PointMap::Rows(_) => Ok(py.None().into_bound(py)),
            PointMap::Chain(steps) => {
                let steps = steps
                    .iter()
                    .map(|step| Py::new(py, PyPointMap::wrap(step.clone())))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyList::new(py, steps)?.into_any())
            }
        }
    }

    /// Where `index` lands after this edit, or ``None`` when the point it
    /// named is gone.
    fn forward(&self, index: u32) -> Option<u32> {
        self.inner.forward(index)
    }

    /// Where `index` came from before this edit, or ``None`` when the edit
    /// created the point it names.
    fn inverse(&self, index: u32) -> Option<u32> {
        self.inner.inverse(index)
    }

    fn __repr__(&self) -> String {
        format!("PointMap({})", describe_map(&self.inner))
    }
}

/// One map as the text its `__repr__` shows, a chain's steps included.
fn describe_map(map: &PointMap) -> String {
    match map {
        PointMap::Removed(indexes) => format!("removed={indexes:?}"),
        PointMap::Replaced(pairs) => format!("replaced={pairs:?}"),
        PointMap::Created(indexes) => format!("created={indexes:?}"),
        PointMap::Rows(_) => "rows".to_string(),
        PointMap::Chain(steps) => {
            let steps: Vec<String> = steps.iter().map(describe_map).collect();
            format!("chain=[{}]", steps.join(", "))
        }
    }
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
    // `pub(crate)` so the bench bindings, which commit a track into a version,
    // can read the value and hand back the next one without a second wrapper.
    pub(crate) inner: EditedReconstruction,
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

    /// Re-estimate `image`'s pose against structure held out from it, and give
    /// back the answer as this version's successor.
    ///
    /// The estimate is ``geometry.resect_images`` on the one-element target set
    /// (see ``specs/gui/edits/resect-image.md``): the points the image observes are
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
    ///     cluster_patches_path: Optional cluster-patches ``.matches`` file.
    ///         Without it the 2D-3D pairs are the tracks' alone; with it the
    ///         file's clusters are used beside them, each as a track of its
    ///         own (see ``geometry.resect_images``).
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
    #[pyo3(signature = (image, *, cluster_patches_path=None, min_obs=8, accept_gate=0.30, seed=0))]
    fn resect_image_in_place(
        &self,
        py: Python<'_>,
        image: usize,
        cluster_patches_path: Option<std::path::PathBuf>,
        min_obs: usize,
        accept_gate: f64,
        seed: u64,
    ) -> PyResult<(PyEditedReconstruction, Py<PyDict>)> {
        let clusters = crate::geometry::resect_images::read_cluster_patches(
            py,
            cluster_patches_path.as_deref(),
        )?;
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
                resect_image_in_place(
                    &value,
                    image,
                    crate::geometry::resect_images::source(clusters.as_ref()),
                    &options,
                )
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

    /// Put image `image` at `world_from_camera`, and give back the answer as
    /// this version's successor.
    ///
    /// The pose is world-from-camera in **this reconstruction's own frame**:
    /// ``quaternion_wxyz`` carries camera axes onto world axes and
    /// ``translation`` is the camera centre. The tracks that image observes are
    /// then settled on their own evidence (see
    /// ``specs/core/reconstruction/move-camera.md``): a finite point two or more
    /// pixels see is re-triangulated at the new pose, with its patch frame
    /// rescaled and its stored error rewritten; a bearing only this image sees
    /// rotates with the camera; everything else keeps its position, a track that
    /// will not re-triangulate included. A **bulk** edit, so the value that
    /// comes back is a whole new base with an empty overlay, and this object is
    /// not changed.
    ///
    /// Returns:
    ///     ``(EditedReconstruction, report)``. The report carries ``image``,
    ///     ``rotation_deg``, ``translation``, ``translation_scene``,
    ///     ``observed``, ``retriangulated``, ``kept``, ``rotated_bearings``,
    ///     ``residual_before_px`` and ``residual_after_px`` -- the last two the
    ///     median and 90th percentile of this image's own reprojection
    ///     residuals, or ``None`` where the value carries no inline keypoints.
    ///     Raises ``ValueError`` with the reason when the move is refused.
    #[pyo3(signature = (image, quaternion_wxyz, translation))]
    fn move_camera(
        &self,
        py: Python<'_>,
        image: usize,
        quaternion_wxyz: [f64; 4],
        translation: [f64; 3],
    ) -> PyResult<(PyEditedReconstruction, Py<PyDict>)> {
        let pose = Se3Transform::new(
            RotQuaternion::from_wxyz_array(quaternion_wxyz),
            Vector3::from_row_slice(&translation),
            1.0,
        );
        let value = materialised(&self.inner);
        let (next, report) = py
            .detach(|| core_move_camera(&value, image, &pose))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let d = PyDict::new(py);
        d.set_item("image", report.image)?;
        d.set_item("rotation_deg", report.rotation_deg)?;
        d.set_item("translation", report.translation)?;
        d.set_item("translation_scene", report.translation_scene)?;
        d.set_item("observed", report.observed)?;
        d.set_item("retriangulated", report.retriangulated)?;
        d.set_item("kept", report.kept)?;
        d.set_item("rotated_bearings", report.rotated_bearings)?;
        d.set_item(
            "residual_before_px",
            report
                .residual_before_px
                .map(|r| PyArray1::from_vec(py, r.to_vec())),
        )?;
        d.set_item(
            "residual_after_px",
            report
                .residual_after_px
                .map(|r| PyArray1::from_vec(py, r.to_vec())),
        )?;
        Ok((
            PyEditedReconstruction {
                inner: EditedReconstruction::new(Arc::new(next)),
            },
            d.unbind(),
        ))
    }

    /// Retire every observation of this version that a finer tracked one
    /// covers, and give back the value that is left.
    ///
    /// Per observation the rule reads two lengths off one projection of the
    /// point's patch frame into the observing camera (see
    /// ``specs/core/reconstruction/prune-covered-observations.md``). The
    /// **radius** is the mean of the projected frame's two column norms, and
    /// the **footprint** containment is asked within is ``footprint_fraction``
    /// of it. An observation is retired where another observation in the same
    /// image, on another point, sits inside that footprint with a radius at
    /// least ``ratio`` times smaller; the coarse side goes, never the fine one.
    /// A point the value ranges or holds is never retired and still covers, and
    /// a point left under ``min_observations`` is dropped with its survivors.
    ///
    /// Nothing is re-solved: no point, camera or lens moves, and a surviving
    /// point keeps its position, frame, bitmap, colour and constraint. A
    /// **bulk** edit, so the value that comes back is a whole new base with an
    /// empty overlay, and this object is not changed. A prune that retires
    /// nothing gives this version back as it stands, with ``changed`` false.
    ///
    /// Args:
    ///     footprint_fraction: The fraction of an observation's projected
    ///         radius its footprint is (default 0.5). A patch embedded at
    ///         patch size 11 spans 5.5 feature sizes, and a keypoint's support
    ///         is stated at 2.5 of them, so the faithful value on such a file
    ///         is ``2.5 / 5.5 = 0.4545``.
    ///     ratio: How many times finer the covering observation has to be
    ///         (default 2.0, one octave).
    ///     min_fine_radius_px: A covering observation whose projected radius is
    ///         below this says nothing (default 1.0): a feature that projects
    ///         to a fraction of a pixel is a collapsed measurement.
    ///     min_observations: Surviving observations a point needs to be kept
    ///         (default 2).
    ///
    /// Returns:
    ///     ``(EditedReconstruction, report)``. The report carries ``changed``,
    ///     ``map`` (a ``PointMap``), ``points_before``, ``points_after``,
    ///     ``observations_before``, ``observations_after``,
    ///     ``degenerate_rows``, ``protected_rows``, ``protected_rows_spared``,
    ///     ``census`` (the rule's own counts) and ``bands``, a list of
    ///     ``{"band", "lower_px", "upper_px", "rows", "rows_retired",
    ///     "points_dropped"}`` coarsest first. Raises ``ValueError`` with the
    ///     reason when the prune is refused.
    // This is a Python docstring (rendered by `help()`), not Rust prose: its
    // indented `Args:` / `Returns:` continuation paragraphs read as Markdown
    // indented code blocks, which rustdoc then tries to parse as Rust.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[pyo3(signature = (
        *,
        footprint_fraction=0.5,
        ratio=2.0,
        min_fine_radius_px=1.0,
        min_observations=2,
    ))]
    fn prune_covered_observations(
        &self,
        py: Python<'_>,
        footprint_fraction: f64,
        ratio: f64,
        min_fine_radius_px: f64,
        min_observations: usize,
    ) -> PyResult<(PyEditedReconstruction, Py<PyDict>)> {
        let options = PruneCoveredOptions {
            footprint_fraction,
            ratio,
            min_fine_radius_px,
            min_observations,
        };
        let (next, map, report) = py
            .detach(|| core_prune_covered(&self.inner, &options, &Progress::none()))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let census = PyDict::new(py);
        census.set_item("rows", report.census.rows)?;
        census.set_item("pairs_contained", report.census.pairs_contained)?;
        census.set_item("pairs_finer", report.census.pairs_finer)?;
        census.set_item("rows_flagged", report.census.rows_flagged)?;
        census.set_item("rows_spared", report.census.rows_spared)?;
        census.set_item("rows_removed", report.census.rows_removed)?;
        census.set_item(
            "owners_dropped_all_covered",
            report.census.owners_dropped_all_covered,
        )?;
        census.set_item(
            "owners_dropped_by_sweep",
            report.census.owners_dropped_by_sweep,
        )?;
        census.set_item("owners_kept", report.census.owners_kept)?;

        let bands = PyList::empty(py);
        for band in &report.bands {
            let b = PyDict::new(py);
            b.set_item("band", band.band)?;
            b.set_item("lower_px", band.lower_px)?;
            b.set_item("upper_px", band.upper_px)?;
            b.set_item("rows", band.rows)?;
            b.set_item("rows_retired", band.rows_retired)?;
            b.set_item("points_dropped", band.points_dropped)?;
            bands.append(b)?;
        }

        let d = PyDict::new(py);
        d.set_item("changed", report.changed)?;
        d.set_item("map", PyPointMap::wrap(map))?;
        d.set_item("points_before", report.points_before)?;
        d.set_item("points_after", report.points_after)?;
        d.set_item("observations_before", report.observations_before)?;
        d.set_item("observations_after", report.observations_after)?;
        d.set_item("degenerate_rows", report.degenerate_rows)?;
        d.set_item("protected_rows", report.protected_rows)?;
        d.set_item("protected_rows_spared", report.protected_rows_spared)?;
        d.set_item("census", census)?;
        d.set_item("bands", bands)?;
        Ok((PyEditedReconstruction { inner: next }, d.unbind()))
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
            .detach(|| core_bundle_adjust(&value, &options, &Progress::none()))
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
