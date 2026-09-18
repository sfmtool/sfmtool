// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the bench and the editable track.
//!
//! Two value classes, [`PyBench`] and [`PyEditableTrack`], and the steps as
//! module-level functions, exactly as the core module is shaped: a step takes a
//! value and gives back the next one plus a report, and nothing is mutated in
//! place. A script holding several candidate tracks gets the same list, the
//! same labels and the same "the active one" default a window does, with no
//! window.
//!
//! An observation crosses the boundary as a dict, one key per column of the
//! table the panel draws, with the two stages' measurements under `"cluster"`
//! and `"track"` and a key present exactly when something has measured it.

use std::sync::Arc;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList};

use sfmtool_core::bench::{
    add_observation as core_add_observation, apply_thresholds as core_apply_thresholds,
    commit as core_commit, create_cluster as core_create_cluster,
    create_track as core_create_track, duplicate as core_duplicate, evaluate as core_evaluate,
    fit as core_fit, resize_frame as core_resize_frame, resize_from_edge as core_resize_from_edge,
    rotate_frame as core_rotate_frame, search_descriptors as core_search_descriptors,
    set_observation_keypoint as core_set_observation_keypoint,
    set_observation_shape as core_set_observation_shape, set_stage as core_set_stage,
    set_verdict as core_set_verdict, split as core_split, translate_frame as core_translate_frame,
    Bench, BenchItem, ClassificationReason, ClusterSeed, CreateTrackOptions, Edge, EditableTrack,
    EvaluateOptions, EvaluateReport, FitOptions, FitReport, Found, ItemKind, Observation,
    ObservationSeed, Provenance, ResizeReport, SearchOptions, SearchReport, StageKind,
    TrackClassification, Verdict, DEFAULT_RADIUS_PX,
};
use sfmtool_core::features::kdforest::{ConstellationParams, ImageKeypoints};
use sfmtool_core::patch::normal_refine::ProjectedImage;
use sfmtool_core::progress::Progress;

use crate::patches::views::{resolve_pyramids, PosedViews};
use crate::reconstruction::edited::{PyEditedReconstruction, PyPointMap};
use crate::spatial::constellation_query::DEFAULTS as QUERY_DEFAULTS;
use crate::spatial::kdf::PyLazyKdForest;

/// Turn any core refusal into a Python `ValueError` carrying its sentence.
fn refused<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The verdict `word` names.
fn parse_verdict(word: &str) -> PyResult<Verdict> {
    match word {
        "in" => Ok(Verdict::In),
        "out" => Ok(Verdict::Out),
        "candidate" => Ok(Verdict::Candidate),
        other => Err(PyValueError::new_err(format!(
            "unknown verdict: {other:?} (expected in|out|candidate)"
        ))),
    }
}

/// The provenance `word` names, with `feature`, `inliers` and `point`
/// supplying what the variant that needs one needs.
fn parse_provenance(
    word: &str,
    feature: Option<u32>,
    inliers: Option<u32>,
    point: Option<u32>,
) -> PyResult<Provenance> {
    match word {
        "origin" => Ok(Provenance::Origin),
        "sweep" => Ok(Provenance::Sweep),
        "pixel" => Ok(Provenance::Pixel),
        "descriptor" => feature
            .map(|feature| Provenance::Descriptor { feature })
            .ok_or_else(|| {
                PyValueError::new_err("a descriptor provenance needs the 'feature' it returned")
            }),
        "search" => inliers
            .map(|inliers| Provenance::Search { inliers })
            .ok_or_else(|| {
                PyValueError::new_err("a search provenance needs the 'inliers' that voted for it")
            }),
        "point" => point
            .map(|point| Provenance::Point { point })
            .ok_or_else(|| {
                PyValueError::new_err("a point provenance needs the 'point' it was pulled from")
            }),
        other => Err(PyValueError::new_err(format!(
            "unknown provenance: {other:?} (expected \r
             origin|descriptor|search|sweep|pixel|point)"
        ))),
    }
}

/// The dict form of a provenance: the kind, plus whatever the kind names.
fn provenance_to_dict<'py>(py: Python<'py>, p: Provenance) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match p {
        Provenance::Origin => d.set_item("kind", "origin")?,
        Provenance::Sweep => d.set_item("kind", "sweep")?,
        Provenance::Pixel => d.set_item("kind", "pixel")?,
        Provenance::Descriptor { feature } => {
            d.set_item("kind", "descriptor")?;
            d.set_item("feature", feature)?;
        }
        Provenance::Search { inliers } => {
            d.set_item("kind", "search")?;
            d.set_item("inliers", inliers)?;
        }
        Provenance::Point { point } => {
            d.set_item("kind", "point")?;
            d.set_item("point", point)?;
        }
    }
    Ok(d)
}

/// A 2x2 affine shape as the `(2, 2)` array Python reads it as: row `r` is the
/// image of the detector frame's `r`-th axis, which is the layout the
/// `.matches` cluster-patches section stores.
///
/// The shape is per keypoint-frame unit and the patch is `[-radius, radius]` of
/// them at the track's `radius`, so a caller that wants a size in pixels
/// multiplies a column's norm by that.
fn shape_array(shape: [[f64; 2]; 2]) -> Array2<f64> {
    Array2::from_shape_vec((2, 2), shape.concat()).expect("four values in a 2x2")
}

/// The dict form of one observation, with each stage's measurements under its
/// own key and absent where that stage has not run.
fn observation_to_dict<'py>(py: Python<'py>, o: &Observation) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("image", o.image)?;
    d.set_item("provenance", provenance_to_dict(py, o.provenance)?)?;
    d.set_item("verdict", o.verdict.to_string())?;
    d.set_item("pinned", o.pinned)?;
    if let Some(m) = &o.cluster {
        let c = PyDict::new(py);
        c.set_item(
            "seed_position",
            PyArray1::from_vec(py, m.seed_position.to_vec()),
        )?;
        c.set_item("seed_shape", shape_array(m.seed_shape).into_pyarray(py))?;
        if let Some(p) = m.position {
            c.set_item("position", PyArray1::from_vec(py, p.to_vec()))?;
        }
        if let Some(s) = m.shape {
            c.set_item("shape", shape_array(s).into_pyarray(py))?;
        }
        if let Some(v) = m.zncc {
            c.set_item("zncc", v)?;
        }
        if let Some(v) = m.shift_px {
            c.set_item("shift_px", v)?;
        }
        if let Some(v) = m.localizability {
            c.set_item("localizability", v)?;
        }
        if let Some(s) = m.status {
            c.set_item("status", format!("{s:?}"))?;
        }
        d.set_item("cluster", c)?;
    }
    if let Some(m) = &o.track {
        let t = PyDict::new(py);
        if let Some(k) = m.keypoint {
            t.set_item("keypoint", PyArray1::from_vec(py, k.to_vec()))?;
        }
        for (key, value) in [
            ("zncc", m.zncc),
            ("seed_shift_px", m.seed_shift_px),
            ("projection_offset_px", m.projection_offset_px),
            ("reprojection_error", m.reprojection_error),
            ("ray_angle_deg", m.ray_angle_deg),
            ("localizability", m.localizability),
            // Present exactly when the last fit refused the walk and left this
            // sighting at its seed; the number is how far the peak sat.
            ("walked_px", m.walked_px),
        ] {
            if let Some(value) = value {
                t.set_item(key, value)?;
            }
        }
        // Present exactly when there is no score, and the sentence is the one
        // the panel shows in its Status cell.
        if let Some(reason) = m.reason {
            t.set_item("reason", reason.to_string())?;
        }
        d.set_item("track", t)?;
    }
    Ok(d)
}

/// A track being worked on: its observations, what has been measured about
/// each, the verdicts, the stage and the origin.
///
/// A plain value. Every step gives back a new one and leaves this object
/// exactly as it was, so a caller that keeps both can go back to the one it
/// had.
#[pyclass(name = "EditableTrack", module = "sfmtool.bench", skip_from_py_object)]
#[derive(Clone)]
pub struct PyEditableTrack {
    pub(crate) inner: Arc<EditableTrack>,
}

#[pymethods]
impl PyEditableTrack {
    /// Which representation the track is in: ``"cluster"`` or ``"track"``.
    #[getter]
    fn stage(&self) -> String {
        self.inner.stage_kind().to_string()
    }

    /// How many observations the track holds.
    #[getter]
    fn observation_count(&self) -> usize {
        self.inner.observations.len()
    }

    /// ``(in, candidate, out)``.
    #[getter]
    fn verdict_counts(&self) -> (usize, usize, usize) {
        self.inner.verdict_counts()
    }

    /// The indexes of the ``in`` observations, ascending.
    #[getter]
    fn in_observations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray1::from_vec(
            py,
            self.inner
                .in_observations()
                .into_iter()
                .map(|i| i as u64)
                .collect(),
        )
    }

    /// The point this track was put on the bench from, as
    /// ``{"version": ..., "point": ...}``, or ``None`` for a track that is a
    /// point of its own.
    #[getter]
    fn origin<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.origin {
            None => Ok(None),
            Some(origin) => {
                let d = PyDict::new(py);
                d.set_item("version", origin.version)?;
                d.set_item("point", origin.point)?;
                Ok(Some(d))
            }
        }
    }

    /// Whether the track's surfel is a bearing (``w == 0``) rather than a point.
    ///
    /// ``False`` at the cluster stage and for a track with no frame, neither of
    /// which states a direction.
    #[getter]
    fn at_infinity(&self) -> bool {
        self.inner
            .track()
            .and_then(|p| p.frame.as_ref())
            .is_some_and(|frame| frame.w == 0.0)
    }

    /// Where the track's point stands, or ``None`` at the cluster stage, before
    /// anything has triangulated it, **or when the track is at infinity** --
    /// which has no position, and whose coordinate is :attr:`direction`.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        if self.at_infinity() {
            return None;
        }
        let p = self.inner.track()?.position?;
        Some(PyArray1::from_vec(py, vec![p.x, p.y, p.z]))
    }

    /// The unit bearing a track at infinity points along, or ``None`` when the
    /// track is not at infinity.
    ///
    /// The coordinate a `w = 0` row of a `.sfmr` stores: the same three numbers
    /// :attr:`position` would hold for a finite track, under the other rule.
    #[getter]
    fn direction<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        if !self.at_infinity() {
            return None;
        }
        let p = self.inner.track()?.position?;
        Some(PyArray1::from_vec(py, vec![p.x, p.y, p.z]))
    }

    /// The track's surfel, or ``None`` at the cluster stage or before anything
    /// has fitted one.
    ///
    /// A dict of ``center``, ``u_halfvec``, ``v_halfvec`` and ``w`` -- the two
    /// half-vectors being the axes scaled by the half-extents, which is how a
    /// `.sfmr` stores them, and ``w`` being ``1.0`` for a position and ``0.0``
    /// for a bearing, whose ``center`` is a unit direction. The patch covers
    /// ``center + s * u + t * v`` for ``(s, t)`` in ``[-1, 1]^2``, so this is
    /// what :func:`resize_frame`, :func:`resize_from_edge` and
    /// :func:`rotate_frame` are read back through.
    #[getter]
    fn frame<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(frame) = self.inner.track().and_then(|p| p.frame.as_ref()) else {
            return Ok(None);
        };
        let vector = |v: nalgebra::Vector3<f64>| PyArray1::from_vec(py, vec![v.x, v.y, v.z]);
        let d = PyDict::new(py);
        d.set_item("center", vector(frame.center.coords))?;
        d.set_item("u_halfvec", vector(frame.u_axis * frame.half_extent[0]))?;
        d.set_item("v_halfvec", vector(frame.v_axis * frame.half_extent[1]))?;
        d.set_item("w", frame.w)?;
        Ok(Some(d))
    }

    /// Which observation the cluster stage cuts its template around, or
    /// ``None`` at the track stage.
    #[getter]
    fn reference(&self) -> Option<usize> {
        Some(self.inner.cluster()?.reference)
    }

    /// The cluster stage's template half-width in keypoint-frame units, or
    /// ``None`` at the track stage.
    ///
    /// This is the cluster's one scale: every ``seed_shape`` and ``shape`` on
    /// its observations maps one keypoint-frame unit to that image's pixels and
    /// the patch is the square ``[-radius, radius]`` of them, so a sighting's
    /// pixel half-width along a column is ``radius * norm(column)``.
    #[getter]
    fn radius(&self) -> Option<f64> {
        Some(self.inner.cluster()?.radius)
    }

    /// The bars the threshold painting judges against.
    #[getter]
    fn thresholds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let t = &self.inner.thresholds;
        let d = PyDict::new(py);
        d.set_item("min_zncc", t.min_zncc)?;
        d.set_item("max_shift_px", t.max_shift_px)?;
        d.set_item("max_keypoint_uncertainty", t.max_keypoint_uncertainty)?;
        d.set_item("min_relative_zncc", t.min_relative_zncc)?;
        Ok(d)
    }

    /// Every observation as a dict, in the order they were added.
    #[getter]
    fn observations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.inner
                .observations
                .iter()
                .map(|o| observation_to_dict(py, o))
                .collect::<PyResult<Vec<_>>>()?,
        )
    }

    /// One observation as a dict, or ``None`` when the index is past the end.
    fn observation<'py>(
        &self,
        py: Python<'py>,
        observation: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.observations.get(observation) {
            None => Ok(None),
            Some(o) => Ok(Some(observation_to_dict(py, o)?)),
        }
    }

    /// A copy of this track whose origin is `point` in version `version`.
    ///
    /// What a caller re-seats a committed track with, so a second commit of it
    /// replaces what the first wrote.
    fn with_origin(&self, version: u64, point: u32) -> Self {
        Self {
            inner: Arc::new(self.inner.with_origin(version, point)),
        }
    }

    fn __repr__(&self) -> String {
        let (inside, candidates, out) = self.inner.verdict_counts();
        format!(
            "EditableTrack(stage={}, {inside} in, {candidates} candidates, {out} out)",
            self.inner.stage_kind()
        )
    }
}

/// The things being worked on beside one reconstruction, in the order they were
/// put there, and which one of each kind is active.
///
/// A plain value: every operation returns the next bench and leaves this object
/// as it was. Items the operation did not touch are shared, not copied.
#[pyclass(name = "Bench", module = "sfmtool.bench", skip_from_py_object)]
#[derive(Clone, Default)]
pub struct PyBench {
    pub(crate) inner: Bench,
}

impl PyBench {
    /// The wrapper around a core bench.
    fn wrap(inner: Bench) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyBench {
    /// A bench with nothing on it.
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Every label, in the order the items were put on.
    #[getter]
    fn labels(&self) -> Vec<String> {
        self.inner.labels().map(str::to_string).collect()
    }

    /// The label of the active item of `kind`, or ``None``.
    #[pyo3(signature = (kind = "track"))]
    fn active_label(&self, kind: &str) -> PyResult<Option<String>> {
        let kind = match kind {
            "track" => ItemKind::Track,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown item kind: {other:?} (expected track)"
                )))
            }
        };
        Ok(self.inner.active_label(kind).map(str::to_string))
    }

    /// The active track, or ``None`` when no track is on the bench.
    #[getter]
    fn active_track(&self) -> Option<PyEditableTrack> {
        self.inner.active_track().map(|t| PyEditableTrack {
            inner: Arc::clone(t),
        })
    }

    /// The track called `label`, or ``None`` when nothing is.
    fn track(&self, label: &str) -> Option<PyEditableTrack> {
        self.inner.track(label).map(|t| PyEditableTrack {
            inner: Arc::clone(t),
        })
    }

    /// Put `track` back in the place of the item called `label`, leaving the
    /// order, the label and the activation as they were.
    ///
    /// How a step on one track is installed: the step gives back a value, and
    /// this is what puts that value on the bench.
    fn replace(&self, label: &str, track: &PyEditableTrack) -> PyResult<Self> {
        self.inner
            .replace(label, BenchItem::Track(Arc::clone(&track.inner)))
            .map(Self::wrap)
            .map_err(refused)
    }

    /// Make the item called `label` the active one of its kind.
    fn activate(&self, label: &str) -> PyResult<Self> {
        self.inner.activate(label).map(Self::wrap).map_err(refused)
    }

    /// Take the item called `label` off the bench. Its label is then free to be
    /// minted again.
    fn discard(&self, label: &str) -> PyResult<Self> {
        self.inner.discard(label).map(Self::wrap).map_err(refused)
    }

    /// Rename the item called `label` to `to`.
    fn rename(&self, label: &str, to: &str) -> PyResult<Self> {
        self.inner
            .rename(label, to)
            .map(Self::wrap)
            .map_err(refused)
    }

    fn __repr__(&self) -> String {
        format!("Bench({} items)", self.inner.len())
    }
}

/// Put the point at `point` on the bench as a track-stage editable track.
///
/// The track arrives with the point's own frame, bitmap and keypoints, its
/// origin set to that point, and every observation ``in``. The measurements are
/// carried from what the record stores and nothing is recomputed, so putting a
/// track on the bench and doing nothing shows the numbers the reconstruction
/// already holds.
///
/// `label` is what the track is put on the bench under, before any collision
/// suffix; the viewer passes the point's portable id. With none, the
/// label is ``pt3d_<hash>_<index>`` over the base's own content hash for a point
/// that is a row of it, and ``point_<index>`` for one an edit added.
/// `version` is the version serial the origin records, which is the caller's own
/// numbering and is not interpreted here.
///
/// Returns ``(Bench, EditableTrack)``. A ``sift_files`` reconstruction is put on
/// the bench like any other -- inspecting a track is allowed everywhere -- and
/// it is :func:`commit` that refuses to write one back.
#[pyfunction]
#[pyo3(signature = (bench, edited, point, *, version = 0, label = None))]
fn create_track(
    bench: &PyBench,
    edited: &PyEditedReconstruction,
    point: u32,
    version: u64,
    label: Option<String>,
) -> PyResult<(PyBench, PyEditableTrack)> {
    let (next, report) = core_create_track(
        &bench.inner,
        &edited.inner,
        point,
        &CreateTrackOptions { version, label },
    )
    .map_err(refused)?;
    let track = PyEditableTrack {
        inner: Arc::clone(next.track(&report.label).expect("just put on")),
    };
    Ok((PyBench::wrap(next), track))
}

/// Put a new cluster-stage track on the bench with one observation in `image`.
///
/// `image_stem` is what the label is minted from: a hand-placed seed is labelled
/// ``<stem>@<x>,<y>`` and a ``.sift`` feature ``<stem>#<feature>``. Either
/// `radius_px` (an isotropic seed around a pixel nobody detected, in
/// source-image pixels from the pixel to the patch's edge) or `shape` (a
/// ``2x2`` affine shape, the map from the detector's canonical **keypoint
/// frame** onto this image's pixels) says how large the patch is. A shape is
/// read over the square ``[-radius, radius]`` of keypoint-frame units, where
/// ``radius`` is the new track's :attr:`EditableTrack.radius`, so a
/// `radius_px` becomes the shape ``radius_px / radius`` times the identity.
///
/// The template is left uncut: cutting it reads the reference's pixels, and this
/// step reads no photograph. Returns ``(Bench, EditableTrack)``.
#[pyfunction]
#[pyo3(signature = (bench, image, image_stem, pixel, *, radius_px = None, shape = None, feature = None))]
fn create_cluster(
    bench: &PyBench,
    image: u32,
    image_stem: &str,
    pixel: [f64; 2],
    radius_px: Option<f64>,
    shape: Option<[[f64; 2]; 2]>,
    feature: Option<u32>,
) -> PyResult<(PyBench, PyEditableTrack)> {
    let shape = match (shape, radius_px) {
        (Some(shape), _) => shape,
        (None, Some(r)) => ClusterSeed::shape_from_radius_px(r),
        (None, None) => {
            return Err(PyValueError::new_err(
                "a cluster seed needs a 'radius_px' or a 2x2 'shape': nothing in a \
                 pixel says how large the patch around it is",
            ))
        }
    };
    let seed = ClusterSeed {
        image,
        image_stem: image_stem.to_string(),
        pixel,
        shape,
        feature,
    };
    let (next, report) = core_create_cluster(&bench.inner, &seed).map_err(refused)?;
    let track = PyEditableTrack {
        inner: Arc::clone(next.track(&report.label).expect("just put on")),
    };
    Ok((PyBench::wrap(next), track))
}

/// Add a candidate observation to `track`.
///
/// It joins as a ``candidate``: something proposed it and nobody has ruled on
/// it. A second observation in an image the track already holds is allowed and
/// is scored like any other; what it cannot do is be turned ``in`` while the
/// other is.
///
/// `shape` is in the cluster stage's own convention (keypoint-frame units to
/// pixels, over ``[-radius, radius]``) and defaults to the reference
/// observation's own, so a pixel gesture on a track that already has a scale
/// needs no radius. `provenance` is one of
/// ``origin``, ``descriptor`` (with `feature`), ``search`` (with `inliers`),
/// ``sweep``, ``pixel`` or ``point`` (with `point`, which a commit then
/// absorbs).
///
/// Returns ``(EditableTrack, report)``.
#[pyfunction]
#[pyo3(signature = (track, image, pixel, *, shape = None, provenance = "pixel", feature = None,
                    inliers = None, point = None))]
#[allow(clippy::too_many_arguments)]
fn add_observation(
    py: Python<'_>,
    track: &PyEditableTrack,
    image: u32,
    pixel: [f64; 2],
    shape: Option<[[f64; 2]; 2]>,
    provenance: &str,
    feature: Option<u32>,
    inliers: Option<u32>,
    point: Option<u32>,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let seed = ObservationSeed {
        image,
        pixel,
        shape,
        provenance: parse_provenance(provenance, feature, inliers, point)?,
    };
    let (next, report) = core_add_observation(&track.inner, &seed).map_err(refused)?;
    let d = PyDict::new(py);
    d.set_item("observation", report.observation)?;
    d.set_item("image", report.image)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// Set the verdict of one observation, by hand.
///
/// The verdict is pinned by this, so :func:`apply_thresholds` leaves it where it
/// is. Turning an observation ``in`` is refused when another ``in`` observation
/// already holds its image, because a track observes an image once.
///
/// Returns ``(EditableTrack, report)``.
#[pyfunction]
fn set_verdict(
    py: Python<'_>,
    track: &PyEditableTrack,
    observation: usize,
    verdict: &str,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let (next, report) =
        core_set_verdict(&track.inner, observation, parse_verdict(verdict)?).map_err(refused)?;
    let d = PyDict::new(py);
    d.set_item("observation", report.observation)?;
    d.set_item("was", report.was.to_string())?;
    d.set_item("is", report.is.to_string())?;
    d.set_item("changed", report.changed)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// Slide the track's surfel across its own plane until its centre sits under
/// ``pixel`` in ``observation``'s photograph.
///
/// This moves the **patch**, not one sighting: a track-stage track has one
/// surfel and every observation is a view of it, so the centre moves in-plane,
/// the half-vectors and the normal are kept, and every observation's keypoint
/// becomes the projection of the new centre through its own camera. Nothing is
/// pinned -- a translation says where the patch is, not whether a sighting
/// belongs to it -- and the measurements and the bitmap go, because all of them
/// were read at a place the patch has left. A sighting the moved centre no
/// longer projects into is left with no keypoint and ``NoProjection`` as its
/// reason.
///
/// The pointer is read against the outline as drawn: the frame re-anchored on
/// that observation's own sighting. :func:`set_observation_keypoint` is the step
/// for **one** keypoint.
///
/// Returns ``(EditableTrack, report)`` carrying ``observation``, ``image``,
/// ``pixel`` (where the centre now projects in it), ``center``, ``moved``,
/// ``placed`` and ``changed``.
#[pyfunction]
fn translate_frame(
    py: Python<'_>,
    track: &PyEditableTrack,
    edited: &PyEditedReconstruction,
    observation: usize,
    pixel: [f64; 2],
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let (next, report) =
        core_translate_frame(&track.inner, &edited.inner, observation, pixel).map_err(refused)?;
    let d = PyDict::new(py);
    d.set_item("observation", report.observation)?;
    d.set_item("image", report.image)?;
    d.set_item("pixel", report.pixel)?;
    d.set_item(
        "center",
        PyArray1::from_vec(py, vec![report.center.x, report.center.y, report.center.z]),
    )?;
    d.set_item("moved", report.moved)?;
    d.set_item("placed", report.placed)?;
    d.set_item("changed", report.changed)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// Put **one** observation's own sighting at ``pixel``, by hand, leaving every
/// other where it is.
///
/// At the track stage this writes the observation's keypoint, which is the pixel
/// a commit writes; at the cluster stage it moves its seed and keeps the shape
/// it is read at. Either way every measurement that was read at the old pixel is
/// dropped -- none of them says anything about the new one -- and the
/// observation is pinned, so :func:`apply_thresholds` leaves its verdict alone.
///
/// :func:`translate_frame` is the step that moves the **patch**, which is what
/// the viewer's dot drag means at the track stage.
///
/// Returns ``(EditableTrack, report)``.
#[pyfunction]
fn set_observation_keypoint(
    py: Python<'_>,
    track: &PyEditableTrack,
    observation: usize,
    pixel: [f64; 2],
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let (next, report) =
        core_set_observation_keypoint(&track.inner, observation, pixel).map_err(refused)?;
    let d = PyDict::new(py);
    d.set_item("observation", report.observation)?;
    d.set_item("image", report.image)?;
    d.set_item("was", report.was)?;
    d.set_item("pixel", report.pixel)?;
    d.set_item("moved_px", report.moved_px)?;
    d.set_item("changed", report.changed)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// Resize the track's surfel to ``half_length`` on both of its axes, about its
/// own centre.
///
/// One scalar, because a patch frame is square: the stored half-vector pair has
/// ``|u| == |v|`` and the tile grid is square with it, so a resize that moved
/// one axis alone would stretch the template rather than enlarge it. The centre,
/// the axes' directions and the normal are untouched. The consensus bitmap and
/// every track measurement but the keypoints are dropped, because all of them
/// were read over the square as it stood.
///
/// Returns ``(EditableTrack, report)``.
#[pyfunction]
fn resize_frame(
    py: Python<'_>,
    track: &PyEditableTrack,
    half_length: f64,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let (next, report) = core_resize_frame(&track.inner, half_length).map_err(refused)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        resize_report_dict(py, &report)?,
    ))
}

/// Resize the patch by putting one edge of the outline drawn at ``observation``
/// under ``pixel``, with the **opposite edge left where it is**.
///
/// ``edge`` is ``"+u"``, ``"-u"``, ``"+v"`` or ``"-v"``. At the track stage the
/// pixel is unprojected onto the patch's own plane through that observation's
/// camera, so the edge lands there exactly under any lens; the surfel takes the
/// centre the outline had plus the edge's shift, the track's position follows
/// it, and that observation's keypoint is set to the projection of the new
/// centre and pinned. At the cluster stage the same arithmetic runs in that
/// image's pixels over the observation's own affine shape.
///
/// Returns ``(EditableTrack, report)``.
#[pyfunction]
fn resize_from_edge(
    py: Python<'_>,
    track: &PyEditableTrack,
    edited: &PyEditedReconstruction,
    observation: usize,
    edge: &str,
    pixel: [f64; 2],
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let edge: Edge = edge.parse().map_err(refused)?;
    let (next, report) =
        core_resize_from_edge(&track.inner, &edited.inner, observation, edge, pixel)
            .map_err(refused)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        resize_report_dict(py, &report)?,
    ))
}

/// Turn the track's surfel by ``angle_rad`` about its own outward normal.
///
/// Both axes are rotated by a rotation whose axis is the normal, so they keep
/// their lengths and the patch keeps its plane and the face it shows: what
/// changes is which way up the square sits. The centre is untouched, so no
/// sighting moves.
///
/// Returns ``(EditableTrack, report)`` with ``degrees`` and ``changed``.
#[pyfunction]
fn rotate_frame(
    py: Python<'_>,
    track: &PyEditableTrack,
    angle_rad: f64,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let (next, report) = core_rotate_frame(&track.inner, angle_rad).map_err(refused)?;
    let d = PyDict::new(py);
    d.set_item("degrees", report.degrees)?;
    d.set_item("changed", report.changed)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// Give one cluster-stage observation the affine shape ``shape``, by hand.
///
/// The shape is the cluster stage's own convention: the detector's canonical
/// keypoint frame mapped onto this image's pixels, read over ``[-r, r]^2`` at
/// the cluster's radius. The observation is re-seeded where it is already drawn
/// and its refinement is dropped, because those numbers were the refinement's
/// answer about another shape. The verdict is not pinned: a size or a turn is
/// not a ruling on whether the sighting belongs.
///
/// Returns ``(EditableTrack, report)``.
#[pyfunction]
fn set_observation_shape(
    py: Python<'_>,
    track: &PyEditableTrack,
    observation: usize,
    shape: [[f64; 2]; 2],
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let (next, report) =
        core_set_observation_shape(&track.inner, observation, shape).map_err(refused)?;
    let d = PyDict::new(py);
    d.set_item("observation", report.observation)?;
    d.set_item("image", report.image)?;
    d.set_item("shape", shape_array(report.shape).into_pyarray(py))?;
    d.set_item("half_px", report.half_px)?;
    d.set_item("was_half_px", report.was_half_px)?;
    d.set_item("changed", report.changed)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// What a resize reported, as a dict.
fn resize_report_dict(py: Python<'_>, report: &ResizeReport) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("observation", report.observation)?;
    d.set_item("image", report.image)?;
    d.set_item("half", report.half)?;
    d.set_item("was", report.was)?;
    d.set_item("changed", report.changed)?;
    Ok(d.unbind())
}

/// Paint the proposed verdicts from the stored measurements onto the unpinned
/// observations.
///
/// Each keyword moves one bar and leaves the others where the track has them,
/// so a script can differ from the pipeline's default in one number. An
/// observation nothing has measured at the track's current stage is left where
/// it is: there is no proposal to apply. One ``in`` per image survives the
/// painting -- where several would pass, the best-scoring takes the image.
///
/// Returns ``(EditableTrack, report)``.
#[pyfunction]
#[pyo3(signature = (
    track, *, min_zncc = None, max_shift_px = None, max_keypoint_uncertainty = None,
    min_relative_zncc = None
))]
fn apply_thresholds(
    py: Python<'_>,
    track: &PyEditableTrack,
    min_zncc: Option<f64>,
    max_shift_px: Option<f64>,
    max_keypoint_uncertainty: Option<f64>,
    min_relative_zncc: Option<f64>,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let mut seeded = (*track.inner).clone();
    let t = &mut seeded.thresholds;
    for (slot, value) in [
        (&mut t.min_zncc, min_zncc),
        (&mut t.max_shift_px, max_shift_px),
        (&mut t.max_keypoint_uncertainty, max_keypoint_uncertainty),
        (&mut t.min_relative_zncc, min_relative_zncc),
    ] {
        if let Some(value) = value {
            *slot = value;
        }
    }
    let (next, report) = core_apply_thresholds(&seeded);
    let d = PyDict::new(py);
    d.set_item("turned_in", report.turned_in)?;
    d.set_item("turned_out", report.turned_out)?;
    d.set_item("pinned", report.pinned)?;
    d.set_item("unmeasured", report.unmeasured)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// The decoded views a photometric step reads, one per image of `edited`.
///
/// `images` is what every patch kernel takes -- a list of ``HxW[xC]`` ``uint8``
/// arrays, one per image of the base, or a prebuilt :class:`ImagePyramidSet` --
/// because registering a patch needs pixels and a reconstruction carries poses
/// and lenses rather than photographs.
fn views_of<'a>(
    posed: &'a PosedViews,
    pyramids: &'a crate::patches::views::PyramidSet,
) -> Vec<ProjectedImage<'a>> {
    posed
        .cameras
        .iter()
        .zip(&posed.poses)
        .zip(pyramids.as_slice())
        .map(|((camera, cam_from_world), pyramid)| ProjectedImage {
            camera,
            cam_from_world,
            pyramid,
        })
        .collect()
}

/// The dict form of one evaluation's report.
fn evaluate_report_dict<'py>(
    py: Python<'py>,
    report: &EvaluateReport,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("stage", report.stage.to_string())?;
    d.set_item("measured", report.measured)?;
    d.set_item("unmeasured", report.unmeasured)?;
    if let Some(reference) = report.reference {
        d.set_item("reference", reference)?;
    }
    d.set_item("at_infinity", report.at_infinity)?;
    if let Some(p) = report.position {
        let key = if report.at_infinity {
            "direction"
        } else {
            "position"
        };
        d.set_item(key, PyArray1::from_vec(py, vec![p.x, p.y, p.z]))?;
    }
    if let Some(condition_number) = report.condition_number {
        d.set_item("condition_number", condition_number)?;
    }
    Ok(d)
}

/// The dict form of one classification: which representation the rays earned,
/// and the numbers behind the call.
fn classification_dict<'py>(
    py: Python<'py>,
    call: &TrackClassification,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("at_infinity", call.at_infinity)?;
    let c = call.coordinate;
    let key = if call.at_infinity {
        "direction"
    } else {
        "position"
    };
    d.set_item(key, PyArray1::from_vec(py, vec![c.x, c.y, c.z]))?;
    d.set_item(
        "reason",
        match call.reason {
            ClassificationReason::WellConditioned => "well_conditioned",
            ClassificationReason::DepthResolved => "depth_resolved",
            ClassificationReason::DepthUnresolved => "depth_unresolved",
            ClassificationReason::BaselineTooShort => "baseline_too_short",
            ClassificationReason::FiniteDoesNotExplainTheSightings => {
                "finite_does_not_explain_the_sightings"
            }
            ClassificationReason::BearingDoesNotExplainTheSightings => {
                "bearing_does_not_explain_the_sightings"
            }
        },
    )?;
    d.set_item("condition_number", call.condition_number)?;
    d.set_item("inverse_depth_z", call.inverse_depth_z)?;
    d.set_item("inverse_depth_z_cutoff", call.inverse_depth_z_cutoff)?;
    d.set_item("resolvable_distance", call.resolvable_distance)?;
    d.set_item("finite_horizon", call.finite_horizon)?;
    d.set_item("max_pair_angle_deg", call.max_pair_angle_deg)?;
    d.set_item("finite_rms_px", call.finite_rms_px)?;
    d.set_item("bearing_rms_px", call.bearing_rms_px)?;
    d.set_item("residual_margin", call.residual_margin)?;
    d.set_item("text", call.to_string())?;
    Ok(d)
}

/// The dict form of one fit's report, the reading it ended with inside it.
fn fit_report_dict<'py>(py: Python<'py>, report: &FitReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("placed", report.placed)?;
    d.set_item("kept_at_seed", report.kept_at_seed)?;
    let at_infinity = report.classification.is_some_and(|c| c.at_infinity);
    d.set_item("at_infinity", at_infinity)?;
    if let Some(p) = report.position {
        let key = if at_infinity { "direction" } else { "position" };
        d.set_item(key, PyArray1::from_vec(py, vec![p.x, p.y, p.z]))?;
    }
    if let Some(condition_number) = report.condition_number {
        d.set_item("condition_number", condition_number)?;
    }
    if let Some(call) = &report.classification {
        d.set_item("classification", classification_dict(py, call)?)?;
    }
    d.set_item("evaluate", evaluate_report_dict(py, &report.evaluate)?)?;
    Ok(d)
}

/// The reading options a call runs with: the defaults, with the caller's own
/// search radius and memory bounds where they named them.
fn evaluate_options(
    search_px: Option<f64>,
    max_seed_offset_px: Option<f64>,
    max_cache_bytes: Option<usize>,
) -> EvaluateOptions {
    let mut options = EvaluateOptions::default();
    if let Some(search_px) = search_px {
        options.search_px = search_px;
    }
    if let Some(max_seed_offset_px) = max_seed_offset_px {
        options.max_seed_offset_px = max_seed_offset_px;
    }
    if let Some(max_cache_bytes) = max_cache_bytes {
        options.max_cache_bytes = max_cache_bytes;
    }
    options
}

/// The fit options a call runs with. The reading the fit ends with takes the
/// same search radius and the same bounds, so a fit and an `evaluate` of its
/// result are stated in one set of terms.
fn fit_options(
    search_px: Option<f64>,
    max_seed_offset_px: Option<f64>,
    max_cache_bytes: Option<usize>,
    noise_floor_px: Option<f64>,
    inverse_depth_z_cutoff: Option<f64>,
    residual_margin: Option<f64>,
) -> FitOptions {
    let mut options = FitOptions {
        evaluate: evaluate_options(search_px, max_seed_offset_px, max_cache_bytes),
        ..FitOptions::default()
    };
    if let Some(noise_floor_px) = noise_floor_px {
        options.noise_floor_px = noise_floor_px;
    }
    if let Some(cutoff) = inverse_depth_z_cutoff {
        options.inverse_depth_z_cutoff = cutoff;
    }
    if let Some(margin) = residual_margin {
        options.residual_margin = margin;
    }
    options
}

/// The stage `word` names.
fn parse_stage(word: &str) -> PyResult<StageKind> {
    match word {
        "cluster" => Ok(StageKind::Cluster),
        "track" => Ok(StageKind::Track),
        other => Err(PyValueError::new_err(format!(
            "unknown stage: {other:?} (expected cluster|track)"
        ))),
    }
}

/// Read `track` as it stands: fill the measurement slots of every observation,
/// whatever its verdict, at the stage it is in, and **move nothing else**. The
/// position, the frame, the bitmap, every keypoint and every verdict come back
/// exactly as they went in. An ``out`` observation is scored the way a
/// candidate is.
///
/// Nothing is dropped. The kernels run with their per-view gates off and the
/// consensus-basis cap lifted, because a gate is a decision and this makes
/// none; an observation that cannot be read at all comes back with a ``reason``
/// sentence instead of a blank row.
///
/// At the **cluster stage** the seeds are an in-memory ``.matches`` cluster and
/// the refinement kernel is run over it: the kernel picks the reference, cuts
/// the template there and warps every other seed onto it, and each observation
/// gets the refined position and shape, the achieved ZNCC, the drift from its
/// seed, its own tile localizability and the kernel's ``member_status``. No
/// pose is read.
///
/// At the **track stage** one round of the localizer scores every observation
/// against the leave-one-out consensus of the others, at the pixel it already
/// sits at: each gets that ZNCC, ``seed_shift_px`` (how far the correlation
/// peak sits from the observation itself), ``projection_offset_px`` (how far
/// the observation sits from the point's projection -- the number that says how
/// far the *point* is off), the reprojection error, the ray angle and its tile
/// localizability.
///
/// `search_px` is how far from each observation the peak is looked for, in
/// patch-grid px; the default is the localizer's own search radius.
///
/// Two bounds keep a wide window from asking for memory the machine does not
/// have -- the window is widened to reach the furthest seed and each view's
/// tile costs the square of it. `max_seed_offset_px` (64 patch-grid px by
/// default) is how far from the point's projection a seed may sit and still be
/// read: past it the row comes back with a ``reason`` naming the bound instead
/// of a score. `max_cache_bytes` (256 MiB by default) is what one round's tiles
/// may take together; a round past it is refused rather than attempted.
///
/// Nothing here decides anything: the thresholds propose and
/// :func:`apply_thresholds` applies the proposal.
///
/// Returns ``(EditableTrack, report)``. The report carries ``stage``,
/// ``measured`` and ``unmeasured``; ``reference`` at the cluster stage; and
/// ``position`` and ``condition_number`` at the track stage. Raises
/// ``ValueError`` with the reason when the reading is refused.
#[pyfunction]
#[pyo3(signature = (
    track,
    edited,
    images,
    *,
    search_px = None,
    max_seed_offset_px = None,
    max_cache_bytes = None,
))]
fn evaluate(
    py: Python<'_>,
    track: &PyEditableTrack,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    search_px: Option<f64>,
    max_seed_offset_px: Option<f64>,
    max_cache_bytes: Option<usize>,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let views = views_of(&posed, &pyramids);
    let (next, report) = core_evaluate(
        &track.inner,
        &edited.inner,
        &views,
        &evaluate_options(search_px, max_seed_offset_px, max_cache_bytes),
        &Progress::none(),
    )
    .map_err(refused)?;
    let d = evaluate_report_dict(py, &report)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// Fit `track` at the stage it is in: the step that **moves** it.
///
/// At the **track stage** the surfel is localized into every view, refined to
/// sub-pixel, the ``in`` results are re-triangulated, the frame is placed at
/// what they resolve to and the consensus bitmap is fused over them; the
/// keypoints, the coordinate, the frame and the bitmap are written. At the
/// **cluster stage** a fit is the refinement, which is what a reading is too: a
/// cluster has no geometry behind it to move.
///
/// **A track-stage fit decides finite versus at infinity afresh.** The rays are
/// put through the reconstruction's own criterion, so a track whose sightings
/// have just given it a depth becomes a point and one whose rays no longer fix
/// one becomes a bearing. ``noise_floor_px`` is the per-sighting measurement
/// noise that criterion assumes (1.0 px) and ``inverse_depth_z_cutoff`` the
/// z-score a depth has to reach to be called finite (4.0); both default to the
/// reconstruction pass's own values.
///
/// **And the criterion's answer is checked against the sightings.** Both
/// candidates -- the triangulated point and the bearing -- are reprojected into
/// every sighting's own photograph, and the depth is believed only where the
/// point's rms residual comes under ``residual_margin`` (0.8) of the bearing's
/// *and* under it by more than ``noise_floor_px``. An ill-conditioned midpoint
/// that landed wherever the rays' inconsistency threw it therefore does not
/// become a point, however well the z-score reads; the report's
/// ``classification`` carries both residuals and says which way the check went.
///
/// A fit ends by evaluating its own result, so every number in the observations'
/// slots and in the report is that reading's and :func:`evaluate` called after
/// it agrees to the last digit.
///
/// Returns ``(EditableTrack, report)``. The report carries ``placed`` (how many
/// observations the kernels moved), ``kept_at_seed`` (how many the kernels
/// wanted to walk further than ``max_shift_px`` and were left where they were),
/// ``at_infinity`` with ``position`` or ``direction``, ``condition_number`` and
/// ``classification`` at the track stage, and ``evaluate``: the reading's own
/// report. Raises ``ValueError`` with the reason when the fit is refused -- a
/// track stage with fewer than two ``in`` observations among them, which a
/// reading permits.
#[pyfunction]
#[pyo3(signature = (
    track,
    edited,
    images,
    *,
    search_px = None,
    max_seed_offset_px = None,
    max_cache_bytes = None,
    noise_floor_px = None,
    inverse_depth_z_cutoff = None,
    residual_margin = None,
))]
#[allow(clippy::too_many_arguments)]
fn fit(
    py: Python<'_>,
    track: &PyEditableTrack,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    search_px: Option<f64>,
    max_seed_offset_px: Option<f64>,
    max_cache_bytes: Option<usize>,
    noise_floor_px: Option<f64>,
    inverse_depth_z_cutoff: Option<f64>,
    residual_margin: Option<f64>,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let views = views_of(&posed, &pyramids);
    let (next, report) = core_fit(
        &track.inner,
        &edited.inner,
        &views,
        &fit_options(
            search_px,
            max_seed_offset_px,
            max_cache_bytes,
            noise_floor_px,
            inverse_depth_z_cutoff,
            residual_margin,
        ),
        &Progress::none(),
    )
    .map_err(refused)?;
    let d = fit_report_dict(py, &report)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// Put `track` into `stage`, which is ``"cluster"`` or ``"track"``.
///
/// **Up**, cluster to track: the ``in`` observations' refined cluster positions
/// are triangulated, the patch is framed at that position from the reference
/// observation's affine shape at the triangulated depth, and the track-stage
/// fit then runs over it. The cluster-stage measurements are dropped with the
/// stage.
///
/// **Down**, track to cluster: always possible and lossy on purpose. The
/// reference becomes the ``in`` observation with the largest projected patch
/// scale, every observation is re-seeded at its keypoint with the shape the
/// frame projects to there, and the position, the frame, the bitmap and the
/// track-stage measurements are dropped.
///
/// Setting the stage a track is already at gives the track back unchanged, with
/// ``changed`` false, and the caller pushes no version for it.
///
/// The upgrade puts the triangulated rays through the same finite-versus-
/// infinity criterion a fit does, with the same check against the sightings, so
/// a cluster whose sightings only ever stated a direction becomes a ``w = 0``
/// track rather than a point at a depth they never carried. ``noise_floor_px``,
/// ``inverse_depth_z_cutoff`` and ``residual_margin`` are that criterion's,
/// exactly as on :func:`fit`.
///
/// Returns ``(EditableTrack, report)``, whose report carries ``from``, ``to``,
/// ``changed``, the upgrade's ``fit`` report (``classification`` inside it) and
/// the downgrade's ``reference``.
#[pyfunction]
#[pyo3(signature = (
    track,
    edited,
    images,
    stage,
    *,
    noise_floor_px = None,
    inverse_depth_z_cutoff = None,
    residual_margin = None,
))]
#[allow(clippy::too_many_arguments)]
fn set_stage(
    py: Python<'_>,
    track: &PyEditableTrack,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    stage: &str,
    noise_floor_px: Option<f64>,
    inverse_depth_z_cutoff: Option<f64>,
    residual_margin: Option<f64>,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let stage = parse_stage(stage)?;
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let views = views_of(&posed, &pyramids);
    let (next, report) = core_set_stage(
        &track.inner,
        &edited.inner,
        &views,
        stage,
        &fit_options(
            None,
            None,
            None,
            noise_floor_px,
            inverse_depth_z_cutoff,
            residual_margin,
        ),
        &Progress::none(),
    )
    .map_err(refused)?;
    let d = PyDict::new(py);
    d.set_item("from", report.from.to_string())?;
    d.set_item("to", report.to.to_string())?;
    d.set_item("changed", report.changed)?;
    if let Some(fitted) = &report.fit {
        d.set_item("fit", fit_report_dict(py, fitted)?)?;
    }
    if let Some(reference) = report.reference {
        d.set_item("reference", reference)?;
    }
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// Split the observations at `observations` off the track called `label` into a
/// second track beside it on the bench.
///
/// The observations are named explicitly rather than read off the verdicts,
/// because ``out`` says an observation does not belong *here* and cannot say
/// which of two surfaces it belongs to. The moved observations keep their
/// verdicts, their provenance and both stages' measurements; the second track
/// has no origin, so a commit of it creates a point while a commit of the first
/// still replaces the one it came from. An empty list, or every observation, is
/// refused.
///
/// **The second track is a cluster.** The half being taken off is a set of
/// sightings that agree with each other and not with a 3D hypothesis fitted to
/// both, so a track-stage half is put down to the cluster stage through the
/// same downgrade :func:`set_stage` runs -- which is why `edited` is needed:
/// the downgrade projects the frame through each observation's camera.
///
/// Returns ``(Bench, report)``; the report's ``label`` names the second track.
#[pyfunction]
fn split(
    py: Python<'_>,
    bench: &PyBench,
    edited: &PyEditedReconstruction,
    label: &str,
    observations: Vec<usize>,
) -> PyResult<(PyBench, Py<PyDict>)> {
    let (next, report) =
        core_split(&bench.inner, &edited.inner, label, &observations).map_err(refused)?;
    let d = PyDict::new(py);
    d.set_item("label", report.label)?;
    d.set_item("moved", report.moved)?;
    d.set_item("kept", report.kept)?;
    Ok((PyBench::wrap(next), d.unbind()))
}

/// Put a copy of the item called ``label`` on the bench beside it.
///
/// What a second patch over neighbouring ground is started from: the copy
/// carries the stage and all of its data, every observation with its keypoint,
/// seed, shape, verdict and pin, the measurements, and the thresholds. The one
/// field it does not carry is the **origin**, so a commit of the copy creates a
/// point rather than replacing the one the original came from. Its label is the
/// original's with ``" copy"`` after it, through the bench's own collision
/// rule, and the copy is the active track.
///
/// Returns ``(Bench, report)`` with ``label``, ``from`` and
/// ``observation_count``.
#[pyfunction]
fn duplicate(py: Python<'_>, bench: &PyBench, label: &str) -> PyResult<(PyBench, Py<PyDict>)> {
    let (next, report) = core_duplicate(&bench.inner, label).map_err(refused)?;
    let d = PyDict::new(py);
    d.set_item("label", report.label)?;
    d.set_item("from", report.from)?;
    d.set_item("observation_count", report.observation_count)?;
    Ok((PyBench::wrap(next), d.unbind()))
}

/// Write `track` into `edited` as one point.
///
/// The record is the track's payload plus its ``in`` observations' keypoints.
/// With no origin that resolves the point is appended; with one that does it
/// takes that point's place; and ``in`` observations pulled from other points
/// delete those points, which is what a merge is.
///
/// Nothing here triangulates: the track commits with the position it carries,
/// and a track that carries none refuses naming the evaluation as the step that
/// is missing.
///
/// Returns ``(EditedReconstruction, report)``. The report carries ``point``,
/// the index the written point took; ``replaced``, the index it took the place
/// of, present only when the track's origin resolved; the ``absorbed`` point
/// indexes; ``observation_count``; ``label``, the sentence a log records; and
/// ``map``, the :class:`PointMap` the commit made -- the write, with the
/// absorbed points' removal chained after it when there was one.
#[pyfunction]
#[pyo3(signature = (edited, track, *, node = "the reconstruction"))]
fn commit(
    py: Python<'_>,
    edited: &PyEditedReconstruction,
    track: &PyEditableTrack,
    node: &str,
) -> PyResult<(PyEditedReconstruction, Py<PyDict>)> {
    let (next, report) = core_commit(&edited.inner, &track.inner).map_err(refused)?;
    let d = PyDict::new(py);
    d.set_item("point", report.point)?;
    if let Some(replaced) = report.replaced {
        d.set_item("replaced", replaced)?;
    }
    d.set_item("absorbed", report.absorbed().to_vec().into_pyarray(py))?;
    d.set_item("observation_count", report.observation_count)?;
    d.set_item("label", report.label(node))?;
    d.set_item("map", PyPointMap::wrap(report.map))?;
    Ok((PyEditedReconstruction { inner: next }, d.unbind()))
}

/// Ask a descriptor index which other images hold the patch around one
/// observation, and add each as a candidate.
///
/// This is a **constellation query**, not a lookup of one descriptor: the
/// keypoints inside `radius_px` of the observation are looked up in `forest`,
/// the hits are grouped by image, and an image whose hits agree on one affine
/// warp with at least `min_inliers` of them is a candidate. The warp applied to
/// the observation's own pixel and keypoint-frame shape is the seed the new
/// observation takes, so a candidate arrives at the place and the size the warp
/// says the patch has in that image.
///
/// Args:
///     track: The track to search from.
///     observation: Which of its observations, by position in the list.
///     positions: `(N, 2)` float32 keypoint centres of the **searched image**,
///         in its own `.sift` row order, as
///         :meth:`SiftReader.read_positions_and_shapes` returns them. Nothing
///         here opens a `.sift` file.
///     affine_shapes: `(N, 2, 2)` float32 shapes for those same keypoints.
///     forest: An open :class:`LazyKdForest` whose corpus indexes this
///         reconstruction's images **in the same order**: a match names a
///         corpus image index and the observation it becomes names a
///         reconstruction image index.
///     radius_px: The constellation's radius around the observation, in that
///         image's pixels.
///     min_inliers: Fewest agreeing correspondences an image needs.
///     Remaining arguments are the constellation query's, as
///     :meth:`LazyKdForest.constellation_query` takes them.
///
/// Returns:
///     ``(EditableTrack, report)``. The report carries ``observation``,
///     ``observation_count``, ``image``, ``center``, ``constellation`` (how
///     many keypoints were asked about), ``added``, ``already_in_track``,
///     ``sentence`` and ``matches``: one dict per found image with its
///     ``image``, ``inliers``, ``correspondences``, ``affine``, ``pixel`` and
///     ``found`` -- ``"added"`` with the index it took, ``"already_in_track"``
///     with the observation that holds the image, or ``"own_image"``.
///     Raises ``ValueError`` with the reason when the search is refused.
#[pyfunction]
#[pyo3(signature = (track, observation, positions, affine_shapes, forest, *,
                    radius_px = DEFAULT_RADIUS_PX, min_inliers = QUERY_DEFAULTS.min_inliers,
                    k = QUERY_DEFAULTS.k, max_leaf_checks = QUERY_DEFAULTS.max_leaf_checks,
                    threshold_px = QUERY_DEFAULTS.threshold_px,
                    iterations = QUERY_DEFAULTS.iterations,
                    min_correspondences = QUERY_DEFAULTS.min_correspondences,
                    one_hit_per_image = QUERY_DEFAULTS.one_hit_per_image,
                    same_image_ratio = QUERY_DEFAULTS.same_image_ratio,
                    max_scale = QUERY_DEFAULTS.max_scale, seed = QUERY_DEFAULTS.seed))]
#[allow(clippy::too_many_arguments)]
fn search_descriptors(
    py: Python<'_>,
    track: &PyEditableTrack,
    observation: usize,
    positions: PyReadonlyArray2<'_, f32>,
    affine_shapes: PyReadonlyArray3<'_, f32>,
    forest: &PyLazyKdForest,
    radius_px: f32,
    min_inliers: usize,
    k: usize,
    max_leaf_checks: usize,
    threshold_px: f64,
    iterations: usize,
    min_correspondences: usize,
    one_hit_per_image: bool,
    same_image_ratio: f32,
    max_scale: f64,
    seed: u64,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let keypoints = read_keypoints(&positions, &affine_shapes)?;
    let options = SearchOptions {
        constellation: ConstellationParams {
            k,
            max_leaf_checks,
            threshold_px,
            iterations,
            min_correspondences,
            one_hit_per_image,
            same_image_ratio,
            min_inliers,
            max_scale,
            // The search's centre is the observation's own pixel, and the query
            // takes it from the radius it is given, so the default weighted
            // refit needs nothing said here.
            refit: QUERY_DEFAULTS.refit,
            seed,
        },
        radius_px,
        min_inliers,
    };
    let (next, report) = py
        .detach(|| {
            core_search_descriptors(
                &track.inner,
                observation,
                &keypoints,
                forest.inner(),
                &options,
                &Progress::none(),
            )
        })
        .map_err(refused)?;
    let d = search_report_dict(py, &report)?;
    Ok((
        PyEditableTrack {
            inner: Arc::new(next),
        },
        d.unbind(),
    ))
}

/// The searched image's keypoints, from the two arrays a `.sift` read gives.
fn read_keypoints(
    positions: &PyReadonlyArray2<'_, f32>,
    affine_shapes: &PyReadonlyArray3<'_, f32>,
) -> PyResult<ImageKeypoints> {
    let positions = positions.as_array();
    let affine_shapes = affine_shapes.as_array();
    let n = positions.nrows();
    if positions.ncols() != 2 || affine_shapes.shape() != [n, 2, 2] {
        return Err(PyValueError::new_err(format!(
            "positions must be (N, 2) and affine_shapes (N, 2, 2) for N={n}"
        )));
    }
    Ok(ImageKeypoints {
        positions: (0..n)
            .map(|i| [positions[[i, 0]], positions[[i, 1]]])
            .collect(),
        affine_shapes: (0..n)
            .map(|i| {
                [
                    [affine_shapes[[i, 0, 0]], affine_shapes[[i, 0, 1]]],
                    [affine_shapes[[i, 1, 0]], affine_shapes[[i, 1, 1]]],
                ]
            })
            .collect(),
    })
}

/// The dict form of a search report, with one entry per found image.
fn search_report_dict<'py>(py: Python<'py>, report: &SearchReport) -> PyResult<Bound<'py, PyDict>> {
    let matches = PyList::empty(py);
    for found in &report.matches {
        let entry = PyDict::new(py);
        entry.set_item("image", found.image)?;
        entry.set_item("inliers", found.inliers)?;
        entry.set_item("correspondences", found.correspondences)?;
        entry.set_item(
            "affine",
            PyArray1::from_vec(py, found.affine.iter().flatten().copied().collect())
                .reshape([2, 3])?,
        )?;
        entry.set_item("pixel", found.pixel)?;
        match found.found {
            Found::Added { observation } => {
                entry.set_item("found", "added")?;
                entry.set_item("observation", observation)?;
            }
            Found::AlreadyInTrack { observation } => {
                entry.set_item("found", "already_in_track")?;
                entry.set_item("observation", observation)?;
            }
            Found::OwnImage => entry.set_item("found", "own_image")?,
        }
        matches.append(entry)?;
    }
    let d = PyDict::new(py);
    d.set_item("observation", report.observation)?;
    d.set_item("observation_count", report.observation_count)?;
    d.set_item("image", report.image)?;
    d.set_item("center", report.center)?;
    d.set_item("constellation", report.constellation)?;
    d.set_item("added", report.added())?;
    d.set_item("already_in_track", report.already_in_track())?;
    d.set_item("sentence", report.to_string())?;
    d.set_item("matches", matches)?;
    Ok(d)
}

/// Register the bench bindings on the `sfmtool.bench` submodule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBench>()?;
    m.add_class::<PyEditableTrack>()?;
    m.add_function(wrap_pyfunction!(create_track, m)?)?;
    m.add_function(wrap_pyfunction!(create_cluster, m)?)?;
    m.add_function(wrap_pyfunction!(add_observation, m)?)?;
    m.add_function(wrap_pyfunction!(set_verdict, m)?)?;
    m.add_function(wrap_pyfunction!(translate_frame, m)?)?;
    m.add_function(wrap_pyfunction!(set_observation_keypoint, m)?)?;
    m.add_function(wrap_pyfunction!(resize_frame, m)?)?;
    m.add_function(wrap_pyfunction!(resize_from_edge, m)?)?;
    m.add_function(wrap_pyfunction!(rotate_frame, m)?)?;
    m.add_function(wrap_pyfunction!(set_observation_shape, m)?)?;
    m.add_function(wrap_pyfunction!(apply_thresholds, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(set_stage, m)?)?;
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(duplicate, m)?)?;
    m.add_function(wrap_pyfunction!(search_descriptors, m)?)?;
    m.add_function(wrap_pyfunction!(commit, m)?)?;
    Ok(())
}
