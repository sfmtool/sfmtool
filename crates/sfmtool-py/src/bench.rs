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
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList};

use sfmtool_core::bench::{
    add_observation as core_add_observation, apply_thresholds as core_apply_thresholds,
    commit as core_commit, create_cluster as core_create_cluster,
    create_track as core_create_track, set_verdict as core_set_verdict, split as core_split, Bench,
    BenchItem, ClusterSeed, CommitOutcome, CreateTrackOptions, EditableTrack, ItemKind,
    Observation, ObservationSeed, Provenance, Verdict,
};

use crate::reconstruction::edited::PyEditedReconstruction;

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

/// The provenance `word` names, with `feature` and `point` supplying what the
/// variant that needs one needs.
fn parse_provenance(word: &str, feature: Option<u32>, point: Option<u32>) -> PyResult<Provenance> {
    match word {
        "origin" => Ok(Provenance::Origin),
        "sweep" => Ok(Provenance::Sweep),
        "pixel" => Ok(Provenance::Pixel),
        "descriptor" => feature
            .map(|feature| Provenance::Descriptor { feature })
            .ok_or_else(|| {
                PyValueError::new_err("a descriptor provenance needs the 'feature' it returned")
            }),
        "point" => point
            .map(|point| Provenance::Point { point })
            .ok_or_else(|| {
                PyValueError::new_err("a point provenance needs the 'point' it was pulled from")
            }),
        other => Err(PyValueError::new_err(format!(
            "unknown provenance: {other:?} (expected origin|descriptor|sweep|pixel|point)"
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
            ("shift_px", m.shift_px),
            ("reprojection_error", m.reprojection_error),
            ("ray_angle_deg", m.ray_angle_deg),
            ("localizability", m.localizability),
        ] {
            if let Some(value) = value {
                t.set_item(key, value)?;
            }
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

    /// Where the track's point stands, or ``None`` at the cluster stage or
    /// before anything has triangulated it.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        let p = self.inner.track()?.position?;
        Some(PyArray1::from_vec(py, vec![p.x, p.y, p.z]))
    }

    /// Which observation the cluster stage cuts its template around, or
    /// ``None`` at the track stage.
    #[getter]
    fn reference(&self) -> Option<usize> {
        Some(self.inner.cluster()?.reference)
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
/// `radius_px` (an isotropic seed around a pixel nobody detected) or `shape` (a
/// ``2x2`` affine shape, the map from the detector's canonical unit frame onto
/// this image's pixels) says how large the patch is.
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
        (None, Some(r)) => [[r, 0.0], [0.0, r]],
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
/// `shape` defaults to the reference observation's own, so a pixel gesture on a
/// track that already has a scale needs no radius. `provenance` is one of
/// ``origin``, ``descriptor`` (with `feature`), ``sweep``, ``pixel`` or
/// ``point`` (with `point`, which a commit then absorbs).
///
/// Returns ``(EditableTrack, report)``.
#[pyfunction]
#[pyo3(signature = (track, image, pixel, *, shape = None, provenance = "pixel", feature = None, point = None))]
#[allow(clippy::too_many_arguments)]
fn add_observation(
    py: Python<'_>,
    track: &PyEditableTrack,
    image: u32,
    pixel: [f64; 2],
    shape: Option<[[f64; 2]; 2]>,
    provenance: &str,
    feature: Option<u32>,
    point: Option<u32>,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let seed = ObservationSeed {
        image,
        pixel,
        shape,
        provenance: parse_provenance(provenance, feature, point)?,
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
/// Returns ``(Bench, report)``; the report's ``label`` names the second track.
#[pyfunction]
fn split(
    py: Python<'_>,
    bench: &PyBench,
    label: &str,
    observations: Vec<usize>,
) -> PyResult<(PyBench, Py<PyDict>)> {
    let (next, report) = core_split(&bench.inner, label, &observations).map_err(refused)?;
    let d = PyDict::new(py);
    d.set_item("label", report.label)?;
    d.set_item("moved", report.moved)?;
    d.set_item("kept", report.kept)?;
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
/// Returns ``(EditedReconstruction, report)``. The report carries ``outcome``
/// (``"created"`` or ``"replaced"``), ``point``, ``replaced`` where there is
/// one, the ``absorbed`` point indexes, ``observation_count`` and ``label``, the
/// sentence a log records.
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
    match report.outcome {
        CommitOutcome::Created(point) => {
            d.set_item("outcome", "created")?;
            d.set_item("point", point)?;
        }
        CommitOutcome::Replaced { point, replaced } => {
            d.set_item("outcome", "replaced")?;
            d.set_item("point", point)?;
            d.set_item("replaced", replaced)?;
        }
    }
    d.set_item("absorbed", report.absorbed.clone().into_pyarray(py))?;
    d.set_item("observation_count", report.observation_count)?;
    d.set_item("label", report.label(node))?;
    Ok((PyEditedReconstruction { inner: next }, d.unbind()))
}

/// Register the bench bindings on the `sfmtool.bench` submodule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBench>()?;
    m.add_class::<PyEditableTrack>()?;
    m.add_function(wrap_pyfunction!(create_track, m)?)?;
    m.add_function(wrap_pyfunction!(create_cluster, m)?)?;
    m.add_function(wrap_pyfunction!(add_observation, m)?)?;
    m.add_function(wrap_pyfunction!(set_verdict, m)?)?;
    m.add_function(wrap_pyfunction!(apply_thresholds, m)?)?;
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(commit, m)?)?;
    Ok(())
}
