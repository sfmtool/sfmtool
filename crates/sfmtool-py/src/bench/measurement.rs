// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the editable measurement: the item, putting a stored
//! measurement on the bench or starting a new one from a pixel, the steps
//! that set its label, its stated distance and an endpoint, the parser of a
//! typed distance, and the commit that writes it back.
//!
//! `specs/core/bench/editable-measurement.md` is the design. The item is a
//! plain value like the track, and each step gives back the next value and
//! leaves its inputs as they were.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use sfmtool_core::bench::{
    commit_measurement as core_commit_measurement,
    default_measurement_label as core_default_measurement_label,
    parse_stated_distance as core_parse_stated_distance, put_measurement as core_put_measurement,
    set_measurement_endpoint as core_set_measurement_endpoint,
    set_measurement_label as core_set_measurement_label,
    set_stated_distance as core_set_stated_distance, start_measurement as core_start_measurement,
    EditableMeasurement, EditableMeasurementBody, MeasurementEndpoint, MeasurementKind,
};
use sfmtool_core::LengthUnit;

use super::{refused, PyBench};
use crate::reconstruction::edited::PyEditedReconstruction;

/// The kind `name` spells, or a ``ValueError`` naming the kinds there are.
fn kind_named(name: &str) -> PyResult<MeasurementKind> {
    MeasurementKind::from_name(name).ok_or_else(|| {
        let kinds: Vec<&str> = MeasurementKind::ALL.iter().map(|k| k.name()).collect();
        PyValueError::new_err(format!(
            "{name:?} is not a kind of measurement; the kinds are {kinds:?}"
        ))
    })
}

/// A measurement being worked on: where it came from, and what it will write.
///
/// It holds only what the file stores of a measurement: each endpoint's image
/// and pixel, the stated distance and its unit, and the label. An endpoint's
/// point and 3D position are not held; they follow from its image and pixel
/// through :meth:`SfmrReconstruction.resolve_measurement_endpoint`.
///
/// A plain value, as :class:`EditableTrack` is.
#[pyclass(name = "EditableMeasurement", module = "sfmtool.bench")]
#[derive(Clone)]
pub struct PyEditableMeasurement {
    pub(crate) inner: Arc<EditableMeasurement>,
}

#[pymethods]
impl PyEditableMeasurement {
    /// The kind of measurement, ``"distance"``.
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind().name()
    }

    /// The measurement's own label, written into the file on commit. A commit
    /// needs it not empty and not only white space, and it is not the item's
    /// label on the bench.
    #[getter]
    fn label(&self) -> String {
        self.inner.label.clone()
    }

    /// The stored row this measurement was put on the bench from, as
    /// ``{"version": ..., "kind": ..., "index": ...}``, or ``None`` for one
    /// with no row to replace.
    #[getter]
    fn origin<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.origin {
            None => Ok(None),
            Some(origin) => {
                let d = PyDict::new(py);
                d.set_item("version", origin.version)?;
                d.set_item("kind", origin.kind.name())?;
                d.set_item("index", origin.index)?;
                Ok(Some(d))
            }
        }
    }

    /// Endpoints A and B, each ``(image, (x, y))``, or ``None`` for an
    /// endpoint that is not placed.
    #[getter]
    fn endpoints(&self) -> Vec<Option<(u32, (f64, f64))>> {
        let EditableMeasurementBody::Distance(distance) = &self.inner.body;
        distance
            .endpoints
            .iter()
            .map(|e| e.map(|e| (e.image, (e.pixel[0], e.pixel[1]))))
            .collect()
    }

    /// The stated distance in :attr:`unit`, ``12.0``, or ``None`` when none
    /// is given.
    #[getter]
    fn distance(&self) -> Option<f64> {
        let EditableMeasurementBody::Distance(distance) = &self.inner.body;
        distance.distance
    }

    /// The name of the unit the stated distance is in, or is given in once
    /// there is one, ``"cm"``: present whether or not a distance is given.
    #[getter]
    fn unit(&self) -> &'static str {
        let EditableMeasurementBody::Distance(distance) = &self.inner.body;
        distance.unit.name()
    }

    /// This measurement with its origin set to row `index` at `version`, and
    /// nothing else changed: what a caller re-seats a committed measurement
    /// on, so a second commit of it replaces what the first wrote.
    fn with_origin(&self, version: u64, index: u32) -> Self {
        Self {
            inner: Arc::new(self.inner.with_origin(version, index)),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EditableMeasurement(kind={}, label={:?}, {} endpoints placed)",
            self.inner.kind(),
            self.inner.label,
            self.inner.placed_endpoints()
        )
    }
}

/// Put the stored measurement at row `index` of `kind` on the bench.
///
/// The item copies the row's endpoints, stated distance, unit and label, and
/// its origin is that row at `version`, the caller's own numbering of the
/// version `edited` is. Its label on the bench is minted from the kind and
/// the row, ``"distance #2"``, with a collision suffix where another item
/// holds it. A row past the kind's rows is a ``ValueError`` naming the count.
///
/// Returns ``(Bench, EditableMeasurement)``.
#[pyfunction]
#[pyo3(signature = (bench, edited, kind, index, *, version = 0))]
fn put_measurement(
    bench: &PyBench,
    edited: &PyEditedReconstruction,
    kind: &str,
    index: u32,
    version: u64,
) -> PyResult<(PyBench, PyEditableMeasurement)> {
    let kind = kind_named(kind)?;
    let (next, label) =
        core_put_measurement(&bench.inner, &edited.inner, kind, index, version).map_err(refused)?;
    let measurement = PyEditableMeasurement {
        inner: Arc::clone(next.measurement(&label).expect("just put on")),
    };
    Ok((PyBench { inner: next }, measurement))
}

/// Put a new distance measurement on the bench with endpoint A at pixel `xy`
/// of image `image`, and nothing else.
///
/// The measurement has no origin, so its commit appends a row, and no
/// endpoint B or stated distance until the other steps give them. Its own
/// label is :func:`default_measurement_label`'s, ``"measurement 1"``, and its
/// unit the reconstruction's ``world_space_unit``, or metres where it names
/// none. Its label on the bench is minted from where it starts,
/// ``"distance from IMG_0042@142,198"`` (the image stem and the rounded
/// pixel), with a collision suffix where another item holds it. An image past
/// `edited`'s images, and a pixel outside ``[0, width) x [0, height)`` of its
/// image, are each a ``ValueError`` in the commit's words.
///
/// Returns ``(Bench, EditableMeasurement)``.
#[pyfunction]
fn start_measurement(
    bench: &PyBench,
    edited: &PyEditedReconstruction,
    image: u32,
    xy: (f64, f64),
) -> PyResult<(PyBench, PyEditableMeasurement)> {
    let endpoint = MeasurementEndpoint {
        image,
        pixel: [xy.0, xy.1],
    };
    let (next, label) =
        core_start_measurement(&bench.inner, &edited.inner, endpoint).map_err(refused)?;
    let measurement = PyEditableMeasurement {
        inner: Arc::clone(next.measurement(&label).expect("just put on")),
    };
    Ok((PyBench { inner: next }, measurement))
}

/// The label a new measurement on `bench` over `edited` starts with,
/// ``"measurement N"``: the smallest ``N`` from 1 that no stored distance
/// measurement of `edited` and no measurement on `bench` holds as its own
/// label.
#[pyfunction]
fn default_measurement_label(bench: &PyBench, edited: &PyEditedReconstruction) -> String {
    core_default_measurement_label(&bench.inner, &edited.inner)
}

/// Write `measurement` into `edited`: replace row `origin_row` of its kind, or
/// append a row where `origin_row` is ``None``.
///
/// The caller resolves the measurement's origin to a row of `edited` first.
/// The new value shares `edited`'s points, tracks and patch bitmaps and
/// carries its point edits across unchanged, since no point index moves. A
/// row that already holds exactly what the measurement holds, as the file
/// stores it, is not written.
///
/// An endpoint not placed and no stated distance are written as absent.
/// Refused with a ``ValueError`` naming what is wrong: a label that is empty
/// or only white space, a distance that is not finite and positive, an image
/// past the image set, a pixel outside its image, or a row past the rows.
///
/// Returns ``(EditedReconstruction, report)``. The report carries ``index``,
/// the row written (or the row that already holds it); ``replaced``, whether
/// a row was replaced rather than appended; ``wrote``, whether anything was
/// written; and ``label``, the sentence a log records.
#[pyfunction]
#[pyo3(signature = (edited, measurement, origin_row = None, *, node = "the reconstruction"))]
fn commit_measurement(
    py: Python<'_>,
    edited: &PyEditedReconstruction,
    measurement: &PyEditableMeasurement,
    origin_row: Option<u32>,
    node: &str,
) -> PyResult<(PyEditedReconstruction, Py<PyDict>)> {
    let commit =
        core_commit_measurement(&edited.inner, &measurement.inner, origin_row).map_err(refused)?;
    let d = PyDict::new(py);
    d.set_item("index", commit.index)?;
    d.set_item("replaced", commit.replaced)?;
    d.set_item("wrote", commit.wrote)?;
    d.set_item("label", commit.label(node))?;
    Ok((
        PyEditedReconstruction {
            inner: commit.value,
        },
        d.unbind(),
    ))
}

/// The unit `name` spells, or a ``ValueError`` naming the units there are.
fn unit_named(name: &str) -> PyResult<LengthUnit> {
    LengthUnit::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "{name:?} is not a unit: the units are {}",
            LengthUnit::NAMES.join(", ")
        ))
    })
}

/// `measurement` with its own label set to `label`, and nothing else changed.
///
/// Any text is a label, kept as given, except one that is empty or only white
/// space, which is a ``ValueError``. Returns the new
/// :class:`EditableMeasurement` and leaves `measurement` as it was.
#[pyfunction]
fn set_measurement_label(
    measurement: &PyEditableMeasurement,
    label: &str,
) -> PyResult<PyEditableMeasurement> {
    let next = core_set_measurement_label(&measurement.inner, label).map_err(refused)?;
    Ok(PyEditableMeasurement {
        inner: Arc::new(next),
    })
}

/// `measurement` with its stated distance set to `distance` in `unit` (a unit
/// name, ``"cm"``), and nothing else changed.
///
/// A distance that is not finite and strictly positive as the ``float32`` the
/// file holds, or a unit that is not one of ``mm``, ``cm``, ``m``, ``in`` and
/// ``ft``, is a ``ValueError``. Returns the new :class:`EditableMeasurement`
/// and leaves `measurement` as it was.
#[pyfunction]
fn set_stated_distance(
    measurement: &PyEditableMeasurement,
    distance: f64,
    unit: &str,
) -> PyResult<PyEditableMeasurement> {
    let unit = unit_named(unit)?;
    let next = core_set_stated_distance(&measurement.inner, distance, unit).map_err(refused)?;
    Ok(PyEditableMeasurement {
        inner: Arc::new(next),
    })
}

/// `measurement` with endpoint `side` (``0`` for A, ``1`` for B) moved to pixel
/// `xy` of image `image`, and nothing else changed: the step that switches an
/// endpoint to another image, or places one that is not placed.
///
/// Only the image and the pixel are set; the point, texel and position follow
/// from them through :meth:`SfmrReconstruction.resolve_measurement_endpoint`.
/// A side other than 0 and 1, an image past `edited`'s images, and a pixel
/// outside ``[0, width) x [0, height)`` of its image are each a ``ValueError``
/// in the commit's words. Returns the new :class:`EditableMeasurement` and
/// leaves `measurement` as it was.
#[pyfunction]
fn set_measurement_endpoint(
    edited: &PyEditedReconstruction,
    measurement: &PyEditableMeasurement,
    side: usize,
    image: u32,
    xy: (f64, f64),
) -> PyResult<PyEditableMeasurement> {
    let endpoint = MeasurementEndpoint {
        image,
        pixel: [xy.0, xy.1],
    };
    let next = core_set_measurement_endpoint(&edited.inner, &measurement.inner, side, endpoint)
        .map_err(refused)?;
    Ok(PyEditableMeasurement {
        inner: Arc::new(next),
    })
}

/// Read a distance written as a number with an optional unit suffix,
/// ``"12in"``, ``"30.48 cm"`` or ``"304.8"``, into ``(value, unit_name)``.
///
/// A bare number is in `bare_unit`. The suffixes are ``mm``, ``cm``, ``m``,
/// ``in`` and ``ft``, with or without a space before them. Text that is not
/// such a number, an unknown suffix, and a value that is not finite and
/// strictly positive are each a ``ValueError`` saying which.
#[pyfunction]
#[pyo3(signature = (text, bare_unit = "m"))]
fn parse_stated_distance(text: &str, bare_unit: &str) -> PyResult<(f64, &'static str)> {
    let bare_unit = unit_named(bare_unit)?;
    let (value, unit) = core_parse_stated_distance(text, bare_unit).map_err(refused)?;
    Ok((value, unit.name()))
}

/// Register the measurement bindings on the `sfmtool.bench` submodule.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEditableMeasurement>()?;
    m.add_function(wrap_pyfunction!(put_measurement, m)?)?;
    m.add_function(wrap_pyfunction!(start_measurement, m)?)?;
    m.add_function(wrap_pyfunction!(default_measurement_label, m)?)?;
    m.add_function(wrap_pyfunction!(commit_measurement, m)?)?;
    m.add_function(wrap_pyfunction!(set_measurement_label, m)?)?;
    m.add_function(wrap_pyfunction!(set_stated_distance, m)?)?;
    m.add_function(wrap_pyfunction!(set_measurement_endpoint, m)?)?;
    m.add_function(wrap_pyfunction!(parse_stated_distance, m)?)?;
    Ok(())
}
