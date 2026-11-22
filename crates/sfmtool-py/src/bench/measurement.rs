// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the editable measurement: the item, putting a stored
//! measurement on the bench, and the commit that writes it back.
//!
//! `specs/core/bench/editable-measurement.md` is the design. The item is a
//! plain value like the track, and both steps give back the next value and
//! leave their inputs as they were.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use sfmtool_core::bench::{
    commit_measurement as core_commit_measurement, put_measurement as core_put_measurement,
    EditableMeasurement, EditableMeasurementBody, MeasurementKind,
};

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

    /// The measurement's own label, written into the file on commit. It may be
    /// empty, and it is not the item's label on the bench.
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

    /// The stated distance and its unit name, ``(12.0, "cm")``, or ``None``
    /// when none is given.
    #[getter]
    fn distance(&self) -> Option<(f64, &'static str)> {
        let EditableMeasurementBody::Distance(distance) = &self.inner.body;
        distance.distance.map(|(d, unit)| (d, unit.name()))
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

/// Write `measurement` into `edited`: replace row `origin_row` of its kind, or
/// append a row where `origin_row` is ``None``.
///
/// The caller resolves the measurement's origin to a row of `edited` first.
/// The new value shares `edited`'s points, tracks and patch bitmaps and
/// carries its point edits across unchanged, since no point index moves. A
/// row that already holds exactly what the measurement holds, as the file
/// stores it, is not written.
///
/// Refused with a ``ValueError`` naming what is missing or wrong: an endpoint
/// not placed, no stated distance, a distance that is not finite and
/// positive, an image past the image set, a pixel outside its image, or a row
/// past the rows.
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

/// Register the measurement bindings on the `sfmtool.bench` submodule.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEditableMeasurement>()?;
    m.add_function(wrap_pyfunction!(put_measurement, m)?)?;
    m.add_function(wrap_pyfunction!(commit_measurement, m)?)?;
    Ok(())
}
