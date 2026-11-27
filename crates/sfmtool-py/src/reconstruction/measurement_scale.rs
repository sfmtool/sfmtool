// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `SfmrReconstruction.distance_measurement_scale`: the uniform scale factor
//! the reconstruction's stored distance measurements give it in a physical
//! unit, the median of the per-measurement factors, with the statistics of
//! their spread and the measurements that give none (see
//! ``specs/core/reconstruction/measurement-scale.md``).

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::reconstruction::measurement_endpoint::{
    resolve_distance_measurements, MeasurementEndpointOptions,
};
use sfmtool_core::reconstruction::measurement_scale::{
    distance_measurement_scale, SCALE_DISAGREEMENT_PERCENT,
};
use sfmtool_core::reconstruction::EditedReconstruction;
use sfmtool_core::LengthUnit;

use super::sfmr_reconstruction::PySfmrReconstruction;

#[pymethods]
impl PySfmrReconstruction {
    /// The uniform scale factor that puts this reconstruction in ``unit``, the
    /// median of the factors its stored distance measurements imply, with how
    /// well they agree.
    ///
    /// Each measurement's endpoints are resolved with the default options, as
    /// ``resolve_distance_measurements`` resolves them, and each measurement
    /// with a finite scene distance gives the factor ``stated / scene_distance``,
    /// its stated distance converted to ``unit``. Multiplying the
    /// coordinates by the median puts them in ``unit``, whatever
    /// ``world_space_unit`` the file declares. See
    /// ``specs/core/reconstruction/measurement-scale.md``.
    ///
    /// Args:
    ///     unit: The unit to scale to: ``"mm"``, ``"cm"``, ``"m"``, ``"in"``
    ///         or ``"ft"``.
    ///
    /// Returns:
    ///     A dict with ``unit``; ``factor``, the median; ``factors``, one dict
    ///     per usable measurement in stored order with ``index``, ``label``
    ///     (the stored label, ``""`` for none), ``stated`` (in ``unit``),
    ///     ``scene_distance``, ``factor`` and ``deviation_percent`` (signed,
    ///     of the median); ``excluded``, one dict per measurement that gives
    ///     no factor with ``index``, ``label`` and ``reason``, a sentence; and
    ///     ``statistics``, a dict with ``count``, ``median``, ``mean``,
    ///     ``sample_std`` and ``sample_std_percent``
    ///     (``None`` for one measurement), ``median_absolute_deviation``,
    ///     ``min``, ``max``, ``max_abs_deviation_percent``, ``disagrees`` and
    ///     ``disagreement_percent``, the bar ``disagrees`` is measured against.
    ///     Raises ``ValueError`` for a unit that is not one of the five, where
    ///     the reconstruction stores no distance measurement, and where none
    ///     gives a factor, the last naming each measurement and why.
    // A Python docstring, not Rust prose: its indented `Args:` paragraphs read
    // as Markdown code blocks, which rustdoc would parse as Rust.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    fn distance_measurement_scale<'py>(
        &self,
        py: Python<'py>,
        unit: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let length_unit = LengthUnit::from_name(unit).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown unit {unit:?}: expected one of {:?}",
                LengthUnit::NAMES
            ))
        })?;
        let value = EditedReconstruction::new(Arc::new(self.inner().clone()));
        let scale = py
            .detach(|| {
                let readings =
                    resolve_distance_measurements(&value, &MeasurementEndpointOptions::default());
                distance_measurement_scale(value.measurements(), &readings, length_unit)
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        // The call succeeded, so the distance kind is present and each index
        // is one of its rows.
        let labels: &[String] = value
            .measurements()
            .distance
            .as_ref()
            .map_or(&[], |distance| &distance.labels);
        let label = |index: usize| labels.get(index).map_or("", String::as_str);

        let d = PyDict::new(py);
        d.set_item("unit", length_unit.name())?;
        d.set_item("factor", scale.factor())?;
        let factors = PyList::empty(py);
        for f in &scale.factors {
            let item = PyDict::new(py);
            item.set_item("index", f.index)?;
            item.set_item("label", label(f.index))?;
            item.set_item("stated", f.stated)?;
            item.set_item("scene_distance", f.scene_distance)?;
            item.set_item("factor", f.factor)?;
            item.set_item("deviation_percent", f.deviation_percent)?;
            factors.append(item)?;
        }
        d.set_item("factors", factors)?;
        let excluded = PyList::empty(py);
        for e in &scale.excluded {
            let item = PyDict::new(py);
            item.set_item("index", e.index)?;
            item.set_item("label", label(e.index))?;
            item.set_item("reason", e.reason.to_string())?;
            excluded.append(item)?;
        }
        d.set_item("excluded", excluded)?;
        let s = &scale.statistics;
        let statistics = PyDict::new(py);
        statistics.set_item("count", s.count)?;
        statistics.set_item("median", s.median)?;
        statistics.set_item("mean", s.mean)?;
        statistics.set_item("sample_std", s.sample_std)?;
        statistics.set_item("sample_std_percent", s.sample_std_percent)?;
        statistics.set_item("median_absolute_deviation", s.median_absolute_deviation)?;
        statistics.set_item("min", s.min)?;
        statistics.set_item("max", s.max)?;
        statistics.set_item("max_abs_deviation_percent", s.max_abs_deviation_percent)?;
        statistics.set_item("disagrees", s.disagrees)?;
        statistics.set_item("disagreement_percent", SCALE_DISAGREEMENT_PERCENT)?;
        d.set_item("statistics", statistics)?;
        Ok(d)
    }
}
