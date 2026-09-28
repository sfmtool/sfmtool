// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the matching sources of finding the tracks near a
//! pixel.
//!
//! Each source runs one query and gives back its candidates as dicts with the
//! keys the track-at-pixel harness's anchors carry, so the harness can call it
//! in place of its own source.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::bench::{
    nearby_points as core_nearby_points, NearbyCandidate, NearbySource, PointsOptions,
};

use super::views_of;
use crate::patches::views::{resolve_pyramids, PosedViews};
use crate::reconstruction::edited::PyEditedReconstruction;

/// Apply the `options` overrides through `set`, one key at a time.
fn with_overrides<T>(
    mut options: T,
    overrides: Option<&Bound<'_, PyDict>>,
    set: impl Fn(&mut T, &str, &Bound<'_, PyAny>) -> PyResult<bool>,
) -> PyResult<T> {
    if let Some(overrides) = overrides {
        for (key, value) in overrides.iter() {
            let key: String = key.extract()?;
            if !set(&mut options, &key, &value)? {
                return Err(PyValueError::new_err(format!("unknown option {key:?}")));
            }
        }
    }
    Ok(options)
}

/// The name the harness gives a source's anchors.
fn harness_name(source: NearbySource) -> &'static str {
    match source {
        // The harness calls the reconstruction's own points its tracks.
        NearbySource::Points => "tracks",
    }
}

/// One candidate as the harness's anchor dict.
fn candidate_dict<'py>(py: Python<'py>, c: &NearbyCandidate) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("source", harness_name(c.source))?;
    d.set_item("id", c.id)?;
    d.set_item("position", [c.position.x, c.position.y, c.position.z])?;
    // Each row a list `[image, x, y]`, as the harness writes them.
    let views = PyList::empty(py);
    for &(image, px) in &c.sightings {
        let row = PyList::empty(py);
        row.append(image)?;
        row.append(px[0])?;
        row.append(px[1])?;
        views.append(row)?;
    }
    d.set_item("views", views)?;
    d.set_item("query_pixel", c.query_pixel)?;
    d.set_item("distance_px", c.distance_px)?;
    d.set_item("n_views", c.n_views())?;
    d.set_item("max_reproj_px", c.max_reproj_px)?;
    d.set_item("max_ray_angle_deg", c.max_ray_angle_deg)?;
    d.set_item("depth", c.depth)?;
    Ok(d)
}

fn candidate_list(py: Python<'_>, found: &[NearbyCandidate]) -> PyResult<Py<PyList>> {
    let out = PyList::empty(py);
    for c in found {
        out.append(candidate_dict(py, c)?)?;
    }
    Ok(out.unbind())
}

/// The reconstruction's points observed near ``pixel`` in ``image``, nearest
/// first, as candidate tracks.
///
/// A point is kept when it is finite, has ``min_views`` or more observations
/// and every observation lies within ``max_reproj_px`` of where it projects;
/// at most ``max_points`` are kept.
///
/// Args:
///     edited: The reconstruction, whose deleted points are never found.
///     images: One decoded image per image of ``edited``, or a prebuilt
///         :class:`ImagePyramidSet`, as :func:`far_field_sweep` takes them.
///         Only the cameras are read.
///     image: The queried image's index.
///     pixel: ``(x, y)`` in that image.
///     options: Overrides keyed by the field of the Rust ``PointsOptions``:
///         ``radius_px`` (40), ``max_points`` (8), ``min_views`` (2) and
///         ``max_reproj_px`` (2). An unknown key is an error.
///
/// Returns:
///     A list of dicts with the harness's anchor keys: ``source``
///     (``"tracks"``), ``id`` (the point), ``position``, ``views``
///     (``[image, x, y]`` rows, the queried image first), ``query_pixel``,
///     ``distance_px``, ``n_views``, ``max_reproj_px``, ``max_ray_angle_deg``
///     and ``depth``.
///
/// Raises:
///     ValueError: the image or pixel names no place, or an input does not
///         match the reconstruction.
#[pyfunction]
#[pyo3(signature = (edited, images, image, pixel, *, options = None))]
pub(super) fn nearby_points(
    py: Python<'_>,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    image: u32,
    pixel: [f64; 2],
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyList>> {
    let options = with_overrides(PointsOptions::default(), options, |o, key, value| {
        match key {
            "radius_px" => o.radius_px = value.extract()?,
            "max_points" => o.max_points = value.extract()?,
            "min_views" => o.min_views = value.extract()?,
            "max_reproj_px" => o.max_reproj_px = value.extract()?,
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let views = views_of(&posed, &pyramids);
    let found = py
        .detach(|| core_nearby_points(&edited.inner, &views, image, pixel, &options))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    candidate_list(py, &found)
}

/// Register the matching-source bindings on the `sfmtool.bench` submodule.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nearby_points, m)?)?;
    Ok(())
}
