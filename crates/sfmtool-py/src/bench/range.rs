// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for distance ranges: the range a set of sightings allows
//! along a pixel's ray, the camera spread, and what a range says about its
//! point.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::bench::{
    camera_spread as core_camera_spread, classify_range as core_classify_range,
    distance_range as core_distance_range, RangeOptions,
};

use super::{refused, views_of};
use crate::patches::views::{resolve_pyramids, PosedViews};
use crate::reconstruction::edited::PyEditedReconstruction;

/// The distances along ``pixel``'s ray in ``image`` at which every other
/// sighting stays within its tolerance.
///
/// The error at a distance is the largest pixel distance between a sighting
/// and where the point at that distance projects in the sighting's image; an
/// image that cannot see the point counts as an infinite error. The tolerance
/// is ``tolerance_px``, or half a pixel more than the error at ``distance`` when
/// that is larger. Each end is found by doubling away from ``distance`` (from
/// ``1e7`` when it is infinite) until the error leaves the tolerance, then
/// bisecting ten times in log distance.
///
/// Args:
///     edited: The reconstruction whose cameras and poses are read.
///     images: One decoded image per image of ``edited``, or a prebuilt
///         :class:`ImagePyramidSet`, as :func:`far_field_sweep` takes them.
///         Only the cameras are read; the images are checked against them.
///     image: The queried image's index.
///     pixel: ``(x, y)`` in that image.
///     sightings: ``(image, (x, y))`` pairs of the point. Sightings in
///         ``image`` are not checked.
///     distance: The point's distance along the pixel's unit ray from the
///         queried camera's centre, ``math.inf`` for a point at infinity.
///     tolerance_px: The error each sighting may reach, before widening.
///
/// Returns:
///     ``(near, far)``: ``near`` is ``0`` when nothing is too near, and
///     ``far`` is ``math.inf`` when ``distance`` is or when the error at
///     infinity is within the tolerance.
///
/// Raises:
///     ValueError: an image is not one of the reconstruction's, or ``images``
///         does not match it.
#[pyfunction]
#[pyo3(signature = (edited, images, image, pixel, sightings, distance, tolerance_px = 1.0))]
#[allow(clippy::too_many_arguments)]
pub(super) fn distance_range(
    py: Python<'_>,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    image: u32,
    pixel: [f64; 2],
    sightings: Vec<(u32, [f64; 2])>,
    distance: f64,
    tolerance_px: f64,
) -> PyResult<(f64, f64)> {
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let views = views_of(&posed, &pyramids);
    let [near, far] = py
        .detach(|| core_distance_range(&views, image, pixel, &sightings, distance, tolerance_px))
        .map_err(refused)?;
    Ok((near, far))
}

/// The largest distance between two camera centres of ``edited``, the scale
/// ``far`` in :func:`classify_range` is measured against.
///
/// Args:
///     edited: The reconstruction.
///     images: As :func:`distance_range` takes them.
#[pyfunction]
pub(super) fn camera_spread(
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    Ok(core_camera_spread(&views_of(&posed, &pyramids)))
}

/// What a range says about its point.
///
/// Args:
///     range: ``(near, far)``, as :func:`distance_range` returns it.
///     camera_spread: The reconstruction's :func:`camera_spread`.
///     max_span: The widest a bounded range may be, far end over near end.
///     far_spread: A range with no far end is far when its near end is at
///         least this many times ``camera_spread``.
///
/// Returns:
///     A dict with ``bounded`` (finite at both ends, near above zero, and no
///     wider than ``max_span``), ``far`` (no far end and a near end at least
///     ``far_spread`` times the spread) and ``usable`` (either).
#[pyfunction]
#[pyo3(signature = (range, camera_spread, *, max_span = 3.0, far_spread = 5.0))]
pub(super) fn classify_range(
    py: Python<'_>,
    range: [f64; 2],
    camera_spread: f64,
    max_span: f64,
    far_spread: f64,
) -> PyResult<Py<PyDict>> {
    let options = RangeOptions {
        max_span,
        far_spread,
        ..RangeOptions::default()
    };
    let class = core_classify_range(range, camera_spread, &options);
    let d = PyDict::new(py);
    d.set_item("bounded", class.bounded)?;
    d.set_item("far", class.far)?;
    d.set_item("usable", class.usable())?;
    Ok(d.unbind())
}

/// Register the distance-range bindings on the `sfmtool.bench` submodule.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance_range, m)?)?;
    m.add_function(wrap_pyfunction!(camera_spread, m)?)?;
    m.add_function(wrap_pyfunction!(classify_range, m)?)?;
    Ok(())
}
