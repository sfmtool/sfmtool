// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the far-field sweep.
//!
//! [`far_field_sweep`] runs one sweep and gives back its readings as dicts
//! with the keys the track-at-pixel harness's anchors carry, so the harness can
//! call it in place of its own `from_farfield`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::bench::{
    far_field_sweep as core_far_field_sweep, FarFieldOptions, FarFieldReading, Refit, WideAmong,
};
use sfmtool_core::progress::Progress;

use super::views_of;
use crate::patches::views::{resolve_grey, resolve_pyramids, PosedViews};
use crate::reconstruction::edited::PyEditedReconstruction;

/// Set one field of `options` from a Python value.
pub(super) fn set_option(
    options: &mut FarFieldOptions,
    key: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let o = options;
    match key {
        "disparities" => o.disparities = value.extract()?,
        "radius_px" => o.radius_px = value.extract()?,
        "wide" => o.wide = value.extract()?,
        "wide_among" => {
            let word: String = value.extract()?;
            o.wide_among = word.parse::<WideAmong>().map_err(PyValueError::new_err)?;
        }
        "min_whole" => o.min_whole = value.extract()?,
        "min_middle" => o.min_middle = value.extract()?,
        "middle_min_std" => o.middle_min_std = value.extract()?,
        "max_peaks" => o.max_peaks = value.extract()?,
        "min_prominence" => o.min_prominence = value.extract()?,
        "refit" => o.refit = value.extract()?,
        "group_cut" => o.group_cut = value.extract()?,
        "group_max" => o.group_max = value.extract()?,
        "refit_px" => o.refit_px = value.extract()?,
        "refit_max_px" => o.refit_max_px = value.extract()?,
        "refit_max_err_px" => o.refit_max_err_px = value.extract()?,
        _ => return Err(PyValueError::new_err(format!("unknown option {key:?}"))),
    }
    Ok(())
}

/// Read ``pixel``'s patch in ``image`` from infinity in, in every image it
/// lands in, and return one reading per peak of that reading.
///
/// Disparities are counted in the image that moves the pixel most for a change
/// of inverse distance: a disparity ``d`` is the distance ``R / d`` along the
/// pixel's ray, ``R`` that image's pixels per unit of inverse distance, and
/// ``d = 0`` is infinity. The patch (an 11 x 11 grid of ``radius_px``, 8 px) is
/// read at every disparity on a plane facing the queried camera, sampled in
/// grey and blurred, and each peak of the reading over the wide images is a
/// reading. Each reading's images are grouped by the ZNCC of their middles with
/// each other, and a reading whose queried image stands alone is refitted on
/// the largest other group: it keeps its place, moves to where the fit lands,
/// or is dropped.
///
/// Args:
///     edited: The reconstruction; only the refit's fit reads it.
///     images: One decoded image per image of ``edited``, or a prebuilt
///         :class:`ImagePyramidSet`, as :func:`evaluate` takes them. A set
///         also keeps the grey images between calls.
///     image: The queried image's index.
///     pixel: ``(x, y)`` in that image.
///     options: Overrides of the sweep's parameters, keyed by the field of the
///         Rust ``FarFieldOptions``: ``disparities``, ``radius_px``, ``wide``,
///         ``wide_among`` (``"matching"`` or ``"all"``), ``min_whole``,
///         ``min_middle``, ``middle_min_std``, ``max_peaks``,
///         ``min_prominence``, ``refit``, ``group_cut``, ``group_max``,
///         ``refit_px``, ``refit_max_px`` and ``refit_max_err_px``. An unknown
///         key is an error.
///
/// Returns:
///     A list of dicts, highest peak first, each with ``source``
///     (``"farfield"``), ``id`` (``None``), ``position`` and ``w`` (``0`` for a
///     bearing), ``views`` (``[image, x, y]`` rows, the queried image first),
///     ``query_pixel``, ``distance_px``, ``n_views``, ``max_reproj_px``,
///     ``max_ray_angle_deg``, ``depth``, ``farfield`` (what the reading rests
///     on) and, with the refit, ``groups``, ``query_middle``, ``refit`` and
///     where the refit landed, ``refit_px``. A reading that stays at the
///     sweep's distance carries ``disparity`` and ``range_override``, the
///     distances it allows; a moved one carries ``sweep_disparity`` instead.
///
/// Raises:
///     ValueError: the image or pixel names no place, or an input does not
///         match the reconstruction.
#[pyfunction]
#[pyo3(signature = (edited, images, image, pixel, *, options = None))]
pub(super) fn far_field_sweep(
    py: Python<'_>,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    image: u32,
    pixel: [f64; 2],
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyList>> {
    let overrides = options;
    let mut options = FarFieldOptions::default();
    if let Some(overrides) = overrides {
        for (key, value) in overrides.iter() {
            let key: String = key.extract()?;
            set_option(&mut options, &key, &value)?;
        }
    }
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let grey = resolve_grey(images, posed.cameras.len());
    let views = views_of(&posed, &pyramids);
    let sweep = py
        .detach(|| {
            core_far_field_sweep(
                &edited.inner,
                &views,
                &grey,
                image,
                pixel,
                &options,
                &Progress::none(),
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out = PyList::empty(py);
    for reading in &sweep.readings {
        out.append(reading_dict(py, reading)?)?;
    }
    Ok(out.unbind())
}

/// One reading as the harness's anchor dict.
pub(super) fn reading_dict<'py>(
    py: Python<'py>,
    r: &FarFieldReading,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("source", "farfield")?;
    d.set_item("id", py.None())?;
    d.set_item("position", [r.position.x, r.position.y, r.position.z])?;
    d.set_item("w", if r.at_infinity { 0.0 } else { 1.0 })?;
    // Each row a list `[image, x, y]`, as the harness writes them.
    let views = PyList::empty(py);
    for &(image, px) in &r.views {
        let row = PyList::empty(py);
        row.append(image)?;
        row.append(px[0])?;
        row.append(px[1])?;
        views.append(row)?;
    }
    d.set_item("views", views)?;
    d.set_item("query_pixel", r.query_pixel)?;
    d.set_item("distance_px", r.distance_px)?;
    d.set_item("n_views", r.views.len())?;
    d.set_item("max_reproj_px", r.max_reproj_px)?;
    d.set_item("max_ray_angle_deg", r.max_ray_angle_deg)?;
    d.set_item("depth", r.depth)?;
    if r.refit == Some(Refit::Moved) {
        d.set_item("sweep_disparity", r.disparity)?;
    } else {
        d.set_item("disparity", r.disparity)?;
    }
    if let Some(range) = r.range {
        d.set_item("range_override", range)?;
    }
    let m = &r.metrics;
    let f = PyDict::new(py);
    f.set_item("whole", m.whole)?;
    f.set_item("middle", m.middle)?;
    f.set_item("prominence", m.prominence)?;
    f.set_item("peak_rank", m.peak_rank)?;
    f.set_item("peaks", m.peaks)?;
    f.set_item("middle_flat", m.middle_flat)?;
    f.set_item("middle_std", m.middle_std)?;
    f.set_item("images", m.images)?;
    f.set_item("parallax_px", m.parallax_px)?;
    f.set_item("widest_px", m.widest_px)?;
    f.set_item("profile_whole", &m.profile_whole)?;
    f.set_item("profile_middle", &m.profile_middle)?;
    if let Some(g) = &r.grouping {
        f.set_item("group_middle", g.group_middle)?;
        f.set_item("left_out", g.left_out)?;
        d.set_item("groups", &g.groups)?;
        d.set_item("query_middle", &g.query_middle)?;
    }
    d.set_item("farfield", f)?;
    if let Some(refit) = r.refit {
        d.set_item("refit", refit.name())?;
    }
    if let Some(off) = r.refit_px {
        d.set_item("refit_px", off)?;
    }
    Ok(d)
}

/// Register the far-field binding on the `sfmtool.bench` submodule.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(far_field_sweep, m)?)?;
    Ok(())
}
