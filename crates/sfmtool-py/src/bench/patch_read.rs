// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for reading a pixel's patch along its ray.

use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::bench::{read_patch_along_ray as core_read_patch_along_ray, RayPatch};

use super::views_of;
use crate::patches::views::{resolve_grey, resolve_pyramids, PosedViews};
use crate::reconstruction::edited::PyEditedReconstruction;

/// Read ``pixel``'s patch in ``image`` at each of ``distances`` along its
/// ray, in each of ``read_images``.
///
/// The patch is an 11 x 11 grid of ``radius_px`` around the pixel, sampled in
/// grey and blurred. Each grid sample's ray is cut by the plane facing the
/// queried camera at the distance, and the point sampled in the other image;
/// at ``math.inf`` the rays are projected as directions. Each read gives the
/// ZNCC of the whole grid and of its middle 5 x 5 from the same samples. A read
/// is skipped, and scores ``-1``, where the patch does not land inside the
/// other photograph.
///
/// Args:
///     edited: The reconstruction whose cameras and poses are read.
///     images: One decoded image per image of ``edited``, or a prebuilt
///         :class:`ImagePyramidSet`, which also keeps the grey images.
///     image: The queried image's index.
///     pixel: ``(x, y)`` in that image.
///     radius_px: From the pixel to the grid's edge, in px.
///     distances: Distances along the pixel's unit ray from the queried
///         camera's centre; ``math.inf`` for infinity.
///     read_images: The images to read in; every image but ``image`` when
///         left out.
///     samples: Also return the samples.
///
/// Returns:
///     ``None`` when the queried patch is flat or runs off its photograph;
///     otherwise a dict with ``images``, ``distances``, ``whole`` and
///     ``middle`` (``(distances, images)`` float64 arrays), ``centres``
///     (``(distances, images, 2)``, NaN where unread), ``middle_std`` and, with
///     ``samples``, ``template``, ``values`` (``(distances, images, 121)``) and
///     ``middle_mask``.
///
/// Raises:
///     ValueError: an image is not one of the reconstruction's, or ``images``
///         does not match it.
#[pyfunction]
#[pyo3(signature = (edited, images, image, pixel, radius_px, distances, read_images = None, *, samples = false))]
#[allow(clippy::too_many_arguments)]
pub(super) fn read_patch_along_ray(
    py: Python<'_>,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    image: u32,
    pixel: [f64; 2],
    radius_px: f64,
    distances: Vec<f64>,
    read_images: Option<Vec<u32>>,
    samples: bool,
) -> PyResult<Option<Py<PyDict>>> {
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let grey = resolve_grey(images, posed.cameras.len());
    let views = views_of(&posed, &pyramids);
    let count = views.len() as u32;
    let read_images = read_images.unwrap_or_else(|| (0..count).filter(|&i| i != image).collect());
    if let Some(&bad) = std::iter::once(&image)
        .chain(&read_images)
        .find(|&&i| i >= count)
    {
        return Err(PyValueError::new_err(format!(
            "image {bad} is not one of the reconstruction's {count} images"
        )));
    }
    let patch = RayPatch {
        image,
        pixel,
        radius_px,
    };
    let Some(read) = py.detach(|| {
        core_read_patch_along_ray(&views, &grey, &patch, &distances, &read_images, samples)
    }) else {
        return Ok(None);
    };
    let d = PyDict::new(py);
    d.set_item("images", &read.images)?;
    d.set_item("distances", &read.distances)?;
    d.set_item("whole", read.whole.into_pyarray(py))?;
    d.set_item("middle", read.middle.into_pyarray(py))?;
    d.set_item("centres", read.centres.into_pyarray(py))?;
    d.set_item("middle_std", read.middle_std)?;
    if let Some(s) = read.samples {
        d.set_item("template", s.template.into_pyarray(py))?;
        d.set_item("values", s.values.into_pyarray(py))?;
        d.set_item("middle_mask", s.middle.into_pyarray(py))?;
    }
    Ok(Some(d.unbind()))
}

/// Register the patch-read binding on the `sfmtool.bench` submodule.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_patch_along_ray, m)?)?;
    Ok(())
}
