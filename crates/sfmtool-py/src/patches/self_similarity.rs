// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `zncc_self_similarity_parts`: the ZNCC self-similarity radius of
//! one bitmap, whole, over its middle and over each cell of the ZNCC grid's
//! split, read the overlap way; and `zncc_self_similarity_parts_stack`:
//! the same reading of each bitmap of a stack.

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use sfmtool_core::patch::self_similarity::{
    zncc_self_similarity_parts as core_parts, PatchTile, SelfSimilarityParams, SelfSimilarityParts,
};

/// The ZNCC self-similarity radius of one ``R x R`` bitmap: how far, in
/// bitmap pixels, it can slide over itself by whole pixels and still match
/// itself as well as a true match between two views would (see
/// ``specs/core/patch/zncc-self-similarity-radius.md``).
///
/// The bitmap is read as it is, the overlap way, with no pixels from outside
/// it: for every shift ``d`` of the square ``|dx|, |dy| <= r``, the bitmap is
/// compared with itself moved by ``d`` over only the samples both windows
/// hold, and carry data on both sides, by the per-channel ZNCC averaged over
/// the bitmap's textured channels. A shift is indistinguishable when ``1 -
/// z(d) <= relative_tolerance + mean_c (noise / s_c)^2``, with ``s_c`` the
/// bitmap's own spread in channel ``c``. The region of those shifts, the ZNCC
/// taken as bilinear between whole-pixel shifts, is summarised by the ellipse
/// with the same second moments per unit area about the true position; the radius
/// is its semi-major axis, capped at ``r``, which reads as "``r`` or more".
///
/// Args:
///     bitmap: An ``(R, R)`` single-channel bitmap or an ``(R, R, C)`` patch,
///         uint8 or float32, in grey levels. A fourth channel is read as
///         alpha: a sample whose alpha is 0 carries no data. Without one, every
///         sample carries data.
///     max_radius: ``r``, the length of the largest shift searched, in bitmap
///         pixels.
///     relative_tolerance: The ZNCC deficit two views of the same surface
///         show from warp, blur and lighting, as a fraction.
///     noise: The noise between two views, in grey levels.
///
/// Returns a dict: ``radius`` (float, the whole bitmap), ``radius_middle``
/// (float, the middle square, rows and columns ``R/4 .. R - R/4``),
/// ``radius_grid`` (``(3, 3)`` float64, each cell of the split at ``R/3`` and
/// ``R - R/3``, from the top-left), ``radius_is_at_least`` (bool, whether the
/// whole bitmap's true radius may be larger); the whole bitmap's ellipse as
/// ``ellipse_axes`` (``(2,)`` float64, ``[semi-major, semi-minor]`` in bitmap
/// pixels, each capped at ``r``), ``ellipse_axes_is_at_least`` (``(2,)`` bool,
/// per axis whether the true length may be larger), ``ellipse_major_angle``
/// (float, the major axis's angle in radians in ``[0, pi)`` from ``x``
/// column-right towards ``y`` row-down, NaN for a circle) and
/// ``ellipse_matrix`` (``(2, 2)`` float64, ``E`` with ``d^T E^-1 d = 1`` on the
/// ellipse); the middle square's as ``ellipse_axes_middle``,
/// ``ellipse_axes_is_at_least_middle`` and ``ellipse_major_angle_middle``, and
/// each cell's as ``ellipse_axes_grid`` (``(3, 3, 2)``),
/// ``ellipse_axes_is_at_least_grid`` (``(3, 3, 2)`` bool) and
/// ``ellipse_major_angle_grid`` (``(3, 3)``); ``tolerance`` (float, the whole
/// bitmap's tolerance; infinite for a bitmap with no texture) and ``surface``
/// (``(2r + 1, 2r + 1)`` float64, the whole bitmap's ZNCC at every shift from
/// ``(dx, dy) = (-r, -r)``, 1 at the centre). A bitmap with no sample carrying
/// data has no reading, and its values are NaN and its flags false.
///
/// Raises:
///     ValueError: If the bitmap is not a 2-D or 3-D uint8 or float32 array, is
///         not square and at least 3 x 3, or has more than four channels.
#[pyfunction]
#[pyo3(signature = (bitmap, *, max_radius=3, relative_tolerance=0.05, noise=2.0))]
pub fn zncc_self_similarity_parts<'py>(
    py: Python<'py>,
    bitmap: &Bound<'py, PyAny>,
    max_radius: u32,
    relative_tolerance: f64,
    noise: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (values, shape) = if let Ok(array) = bitmap.extract::<PyReadonlyArrayDyn<'py, u8>>() {
        let shape = array.shape().to_vec();
        let values: Vec<f32> = array.as_array().iter().map(|&v| f32::from(v)).collect();
        (values, shape)
    } else if let Ok(array) = bitmap.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        let shape = array.shape().to_vec();
        let values: Vec<f32> = array.as_array().iter().copied().collect();
        (values, shape)
    } else {
        return Err(PyValueError::new_err(
            "bitmap must be a uint8 or float32 numpy array",
        ));
    };
    let (height, width, channels) = match shape.as_slice() {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        _ => {
            return Err(PyValueError::new_err(format!(
                "bitmap must be (R, R) or (R, R, C), got shape {shape:?}"
            )))
        }
    };
    if !(1..=4).contains(&channels) {
        return Err(PyValueError::new_err(format!(
            "bitmap must have 1 to 4 channels, got {channels}"
        )));
    }
    if height != width || width < 3 {
        return Err(PyValueError::new_err(format!(
            "bitmap must be square and at least 3 x 3, got {height} x {width}"
        )));
    }
    let params = SelfSimilarityParams {
        max_radius,
        relative_tolerance,
        noise,
    };
    let parts = py.detach(|| {
        let (planes, colour) = PatchTile::planes_from_interleaved(&values, width, height, channels);
        let data = PatchTile::data_from_interleaved(&values, width, height, channels);
        let tile = PatchTile {
            values: &planes,
            channels: colour,
            width,
            height,
        };
        core_parts(&tile, data.as_deref(), &params)
    });
    let grid = |f: &dyn Fn(usize, usize) -> f64| Array2::from_shape_fn((3, 3), |(r, c)| f(r, c));
    let out = PyDict::new(py);
    out.set_item("radius", parts.whole.radius)?;
    out.set_item("radius_middle", parts.middle.radius)?;
    out.set_item(
        "radius_grid",
        grid(&|r, c| parts.grid[r][c].radius).into_pyarray(py),
    )?;
    out.set_item("radius_is_at_least", parts.whole.radius_is_at_least())?;
    for (suffix, part) in [("", &parts.whole), ("_middle", &parts.middle)] {
        let e = &part.ellipse;
        out.set_item(
            format!("ellipse_axes{suffix}"),
            Array1::from_vec(e.axes.to_vec()).into_pyarray(py),
        )?;
        out.set_item(
            format!("ellipse_axes_is_at_least{suffix}"),
            Array1::from_vec(e.axes_is_at_least.to_vec()).into_pyarray(py),
        )?;
        out.set_item(format!("ellipse_major_angle{suffix}"), e.major_angle)?;
    }
    out.set_item(
        "ellipse_matrix",
        Array2::from_shape_fn((2, 2), |(r, c)| parts.whole.ellipse.matrix[r][c]).into_pyarray(py),
    )?;
    out.set_item(
        "ellipse_axes_grid",
        Array3::from_shape_fn((3, 3, 2), |(r, c, k)| parts.grid[r][c].ellipse.axes[k])
            .into_pyarray(py),
    )?;
    out.set_item(
        "ellipse_axes_is_at_least_grid",
        Array3::from_shape_fn((3, 3, 2), |(r, c, k)| {
            parts.grid[r][c].ellipse.axes_is_at_least[k]
        })
        .into_pyarray(py),
    )?;
    out.set_item(
        "ellipse_major_angle_grid",
        grid(&|r, c| parts.grid[r][c].ellipse.major_angle).into_pyarray(py),
    )?;
    out.set_item("tolerance", parts.whole.tolerance)?;
    let n = 2 * max_radius as usize + 1;
    out.set_item(
        "surface",
        Array2::from_shape_vec((n, n), parts.whole.surface)
            .expect("the surface holds (2r + 1)^2 values")
            .into_pyarray(py),
    )?;
    Ok(out)
}

/// The ZNCC self-similarity radius of each bitmap of a stack, read the overlap
/// way, as it is, with no pixels from outside it: at each shift only
/// the samples that lie inside the bitmap on both sides, and carry data on both
/// sides, are correlated, with the template's and the moved window's mean and
/// spread taken over that overlap (see
/// ``specs/core/patch/zncc-self-similarity-radius.md``, "The overlap reading").
/// The middle square and the cells take their shifted windows from the rest of
/// the bitmap where it reaches. No source images are read.
///
/// Args:
///     bitmaps: An ``(N, R, R, C)`` uint8 or float32 stack, in grey levels,
///         such as ``recon.patch_bitmaps``. A fourth channel is read as alpha:
///         a sample whose alpha is 0 carries no data. Without one, every sample
///         carries data.
///     max_radius: ``r``, the length of the largest shift searched, in bitmap
///         pixels.
///     relative_tolerance: The ZNCC deficit two views of the same surface
///         show from warp, blur and lighting, as a fraction.
///     noise: The noise between two views, in grey levels.
///
/// Returns a dict of per-bitmap numpy arrays: ``radius`` (``(N,)`` float64,
/// the whole bitmap), ``radius_middle`` (``(N,)``, rows and columns ``R/4 ..
/// R - R/4``), ``radius_grid`` (``(N, 3, 3)``, each cell of the split at
/// ``R/3`` and ``R - R/3``), ``radius_is_at_least`` (``(N,)`` bool, whether
/// the whole bitmap's true radius may be larger), the whole bitmap's ellipse
/// as ``ellipse_axes`` (``(N, 2)``, ``[semi-major, semi-minor]``),
/// ``ellipse_axes_is_at_least`` (``(N, 2)`` bool) and ``ellipse_major_angle``
/// (``(N,)``, radians in ``[0, pi)`` from ``x`` column-right towards ``y``
/// row-down, NaN for a circle), ``tolerance`` (``(N,)``, the whole bitmap's;
/// infinite for a bitmap with no texture) and ``covered`` (``(N,)`` bool,
/// whether any sample carries data). A bitmap with no sample carrying data has
/// no reading: ``covered`` is false, its values are NaN and its flags false.
///
/// Raises:
///     ValueError: If the stack is not a 4-D uint8 or float32 array of square
///         bitmaps at least 3 on a side, with 1 to 4 channels.
#[pyfunction]
#[pyo3(signature = (bitmaps, *, max_radius=3, relative_tolerance=0.05, noise=2.0))]
pub fn zncc_self_similarity_parts_stack<'py>(
    py: Python<'py>,
    bitmaps: &Bound<'py, PyAny>,
    max_radius: u32,
    relative_tolerance: f64,
    noise: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (values, shape) = if let Ok(array) = bitmaps.extract::<PyReadonlyArrayDyn<'py, u8>>() {
        let shape = array.shape().to_vec();
        let values: Vec<f32> = array.as_array().iter().map(|&v| f32::from(v)).collect();
        (values, shape)
    } else if let Ok(array) = bitmaps.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        let shape = array.shape().to_vec();
        let values: Vec<f32> = array.as_array().iter().copied().collect();
        (values, shape)
    } else {
        return Err(PyValueError::new_err(
            "bitmaps must be a uint8 or float32 numpy array",
        ));
    };
    let [n, height, width, channels] = shape.as_slice() else {
        return Err(PyValueError::new_err(format!(
            "bitmaps must be (N, R, R, C), got shape {shape:?}"
        )));
    };
    let (n, height, width, channels) = (*n, *height, *width, *channels);
    if height != width || width < 3 {
        return Err(PyValueError::new_err(format!(
            "bitmaps must be square and at least 3 x 3, got {height} x {width}"
        )));
    }
    if !(1..=4).contains(&channels) {
        return Err(PyValueError::new_err(format!(
            "bitmaps must have 1 to 4 channels, got {channels}"
        )));
    }
    let params = SelfSimilarityParams {
        max_radius,
        relative_tolerance,
        noise,
    };
    let per = width * height * channels;
    let parts: Vec<SelfSimilarityParts> = py.detach(|| {
        values
            .par_chunks(per.max(1))
            .take(n)
            .map(|bitmap| {
                let (planes, colour) =
                    PatchTile::planes_from_interleaved(bitmap, width, height, channels);
                let data = PatchTile::data_from_interleaved(bitmap, width, height, channels);
                let tile = PatchTile {
                    values: &planes,
                    channels: colour,
                    width,
                    height,
                };
                core_parts(&tile, data.as_deref(), &params)
            })
            .collect()
    });

    let out = PyDict::new(py);
    let radius: Vec<f64> = parts.iter().map(|p| p.whole.radius).collect();
    let covered: Vec<bool> = parts.iter().map(|p| !p.whole.tolerance.is_nan()).collect();
    out.set_item("radius", radius.into_pyarray(py))?;
    out.set_item(
        "radius_middle",
        parts
            .iter()
            .map(|p| p.middle.radius)
            .collect::<Vec<f64>>()
            .into_pyarray(py),
    )?;
    out.set_item(
        "radius_grid",
        Array3::from_shape_fn((n, 3, 3), |(i, r, c)| parts[i].grid[r][c].radius).into_pyarray(py),
    )?;
    out.set_item(
        "radius_is_at_least",
        parts
            .iter()
            .map(|p| p.whole.radius_is_at_least())
            .collect::<Vec<bool>>()
            .into_pyarray(py),
    )?;
    out.set_item(
        "ellipse_axes",
        Array2::from_shape_fn((n, 2), |(i, k)| parts[i].whole.ellipse.axes[k]).into_pyarray(py),
    )?;
    out.set_item(
        "ellipse_axes_is_at_least",
        Array2::from_shape_fn((n, 2), |(i, k)| parts[i].whole.ellipse.axes_is_at_least[k])
            .into_pyarray(py),
    )?;
    out.set_item(
        "ellipse_major_angle",
        parts
            .iter()
            .map(|p| p.whole.ellipse.major_angle)
            .collect::<Vec<f64>>()
            .into_pyarray(py),
    )?;
    out.set_item(
        "tolerance",
        parts
            .iter()
            .map(|p| p.whole.tolerance)
            .collect::<Vec<f64>>()
            .into_pyarray(py),
    )?;
    out.set_item("covered", covered.into_pyarray(py))?;
    Ok(out)
}
