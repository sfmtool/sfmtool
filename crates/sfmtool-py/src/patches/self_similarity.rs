// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `zncc_self_similarity_parts`: the ZNCC self-similarity radius of one tile,
//! whole, over its middle and over each cell of the ZNCC grid's split.

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::patch::self_similarity::{
    zncc_self_similarity_parts as core_parts, PatchTile, SelfSimilarityParams,
};

/// The ZNCC self-similarity radius of a tile's ``R x R`` core: how far, in
/// tile pixels, the core can slide over itself by whole pixels and still
/// match itself as well as a true match between two views would (see
/// ``specs/core/patch/zncc-self-similarity-radius.md``).
///
/// For every shift ``d`` in the disk ``dx^2 + dy^2 <= r^2``, the core is
/// compared with the window of the same size moved by ``d`` by the per-channel
/// ZNCC averaged over the core's textured channels. A shift is
/// indistinguishable when ``1 - z(d) <= relative_tolerance + mean_c (noise /
/// s_c)^2``, with ``s_c`` the core's own spread in channel ``c``. The radius is
/// the length of the furthest indistinguishable shift, or ``r`` when one lies
/// in the disk's outer ring, read as "``r`` or more".
///
/// Args:
///     tile: An ``(R + 2r, R + 2r)`` single-channel tile or an ``(R + 2r, R +
///         2r, C)`` patch, uint8 or float32, in grey levels. A fourth channel is
///         read as alpha and dropped.
///     resolution: ``R``, the side of the core centred in the tile.
///     max_radius: ``r``, the length of the largest shift searched, in tile
///         pixels.
///     relative_tolerance: The ZNCC deficit two views of the same surface
///         show from warp, blur and lighting, as a fraction.
///     noise: The noise between two views, in grey levels.
///
/// Returns a dict: ``radius`` (float, the whole core), ``radius_middle``
/// (float, the middle square, rows and columns ``R/4 .. R - R/4``),
/// ``radius_grid`` (``(3, 3)`` float64, each cell of the split at ``R/3`` and
/// ``R - R/3``, from the top-left), ``slide`` (``(2,)`` float64, the direction
/// the whole core's indistinguishable shifts line up in, ``[x, y]`` with ``x``
/// column-right and ``y`` row-down, scaled by how strongly), ``slide_grid``
/// (``(3, 3, 2)`` float64, the same per cell), ``tolerance`` (float, the whole
/// core's tolerance; infinite for a core with no texture) and ``surface``
/// (``(2r + 1, 2r + 1)`` float64, the whole core's ZNCC at every shift from
/// ``(dx, dy) = (-r, -r)``, 1 at the centre and NaN outside the disk).
///
/// Raises:
///     ValueError: If the tile is not a 2-D or 3-D uint8 or float32 array, is
///         not ``R + 2r`` square, or has more than four channels.
#[pyfunction]
#[pyo3(signature = (tile, resolution, *, max_radius=3, relative_tolerance=0.05, noise=2.0))]
pub fn zncc_self_similarity_parts<'py>(
    py: Python<'py>,
    tile: &Bound<'py, PyAny>,
    resolution: usize,
    max_radius: u32,
    relative_tolerance: f64,
    noise: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (values, shape) = if let Ok(array) = tile.extract::<PyReadonlyArrayDyn<'py, u8>>() {
        let shape = array.shape().to_vec();
        let values: Vec<f32> = array.as_array().iter().map(|&v| f32::from(v)).collect();
        (values, shape)
    } else if let Ok(array) = tile.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        let shape = array.shape().to_vec();
        let values: Vec<f32> = array.as_array().iter().copied().collect();
        (values, shape)
    } else {
        return Err(PyValueError::new_err(
            "tile must be a uint8 or float32 numpy array",
        ));
    };
    let (height, width, channels) = match shape.as_slice() {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        _ => {
            return Err(PyValueError::new_err(format!(
                "tile must be (H, W) or (H, W, C), got shape {shape:?}"
            )))
        }
    };
    if !(1..=4).contains(&channels) {
        return Err(PyValueError::new_err(format!(
            "tile must have 1 to 4 channels, got {channels}"
        )));
    }
    let side = resolution + 2 * max_radius as usize;
    if resolution < 3 || width != side || height != side {
        return Err(PyValueError::new_err(format!(
            "an R = {resolution} core with max_radius = {max_radius} needs a {side} x {side} \
             tile and R >= 3, got {height} x {width}"
        )));
    }
    let params = SelfSimilarityParams {
        max_radius,
        relative_tolerance,
        noise,
    };
    let parts = py.detach(|| {
        let (planes, colour) = PatchTile::planes_from_interleaved(&values, width, height, channels);
        let tile = PatchTile {
            values: &planes,
            channels: colour,
            width,
            height,
        };
        core_parts(&tile, resolution, &params)
    });

    let grid = |f: &dyn Fn(usize, usize) -> f64| Array2::from_shape_fn((3, 3), |(r, c)| f(r, c));
    let out = PyDict::new(py);
    out.set_item("radius", parts.whole.radius)?;
    out.set_item("radius_middle", parts.middle.radius)?;
    out.set_item(
        "radius_grid",
        grid(&|r, c| parts.grid[r][c].radius).into_pyarray(py),
    )?;
    out.set_item(
        "slide",
        Array1::from_vec(parts.whole.slide.to_vec()).into_pyarray(py),
    )?;
    out.set_item(
        "slide_grid",
        Array3::from_shape_fn((3, 3, 2), |(r, c, k)| parts.grid[r][c].slide[k]).into_pyarray(py),
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
