// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `blur_matched_zncc_matrix`: the pairwise ZNCC of a track's views' tiles,
//! each pair blur-matched.

use numpy::ndarray::{Array2, Array3, Array4};
use numpy::{IntoPyArray, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::patch::blur_matched::{
    blur_matched_pairs, read_tile_ellipse, BlurMatchKernel, TilePlanes,
};
use sfmtool_core::progress::Progress;

use super::args::{parse_matching, parse_patch_window};

/// The ZNCC between every pair of a track's views' tiles, each pair
/// blur-matched: along each direction in which the two tiles' ZNCC
/// self-similarity ellipses differ, the tile whose ellipse is shorter is
/// blurred to the other's sharpness before the two are correlated. The width
/// of each blur is found by blurring the tile, reading its ellipse again, and
/// correcting the width until the two lengths agree (up to 2 grid px). See
/// ``specs/core/patch/blur-matched-zncc.md``.
///
/// The ellipses compared are whole-tile readings over the samples with data,
/// with the default parameters, which is how the blurred tiles are read; pass
/// ``ellipses`` read another way and the blur is set against a different
/// reading.
///
/// Each pair is read over the samples with data in both tiles: over the whole
/// tile with the window, and over each cell of the ZNCC grid's three-by-three
/// split with every sample weighted equally, per colour channel, the channels
/// averaged.
///
/// The bench's reference-view rule reads its blur-matched readings this way,
/// from the tiles ``OrientedPatch.render_view_tile`` renders of the track's
/// frame at each view's keypoint, at the evaluation's resolution and with its
/// sampler. To read the same numbers from those tiles, stack each view's
/// ``samples`` as ``tiles`` and its ``valid`` as ``valid``, leave
/// ``ellipses`` out, and pass
/// ``matching="blur_matched_above_ratio"`` with the default
/// ``min_ellipse_ratio`` of 1.25, the default kernel and the default window.
/// The ellipses read here are then the bench's, its track-stage
/// ``zncc_self_similarity_ellipse["grid_px"]["matrix"]``, up to rounding.
///
/// Args:
///     tiles: A ``(k, R, R, C)`` uint8 stack, one view's tile each, as
///         ``OrientedPatch.render_view_tile`` renders them. One channel is
///         grey, two grey and alpha, three RGB and four RGB and alpha. Alpha
///         is not correlated: a sample whose alpha is 0 carries no data.
///     valid: A ``(k, R, R)`` bool stack, ``True`` where a sample carries
///         data, as ``render_view_tile`` returns it in ``valid``; a sample
///         carries data where it is ``True`` and its alpha, if any, is above
///         0. When ``None`` (default), alpha alone says.
///     ellipses: Each tile's self-similarity ellipse matrix, ``(k, 2, 2)``
///         float64 in grid px squared (``ellipse_matrix`` of
///         ``zncc_self_similarity_parts``), NaN for a tile without one. When
///         ``None`` (default), each tile's whole reading is taken here with the
///         default parameters, over the samples that carry data.
///     matching: ``"blur_matched"`` (default), ``"blur_matched_above_ratio"``
///         or ``"plain"``.
///     min_ellipse_ratio: The factor two lengths along a direction must differ
///         by for ``"blur_matched_above_ratio"`` to blur along it (default
///         1.25), at least 1.
///     kernel: ``"anisotropic"`` (default), each tile blurred along the
///         directions in which it is the sharper, or ``"isotropic_ladder"``,
///         the sharper tile by semi-major axis blurred isotropically at
///         whichever of eight widths brings its semi-major axis closest to the
///         other tile's, each view blurred and read once per width.
///     window: The whole-tile reading's window, ``"gaussian_disk"``
///         (default), ``"gaussian"`` or ``"uniform"``.
///     window_sigma: Its sigma.
///
/// Returns a dict: ``zncc`` (``(k, k)`` float64, unit diagonal, NaN where a
/// pair could not be read), ``zncc_grid`` (``(k, k, 3, 3)`` float64, NaN on
/// the diagonal), ``blurred`` (``(k, k)`` bool, whether either tile of the
/// pair was blurred), ``pairs`` and ``pairs_blurred`` (int), and
/// ``ellipse_matrix`` (``(k, 2, 2)``, the ellipses read).
///
/// Raises:
///     TypeError: If ``tiles`` is not a 4-D uint8 array, or ``valid`` is not a
///         3-D bool array.
///     ValueError: If ``tiles`` is not a stack of square tiles at least 3 on a
///         side with 1 to 4 channels, ``valid`` is not ``(k, R, R)``,
///         ``ellipses`` is not ``(k, 2, 2)``, ``min_ellipse_ratio`` is under 1
///         or not finite, or a name is unknown.
#[pyfunction]
#[pyo3(signature = (
    tiles, *, valid=None, ellipses=None, matching="blur_matched", min_ellipse_ratio=1.25,
    kernel="anisotropic", window="gaussian_disk", window_sigma=0.6
))]
#[allow(clippy::too_many_arguments)]
pub fn blur_matched_zncc_matrix<'py>(
    py: Python<'py>,
    tiles: PyReadonlyArray4<'py, u8>,
    valid: Option<PyReadonlyArray3<'py, bool>>,
    ellipses: Option<PyReadonlyArray3<'py, f64>>,
    matching: &str,
    min_ellipse_ratio: f64,
    kernel: &str,
    window: &str,
    window_sigma: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let matching = parse_matching(matching, min_ellipse_ratio)?;
    let kernel = BlurMatchKernel::from_name(kernel).ok_or_else(|| {
        PyValueError::new_err(format!(
            "kernel must be \"anisotropic\" or \"isotropic_ladder\", not {kernel:?}"
        ))
    })?;
    let window = parse_patch_window(window, window_sigma)?;
    let tiles = tiles.as_array();
    let &[k, height, width, stride] = tiles.shape() else {
        unreachable!("a 4-D array has four dimensions");
    };
    if height != width || width < 3 {
        return Err(PyValueError::new_err(format!(
            "tiles must be square and at least 3 x 3, got {height} x {width}"
        )));
    }
    if !(1..=4).contains(&stride) {
        return Err(PyValueError::new_err(format!(
            "tiles must have 1 to 4 channels, got {stride}"
        )));
    }
    let side = width;
    let valid = valid.as_ref().map(|v| v.as_array());
    if let Some(v) = &valid {
        if v.shape() != [k, side, side] {
            return Err(PyValueError::new_err(format!(
                "valid must be ({k}, {side}, {side}), got {:?}",
                v.shape()
            )));
        }
    }
    // Two channels are grey and alpha, four RGB and alpha.
    let alpha = (stride == 2 || stride == 4).then_some(stride - 1);
    let planes: Vec<TilePlanes> = (0..k)
        .map(|v| {
            let tile = tiles.index_axis(numpy::ndarray::Axis(0), v);
            let samples: Vec<u8> = tile.iter().copied().collect();
            let data: Vec<bool> = (0..side * side)
                .map(|i| {
                    let on = valid
                        .as_ref()
                        .is_none_or(|valid| valid[[v, i / side, i % side]]);
                    on && alpha.is_none_or(|a| samples[i * stride + a] > 0)
                })
                .collect();
            TilePlanes::from_interleaved(&samples, side, stride, &data)
        })
        .collect();
    let ellipses: Vec<Option<[[f64; 2]; 2]>> = match ellipses {
        Some(e) => {
            let e = e.as_array();
            if e.shape() != [k, 2, 2] {
                return Err(PyValueError::new_err(format!(
                    "ellipses must be ({k}, 2, 2), got {:?}",
                    e.shape()
                )));
            }
            (0..k)
                .map(|v| {
                    let m = [[e[[v, 0, 0]], e[[v, 0, 1]]], [e[[v, 1, 0]], e[[v, 1, 1]]]];
                    m.iter().flatten().all(|x| x.is_finite()).then_some(m)
                })
                .collect()
        }
        None => planes
            .iter()
            .map(|t| read_tile_ellipse(&t.values, t.channels, side, &t.data))
            .collect(),
    };
    let pairs = py.detach(|| {
        let refs: Vec<&TilePlanes> = planes.iter().collect();
        blur_matched_pairs(
            &refs,
            &ellipses,
            matching,
            kernel,
            window,
            None,
            &Progress::none(),
        )
    });
    let out = PyDict::new(py);
    out.set_item(
        "zncc",
        Array2::from_shape_vec((k, k), pairs.whole.clone())
            .expect("k*k readings")
            .into_pyarray(py),
    )?;
    out.set_item(
        "zncc_grid",
        Array4::from_shape_fn((k, k, 3, 3), |(a, b, r, c)| pairs.grid[a * k + b][r][c])
            .into_pyarray(py),
    )?;
    out.set_item(
        "blurred",
        Array2::from_shape_vec((k, k), pairs.blurred.clone())
            .expect("k*k flags")
            .into_pyarray(py),
    )?;
    out.set_item("pairs", pairs.pairs)?;
    out.set_item("pairs_blurred", pairs.pairs_blurred)?;
    out.set_item(
        "ellipse_matrix",
        Array3::from_shape_fn((k, 2, 2), |(v, r, c)| {
            ellipses[v].map_or(f64::NAN, |m| m[r][c])
        })
        .into_pyarray(py),
    )?;
    Ok(out)
}
