// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Blur matching: one tile's blur assessment (`assess_blur`), the width that
//! brings its semi-major axis to a length (`blur_sigma_to_reach`), the tile
//! blurred to that length (`blur_to_length`), and the pairwise ZNCC of a
//! track's views' tiles, each pair blur-matched (`blur_matched_zncc_matrix`).

use numpy::ndarray::{Array1, Array2, Array3, Array4, ArrayView2, ArrayView3};
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::patch::blur_matched::{
    assess_blur as assess_tile_blur, blur_to_length as blur_tile_to_length, read_tile_ellipse,
    BlurAssessment, BlurScratch, TilePlanes, GROWTH_PROBE_SIGMAS,
};
use sfmtool_core::patch::reference_view::blur_matched_pairs;
use sfmtool_core::progress::Progress;

use super::args::{parse_matching, parse_patch_window};

/// Check a tile's shape: square, at least 3 on a side, 1 to 4 channels.
fn check_tile(height: usize, width: usize, stride: usize) -> PyResult<()> {
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
    Ok(())
}

/// One interleaved `(R, R, C)` tile as colour planes: a sample carries data
/// where `valid` (if given) is true and its alpha (if any) is above 0. Two
/// channels are grey and alpha, four RGB and alpha.
fn tile_planes(tile: ArrayView3<'_, u8>, valid: Option<ArrayView2<'_, bool>>) -> TilePlanes {
    let &[side, _, stride] = tile.shape() else {
        unreachable!("a 3-D array has three dimensions");
    };
    let alpha = (stride == 2 || stride == 4).then_some(stride - 1);
    let samples: Vec<u8> = tile.iter().copied().collect();
    let data: Vec<bool> = (0..side * side)
        .map(|i| {
            let on = valid.as_ref().is_none_or(|v| v[[i / side, i % side]]);
            on && alpha.is_none_or(|a| samples[i * stride + a] > 0)
        })
        .collect();
    TilePlanes::from_interleaved(&samples, side, stride, &data)
}

/// One `(R, R, C)` tile and its optional `(R, R)` `valid` flags as colour
/// planes, the shapes checked.
fn one_tile(
    samples: &PyReadonlyArray3<'_, u8>,
    valid: Option<&PyReadonlyArray2<'_, bool>>,
) -> PyResult<TilePlanes> {
    let tile = samples.as_array();
    let &[height, width, stride] = tile.shape() else {
        unreachable!("a 3-D array has three dimensions");
    };
    check_tile(height, width, stride)?;
    let valid = valid.map(|v| v.as_array());
    if let Some(v) = &valid {
        if v.shape() != [width, width] {
            return Err(PyValueError::new_err(format!(
                "valid must be ({width}, {width}), got {:?}",
                v.shape()
            )));
        }
    }
    Ok(tile_planes(tile, valid))
}

/// A `(2, 2)` ellipse matrix, `None` where an entry is not finite.
fn ellipse_of(e: &PyReadonlyArray2<'_, f64>) -> PyResult<Option<[[f64; 2]; 2]>> {
    let e = e.as_array();
    if e.shape() != [2, 2] {
        return Err(PyValueError::new_err(format!(
            "ellipse must be (2, 2), got {:?}",
            e.shape()
        )));
    }
    let m = [[e[[0, 0]], e[[0, 1]]], [e[[1, 0]], e[[1, 1]]]];
    Ok(m.iter().flatten().all(|x| x.is_finite()).then_some(m))
}

/// The [`BlurAssessment`] in an `assess_blur` dict: its ``semi_axes`` and
/// ``growth``.
fn assessment_of(assessment: &Bound<'_, PyDict>) -> PyResult<BlurAssessment> {
    let item = |key: &str| {
        assessment
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("assessment has no {key:?}")))
    };
    let semi_axes: Vec<f64> = item("semi_axes")?.extract()?;
    let growth: Vec<Vec<f64>> = item("growth")?.extract()?;
    let probes = GROWTH_PROBE_SIGMAS.len();
    if semi_axes.len() != 2 || growth.len() != probes || growth.iter().any(|g| g.len() != 2) {
        return Err(PyValueError::new_err(format!(
            "assessment must hold semi_axes of 2 and growth of ({probes}, 2)"
        )));
    }
    let mut out = BlurAssessment {
        semi_axes: [semi_axes[0], semi_axes[1]],
        growth: [[f64::NAN; 2]; GROWTH_PROBE_SIGMAS.len()],
    };
    for (g, row) in out.growth.iter_mut().zip(&growth) {
        *g = [row[0], row[1]];
    }
    Ok(out)
}

/// A tile's blur assessment: how sharp the tile is, and how that changes
/// under round blur. The tile's ZNCC self-similarity ellipse gives its
/// semi-axes; the tile is then blurred by a round Gaussian of each of
/// ``probe_sigmas`` (0.4 and 1 grid px), and each blurred tile's ellipse read
/// the same way, over the samples with data, with the default parameters.
/// The assessment depends on the tile alone, so it is read once per tile,
/// whatever the tile is later compared with. See
/// ``specs/core/patch/blur-matched-zncc.md``.
///
/// Args:
///     samples: An ``(R, R, C)`` uint8 tile, as
///         ``OrientedPatch.render_view_tile`` renders it. One channel is grey,
///         two grey and alpha, three RGB and four RGB and alpha. A sample
///         whose alpha is 0 carries no data.
///     valid: An ``(R, R)`` bool array, ``True`` where a sample carries data,
///         as ``render_view_tile`` returns it. When ``None`` (default), alpha
///         alone says.
///     ellipse: The tile's self-similarity ellipse matrix, ``(2, 2)`` float64
///         in grid px squared. When ``None`` (default), it is read here. One
///         passed in should be read the same way, or the growth is set against
///         a different reading.
///
/// Returns a dict, or ``None`` where the tile's ellipse or a probe's cannot
/// be read: ``semi_axes`` (``(2,)``, the tile's [major, minor] in grid px),
/// ``growth`` (``(2, 2)``, one row of [major, minor] per probe),
/// ``probe_sigmas`` (``(2,)``) and ``ellipse_matrix`` (``(2, 2)``).
///
/// Raises:
///     TypeError: If ``samples`` is not a 3-D uint8 array or ``valid`` not a
///         2-D bool array.
///     ValueError: If ``samples`` is not a square tile at least 3 on a side
///         with 1 to 4 channels, or ``valid`` or ``ellipse`` has the wrong
///         shape.
#[pyfunction]
#[pyo3(signature = (samples, *, valid=None, ellipse=None))]
pub fn assess_blur<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray3<'py, u8>,
    valid: Option<PyReadonlyArray2<'py, bool>>,
    ellipse: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let tile = one_tile(&samples, valid.as_ref())?;
    let given = ellipse.as_ref().map(ellipse_of).transpose()?;
    let read = |values: &[f32]| read_tile_ellipse(values, tile.channels, tile.side, &tile.data);
    let Some(e) = (match ellipse {
        Some(_) => given.flatten(),
        None => read(&tile.values),
    }) else {
        return Ok(None);
    };
    let Some(a) = py.detach(|| assess_tile_blur(&tile, &e, read, &mut BlurScratch::default()))
    else {
        return Ok(None);
    };
    let out = PyDict::new(py);
    out.set_item(
        "semi_axes",
        Array1::from(a.semi_axes.to_vec()).into_pyarray(py),
    )?;
    out.set_item(
        "growth",
        Array2::from_shape_fn((a.growth.len(), 2), |(i, j)| a.growth[i][j]).into_pyarray(py),
    )?;
    out.set_item(
        "probe_sigmas",
        Array1::from(GROWTH_PROBE_SIGMAS.to_vec()).into_pyarray(py),
    )?;
    out.set_item(
        "ellipse_matrix",
        Array2::from_shape_fn((2, 2), |(r, c)| e[r][c]).into_pyarray(py),
    )?;
    Ok(Some(out))
}

/// The width, in grid px, of the round blur that brings a tile's
/// self-similarity semi-major axis to ``length``, read off the tile's
/// ``assess_blur`` without blurring it again: the square of the semi-major
/// axis against the square of the width is taken to be piecewise linear
/// through the tile's own reading and the probes', and to go on past the
/// widest probe along its last piece. At most 3.
///
/// Args:
///     assessment: A dict from ``assess_blur`` (its ``semi_axes`` and
///         ``growth`` are read).
///     length: The semi-major axis to reach, in grid px.
///
/// Returns the width, 0 where the semi-major axis is already that long, or
/// ``None`` where the readings do not grow.
///
/// Raises:
///     ValueError: If ``assessment`` lacks ``semi_axes`` or ``growth`` or
///         they have the wrong lengths, or ``length`` is negative or not
///         finite.
#[pyfunction]
pub fn blur_sigma_to_reach(assessment: &Bound<'_, PyDict>, length: f64) -> PyResult<Option<f64>> {
    check_length(length)?;
    Ok(assessment_of(assessment)?.sigma_to_reach(length))
}

/// Check a semi-major axis to reach: finite and not negative.
fn check_length(length: f64) -> PyResult<()> {
    if length.is_finite() && length >= 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "length must be a finite number of at least 0, not {length}"
        )))
    }
}

/// A tile blurred by a round Gaussian until its self-similarity semi-major
/// axis reaches ``length``, by the width ``blur_sigma_to_reach`` gives, over
/// the samples with data (a sample without data neither gives nor takes a
/// value). A tile whose semi-major axis is already that long comes back
/// unblurred, with a width of 0.
///
/// Args:
///     samples: An ``(R, R, C)`` uint8 tile, as for ``assess_blur``.
///     assessment: The tile's own ``assess_blur`` dict.
///     length: The semi-major axis to reach, in grid px.
///     valid: An ``(R, R)`` bool array, as for ``assess_blur``.
///
/// Returns a dict, or ``None`` where the assessment gives no width:
/// ``samples`` (``(R, R, c)`` float32, the colour channels blurred, alpha
/// left out: one for grey, three for RGB) and ``sigma`` (the width, grid px).
///
/// Raises:
///     TypeError: If ``samples`` or ``valid`` has the wrong type.
///     ValueError: If a shape is wrong, ``assessment`` cannot be read, or
///         ``length`` is negative or not finite.
#[pyfunction]
#[pyo3(signature = (samples, assessment, length, *, valid=None))]
pub fn blur_to_length<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray3<'py, u8>,
    assessment: &Bound<'py, PyDict>,
    length: f64,
    valid: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    check_length(length)?;
    let tile = one_tile(&samples, valid.as_ref())?;
    let a = assessment_of(assessment)?;
    let Some((blurred, sigma)) =
        py.detach(|| blur_tile_to_length(&tile, &a, length, &mut BlurScratch::default()))
    else {
        return Ok(None);
    };
    let (side, channels) = (blurred.side, blurred.channels);
    let n = side * side;
    let out = PyDict::new(py);
    out.set_item(
        "samples",
        Array3::from_shape_fn((side, side, channels), |(y, x, c)| {
            blurred.values[c * n + y * side + x]
        })
        .into_pyarray(py),
    )?;
    out.set_item("sigma", sigma)?;
    Ok(Some(out))
}

/// The ZNCC between every pair of a track's views' tiles, each pair
/// blur-matched: where one tile's ZNCC self-similarity semi-major axis is
/// shorter than the other's semi-minor axis, that tile is blurred by a round
/// Gaussian until its semi-major axis reaches the other's semi-minor axis
/// (at most 2 grid px) before the two are correlated; any other pair is
/// correlated as it is. Each tile some pair blurs is assessed once
/// (``assess_blur``), and every pair it is the sharper of blurs it by the
/// width its assessment gives (``blur_to_length``). See
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
/// ``min_ellipse_ratio`` of 1.25 and the default window.
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
///     min_ellipse_ratio: How many times the sharper tile's semi-major axis
///         the other tile's semi-minor axis must at least be for
///         ``"blur_matched_above_ratio"`` to blur the pair (default 1.25), at
///         least 1.
///     window: The whole-tile reading's window, ``"gaussian_disk"``
///         (default), ``"gaussian"`` or ``"uniform"``.
///     window_sigma: Its sigma.
///
/// Returns a dict: ``zncc`` (``(k, k)`` float64, unit diagonal, NaN where a
/// pair could not be read), ``zncc_grid`` (``(k, k, 3, 3)`` float64, NaN on
/// the diagonal), ``blurred`` (``(k, k)`` bool, whether a tile of the pair
/// was blurred), ``blur_sigma`` (``(k, k)`` float64, the width in grid px by
/// which the row's tile was blurred against the column's, 0 where it was not
/// blurred), ``pairs`` and ``pairs_blurred`` (int), and ``ellipse_matrix``
/// (``(k, 2, 2)``, the ellipses read).
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
    window="gaussian_disk", window_sigma=0.6
))]
#[allow(clippy::too_many_arguments)]
pub fn blur_matched_zncc_matrix<'py>(
    py: Python<'py>,
    tiles: PyReadonlyArray4<'py, u8>,
    valid: Option<PyReadonlyArray3<'py, bool>>,
    ellipses: Option<PyReadonlyArray3<'py, f64>>,
    matching: &str,
    min_ellipse_ratio: f64,
    window: &str,
    window_sigma: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let matching = parse_matching(matching, min_ellipse_ratio)?;
    let window = parse_patch_window(window, window_sigma)?;
    let tiles = tiles.as_array();
    let &[k, height, width, stride] = tiles.shape() else {
        unreachable!("a 4-D array has four dimensions");
    };
    check_tile(height, width, stride)?;
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
    let axis = numpy::ndarray::Axis(0);
    let planes: Vec<TilePlanes> = (0..k)
        .map(|v| {
            tile_planes(
                tiles.index_axis(axis, v),
                valid.as_ref().map(|valid| valid.index_axis(axis, v)),
            )
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
        blur_matched_pairs(&refs, &ellipses, matching, window, &Progress::none())
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
    out.set_item(
        "blur_sigma",
        Array2::from_shape_vec((k, k), pairs.sigma.clone())
            .expect("k*k widths")
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
