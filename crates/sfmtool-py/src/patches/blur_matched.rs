// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Blur matching: one tile's blur assessment (`assess_blur`), the width that
//! brings its semi-major axis to a length (`blur_sigma_to_reach`), the tile
//! blurred to that length (`blur_to_length`), and each observation's score
//! against a point's stored bitmap, plain and blur-matched
//! (`score_against_bitmap`).

use numpy::ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::patch::blur_matched::{
    assess_blur as assess_tile_blur, blur_to_length as blur_tile_to_length, read_tile_ellipse,
    BlurAssessment, BlurScratch, TilePlanes, GROWTH_PROBE_SIGMAS,
};
use sfmtool_core::patch::stored_bitmap::{BitmapScore, BitmapScorer};

use super::args::parse_patch_window;

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

/// Each observation's ZNCC with a point's stored bitmap, plain and
/// blur-matched, blurring only the bitmap. See
/// ``specs/core/patch/blur-matched-zncc.md`` § "Scores against the stored
/// bitmap".
///
/// Each pair is read over the samples with data in both tiles, weighted by the
/// window, per colour channel, the channels averaged. Where the observation's
/// ZNCC self-similarity semi-minor axis, capped at 2 grid px, is at least 1.25
/// times the bitmap's semi-major axis, the bitmap is blurred by a round
/// Gaussian until its semi-major axis reaches that capped length, by the width
/// the bitmap's own blur assessment gives, and correlated again. Any other pair, an observation sharper than the bitmap
/// among them, is read plain. The bitmap's assessment is read at most once.
/// The observation ``reference`` names, whose tile the bitmap is, is not
/// computed: its scores read 1.
///
/// The bench scores every row of a track this way
/// (``EditableTrack.observations``' ``plain_zncc`` and
/// ``blur_matched_zncc``, with their middle and grid readings), from the tiles
/// ``OrientedPatch.render_view_tile`` renders at the evaluation's resolution
/// and with its sampler.
///
/// Args:
///     bitmap: The ``(R, R, 4)`` uint8 stored bitmap; a sample whose alpha is
///         0 carries no data.
///     tiles: A ``(k, R, R, C)`` uint8 stack of the observations' tiles. One
///         channel is grey, two grey and alpha, three RGB and four RGB and
///         alpha.
///     valid: A ``(k, R, R)`` bool stack, ``True`` where a sample carries
///         data, as ``render_view_tile`` returns it in ``valid``. When
///         ``None`` (default), alpha alone says.
///     reference: The index into ``tiles`` of the observation whose tile the
///         bitmap is, or ``None`` (default) for a bitmap that names none.
///     window: The window, ``"gaussian_disk"`` (default), ``"gaussian"`` or
///         ``"uniform"``.
///     window_sigma: Its sigma.
///
/// Returns a dict: ``plain_zncc``, ``plain_zncc_middle``,
/// ``blur_matched_zncc`` and ``blur_matched_zncc_middle`` (``(k,)`` float64,
/// NaN where a pair could not be read, 1 for the reference),
/// ``plain_zncc_grid`` and ``blur_matched_zncc_grid`` (``(k, 3, 3)`` float64,
/// each ninth from the top-left; the blur-matched middle and grid are read
/// against the same blurred bitmap as the whole tile, and equal the plain ones
/// where the bitmap is not blurred), ``blur_sigma``
/// (``(k,)`` float64, the width the bitmap was blurred by, 0 where it was
/// not), ``sharper_than_bitmap`` (``(k,)`` bool), and ``bitmap_semi_axes``
/// (``(2,)``, the bitmap's [major, minor] in grid px, NaN where it has no
/// ellipse).
///
/// Raises:
///     TypeError: If ``bitmap`` or ``tiles`` is not a uint8 array of the right
///         rank, or ``valid`` is not a 3-D bool array.
///     ValueError: If a shape is wrong, ``reference`` is past the tiles, or
///         the window name is unknown.
#[pyfunction]
#[pyo3(signature = (
    bitmap, tiles, *, valid=None, reference=None, window="gaussian_disk", window_sigma=0.6
))]
#[allow(clippy::too_many_arguments)]
pub fn score_against_bitmap<'py>(
    py: Python<'py>,
    bitmap: PyReadonlyArray3<'py, u8>,
    tiles: PyReadonlyArray4<'py, u8>,
    valid: Option<PyReadonlyArray3<'py, bool>>,
    reference: Option<usize>,
    window: &str,
    window_sigma: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let window = parse_patch_window(window, window_sigma)?;
    let bitmap_view = bitmap.as_array();
    let &[height, width, stride] = bitmap_view.shape() else {
        unreachable!("a 3-D array has three dimensions");
    };
    check_tile(height, width, stride)?;
    if stride != 4 {
        return Err(PyValueError::new_err(format!(
            "bitmap must be (R, R, 4) RGBA, got {stride} channels"
        )));
    }
    let side = width;
    let bitmap_planes = tile_planes(bitmap_view, None);
    let tiles = tiles.as_array();
    let &[k, th, tw, ts] = tiles.shape() else {
        unreachable!("a 4-D array has four dimensions");
    };
    check_tile(th, tw, ts)?;
    if tw != side {
        return Err(PyValueError::new_err(format!(
            "tiles must be {side} x {side} like the bitmap, got {th} x {tw}"
        )));
    }
    if reference.is_some_and(|r| r >= k) {
        return Err(PyValueError::new_err(format!(
            "reference {} is past the {k} tiles",
            reference.unwrap_or_default()
        )));
    }
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
    // One scorer for the scores and the bitmap's own semi-axes, so its
    // ellipse is read once.
    let (scores, bitmap_axes) = py.detach(|| {
        let mut scorer = BitmapScorer::new(&bitmap_planes, window);
        let scores: Vec<Option<BitmapScore>> = planes
            .iter()
            .enumerate()
            .map(|(v, tile)| (Some(v) != reference).then(|| scorer.score(tile, None)))
            .collect();
        (scores, scorer.bitmap_semi_axes())
    });
    let pick = |f: &dyn Fn(&BitmapScore) -> f64, at_reference: f64| -> Array1<f64> {
        scores
            .iter()
            .map(|s| s.as_ref().map_or(at_reference, f))
            .collect()
    };
    let out = PyDict::new(py);
    let pick_grid = |f: &dyn Fn(&BitmapScore) -> [[f64; 3]; 3]| {
        numpy::ndarray::Array3::from_shape_fn((scores.len(), 3, 3), |(v, i, j)| {
            scores[v].as_ref().map_or(1.0, |s| f(s)[i][j])
        })
    };
    out.set_item("plain_zncc", pick(&|s| s.plain_zncc, 1.0).into_pyarray(py))?;
    out.set_item(
        "plain_zncc_middle",
        pick(&|s| s.plain_zncc_middle, 1.0).into_pyarray(py),
    )?;
    out.set_item(
        "plain_zncc_grid",
        pick_grid(&|s| s.plain_zncc_grid).into_pyarray(py),
    )?;
    out.set_item(
        "blur_matched_zncc",
        pick(&|s| s.blur_matched_zncc, 1.0).into_pyarray(py),
    )?;
    out.set_item(
        "blur_matched_zncc_middle",
        pick(&|s| s.blur_matched_zncc_middle, 1.0).into_pyarray(py),
    )?;
    out.set_item(
        "blur_matched_zncc_grid",
        pick_grid(&|s| s.blur_matched_zncc_grid).into_pyarray(py),
    )?;
    out.set_item("blur_sigma", pick(&|s| s.blur_sigma, 0.0).into_pyarray(py))?;
    out.set_item(
        "sharper_than_bitmap",
        scores
            .iter()
            .map(|s| s.is_some_and(|s| s.sharper_than_bitmap))
            .collect::<Array1<bool>>()
            .into_pyarray(py),
    )?;
    out.set_item(
        "bitmap_semi_axes",
        Array1::from(bitmap_axes.unwrap_or([f64::NAN; 2]).to_vec()).into_pyarray(py),
    )?;
    Ok(out)
}
