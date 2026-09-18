// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The constellation query, shared by the two forest classes.
//!
//! Both `KdForest` and `LazyKdForest` expose the same two methods, so the body
//! lives here once and each class's `#[pymethods]` block forwards to it through
//! `&dyn` references. The only difference between them is where the source
//! tables come from: the file-backed forest is its own, and the resident forest
//! takes them as a `sources` mapping, because loading a `.kdf` into memory
//! rebuilds the trees and the corpus and keeps no origins or geometry.

use std::path::{Path, PathBuf};

use numpy::{PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::features::kdforest::{
    constellation_at_pixel, constellation_query, AffineRefit, Constellation,
    ConstellationDescriptors, ConstellationMatch, ConstellationParams, FeatureGeometry,
    FeatureOrigin, FeatureSources, NeighborIndex, QueryImage, ResidentSources,
};

use super::kdf::to_py_err;
use super::kdforest::extract_u8_2d;

/// The Rust defaults, for the four `#[pyo3(signature = ...)]` blocks to name a
/// field of rather than to repeat a number that would then drift.
pub(crate) const DEFAULTS: ConstellationParams = ConstellationParams::DEFAULT;

/// The default `refit` spelling, so Python's keyword default is the Rust one.
pub(crate) const DEFAULT_REFIT: &str = refit_name(DEFAULTS.refit);

/// The default `refit_sigma`. A default that is not centre-weighted carries no
/// sigma of its own, and the keyword then starts at the value the enum's own
/// documentation recommends, so naming it is still meaningful.
pub(crate) const DEFAULT_REFIT_SIGMA: f64 = match DEFAULTS.refit {
    AffineRefit::CenterWeighted { sigma } => sigma,
    _ => 0.5,
};

/// The spelling Python uses for one refit mode.
const fn refit_name(refit: AffineRefit) -> &'static str {
    match refit {
        AffineRefit::None => "none",
        AffineRefit::LeastSquares => "least_squares",
        AffineRefit::CenterWeighted { .. } => "center_weighted",
    }
}

/// One of the three spellings, with `refit_sigma` read only by the weighted one.
///
/// A misspelling is a `ValueError` naming all three rather than a silent fall
/// back to a default: a caller asking for `"centre_weighted"` wants weighting,
/// and quietly giving it the three-point model would be the one outcome it
/// cannot detect from the result.
fn parse_refit(refit: &str, sigma: f64) -> PyResult<AffineRefit> {
    match refit {
        "none" => Ok(AffineRefit::None),
        "least_squares" => Ok(AffineRefit::LeastSquares),
        "center_weighted" => {
            if !(sigma.is_finite() && sigma > 0.0) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "refit_sigma must be a finite positive fraction of the \
                     constellation radius, got {sigma}"
                )));
            }
            Ok(AffineRefit::CenterWeighted { sigma })
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "refit must be \"center_weighted\", \"least_squares\" or \"none\", got {other:?}"
        ))),
    }
}

/// Every tunable of the query, in one struct, so the two call sites declare the
/// same keyword arguments instead of drifting apart.
pub(crate) struct QueryOptions<'a> {
    pub k: usize,
    pub max_leaf_checks: usize,
    pub threshold_px: f64,
    pub iterations: usize,
    pub min_correspondences: usize,
    pub one_hit_per_image: bool,
    pub same_image_ratio: f32,
    pub min_inliers: usize,
    pub max_scale: f64,
    pub refit: &'a str,
    pub refit_sigma: f64,
    pub seed: u64,
}

impl TryFrom<&QueryOptions<'_>> for ConstellationParams {
    type Error = PyErr;

    fn try_from(value: &QueryOptions<'_>) -> PyResult<Self> {
        Ok(Self {
            k: value.k,
            max_leaf_checks: value.max_leaf_checks,
            threshold_px: value.threshold_px,
            iterations: value.iterations,
            min_correspondences: value.min_correspondences,
            one_hit_per_image: value.one_hit_per_image,
            same_image_ratio: value.same_image_ratio,
            min_inliers: value.min_inliers,
            max_scale: value.max_scale,
            refit: parse_refit(value.refit, value.refit_sigma)?,
            seed: value.seed,
        })
    }
}

/// The radius that holds about `target` keypoints of one image.
///
/// Args:
///     image_width: The image's width in pixels.
///     image_height: Its height in pixels.
///     keypoint_count: How many keypoints the detector found in it.
///     target: Constellation size wanted.
///
/// Returns:
///     `sqrt(target * width * height / (pi * keypoint_count))`, the radius a
///     uniform keypoint density puts `target` keypoints inside, and 0.0 for an
///     image with no keypoints. Keypoints cluster on texture and a patch is
///     usually centred on one, so the measured radius runs 70 to 100% of this.
///     Fifty is the size to ask for: the warp is trustworthy far more often
///     there than at two hundred, where it is wrong more often than right.
#[pyfunction]
#[pyo3(signature = (image_width, image_height, keypoint_count, target))]
fn radius_for_feature_count(
    image_width: u32,
    image_height: u32,
    keypoint_count: usize,
    target: usize,
) -> f32 {
    sfmtool_core::features::kdforest::radius_for_feature_count(
        image_width,
        image_height,
        keypoint_count,
        target,
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(radius_for_feature_count, m)?)?;
    Ok(())
}

/// Build resident source tables from the same mapping `write_kdf` accepts.
///
/// It reads the four per-feature columns and ignores the workspace and image
/// identity a write needs, so a caller can hand the same dict to both.
pub(crate) fn parse_resident_sources(sources: &Bound<'_, PyAny>) -> PyResult<ResidentSources> {
    fn need<'py>(d: &Bound<'py, PyAny>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        d.get_item(key).map_err(|_| {
            pyo3::exceptions::PyKeyError::new_err(format!("sources is missing {key:?}"))
        })
    }
    let image_indexes: Vec<u32> = need(sources, "image_indexes")?.extract()?;
    let image_feature_indexes: Vec<u32> = need(sources, "image_feature_indexes")?.extract()?;
    if image_indexes.len() != image_feature_indexes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "image_indexes has {} entries but image_feature_indexes has {}",
            image_indexes.len(),
            image_feature_indexes.len()
        )));
    }
    let positions: PyReadonlyArray2<'_, f32> = need(sources, "positions")?.extract()?;
    let affine_shapes: PyReadonlyArray3<'_, f32> = need(sources, "affine_shapes")?.extract()?;
    let positions = positions.as_array();
    let affine_shapes = affine_shapes.as_array();
    let n = image_indexes.len();
    if positions.shape() != [n, 2] || affine_shapes.shape() != [n, 2, 2] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "positions must be (N, 2) and affine_shapes (N, 2, 2) for N={n}"
        )));
    }
    let origins = image_indexes
        .into_iter()
        .zip(image_feature_indexes)
        .map(|(image_index, image_feature_index)| FeatureOrigin {
            image_index,
            image_feature_index,
        })
        .collect();
    let geometry: Vec<FeatureGeometry> = (0..n)
        .map(|i| {
            [
                [positions[[i, 0]], positions[[i, 1]]],
                [affine_shapes[[i, 0, 0]], affine_shapes[[i, 0, 1]]],
                [affine_shapes[[i, 1, 0]], affine_shapes[[i, 1, 1]]],
            ]
        })
        .collect();
    ResidentSources::new(origins, geometry).map_err(to_py_err)
}

/// Body of both classes' `constellation_query`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn query<'py>(
    py: Python<'py>,
    index: &(dyn NeighborIndex<u8> + Sync),
    sources: &(dyn FeatureSources + Sync),
    positions: &Bound<'py, PyAny>,
    descriptors: Option<&Bound<'py, PyAny>>,
    feature_ids: Option<Vec<u32>>,
    image_index: Option<u32>,
    center: Option<(f32, f32)>,
    options: &QueryOptions<'_>,
) -> PyResult<Py<PyList>> {
    let positions = read_positions(positions)?;
    let params = ConstellationParams::try_from(options)?;
    let center = center.map(|(x, y)| [x, y]);
    let matches = match (descriptors, feature_ids) {
        (Some(descriptors), None) => {
            let descriptors = extract_u8_2d(descriptors, "constellation descriptors")?;
            let shape = descriptors.shape();
            if shape[1] != index.dim() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "descriptor width {} does not match the index dim {}",
                    shape[1],
                    index.dim()
                )));
            }
            let data: std::borrow::Cow<[u8]> = to_contiguous!(descriptors);
            py.detach(|| {
                constellation_query(
                    index,
                    sources,
                    &Constellation {
                        positions: &positions,
                        descriptors: ConstellationDescriptors::Vectors(&data),
                        image_index,
                        center,
                    },
                    &params,
                )
            })
        }
        (None, Some(ids)) => py.detach(|| {
            constellation_query(
                index,
                sources,
                &Constellation {
                    positions: &positions,
                    descriptors: ConstellationDescriptors::FeatureIds(&ids),
                    image_index,
                    center,
                },
                &params,
            )
        }),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pass exactly one of descriptors or feature_ids",
            ))
        }
    }
    .map_err(to_py_err)?;
    matches_list(py, &matches)
}

/// Body of both classes' `constellation_at_pixel`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn at_pixel<'py>(
    py: Python<'py>,
    index: &(dyn NeighborIndex<u8> + Sync),
    sources: &(dyn FeatureSources + Sync),
    sift_path: PathBuf,
    center: (f32, f32),
    radius: f32,
    image_index: Option<u32>,
    options: &QueryOptions<'_>,
) -> PyResult<Py<PyDict>> {
    let params = ConstellationParams::try_from(options)?;
    let found = py
        .detach(|| {
            constellation_at_pixel(
                index,
                sources,
                &QueryImage {
                    sift_path: Path::new(&sift_path),
                    keypoints: None,
                    image_index,
                },
                [center.0, center.1],
                radius,
                &params,
            )
        })
        .map_err(to_py_err)?;
    let out = PyDict::new(py);
    out.set_item(
        "feature_rows",
        numpy::PyArray1::from_vec(py, found.feature_rows),
    )?;
    out.set_item(
        "feature_ids",
        numpy::PyArray1::from_vec(py, found.feature_ids),
    )?;
    out.set_item("matches", matches_list(py, &found.matches)?)?;
    Ok(out.unbind())
}

/// `(N, 2)` float32 query positions, in the query image's pixels.
fn read_positions(positions: &Bound<'_, PyAny>) -> PyResult<Vec<[f32; 2]>> {
    let array: PyReadonlyArray2<'_, f32> = positions.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("positions must be an (N, 2) float32 array")
    })?;
    let array = array.as_array();
    if array.ncols() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "positions must be (N, 2), got width {}",
            array.ncols()
        )));
    }
    Ok((0..array.nrows())
        .map(|i| [array[[i, 0]], array[[i, 1]]])
        .collect())
}

/// One dict per candidate image, keys named exactly as the Rust fields.
///
/// The inlier correspondences are columns rather than a list of per-row dicts:
/// they are consumed as arrays, by a caller seeding a patch cluster from them,
/// and a few hundred one-key-per-field dicts would cost more to build than the
/// query itself.
fn matches_list<'py>(py: Python<'py>, matches: &[ConstellationMatch]) -> PyResult<Py<PyList>> {
    let list = PyList::empty(py);
    for found in matches {
        let entry = PyDict::new(py);
        entry.set_item("image_index", found.image_index)?;
        entry.set_item(
            "affine",
            numpy::PyArray1::from_vec(py, found.affine.iter().flatten().copied().collect())
                .reshape([2, 3])?,
        )?;
        entry.set_item("inliers", found.inliers)?;
        entry.set_item("correspondences", found.correspondences)?;

        let inliers = &found.inlier_correspondences;
        let columns = PyDict::new(py);
        columns.set_item(
            "query_index",
            numpy::PyArray1::from_vec(py, inliers.iter().map(|c| c.query_index).collect()),
        )?;
        columns.set_item(
            "feature_id",
            numpy::PyArray1::from_vec(py, inliers.iter().map(|c| c.feature_id).collect()),
        )?;
        columns.set_item(
            "position",
            numpy::PyArray1::from_vec(
                py,
                inliers
                    .iter()
                    .flat_map(|c| c.position)
                    .collect::<Vec<f32>>(),
            )
            .reshape([inliers.len(), 2])?,
        )?;
        columns.set_item(
            "affine_shape",
            numpy::PyArray1::from_vec(
                py,
                inliers
                    .iter()
                    .flat_map(|c| c.affine_shape.into_iter().flatten())
                    .collect::<Vec<f32>>(),
            )
            .reshape([inliers.len(), 2, 2])?,
        )?;
        entry.set_item("inlier_correspondences", columns)?;
        list.append(entry)?;
    }
    Ok(list.unbind())
}
