// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the tile-batched consensus atlas orchestrator.
//!
//! See ``specs/core/tile-batched-consensus-atlas.md`` for the design. This
//! binding parses the Python source list **once** and runs
//! [`render_consensus_atlas`], whose peak memory is governed by the heaviest
//! single tile batch rather than the whole-run total.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use sfmtool_core::consensus_atlas::{
    render_consensus_atlas, ConsensusAtlasBatchError, ConsensusAtlasBatchParams,
};
use sfmtool_core::photometric_ransac::RansacPhotometricParams;

use crate::py_per_spherical_tile_source_stack::parse_sources;
use crate::py_spherical_tile_rig::PySphericalTileRig;

fn err_to_py(e: ConsensusAtlasBatchError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e}"))
}

/// `(atlas, tile_primary_count, tile_secondary_count, tile_primary_lum_mad,
/// tile_secondary_lum_mad)` — the Python return shape of
/// [`render_consensus_atlas_py`].
type ConsensusAtlasPyTuple = (Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>);

/// Composite a consensus atlas for ``rig`` over ``sources`` in tile batches.
///
/// Equivalent to building one monolithic
/// :class:`PerSphericalTileSourceStack`, running
/// :func:`refine_photometric_ransac`, and calling
/// :meth:`PerSphericalTileSourceStack.primary_consensus_atlas` — the result is
/// byte-identical for any ``batch_size`` — but peak memory is bounded by the
/// heaviest single batch instead of growing with the source count.
///
/// Args:
///     rig: The :class:`SphericalTileRig`. ``rig.patch_size`` must already be
///         a power of two.
///     sources: Iterable of ``(CameraIntrinsics, RotQuaternion,
///         numpy.ndarray)`` tuples, same as
///         :meth:`PerSphericalTileSourceStack.build_rotation_only`.
///     batch_size: Tiles per batch. Smaller ⇒ lower peak memory. Must be
///         ``>= 1``; a value above ``n_tiles`` acts as a single batch.
///     dtype: Per-batch stack storage — ``"float16"`` (default) or
///         ``"float32"``. ``"uint8"`` is rejected.
///     The remaining keyword arguments mirror :func:`refine_photometric_ransac`.
///
/// Returns:
///     ``(atlas, tile_primary_count, tile_secondary_count,
///     tile_primary_lum_mad, tile_secondary_lum_mad)`` where ``atlas`` is an
///     ``(atlas_h, atlas_w, channels)`` float32 array (``(atlas_h, atlas_w)``
///     when ``channels == 1``) and the four per-tile arrays have length
///     ``n_tiles``.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(name = "render_consensus_atlas", signature = (
    rig,
    sources,
    batch_size = 32,
    dtype = "float16",
    inlier_threshold = 8.0,
    gamma = 1.0,
    target_patch_size = 4,
    scoring_patch_size = 2,
    subset_size = 2,
    max_subsets_per_tile = 64,
    min_inliers = 2,
    saturation_threshold = 254,
    seed = 0,
))]
pub fn render_consensus_atlas_py(
    py: Python<'_>,
    rig: &PySphericalTileRig,
    sources: &Bound<'_, PyAny>,
    batch_size: usize,
    dtype: &str,
    inlier_threshold: f32,
    gamma: f32,
    target_patch_size: u32,
    scoring_patch_size: u32,
    subset_size: u32,
    max_subsets_per_tile: u32,
    min_inliers: u32,
    saturation_threshold: u8,
    seed: u64,
) -> PyResult<ConsensusAtlasPyTuple> {
    let parsed = parse_sources(sources)?;
    // Same channel-fallback rule as the orchestrator / `build_rotation_only`.
    let channels = parsed.first().map_or(1, |(_, _, img)| img.channels()) as usize;

    let params = ConsensusAtlasBatchParams {
        batch_size,
        ransac: RansacPhotometricParams {
            inlier_threshold,
            gamma,
            target_patch_size,
            scoring_patch_size,
            subset_size,
            max_subsets_per_tile,
            min_inliers,
            saturation_threshold,
            seed,
            // Overwritten per batch by the orchestrator.
            tile_index_base: 0,
        },
        build: Default::default(),
    };

    let report = match dtype {
        "float16" | "f16" | "half" => py
            .detach(|| render_consensus_atlas::<half::f16>(&rig.inner, &parsed, &params))
            .map_err(err_to_py)?,
        "float32" | "f32" => py
            .detach(|| render_consensus_atlas::<f32>(&rig.inner, &parsed, &params))
            .map_err(err_to_py)?,
        "uint8" | "u8" => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "render_consensus_atlas requires dtype \"float16\" or \"float32\"; \
                 uint8-backed batches are not supported",
            ));
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dtype must be 'float16' or 'float32', got {other:?}"
            )));
        }
    };

    let (atlas_w, atlas_h) = rig.inner.atlas_size();
    let atlas = if channels == 1 {
        PyArray1::from_vec(py, report.atlas)
            .reshape([atlas_h as usize, atlas_w as usize])?
            .into_any()
            .unbind()
    } else {
        PyArray1::from_vec(py, report.atlas)
            .reshape([atlas_h as usize, atlas_w as usize, channels])?
            .into_any()
            .unbind()
    };

    Ok((
        atlas,
        PyArray1::from_vec(py, report.tile_primary_count)
            .into_any()
            .unbind(),
        PyArray1::from_vec(py, report.tile_secondary_count)
            .into_any()
            .unbind(),
        PyArray1::from_vec(py, report.tile_primary_lum_mad)
            .into_any()
            .unbind(),
        PyArray1::from_vec(py, report.tile_secondary_lum_mad)
            .into_any()
            .unbind(),
    ))
}
