// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for [`sfmtool_core::per_spherical_tile_source_stack::PerSphericalTileSourceStack`].

use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use sfmtool_core::per_spherical_tile_source_stack::{
    BuildError, BuildParams, PerSphericalTileSourceStack,
};
use sfmtool_core::remap::ImageU8;

use crate::py_camera_intrinsics::PyCameraIntrinsics;
use crate::py_rot_quaternion::PyRotQuaternion;
use crate::py_spherical_tile_rig::PySphericalTileRig;

fn err_to_py(e: BuildError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e}"))
}

/// Per-spherical-tile source patch stack: for each tile of a
/// [`SphericalTileRig`], holds the ordered list of contributing source images
/// and their per-source pyramids of warped patches in the tile's local
/// pinhole frame. See ``specs/core/per-spherical-tile-source-stack.md``.
#[pyclass(name = "PerSphericalTileSourceStack", module = "sfmtool._sfmtool")]
pub struct PyPerSphericalTileSourceStack {
    inner: PerSphericalTileSourceStack,
}

#[pymethods]
impl PyPerSphericalTileSourceStack {
    /// Build a rotation-only stack for ``rig`` and ``sources``.
    ///
    /// ``rig.patch_size`` must already be a power of two — call
    /// ``rig.set_patch_size(rig.patch_size`` next-power-of-two ``)`` first if
    /// not.
    ///
    /// Args:
    ///     rig: The :class:`SphericalTileRig`.
    ///     sources: Iterable of ``(CameraIntrinsics, RotQuaternion,
    ///         numpy.ndarray)`` tuples. Each rotation is the source camera's
    ///         ``R_src_from_world``. Each image is ``HxW`` (gray) or
    ///         ``HxWxC`` (RGB / RGBA) ``uint8``; all sources must share a
    ///         channel count.
    ///     max_in_flight_sources: Reserved for future parallel-source
    ///         chunking. Currently a no-op.
    ///
    /// Returns:
    ///     A populated :class:`PerSphericalTileSourceStack`.
    #[staticmethod]
    #[pyo3(signature = (rig, sources, max_in_flight_sources = None))]
    fn build_rotation_only(
        py: Python<'_>,
        rig: &PySphericalTileRig,
        sources: &Bound<'_, PyAny>,
        max_in_flight_sources: Option<usize>,
    ) -> PyResult<Self> {
        let parsed = parse_sources(sources)?;
        let inner = py
            .detach(|| {
                PerSphericalTileSourceStack::build_rotation_only(
                    &rig.inner,
                    &parsed,
                    &BuildParams {
                        max_in_flight_sources,
                    },
                )
            })
            .map_err(err_to_py)?;
        Ok(Self { inner })
    }

    /// Number of tiles (mirrors the rig's tile count).
    #[getter]
    fn n_tiles(&self) -> usize {
        self.inner.n_tiles()
    }

    fn __len__(&self) -> usize {
        self.inner.n_tiles()
    }

    /// Side length of level 0 across every tile (= ``rig.patch_size`` at
    /// build time). Always a power of two.
    #[getter]
    fn base_patch_size(&self) -> u32 {
        self.inner.base_patch_size()
    }

    /// Number of pyramid levels (= ``log2(base_patch_size) + 1``).
    #[getter]
    fn pyramid_levels(&self) -> u32 {
        self.inner.pyramid_levels()
    }

    /// Number of contributing sources for tile ``tile_idx``.
    fn n_contributors(&self, tile_idx: usize) -> PyResult<usize> {
        self.check_tile_idx(tile_idx)?;
        Ok(self.inner.n_contributors(tile_idx))
    }

    /// Source-list indices for tile ``tile_idx`` (sorted ascending), as a
    /// 1-D ``uint32`` numpy array.
    fn src_indices<'py>(&self, py: Python<'py>, tile_idx: usize) -> PyResult<Py<PyAny>> {
        self.check_tile_idx(tile_idx)?;
        let data = self.inner.tile(tile_idx).src_indices.clone();
        Ok(PyArray1::from_vec(py, data).into_any().unbind())
    }

    /// Side length (pixels per side) of pyramid level ``level`` (same for
    /// every tile).
    fn level_size(&self, level: usize) -> PyResult<u32> {
        self.check_level_idx(level)?;
        // Use tile 0's level metadata; sizes are uniform across tiles.
        Ok(self.inner.tile(0).levels[level].size)
    }

    /// Channel count (uniform across all tiles, sources, and levels).
    #[getter]
    fn channels(&self) -> u32 {
        if self.inner.n_tiles() == 0 {
            1
        } else {
            self.inner.tile(0).levels[0].channels
        }
    }

    /// Patches at one ``(tile, level)`` as a ``(n_contributors, size, size,
    /// channels)`` ``uint8`` numpy array (rank 3 if ``channels == 1``).
    /// Empty (size 0) when the tile has no contributors.
    fn level_patches<'py>(
        &self,
        py: Python<'py>,
        tile_idx: usize,
        level: usize,
    ) -> PyResult<Py<PyAny>> {
        self.check_tile_idx(tile_idx)?;
        self.check_level_idx(level)?;
        let level_data = &self.inner.tile(tile_idx).levels[level];
        let n = level_data.n_contributors as usize;
        let s = level_data.size as usize;
        let c = level_data.channels as usize;
        let data = level_data.patches.clone();
        let arr = if c == 1 {
            PyArray1::from_vec(py, data)
                .reshape([n, s, s])?
                .into_any()
                .unbind()
        } else {
            PyArray1::from_vec(py, data)
                .reshape([n, s, s, c])?
                .into_any()
                .unbind()
        };
        Ok(arr)
    }

    /// Valid masks at one ``(tile, level)`` as a ``(n_contributors, size,
    /// size)`` ``bool`` numpy array.
    fn level_valid<'py>(
        &self,
        py: Python<'py>,
        tile_idx: usize,
        level: usize,
    ) -> PyResult<Py<PyAny>> {
        self.check_tile_idx(tile_idx)?;
        self.check_level_idx(level)?;
        let level_data = &self.inner.tile(tile_idx).levels[level];
        let n = level_data.n_contributors as usize;
        let s = level_data.size as usize;
        // numpy's bool dtype is one byte; copy as u8.
        let bytes: Vec<u8> = level_data.valid.iter().map(|&b| b as u8).collect();
        let arr_u8: Bound<'py, numpy::PyArray3<u8>> =
            PyArray1::from_vec(py, bytes).reshape([n, s, s])?;
        // Reinterpret as bool by calling numpy's `.view(bool)`.
        let np = py.import("numpy")?;
        let bool_dtype = np.getattr("bool_")?;
        let arr_bool = arr_u8.into_any().call_method1("view", (bool_dtype,))?;
        Ok(arr_bool.unbind())
    }
}

impl PyPerSphericalTileSourceStack {
    fn check_tile_idx(&self, tile_idx: usize) -> PyResult<()> {
        if tile_idx >= self.inner.n_tiles() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "tile_idx {tile_idx} out of range (n_tiles = {})",
                self.inner.n_tiles()
            )));
        }
        Ok(())
    }

    fn check_level_idx(&self, level: usize) -> PyResult<()> {
        let pl = self.inner.pyramid_levels() as usize;
        if level >= pl {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "level {level} out of range (pyramid_levels = {pl})"
            )));
        }
        Ok(())
    }
}

/// Parse a Python iterable of ``(CameraIntrinsics, RotQuaternion,
/// numpy.ndarray)`` tuples into the Rust slice the builder expects.
fn parse_sources(
    obj: &Bound<'_, PyAny>,
) -> PyResult<
    Vec<(
        sfmtool_core::CameraIntrinsics,
        sfmtool_core::RotQuaternion,
        ImageU8,
    )>,
> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        let item = item?;
        let tup: Bound<'_, PyTuple> = item.downcast_into::<PyTuple>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "each source must be a (CameraIntrinsics, RotQuaternion, ndarray) tuple",
            )
        })?;
        if tup.len() != 3 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "each source tuple must have exactly 3 elements",
            ));
        }
        let cam: PyRef<'_, PyCameraIntrinsics> = tup.get_item(0)?.extract()?;
        let rot: PyRef<'_, PyRotQuaternion> = tup.get_item(1)?.extract()?;
        let img = extract_image_u8(&tup.get_item(2)?)?;
        out.push((cam.inner.clone(), rot.inner.clone(), img));
    }
    Ok(out)
}

/// Extract a numpy uint8 array (HxW or HxWxC) into an [`ImageU8`].
fn extract_image_u8(obj: &Bound<'_, PyAny>) -> PyResult<ImageU8> {
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray3<'_, u8>>() {
        let shape = arr.shape();
        let (h, w, c) = (shape[0] as u32, shape[1] as u32, shape[2] as u32);
        if !matches!(c, 1 | 3 | 4) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "image must have 1, 3, or 4 channels (got {c})"
            )));
        }
        let data: Vec<u8> = match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => arr.as_array().iter().copied().collect(),
        };
        return Ok(ImageU8::new(w, h, c, data));
    }
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray2<'_, u8>>() {
        let shape = arr.shape();
        let (h, w) = (shape[0] as u32, shape[1] as u32);
        let data: Vec<u8> = match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => arr.as_array().iter().copied().collect(),
        };
        return Ok(ImageU8::new(w, h, 1, data));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "image must be a 2D (HxW) or 3D (HxWxC) numpy uint8 array",
    ))
}
