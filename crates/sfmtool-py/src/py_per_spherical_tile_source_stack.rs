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

/// Internal storage: dispatches between u8 and f32 underlying stacks.
enum Inner {
    U8(PerSphericalTileSourceStack<u8>),
    F32(PerSphericalTileSourceStack<f32>),
}

/// Per-spherical-tile source patch stack: for each tile of a
/// [`SphericalTileRig`], holds the ordered list of contributing source images
/// and their per-source pyramids of warped patches in the tile's local
/// pinhole frame. See ``specs/core/per-spherical-tile-source-stack.md``.
///
/// Pixel storage is selected at build time via ``dtype="uint8"`` (default,
/// compact) or ``dtype="float32"`` (autodiff-ready). The ``uint8`` ->
/// ``float32`` conversion preserves 0-255 range, not [0, 1] scale.
#[pyclass(name = "PerSphericalTileSourceStack", module = "sfmtool._sfmtool")]
pub struct PyPerSphericalTileSourceStack {
    inner: Inner,
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
    ///     dtype: Pixel storage type — ``"uint8"`` (default) or
    ///         ``"float32"``. Float32 storage preserves 0–255 range (so
    ///         level-0 values are byte-equivalent to uint8) and uses exact
    ///         arithmetic for box-filter downsampling.
    ///
    /// Returns:
    ///     A populated :class:`PerSphericalTileSourceStack`.
    #[staticmethod]
    #[pyo3(signature = (rig, sources, max_in_flight_sources = None, dtype = "uint8"))]
    fn build_rotation_only(
        py: Python<'_>,
        rig: &PySphericalTileRig,
        sources: &Bound<'_, PyAny>,
        max_in_flight_sources: Option<usize>,
        dtype: &str,
    ) -> PyResult<Self> {
        let parsed = parse_sources(sources)?;
        let params = BuildParams {
            max_in_flight_sources,
        };
        let inner = match dtype {
            "uint8" | "u8" => {
                let stack = py
                    .detach(|| {
                        PerSphericalTileSourceStack::<u8>::build_rotation_only(
                            &rig.inner, &parsed, &params,
                        )
                    })
                    .map_err(err_to_py)?;
                Inner::U8(stack)
            }
            "float32" | "f32" => {
                let stack = py
                    .detach(|| {
                        PerSphericalTileSourceStack::<f32>::build_rotation_only(
                            &rig.inner, &parsed, &params,
                        )
                    })
                    .map_err(err_to_py)?;
                Inner::F32(stack)
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "dtype must be 'uint8' or 'float32', got {other:?}"
                )));
            }
        };
        Ok(Self { inner })
    }

    /// Pixel storage dtype — ``"uint8"`` or ``"float32"``.
    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.inner {
            Inner::U8(_) => "uint8",
            Inner::F32(_) => "float32",
        }
    }

    /// Number of tiles (mirrors the rig's tile count).
    #[getter]
    fn n_tiles(&self) -> usize {
        match &self.inner {
            Inner::U8(s) => s.n_tiles(),
            Inner::F32(s) => s.n_tiles(),
        }
    }

    fn __len__(&self) -> usize {
        self.n_tiles()
    }

    /// Side length of level 0 across every tile (= ``rig.patch_size`` at
    /// build time). Always a power of two.
    #[getter]
    fn base_patch_size(&self) -> u32 {
        match &self.inner {
            Inner::U8(s) => s.base_patch_size(),
            Inner::F32(s) => s.base_patch_size(),
        }
    }

    /// Number of pyramid levels (= ``log2(base_patch_size) + 1``).
    #[getter]
    fn pyramid_levels(&self) -> u32 {
        match &self.inner {
            Inner::U8(s) => s.pyramid_levels(),
            Inner::F32(s) => s.pyramid_levels(),
        }
    }

    /// Channel count (uniform across all tiles, sources, and levels).
    #[getter]
    fn channels(&self) -> u32 {
        match &self.inner {
            Inner::U8(s) => s.channels(),
            Inner::F32(s) => s.channels(),
        }
    }

    /// Total CSR row count summed across all tiles.
    #[getter]
    fn total_contrib_rows(&self) -> usize {
        match &self.inner {
            Inner::U8(s) => s.total_contrib_rows(),
            Inner::F32(s) => s.total_contrib_rows(),
        }
    }

    /// Number of contributing sources for tile ``tile_idx``.
    fn n_contributors(&self, tile_idx: usize) -> PyResult<usize> {
        self.check_tile_idx(tile_idx)?;
        Ok(match &self.inner {
            Inner::U8(s) => s.n_contributors(tile_idx),
            Inner::F32(s) => s.n_contributors(tile_idx),
        })
    }

    /// Source-list indices for tile ``tile_idx`` (sorted ascending), as a
    /// 1-D ``uint32`` numpy array.
    fn src_indices_for_tile<'py>(&self, py: Python<'py>, tile_idx: usize) -> PyResult<Py<PyAny>> {
        self.check_tile_idx(tile_idx)?;
        let data: Vec<u32> = match &self.inner {
            Inner::U8(s) => s.src_indices_for_tile(tile_idx).to_vec(),
            Inner::F32(s) => s.src_indices_for_tile(tile_idx).to_vec(),
        };
        Ok(PyArray1::from_vec(py, data).into_any().unbind())
    }

    /// Side length (pixels per side) of pyramid level ``level`` (same for
    /// every tile).
    fn level_size(&self, level: usize) -> PyResult<u32> {
        self.check_level_idx(level)?;
        Ok(match &self.inner {
            Inner::U8(s) => s.level(level).size,
            Inner::F32(s) => s.level(level).size,
        })
    }

    /// Tile ``tile_idx``'s patches at pyramid ``level``, as a
    /// ``(n_contributors, size, size, channels)`` numpy array (rank 3 if
    /// ``channels == 1``). Dtype is ``uint8`` or ``float32`` depending on
    /// the stack's :attr:`dtype`.
    fn patches_for_tile<'py>(
        &self,
        py: Python<'py>,
        tile_idx: usize,
        level: usize,
    ) -> PyResult<Py<PyAny>> {
        self.check_tile_idx(tile_idx)?;
        self.check_level_idx(level)?;
        let (s, c) = self.size_and_channels(level);
        match &self.inner {
            Inner::U8(stack) => {
                let n = stack.n_contributors(tile_idx);
                let data = stack.patches_for_tile(tile_idx, level).to_vec();
                reshape_patch_array(py, data, n, s, c)
            }
            Inner::F32(stack) => {
                let n = stack.n_contributors(tile_idx);
                let data = stack.patches_for_tile(tile_idx, level).to_vec();
                reshape_patch_array(py, data, n, s, c)
            }
        }
    }

    /// Tile ``tile_idx``'s valid masks at pyramid ``level``, as a
    /// ``(n_contributors, size, size)`` ``bool`` numpy array.
    fn valid_for_tile<'py>(
        &self,
        py: Python<'py>,
        tile_idx: usize,
        level: usize,
    ) -> PyResult<Py<PyAny>> {
        self.check_tile_idx(tile_idx)?;
        self.check_level_idx(level)?;
        let (s, _c) = self.size_and_channels(level);
        let data: Vec<u8> = match &self.inner {
            Inner::U8(stack) => stack.valid_for_tile(tile_idx, level).to_vec(),
            Inner::F32(stack) => stack.valid_for_tile(tile_idx, level).to_vec(),
        };
        let n = data.len() / (s * s).max(1);
        valid_array_to_bool(py, data, n, s)
    }

    /// Whole-level patches buffer at ``level``, as a
    /// ``(total_contrib_rows, size, size, channels)`` numpy array. Dtype
    /// is ``uint8`` or ``float32`` depending on the stack's :attr:`dtype`.
    fn level_patches<'py>(&self, py: Python<'py>, level: usize) -> PyResult<Py<PyAny>> {
        self.check_level_idx(level)?;
        let (s, c) = self.size_and_channels(level);
        match &self.inner {
            Inner::U8(stack) => {
                let n = stack.total_contrib_rows();
                let data = stack.level_patches(level).to_vec();
                reshape_patch_array(py, data, n, s, c)
            }
            Inner::F32(stack) => {
                let n = stack.total_contrib_rows();
                let data = stack.level_patches(level).to_vec();
                reshape_patch_array(py, data, n, s, c)
            }
        }
    }

    /// Whole-level valid buffer at ``level``, as a
    /// ``(total_contrib_rows, size, size)`` ``bool`` numpy array.
    fn level_valid<'py>(&self, py: Python<'py>, level: usize) -> PyResult<Py<PyAny>> {
        self.check_level_idx(level)?;
        let (s, _c) = self.size_and_channels(level);
        let data: Vec<u8> = match &self.inner {
            Inner::U8(stack) => stack.level_valid(level).to_vec(),
            Inner::F32(stack) => stack.level_valid(level).to_vec(),
        };
        let n = data.len() / (s * s).max(1);
        valid_array_to_bool(py, data, n, s)
    }

    /// CSR offsets, length ``n_tiles + 1``. Tile ``t``'s rows are
    /// ``tile_offsets[t]:tile_offsets[t + 1]``. ``uint32``.
    fn tile_offsets<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        let data: Vec<u32> = match &self.inner {
            Inner::U8(s) => s.tile_offsets().to_vec(),
            Inner::F32(s) => s.tile_offsets().to_vec(),
        };
        PyArray1::from_vec(py, data).into_any().unbind()
    }

    /// Per-row tile index, length ``total_contrib_rows``. ``uint32``.
    fn tile_id<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        let data: Vec<u32> = match &self.inner {
            Inner::U8(s) => s.tile_id().to_vec(),
            Inner::F32(s) => s.tile_id().to_vec(),
        };
        PyArray1::from_vec(py, data).into_any().unbind()
    }

    /// Per-row source index, length ``total_contrib_rows``. ``uint32``.
    fn src_id<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        let data: Vec<u32> = match &self.inner {
            Inner::U8(s) => s.src_id().to_vec(),
            Inner::F32(s) => s.src_id().to_vec(),
        };
        PyArray1::from_vec(py, data).into_any().unbind()
    }
}

impl PyPerSphericalTileSourceStack {
    fn check_tile_idx(&self, tile_idx: usize) -> PyResult<()> {
        let n = self.n_tiles();
        if tile_idx >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "tile_idx {tile_idx} out of range (n_tiles = {n})"
            )));
        }
        Ok(())
    }

    fn check_level_idx(&self, level: usize) -> PyResult<()> {
        let pl = self.pyramid_levels() as usize;
        if level >= pl {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "level {level} out of range (pyramid_levels = {pl})"
            )));
        }
        Ok(())
    }

    fn size_and_channels(&self, level: usize) -> (usize, usize) {
        match &self.inner {
            Inner::U8(s) => (s.level(level).size as usize, s.channels() as usize),
            Inner::F32(s) => (s.level(level).size as usize, s.channels() as usize),
        }
    }
}

/// Reshape a flat patch buffer to ``(n, size, size, channels)``, dropping
/// the channel axis when ``channels == 1``.
fn reshape_patch_array<'py, T: numpy::Element>(
    py: Python<'py>,
    data: Vec<T>,
    n: usize,
    size: usize,
    channels: usize,
) -> PyResult<Py<PyAny>> {
    let arr = if channels == 1 {
        PyArray1::from_vec(py, data)
            .reshape([n, size, size])?
            .into_any()
            .unbind()
    } else {
        PyArray1::from_vec(py, data)
            .reshape([n, size, size, channels])?
            .into_any()
            .unbind()
    };
    Ok(arr)
}

/// Reshape a flat ``{0, 1}`` u8 buffer to ``(n, size, size)`` and view as
/// ``bool``.
fn valid_array_to_bool<'py>(
    py: Python<'py>,
    data: Vec<u8>,
    n: usize,
    size: usize,
) -> PyResult<Py<PyAny>> {
    let arr_u8: Bound<'py, numpy::PyArray3<u8>> =
        PyArray1::from_vec(py, data).reshape([n, size, size])?;
    let np = py.import("numpy")?;
    let bool_dtype = np.getattr("bool_")?;
    let arr_bool = arr_u8.into_any().call_method1("view", (bool_dtype,))?;
    Ok(arr_bool.unbind())
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
