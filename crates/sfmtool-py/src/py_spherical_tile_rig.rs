// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for [`sfmtool_core::spherical_tile_rig::SphericalTileRig`].

use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use sfmtool_core::sphere_points::RelaxConfig;
use sfmtool_core::spherical_tile_rig::{
    SphericalTileRig, SphericalTileRigError, SphericalTileRigParams,
};

use crate::py_camera_intrinsics::PyCameraIntrinsics;
use crate::py_rot_quaternion::PyRotQuaternion;
use crate::py_se3_transform::PySe3Transform;
use crate::py_warp_map::PyWarpMap;

fn err_to_py(e: SphericalTileRigError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e}"))
}

/// A rig of `n` shared-centre pinhole tile cameras tiling the unit sphere.
///
/// Every tile has identical intrinsics (focal length, half-FOV, patch
/// resolution); only the rotation differs per tile so the tiles look in
/// different directions on the unit sphere around the rig centre.
#[pyclass(name = "SphericalTileRig", module = "sfmtool._sfmtool")]
pub struct PySphericalTileRig {
    pub(crate) inner: SphericalTileRig,
}

#[pymethods]
impl PySphericalTileRig {
    /// Build a tile rig.
    ///
    /// Args:
    ///     n: Number of tiles (>= 2).
    ///     arc_per_pixel: Angular size of one tile pixel, in radians. For a
    ///         target equirect of width ``W``, pass ``2π / W``.
    ///     centre: 3-element rig centre in world space (default ``[0, 0, 0]``).
    ///     overlap_factor: Multiplicative safety margin on the measured
    ///         worst-case Voronoi cell radius (default ``1.15`` = 15% overlap).
    ///     atlas_cols: Override the atlas column count (default ``ceil(√n)``).
    ///     relax_iterations: Sphere-point relaxer iterations (default 50).
    ///     relax_step_size: Relaxer step size (default 0.05).
    ///     relax_cutoff_multiplier: Relaxer neighbour cutoff (default 5.0).
    ///     seed: Optional u64 seed for the relaxer's random initialisation.
    ///         When set, the entire rig is reproducible.
    #[new]
    #[pyo3(signature = (
        n,
        arc_per_pixel,
        centre=None,
        overlap_factor=1.15,
        atlas_cols=None,
        relax_iterations=50,
        relax_step_size=0.05,
        relax_cutoff_multiplier=5.0,
        seed=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        n: usize,
        arc_per_pixel: f64,
        centre: Option<[f64; 3]>,
        overlap_factor: f64,
        atlas_cols: Option<u32>,
        relax_iterations: usize,
        relax_step_size: f32,
        relax_cutoff_multiplier: f32,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let params = SphericalTileRigParams {
            centre: centre.unwrap_or([0.0, 0.0, 0.0]),
            n,
            arc_per_pixel,
            overlap_factor,
            atlas_cols,
            relax: Some(RelaxConfig {
                iterations: relax_iterations,
                step_size: relax_step_size,
                cutoff_multiplier: relax_cutoff_multiplier,
                seed,
            }),
        };
        let inner = py
            .detach(|| SphericalTileRig::new(&params))
            .map_err(err_to_py)?;
        Ok(Self { inner })
    }

    /// Number of tiles in the rig.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of tiles in the rig.
    #[getter]
    fn n(&self) -> usize {
        self.inner.len()
    }

    /// World-space optical centre shared by every tile.
    #[getter]
    fn centre(&self) -> [f64; 3] {
        self.inner.centre()
    }

    /// Per-tile half-FOV in radians.
    #[getter]
    fn half_fov_rad(&self) -> f64 {
        self.inner.half_fov_rad()
    }

    /// Worst-case nearest-neighbour angular gap between tile directions
    /// (radians). Diagnostic.
    #[getter]
    fn measured_max_nn_angle(&self) -> f64 {
        self.inner.measured_max_nn_angle()
    }

    /// Worst-case Voronoi-cell radius across the sphere (radians).
    /// `half_fov_rad = measured_max_coverage_angle * overlap_factor`.
    #[getter]
    fn measured_max_coverage_angle(&self) -> f64 {
        self.inner.measured_max_coverage_angle()
    }

    /// Patch grid size (pixels per side). Uniform across tiles.
    #[getter]
    fn patch_size(&self) -> u32 {
        self.inner.patch_size()
    }

    /// Override the per-tile patch size after construction.
    ///
    /// Tile directions, bases, half-FOV, and the KD-tree are unaffected;
    /// ``tile_camera`` and ``atlas_size`` shift to reflect the new
    /// resolution. Useful for rounding the constructor's
    /// ``arc_per_pixel``-driven ``patch_size`` up to a power of two so the
    /// per-tile patch can serve as an image-pyramid base.
    ///
    /// Args:
    ///     patch_size: New per-tile patch grid size in pixels (must be > 0).
    fn set_patch_size(&mut self, patch_size: u32) -> PyResult<()> {
        if patch_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "patch_size must be > 0",
            ));
        }
        self.inner.set_patch_size(patch_size);
        Ok(())
    }

    /// Number of tile columns in the packed atlas.
    #[getter]
    fn atlas_cols(&self) -> u32 {
        self.inner.atlas_cols()
    }

    /// Number of tile rows in the packed atlas.
    #[getter]
    fn atlas_rows(&self) -> u32 {
        self.inner.atlas_rows()
    }

    /// Atlas image size as ``(width, height)`` in pixels.
    #[getter]
    fn atlas_size(&self) -> (u32, u32) {
        self.inner.atlas_size()
    }

    /// Top-left pixel ``(x, y)`` of tile ``idx`` in the atlas.
    fn tile_atlas_origin(&self, idx: usize) -> PyResult<(u32, u32)> {
        if idx >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "tile index {idx} out of range (n = {})",
                self.inner.len()
            )));
        }
        Ok(self.inner.tile_atlas_origin(idx))
    }

    /// Unit look direction of tile ``idx`` in world frame.
    fn direction(&self, idx: usize) -> PyResult<[f64; 3]> {
        if idx >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "tile index {idx} out of range (n = {})",
                self.inner.len()
            )));
        }
        Ok(self.inner.direction(idx))
    }

    /// All `n` unit look directions as a `(n, 3)` float64 numpy array.
    fn directions<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let n = self.inner.len();
        let mut flat = Vec::with_capacity(n * 3);
        for i in 0..n {
            let d = self.inner.direction(i);
            flat.extend_from_slice(&d);
        }
        let arr = PyArray1::from_vec(py, flat).reshape([n, 3])?;
        Ok(arr.into_any().unbind())
    }

    /// Tangent basis ``(e_right, e_up)`` of tile ``idx`` in world frame.
    fn basis(&self, idx: usize) -> PyResult<([f64; 3], [f64; 3])> {
        if idx >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "tile index {idx} out of range (n = {})",
                self.inner.len()
            )));
        }
        Ok(self.inner.basis(idx))
    }

    /// 3x3 ``R_world_from_tile`` for tile ``idx`` as a row-major (3, 3) float64
    /// numpy array. Columns are ``[e_right | e_up | direction]``.
    fn tile_rotation<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<Py<PyAny>> {
        if idx >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "tile index {idx} out of range (n = {})",
                self.inner.len()
            )));
        }
        // Internal layout is column-major; numpy expects row-major (3,3).
        let cols = self.inner.tile_rotation(idx);
        let rm = vec![
            cols[0], cols[3], cols[6], cols[1], cols[4], cols[7], cols[2], cols[5], cols[8],
        ];
        let arr = PyArray1::from_vec(py, rm).reshape([3, 3])?;
        Ok(arr.into_any().unbind())
    }

    /// Pinhole `CameraIntrinsics` shared by every tile.
    fn tile_camera(&self) -> PyCameraIntrinsics {
        PyCameraIntrinsics {
            inner: self.inner.tile_camera(),
        }
    }

    /// Apply an `Se3Transform` in place: rotates and translates `centre`,
    /// rotates `directions` and `bases`, rebuilds the direction KD-tree.
    /// Scale is **not** consumed (the rig has no metric scale).
    fn apply_transform(&mut self, t: &PySe3Transform) {
        self.inner.apply_transform(&t.inner);
    }

    /// Build a `WarpMap` with the atlas as the **destination** image.
    /// See [`SphericalTileRig::warp_to_atlas_with_rotation`].
    #[pyo3(signature = (src, rot_src_from_world))]
    fn warp_to_atlas_with_rotation(
        &self,
        py: Python<'_>,
        src: &PyCameraIntrinsics,
        rot_src_from_world: &PyRotQuaternion,
    ) -> PyWarpMap {
        let inner = py.detach(|| {
            self.inner
                .warp_to_atlas_with_rotation(&src.inner, &rot_src_from_world.inner)
        });
        PyWarpMap::from_inner(inner)
    }

    /// Build a `WarpMap` with the atlas as the **source** image.
    /// See [`SphericalTileRig::warp_from_atlas_with_rotation`].
    #[pyo3(signature = (dst, rot_world_from_dst))]
    fn warp_from_atlas_with_rotation(
        &self,
        py: Python<'_>,
        dst: &PyCameraIntrinsics,
        rot_world_from_dst: &PyRotQuaternion,
    ) -> PyWarpMap {
        let inner = py.detach(|| {
            self.inner
                .warp_from_atlas_with_rotation(&dst.inner, &rot_world_from_dst.inner)
        });
        PyWarpMap::from_inner(inner)
    }

    /// Resample an atlas image into the destination camera, blending the
    /// ``k`` angularly-nearest tiles per dst pixel by inverse-angular-distance
    /// weights.
    ///
    /// Args:
    ///     atlas: ``(H_atlas, W_atlas)`` or ``(H_atlas, W_atlas, C)`` float32
    ///         numpy array whose ``(H, W)`` matches ``rig.atlas_size``.
    ///     dst: Destination ``CameraIntrinsics``.
    ///     rot_world_from_dst: Rotation that takes a ray from the dst-camera
    ///         frame into the world frame.
    ///     k: Number of nearest tiles to blend (``>= 1``). ``k = 1`` is
    ///         closest-tile sampling (Voronoi seams visible); ``k > 1`` blends
    ///         across seams.
    ///
    /// Returns:
    ///     ``(dst.height, dst.width)`` (or ``(dst.height, dst.width, C)``)
    ///     float32 numpy array.
    #[pyo3(signature = (atlas, dst, rot_world_from_dst, k))]
    fn resample_atlas<'py>(
        &self,
        py: Python<'py>,
        atlas: &Bound<'py, pyo3::types::PyAny>,
        dst: &PyCameraIntrinsics,
        rot_world_from_dst: &PyRotQuaternion,
        k: usize,
    ) -> PyResult<Py<PyAny>> {
        if k < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "k must be >= 1, got {k}"
            )));
        }

        let (data, atlas_w, atlas_h, channels) = extract_atlas_f32(atlas)?;
        let (rig_w, rig_h) = self.inner.atlas_size();
        if atlas_w != rig_w || atlas_h != rig_h {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "atlas shape (H, W) = ({atlas_h}, {atlas_w}) does not match \
                 rig atlas_size (H, W) = ({rig_h}, {rig_w})"
            )));
        }

        let out = py.detach(|| {
            self.inner
                .resample_atlas(&data, channels, &dst.inner, &rot_world_from_dst.inner, k)
        });

        let dst_h = dst.inner.height as usize;
        let dst_w = dst.inner.width as usize;
        let arr = if channels == 1 {
            PyArray1::from_vec(py, out)
                .reshape([dst_h, dst_w])?
                .into_any()
                .unbind()
        } else {
            PyArray1::from_vec(py, out)
                .reshape([dst_h, dst_w, channels])?
                .into_any()
                .unbind()
        };
        Ok(arr)
    }
}

/// Extract an atlas numpy array (HxW or HxWxC, float32) into a flat
/// row-major `Vec<f32>` plus its `(width, height, channels)` shape.
fn extract_atlas_f32(obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<(Vec<f32>, u32, u32, usize)> {
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray3<'_, f32>>() {
        let shape = arr.shape();
        let (h, w, c) = (shape[0] as u32, shape[1] as u32, shape[2]);
        let data: Vec<f32> = match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => arr.as_array().iter().copied().collect(),
        };
        return Ok((data, w, h, c));
    }
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray2<'_, f32>>() {
        let shape = arr.shape();
        let (h, w) = (shape[0] as u32, shape[1] as u32);
        let data: Vec<f32> = match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => arr.as_array().iter().copied().collect(),
        };
        return Ok((data, w, h, 1));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "atlas must be a 2D (HxW) or 3D (HxWxC) numpy float32 array",
    ))
}
