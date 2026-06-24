// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for the sfmtool-core WarpMap type.

use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use sfmtool_core::camera::remap::{
    remap_aniso, remap_aniso_with_pyramid, remap_bilinear, ImageU8, ImageU8Pyramid,
};
use sfmtool_core::camera::WarpMap;

use crate::geometry::rigid_transform::PyRigidTransform;
use crate::geometry::rot_quaternion::PyRotQuaternion;
use crate::py_patch_cloud::PyOrientedPatch;
use crate::PyCameraIntrinsics;

/// A dense pixel coordinate map for image warping.
///
/// For each pixel (col, row) in the destination image, stores the (x, y)
/// coordinates in the source image to sample from.
///
/// Example::
///
///     camera = reconstruction.cameras[0]
///     pinhole = CameraIntrinsics("PINHOLE", camera.width, camera.height, {
///         "focal_length_x": camera.focal_lengths[0],
///         "focal_length_y": camera.focal_lengths[1],
///         "principal_point_x": camera.principal_point[0],
///         "principal_point_y": camera.principal_point[1],
///     })
///     warp = WarpMap.from_cameras(src=camera, dst=pinhole)
///     undistorted = warp.remap_bilinear(image)
#[pyclass(name = "WarpMap", module = "sfmtool.flow")]
pub struct PyWarpMap {
    pub(crate) inner: WarpMap,
}

impl PyWarpMap {
    /// Wrap a `WarpMap` into the Python type. For use by other PyO3 modules
    /// in this crate that build a `WarpMap` and need to expose it.
    pub(crate) fn from_inner(inner: WarpMap) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyWarpMap {
    /// Build a warp map from source and destination camera intrinsics.
    ///
    /// For each pixel in the destination image, computes where to sample
    /// from in the source image. To undistort, pass the distorted camera
    /// as `src` and a pinhole camera as `dst`.
    ///
    /// Args:
    ///     src: Source (input) camera intrinsics.
    ///     dst: Destination (output) camera intrinsics.
    ///
    /// Returns:
    ///     A WarpMap for use with remap_bilinear or remap_aniso.
    #[staticmethod]
    #[pyo3(signature = (src, dst))]
    fn from_cameras(src: &PyCameraIntrinsics, dst: &PyCameraIntrinsics) -> Self {
        let inner = WarpMap::from_cameras(&src.inner, &dst.inner);
        Self { inner }
    }

    /// Build a warp map that assumes the scene is infinitely far away.
    ///
    /// For each dst pixel center, rotates the corresponding dst ray into
    /// src-camera coordinates (via ``rot_src_from_dst``) and projects. Only
    /// the relative rotation between the two cameras affects the result;
    /// any translation between them is ignored.
    ///
    /// Passing an identity quaternion is equivalent to ``from_cameras``.
    ///
    /// Args:
    ///     src: Source (input) camera intrinsics.
    ///     dst: Destination (output) camera intrinsics.
    ///     rot_src_from_dst: Rotation that takes a ray from the dst-camera
    ///         frame into the src-camera frame.
    #[staticmethod]
    #[pyo3(signature = (src, dst, rot_src_from_dst))]
    fn from_cameras_with_rotation(
        src: &PyCameraIntrinsics,
        dst: &PyCameraIntrinsics,
        rot_src_from_dst: &PyRotQuaternion,
    ) -> Self {
        let inner =
            WarpMap::from_cameras_with_rotation(&src.inner, &dst.inner, &rot_src_from_dst.inner);
        Self { inner }
    }

    /// Build a warp map under the assumption that every dst ray hits a
    /// scene point at radial distance ``depth`` from the dst camera centre.
    ///
    /// ``src_from_world`` and ``dst_from_world`` are world-to-camera
    /// extrinsics, matching ``SfmrReconstruction.quaternions_wxyz`` and
    /// ``SfmrReconstruction.translations`` conventions (i.e. they can be
    /// constructed via ``RigidTransform.from_wxyz_translation(q, t)``).
    ///
    /// ``depth`` is radial distance (sphere radius), not z-depth, so the
    /// method also works with equirectangular or fisheye destination
    /// cameras. Passing ``float('inf')`` short-circuits to the
    /// rotation-only path (the only pose component that still matters is
    /// the relative rotation ``R_src * R_dst^T``).
    ///
    /// Args:
    ///     src: Source (input) camera intrinsics.
    ///     dst: Destination (output) camera intrinsics.
    ///     src_from_world: World-to-src-camera pose.
    ///     dst_from_world: World-to-dst-camera pose.
    ///     depth: Scene radius in world units (may be ``float('inf')``).
    #[staticmethod]
    #[pyo3(signature = (src, dst, src_from_world, dst_from_world, depth))]
    fn from_cameras_with_pose(
        src: &PyCameraIntrinsics,
        dst: &PyCameraIntrinsics,
        src_from_world: &PyRigidTransform,
        dst_from_world: &PyRigidTransform,
        depth: f64,
    ) -> Self {
        let inner = WarpMap::from_cameras_with_pose(
            &src.inner,
            &dst.inner,
            &src_from_world.inner,
            &dst_from_world.inner,
            depth,
        );
        Self { inner }
    }

    /// Build a warp map sampling a camera's image over an oriented 3D patch.
    ///
    /// The destination is the patch's ``resolution × resolution`` canonical grid;
    /// each entry is the source-image ``(x, y)`` where that patch pixel projects
    /// (NaN if behind the camera, outside the model domain, or out of bounds).
    /// Then ``remap_bilinear`` / ``remap_aniso`` render the patch's appearance.
    /// See ``specs/core/patch-cloud.md``.
    ///
    /// Args:
    ///     patch: The OrientedPatch in world coordinates.
    ///     camera: Source camera intrinsics.
    ///     cam_from_world: World-to-camera pose of the source camera.
    ///     resolution: Patch grid side length in pixels.
    #[staticmethod]
    #[pyo3(signature = (patch, camera, cam_from_world, resolution))]
    fn from_patch(
        patch: &PyOrientedPatch,
        camera: &PyCameraIntrinsics,
        cam_from_world: &PyRigidTransform,
        resolution: u32,
    ) -> Self {
        Self {
            inner: WarpMap::from_patch(
                &patch.inner,
                &camera.inner,
                &cam_from_world.inner,
                resolution,
            ),
        }
    }

    /// Build a warp map directly from per-pixel source-coordinate maps.
    ///
    /// The inverse of :meth:`to_numpy`. ``map_x`` and ``map_y`` are
    /// ``(height, width)`` float32 arrays giving, for each destination pixel,
    /// the source-image ``(x, y)`` to sample from. ``(NaN, NaN)`` marks an
    /// invalid / out-of-bounds pixel. Compatible with ``cv2.remap`` maps.
    ///
    /// Args:
    ///     map_x: (H, W) float32 array of source x coordinates.
    ///     map_y: (H, W) float32 array of source y coordinates.
    ///
    /// Returns:
    ///     A WarpMap for use with remap_bilinear or remap_aniso.
    #[staticmethod]
    #[pyo3(signature = (map_x, map_y))]
    fn from_numpy(
        map_x: numpy::PyReadonlyArray2<'_, f32>,
        map_y: numpy::PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Self> {
        let shape_x = map_x.shape();
        if shape_x != map_y.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "map_x and map_y must have the same shape",
            ));
        }
        let (h, w) = (shape_x[0], shape_x[1]);
        let ax = map_x.as_array();
        let ay = map_y.as_array();
        let mut data = Vec::with_capacity(2 * w * h);
        for row in 0..h {
            for col in 0..w {
                data.push(ax[[row, col]]);
                data.push(ay[[row, col]]);
            }
        }
        Ok(Self {
            inner: WarpMap::new(w as u32, h as u32, data),
        })
    }

    /// Width of the destination image (and this map).
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    /// Height of the destination image (and this map).
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }

    /// Return the map data as two numpy arrays (map_x, map_y).
    ///
    /// Each array has shape (height, width) with dtype float32.
    /// Compatible with ``cv2.remap(image, map_x, map_y, ...)``.
    ///
    /// Returns:
    ///     Tuple of (map_x, map_y) numpy arrays.
    #[allow(clippy::type_complexity)]
    fn to_numpy<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray2<f32>>,
        Bound<'py, numpy::PyArray2<f32>>,
    )> {
        let w = self.inner.width() as usize;
        let h = self.inner.height() as usize;
        let mut map_x = Vec::with_capacity(w * h);
        let mut map_y = Vec::with_capacity(w * h);
        for row in 0..h {
            for col in 0..w {
                let (x, y) = self.inner.get(col as u32, row as u32);
                map_x.push(x);
                map_y.push(y);
            }
        }
        let arr_x: Bound<'py, numpy::PyArray2<f32>> =
            numpy::PyArray1::from_vec(py, map_x).reshape([h, w])?;
        let arr_y: Bound<'py, numpy::PyArray2<f32>> =
            numpy::PyArray1::from_vec(py, map_y).reshape([h, w])?;
        Ok((arr_x, arr_y))
    }

    /// Apply the warp map to an image using bilinear interpolation.
    ///
    /// Args:
    ///     image: Input image as HxWxC or HxW numpy uint8 array.
    ///
    /// Returns:
    ///     Warped image as numpy uint8 array with the same number of channels.
    #[pyo3(signature = (image))]
    fn remap_bilinear<'py>(
        &self,
        py: Python<'py>,
        image: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let src = extract_image_u8(image)?;
        let result = remap_bilinear(&src, &self.inner);
        image_u8_to_pyobject(py, &result)
    }

    /// Apply the warp map with anisotropic filtering.
    ///
    /// Builds a Gaussian pyramid and takes multiple samples along the
    /// major axis of compression to reduce aliasing. Call this instead
    /// of remap_bilinear when undistorting fisheye images.
    ///
    /// Args:
    ///     image: Input image as HxWxC or HxW numpy uint8 array.
    ///     max_anisotropy: Maximum samples along the major axis (default 16).
    ///
    /// Returns:
    ///     Warped image as numpy uint8 array.
    #[pyo3(signature = (image, max_anisotropy = 16))]
    fn remap_aniso<'py>(
        &mut self,
        py: Python<'py>,
        image: &Bound<'py, pyo3::types::PyAny>,
        max_anisotropy: u32,
    ) -> PyResult<Py<PyAny>> {
        let src = extract_image_u8(image)?;
        if !self.inner.has_svd() {
            self.inner.compute_svd();
        }
        let result = remap_aniso(&src, &self.inner, max_anisotropy);
        image_u8_to_pyobject(py, &result)
    }
}

/// A Gaussian image pyramid for repeated anisotropic resampling.
///
/// Build it once from a source image, then resample many warp maps through it
/// with :meth:`remap_aniso`. This avoids rebuilding the pyramid (an
/// ``O(image)`` cost) on every warp, which matters when warping one source
/// image through many small maps — e.g. per-keypoint patches.
///
/// Example::
///
///     pyr = ImagePyramid(image)
///     for warp in keypoint_warps:
///         patch = pyr.remap_aniso(warp)
#[pyclass(name = "ImagePyramid", module = "sfmtool.flow")]
pub struct PyImagePyramid {
    inner: ImageU8Pyramid,
}

#[pymethods]
impl PyImagePyramid {
    /// Build a Gaussian pyramid from an image.
    ///
    /// Args:
    ///     image: Input image as HxWxC or HxW numpy uint8 array.
    #[new]
    #[pyo3(signature = (image))]
    fn new(image: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        let src = extract_image_u8(image)?;
        let min_dim = src.width().min(src.height()).max(1);
        let max_levels = ((min_dim as f32).log2().floor() as usize).max(1) + 1;
        Ok(Self {
            inner: ImageU8Pyramid::build(&src, max_levels),
        })
    }

    /// Number of pyramid levels (level 0 is full resolution).
    #[getter]
    fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }

    /// Anisotropically resample a warp map through this prebuilt pyramid.
    ///
    /// Equivalent to :meth:`WarpMap.remap_aniso` but reuses the pyramid instead
    /// of rebuilding it. Computes the warp map's Jacobian SVD if not already
    /// present.
    ///
    /// Args:
    ///     warp_map: The WarpMap to resample through.
    ///     max_anisotropy: Maximum samples along the major axis (default 16).
    ///
    /// Returns:
    ///     Warped image as numpy uint8 array.
    #[pyo3(signature = (warp_map, max_anisotropy = 16))]
    fn remap_aniso<'py>(
        &self,
        py: Python<'py>,
        warp_map: &mut PyWarpMap,
        max_anisotropy: u32,
    ) -> PyResult<Py<PyAny>> {
        if !warp_map.inner.has_svd() {
            warp_map.inner.compute_svd();
        }
        let result = remap_aniso_with_pyramid(&self.inner, &warp_map.inner, max_anisotropy);
        image_u8_to_pyobject(py, &result)
    }
}

/// Extract a numpy uint8 array (HxW or HxWxC) into an ImageU8.
pub(crate) fn extract_image_u8(obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<ImageU8> {
    // Try 3D first (HxWxC), then 2D (HxW)
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

/// Convert an ImageU8 to a numpy Py<PyAny> (HxWxC or HxW for single channel).
fn image_u8_to_pyobject(py: Python<'_>, img: &ImageU8) -> PyResult<Py<PyAny>> {
    let data = img.data().to_vec();
    if img.channels() == 1 {
        let arr: Bound<'_, numpy::PyArray2<u8>> = numpy::PyArray1::from_vec(py, data)
            .reshape([img.height() as usize, img.width() as usize])?;
        Ok(arr.into_any().unbind())
    } else {
        let arr: Bound<'_, numpy::PyArray3<u8>> = numpy::PyArray1::from_vec(py, data).reshape([
            img.height() as usize,
            img.width() as usize,
            img.channels() as usize,
        ])?;
        Ok(arr.into_any().unbind())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWarpMap>()?;
    m.add_class::<PyImagePyramid>()?;
    Ok(())
}
