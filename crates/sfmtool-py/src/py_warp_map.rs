// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for the sfmtool-core WarpMap type.

use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use sfmtool_core::remap::{remap_aniso, remap_bilinear, ImageU8};
use sfmtool_core::warp_map::WarpMap;

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
#[pyclass(name = "WarpMap", module = "sfmtool._core")]
pub struct PyWarpMap {
    inner: WarpMap,
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

/// Extract a numpy uint8 array (HxW or HxWxC) into an ImageU8.
fn extract_image_u8(obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<ImageU8> {
    // Try 3D first (HxWxC), then 2D (HxW)
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray3<'_, u8>>() {
        let shape = arr.shape();
        let (h, w, c) = (shape[0] as u32, shape[1] as u32, shape[2] as u32);
        if c != 1 && c != 3 && c != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "image must have 1, 3, or 4 channels",
            ));
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
