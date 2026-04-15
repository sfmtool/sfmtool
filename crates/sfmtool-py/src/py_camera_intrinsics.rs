// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for the sfmtool-core CameraIntrinsics type.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use sfmr_format::SfmrCamera;
use sfmtool_core::CameraIntrinsics;

/// Camera intrinsic parameters with image dimensions.
///
/// Wraps a camera model (e.g. PINHOLE, OPENCV) with width/height and provides
/// access to focal lengths, principal point, intrinsic matrix, and distortion info.
#[pyclass(name = "CameraIntrinsics", module = "sfmtool._core")]
#[derive(Clone)]
pub struct PyCameraIntrinsics {
    pub(crate) inner: CameraIntrinsics,
}

#[pymethods]
impl PyCameraIntrinsics {
    /// Create a new CameraIntrinsics from model name, dimensions, and parameter dict.
    ///
    /// Args:
    ///     model: COLMAP model name (e.g. "PINHOLE", "OPENCV", "SIMPLE_RADIAL")
    ///     width: Image width in pixels
    ///     height: Image height in pixels
    ///     params: Dict mapping parameter names to float values
    #[new]
    fn new(model: &str, width: u32, height: u32, params: HashMap<String, f64>) -> PyResult<Self> {
        let sfmr_camera = SfmrCamera {
            model: model.to_string(),
            width,
            height,
            parameters: params,
        };
        let inner = CameraIntrinsics::try_from(&sfmr_camera)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// The COLMAP model name (e.g. "PINHOLE", "OPENCV").
    #[getter]
    fn model(&self) -> &str {
        self.inner.model_name()
    }

    /// Image width in pixels.
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }

    /// Image height in pixels.
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }

    /// Focal lengths as (fx, fy). For single-focal models, fx == fy.
    #[getter]
    fn focal_lengths(&self) -> (f64, f64) {
        self.inner.focal_lengths()
    }

    /// Principal point as (cx, cy).
    #[getter]
    fn principal_point(&self) -> (f64, f64) {
        self.inner.principal_point()
    }

    /// Whether this camera model includes distortion parameters.
    #[getter]
    fn has_distortion(&self) -> bool {
        self.inner.has_distortion()
    }

    /// Return the 3x3 intrinsic matrix K as a numpy array.
    fn intrinsic_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let mat = self.inner.intrinsic_matrix();
        let data: Vec<Vec<f64>> = (0..3)
            .map(|r| (0..3).map(|c| mat[(r, c)]).collect())
            .collect();
        numpy::PyArray2::from_vec2(py, &data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Model-specific parameters as a dict mapping parameter names to float values.
    ///
    /// Keys depend on the camera model (e.g. "focal_length_x", "radial_distortion_k1").
    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let sfmr_camera = SfmrCamera::from(&self.inner);
        let params_dict = PyDict::new(py);
        for (key, value) in &sfmr_camera.parameters {
            params_dict.set_item(key, value)?;
        }
        Ok(params_dict)
    }

    /// Convert to a dictionary with keys "model", "width", "height", "parameters".
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("model", self.inner.model_name())?;
        dict.set_item("width", self.inner.width)?;
        dict.set_item("height", self.inner.height)?;
        dict.set_item("parameters", self.parameters(py)?)?;
        Ok(dict)
    }

    /// Create a CameraIntrinsics from a dictionary.
    ///
    /// Args:
    ///     d: Dict with keys "model", "width", "height", "parameters"
    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let model: String = d
            .get_item("model")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'model' key"))?
            .extract()?;

        let width: u32 = d
            .get_item("width")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'width' key"))?
            .extract()?;

        let height: u32 = d
            .get_item("height")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'height' key"))?
            .extract()?;

        let params_obj = d
            .get_item("parameters")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'parameters' key"))?;
        let params_dict: &Bound<'_, PyDict> = params_obj
            .downcast()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("'parameters' must be a dict"))?;

        let mut parameters = HashMap::new();
        for (key, value) in params_dict.iter() {
            let k: String = key.extract()?;
            let v: f64 = value.extract()?;
            parameters.insert(k, v);
        }

        let sfmr_camera = SfmrCamera {
            model,
            width,
            height,
            parameters,
        };
        let inner = CameraIntrinsics::try_from(&sfmr_camera)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Project an undistorted image-plane point to pixel coordinates.
    ///
    /// Applies the camera's distortion model then converts to pixels:
    /// (x, y) -> distort -> (u, v) where u = fx * x_d + cx.
    ///
    /// Args:
    ///     x: Undistorted image-plane x coordinate
    ///     y: Undistorted image-plane y coordinate
    ///
    /// Returns:
    ///     Tuple (u, v) in pixel coordinates.
    fn project(&self, x: f64, y: f64) -> (f64, f64) {
        self.inner.project(x, y)
    }

    /// Unproject pixel coordinates to undistorted image-plane coordinates.
    ///
    /// Converts pixel to distorted image-plane, then removes distortion:
    /// (u, v) -> (x_d, y_d) -> undistort -> (x, y).
    ///
    /// The returned (x, y) can be used as a ray direction (x, y, 1).
    ///
    /// Args:
    ///     u: Pixel x coordinate
    ///     v: Pixel y coordinate
    ///
    /// Returns:
    ///     Tuple (x, y) in undistorted image-plane coordinates.
    fn unproject(&self, u: f64, v: f64) -> (f64, f64) {
        self.inner.unproject(u, v)
    }

    /// Project a batch of undistorted image-plane points to pixel coordinates.
    ///
    /// Args:
    ///     points: Nx2 numpy array of (x, y) image-plane coordinates.
    ///
    /// Returns:
    ///     Nx2 numpy array of (u, v) pixel coordinates.
    fn project_batch<'py>(
        &self,
        py: Python<'py>,
        points: numpy::PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let arr = points.as_array();
        if arr.ncols() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "points must have shape (N, 2)",
            ));
        }
        let input: Vec<[f64; 2]> = arr.rows().into_iter().map(|r| [r[0], r[1]]).collect();
        let output = self.inner.project_batch(&input);
        let flat: Vec<f64> = output.iter().flat_map(|[u, v]| [*u, *v]).collect();
        numpy::PyArray2::from_vec2(
            py,
            &flat.chunks(2).map(|c| vec![c[0], c[1]]).collect::<Vec<_>>(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Unproject a batch of pixel coordinates to undistorted image-plane coordinates.
    ///
    /// Args:
    ///     pixels: Nx2 numpy array of (u, v) pixel coordinates.
    ///
    /// Returns:
    ///     Nx2 numpy array of (x, y) undistorted image-plane coordinates.
    fn unproject_batch<'py>(
        &self,
        py: Python<'py>,
        pixels: numpy::PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let arr = pixels.as_array();
        if arr.ncols() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pixels must have shape (N, 2)",
            ));
        }
        let input: Vec<[f64; 2]> = arr.rows().into_iter().map(|r| [r[0], r[1]]).collect();
        let output = self.inner.unproject_batch(&input);
        let flat: Vec<f64> = output.iter().flat_map(|[x, y]| [*x, *y]).collect();
        numpy::PyArray2::from_vec2(
            py,
            &flat.chunks(2).map(|c| vec![c[0], c[1]]).collect::<Vec<_>>(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Project a unit ray direction in camera space to pixel coordinates.
    ///
    /// For perspective models, equivalent to project(rx/rz, ry/rz).
    /// For fisheye models, computes directly from the incidence angle,
    /// avoiding the tan(theta) singularity. For equirectangular, maps
    /// via longitude/latitude.
    ///
    /// Args:
    ///     ray: List or array [rx, ry, rz] unit ray direction.
    ///
    /// Returns:
    ///     Tuple (u, v) in pixel coordinates, or None if the ray is
    ///     outside the model's valid domain.
    fn ray_to_pixel(&self, ray: [f64; 3]) -> Option<(f64, f64)> {
        self.inner.ray_to_pixel(ray)
    }

    /// Batch version of ray_to_pixel.
    ///
    /// Args:
    ///     rays: Nx3 numpy array of unit ray directions.
    ///
    /// Returns:
    ///     Nx2 numpy array of pixel coordinates (NaN for invalid rays).
    fn ray_to_pixel_batch<'py>(
        &self,
        py: Python<'py>,
        rays: numpy::PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let arr = rays.as_array();
        if arr.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rays must have shape (N, 3)",
            ));
        }
        let input: Vec<[f64; 3]> = arr.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
        let output = self.inner.ray_to_pixel_batch(&input);
        let rows: Vec<Vec<f64>> = output
            .iter()
            .map(|opt| match opt {
                Some([u, v]) => vec![*u, *v],
                None => vec![f64::NAN, f64::NAN],
            })
            .collect();
        numpy::PyArray2::from_vec2(py, &rows)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Convert pixel coordinates to a unit ray direction in camera space.
    ///
    /// For perspective models, equivalent to normalizing (unproject(u, v), 1).
    /// For fisheye models, computes the ray directly from the incidence angle,
    /// correctly handling field of view at and beyond 180°.
    ///
    /// Args:
    ///     u: Pixel x coordinate
    ///     v: Pixel y coordinate
    ///
    /// Returns:
    ///     List [rx, ry, rz] unit ray direction in camera space.
    fn pixel_to_ray(&self, u: f64, v: f64) -> [f64; 3] {
        self.inner.pixel_to_ray(u, v)
    }

    /// Convert a batch of pixel coordinates to unit ray directions.
    ///
    /// Args:
    ///     pixels: Nx2 numpy array of (u, v) pixel coordinates.
    ///
    /// Returns:
    ///     Nx3 numpy array of unit ray directions in camera space.
    fn pixel_to_ray_batch<'py>(
        &self,
        py: Python<'py>,
        pixels: numpy::PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let arr = pixels.as_array();
        if arr.ncols() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pixels must have shape (N, 2)",
            ));
        }
        let input: Vec<[f64; 2]> = arr.rows().into_iter().map(|r| [r[0], r[1]]).collect();
        let output = self.inner.pixel_to_ray_batch(&input);
        let rows: Vec<Vec<f64>> = output.iter().map(|[x, y, z]| vec![*x, *y, *z]).collect();
        numpy::PyArray2::from_vec2(py, &rows)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "CameraIntrinsics(model={}, width={}, height={})",
            self.inner.model_name(),
            self.inner.width,
            self.inner.height,
        )
    }

    fn __eq__(&self, other: &PyCameraIntrinsics) -> bool {
        self.inner == other.inner
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, pyo3::types::PyAny>) -> Self {
        self.clone()
    }

    #[allow(clippy::type_complexity)]
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyType>, (String, u32, u32, HashMap<String, f64>))> {
        let sfmr_camera = SfmrCamera::from(&self.inner);
        let cls = PyType::new::<PyCameraIntrinsics>(py);
        Ok((
            cls,
            (
                sfmr_camera.model,
                sfmr_camera.width,
                sfmr_camera.height,
                sfmr_camera.parameters,
            ),
        ))
    }
}
