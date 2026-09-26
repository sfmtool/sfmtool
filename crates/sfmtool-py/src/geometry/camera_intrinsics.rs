// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for the sfmtool-core CameraIntrinsics type.

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use sfmtool_core::camera::refit_intrinsics::{
    refit_camera_intrinsics, CameraIntrinsicsRefit, MonotoneConstraint, RefitOptions, RefitTarget,
};
use sfmtool_core::CameraIntrinsics;
use sfmtool_sfmr_format::SfmrCamera;

/// Camera intrinsic parameters with image dimensions.
///
/// Wraps a camera model (e.g. PINHOLE, OPENCV) with width/height and provides
/// access to focal lengths, principal point, intrinsic matrix, and distortion info.
#[pyclass(name = "CameraIntrinsics", module = "sfmtool.geometry", from_py_object)]
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
    fn new(model: &str, width: u32, height: u32, params: BTreeMap<String, f64>) -> PyResult<Self> {
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

    /// Whether sfmtool's bundle adjustment can release this camera's focal
    /// length: true for ``SIMPLE_PINHOLE``, ``EQUIDISTANT_FISHEYE``,
    /// ``SIMPLE_RADIAL_FISHEYE``, ``SFMTOOL_FISHEYE`` and ``SFMTOOL_PINHOLE``.
    #[getter]
    fn focal_is_releasable(&self) -> bool {
        sfmtool_core::reconstruction::bundle_adjust::focal_is_releasable(&self.inner)
    }

    /// Whether sfmtool's bundle adjustment can release this camera's lens
    /// distortion: ``k1`` on ``SIMPLE_RADIAL_FISHEYE``, and the spline on an
    /// ``SFMTOOL_FISHEYE`` or ``SFMTOOL_PINHOLE`` with at least two
    /// coefficients on a positive domain.
    #[getter]
    fn distortion_is_releasable(&self) -> bool {
        sfmtool_core::reconstruction::bundle_adjust::distortion_is_releasable(&self.inner)
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
            .cast()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("'parameters' must be a dict"))?;

        let mut parameters = BTreeMap::new();
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

    /// World-space radius at each camera-frame point that projects to
    /// ``radius_px`` pixels.
    ///
    /// ``radius_px / σ_min``, where ``σ_min`` is the smaller singular value of
    /// the pixel Jacobian ``∂(u, v)/∂p_cam`` at that point — the local
    /// pixels-per-world-unit in the least-magnified tangent direction, so the
    /// result meets the pixel budget however the surface is oriented. One rule
    /// for every camera model; it reduces to ``radius_px·|z|/f`` for a pinhole
    /// and ``radius_px·‖p_cam‖/f`` for ``EQUIDISTANT_FISHEYE`` (finite past
    /// 90°, where ``|z|`` collapses), and picks up the local distortion
    /// magnification for every other model. This is the sizing rule for a patch
    /// anchored to a POSITION; :meth:`pixel_radius_to_angle_batch` is its
    /// range-free sibling for a patch anchored to a direction.
    ///
    /// Points are in CANONICAL camera space (``-Z`` forward), the same frame
    /// :meth:`ray_to_pixel_batch` takes.
    ///
    /// Args:
    ///     points: Nx3 numpy array of camera-space points.
    ///     radius_px: Pixel radius, either a scalar or an N-vector.
    ///
    /// Returns:
    ///     Length-N numpy array of world-space radii.
    fn pixel_radius_to_world_batch<'py>(
        &self,
        py: Python<'py>,
        points: numpy::PyReadonlyArray2<'py, f64>,
        radius_px: numpy::PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let arr = points.as_array();
        if arr.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "points must have shape (N, 3)",
            ));
        }
        let radii = radius_px.as_array();
        if radii.len() != 1 && radii.len() != arr.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "radius_px must be a scalar or have length {} (parallel to points), got {}",
                arr.nrows(),
                radii.len()
            )));
        }
        let out: Vec<f64> = arr
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                let px = if radii.len() == 1 { radii[0] } else { radii[i] };
                self.inner.pixel_radius_to_world([p[0], p[1], p[2]], px)
            })
            .collect();
        Ok(numpy::PyArray1::from_vec(py, out))
    }

    /// Angular radius (radians) around each bearing that projects to
    /// ``radius_px`` pixels.
    ///
    /// ``radius_px / (‖ray‖·σ_min)``, where ``σ_min`` is the smaller singular
    /// value of the pixel Jacobian ``∂(u, v)/∂p_cam`` — the local pixels per
    /// radian in the least-magnified tangent direction. Only the DIRECTION of
    /// each ray matters (``σ_min`` goes as ``1/‖p_cam‖``, so the range cancels).
    /// This is the sizing rule for a patch anchored to a direction rather than a
    /// position, i.e. a point at infinity. For a pinhole it is
    /// ``radius_px·cos θ/f``; for ``EQUIDISTANT_FISHEYE`` it is ``radius_px/f``
    /// at every θ.
    ///
    /// Args:
    ///     rays: Nx3 numpy array of bearings in camera space (need not be unit).
    ///     radius_px: Pixel radius, either a scalar or an N-vector.
    ///
    /// Returns:
    ///     Length-N numpy array of angular radii in radians.
    fn pixel_radius_to_angle_batch<'py>(
        &self,
        py: Python<'py>,
        rays: numpy::PyReadonlyArray2<'py, f64>,
        radius_px: numpy::PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let arr = rays.as_array();
        if arr.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rays must have shape (N, 3)",
            ));
        }
        let radii = radius_px.as_array();
        if radii.len() != 1 && radii.len() != arr.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "radius_px must be a scalar or have length {} (parallel to rays), got {}",
                arr.nrows(),
                radii.len()
            )));
        }
        let out: Vec<f64> = arr
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                let px = if radii.len() == 1 { radii[0] } else { radii[i] };
                self.inner.pixel_radius_to_angle([r[0], r[1], r[2]], px)
            })
            .collect();
        Ok(numpy::PyArray1::from_vec(py, out))
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

    /// Build a pinhole camera whose FoV is the largest that maps every
    /// destination pixel to a valid location in this camera (no black borders).
    ///
    /// Args:
    ///     width: Output image width.
    ///     height: Output image height.
    ///
    /// Returns:
    ///     A PINHOLE CameraIntrinsics with the best-fit focal length.
    ///
    /// Raises:
    ///     ValueError: If this camera is fisheye or equirectangular.
    #[pyo3(signature = (width, height))]
    fn best_fit_inside_pinhole(&self, width: u32, height: u32) -> PyResult<PyCameraIntrinsics> {
        let inner = self
            .inner
            .best_fit_inside_pinhole(width, height)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyCameraIntrinsics { inner })
    }

    /// Build a pinhole camera whose FoV is the smallest that covers every
    /// pixel in this camera (no cropping, may have black borders).
    ///
    /// Args:
    ///     width: Output image width.
    ///     height: Output image height.
    ///
    /// Returns:
    ///     A PINHOLE CameraIntrinsics with the best-fit focal length.
    ///
    /// Raises:
    ///     ValueError: If this camera is fisheye or equirectangular.
    #[pyo3(signature = (width, height))]
    fn best_fit_outside_pinhole(&self, width: u32, height: u32) -> PyResult<PyCameraIntrinsics> {
        let inner = self
            .inner
            .best_fit_outside_pinhole(width, height)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyCameraIntrinsics { inner })
    }

    /// Fit a camera of another model to this one.
    ///
    /// Rays are sampled over the angles where this camera is trusted and 64
    /// azimuths, projected with this camera, and the target's parameters chosen
    /// to put every ray as close as they can to the same pixel (see
    /// ``specs/core/camera/refit-camera-intrinsics.md``). The principal point and image size are
    /// copied.
    ///
    /// Args:
    ///     camera_model: The model name, case-insensitive: ``SFMTOOL_FISHEYE``,
    ///         ``SFMTOOL_PINHOLE``, ``EQUIDISTANT_FISHEYE`` or a COLMAP lens
    ///         model.
    ///     coeff_count: Spline coefficients for the two spline models.
    ///         Default: this camera's own count when it already has this
    ///         model's spline, otherwise 8. Refused for any other model.
    ///     theta_fit_deg: The largest incidence angle sampled. Default: this
    ///         camera's trusted bound, or its far image corner for a model with
    ///         none. A value past the trusted bound is refused.
    ///     spline_domain_deg: Where a spline target's domain ends, as an
    ///         incidence angle. Default: the far image corner.
    ///
    /// Returns:
    ///     ``(CameraIntrinsics, report)``. The report carries ``camera_model``,
    ///     ``theta_fit_deg``, ``theta_fit_source`` (``"trusted_bound"``,
    ///     ``"observations"``, ``"image_corner"`` or ``"given"``),
    ///     ``spline_domain_deg`` (``None`` for a non-spline target),
    ///     ``rms_px``, ``max_px``, ``radial_rms_px``, ``dropped`` (one
    ///     sentence per term the target cannot represent) and ``extent``:
    ///     ``edge_deg``, ``corner_deg``, ``source_trusted_deg`` and
    ///     ``source_fold_deg``, and ``monotone_constraint``: ``active``,
    ///     ``active_angles`` and ``range_deg`` (the ``(from, to)`` incidence
    ///     angles where the fit held the lens's slope at its floor to keep it
    ///     invertible, departing from this camera there, or ``None``). Raises
    ///     ``ValueError`` naming the rule and the value when the fit is refused.
    #[pyo3(signature = (camera_model, *, coeff_count=None, theta_fit_deg=None, spline_domain_deg=None))]
    fn refit<'py>(
        &self,
        py: Python<'py>,
        camera_model: &str,
        coeff_count: Option<usize>,
        theta_fit_deg: Option<f64>,
        spline_domain_deg: Option<f64>,
    ) -> PyResult<(PyCameraIntrinsics, Bound<'py, PyDict>)> {
        let target = RefitTarget::from_name(camera_model, coeff_count)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let options = RefitOptions {
            theta_fit_deg,
            spline_domain_deg,
        };
        let refit = py
            .detach(|| refit_camera_intrinsics(&self.inner, &target, &options))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let report = refit_report_to_py(py, &refit)?;
        Ok((
            PyCameraIntrinsics {
                inner: refit.camera,
            },
            report,
        ))
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
    ) -> PyResult<(
        Bound<'py, PyType>,
        (String, u32, u32, BTreeMap<String, f64>),
    )> {
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

/// A [`CameraIntrinsicsRefit`] as the dict the Python surface reports it as, the camera
/// itself left out (the caller returns it beside the dict).
pub(crate) fn refit_report_to_py<'py>(
    py: Python<'py>,
    refit: &CameraIntrinsicsRefit,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("camera_model", refit.camera.model_name())?;
    d.set_item("theta_fit_deg", refit.theta_fit_deg)?;
    d.set_item("theta_fit_source", refit.theta_fit_source.as_str())?;
    d.set_item("spline_domain_deg", refit.spline_domain_deg)?;
    d.set_item("rms_px", refit.rms_px)?;
    d.set_item("max_px", refit.max_px)?;
    d.set_item("radial_rms_px", refit.radial_rms_px)?;
    let dropped: Vec<String> = refit.dropped.iter().map(|t| t.to_string()).collect();
    d.set_item("dropped", dropped)?;
    let extent = PyDict::new(py);
    extent.set_item("edge_deg", refit.extent.edge_deg)?;
    extent.set_item("corner_deg", refit.extent.corner_deg)?;
    extent.set_item("source_trusted_deg", refit.extent.source_trusted_deg)?;
    extent.set_item("source_fold_deg", refit.extent.source_fold_deg)?;
    d.set_item("extent", extent)?;
    d.set_item(
        "monotone_constraint",
        monotone_constraint_to_py(py, &refit.monotone_constraint)?,
    )?;
    Ok(d)
}

/// A [`MonotoneConstraint`] as the dict the Python surface reports it as:
/// `active`, `active_angles` and `range_deg`, the `(from, to)` incidence angles
/// in degrees where the slope floor bound, or `None`.
pub(crate) fn monotone_constraint_to_py<'py>(
    py: Python<'py>,
    constraint: &MonotoneConstraint,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("active", constraint.active)?;
    d.set_item("active_angles", constraint.active_angles)?;
    d.set_item("range_deg", constraint.range_deg.map(|[a, b]| (a, b)))?;
    Ok(d)
}
