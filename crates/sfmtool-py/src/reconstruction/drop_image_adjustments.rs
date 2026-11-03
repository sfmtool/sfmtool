// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `SfmrReconstruction.drop_image_adjustments`: dropping per-image parameters
//! from one camera, with their mean folded into the camera (see
//! ``specs/core/reconstruction/drop-image-adjustments.md``), and
//! `SfmrReconstruction.varied_image_adjustments`: the per-image parameters a
//! camera varies, which are the ones a drop changes something for.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sfmtool_core::reconstruction::bundle_adjust::ImageAdjustmentParameter;
use sfmtool_core::reconstruction::drop_image_adjustments::drop_image_adjustments as core_drop;
use sfmtool_core::reconstruction::switch_camera_model::{
    varied_image_adjustments as core_varied, SwitchCameraModelError,
};

use super::sfmr_reconstruction::PySfmrReconstruction;

/// The parameter a camera's `per_image_adjustments` names `name`.
fn parameter_from_name(name: &str) -> PyResult<ImageAdjustmentParameter> {
    use ImageAdjustmentParameter::*;
    [Focal, PrincipalPoint, RadialBspline, RollingShutter]
        .into_iter()
        .find(|p| p.name() == name)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "drop_image_adjustments(): {name:?} is not one of \"focal\", \
                 \"principal_point\", \"radial_bspline\" and \"rolling_shutter\""
            ))
        })
}

#[pymethods]
impl PySfmrReconstruction {
    /// Drop per-image parameters from one camera, folding their mean into the
    /// camera.
    ///
    /// The camera stops varying each parameter named, every image of the
    /// camera has that part of its adjustment set to zero, and the mean over
    /// the camera's posed images (all its images where none is posed) is
    /// folded into the camera: the focal is multiplied by the geometric mean
    /// of ``1 + s``, ``exp(mean(log1p(s)))``, the value the bundle adjustment
    /// re-centres it to, the principal point moved by the mean offset, and the
    /// mean
    /// radial correction added to the spline. Rolling-shutter rates are
    /// dropped with nothing folded, and the camera keeps its scanlines
    /// setting. A parameter the camera does not vary, with no non-zero value
    /// on its images, is left as it is. Poses and points are not changed; the
    /// stored point errors are rescored from the inline keypoints where the
    /// reconstruction carries them. See
    /// ``specs/core/reconstruction/drop-image-adjustments.md``.
    ///
    /// Args:
    ///     camera: The camera's index.
    ///     parameters: The names to drop, each one of ``"focal"``,
    ///         ``"principal_point"``, ``"radial_bspline"`` and
    ///         ``"rolling_shutter"``, the names ``per_image_adjustments``
    ///         lists.
    ///
    /// Returns:
    ///     A new ``SfmrReconstruction``; this object is not changed. Raises
    ///     ``ValueError`` for an unknown name or camera, or where the folded
    ///     camera is not a valid lens (a spline that is not monotone, a focal
    ///     that is not positive).
    // This is a Python docstring (rendered by `help()`), not Rust prose: its
    // indented `Args:` / `Returns:` continuation paragraphs read as Markdown
    // indented code blocks, which rustdoc then tries to parse as Rust.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[pyo3(signature = (camera, parameters))]
    fn drop_image_adjustments(
        &self,
        py: Python<'_>,
        camera: usize,
        parameters: Vec<String>,
    ) -> PyResult<PySfmrReconstruction> {
        let parameters = parameters
            .iter()
            .map(|name| parameter_from_name(name))
            .collect::<PyResult<Vec<_>>>()?;
        let next = py
            .detach(|| core_drop(self.inner(), camera, &parameters))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PySfmrReconstruction::new(next))
    }

    /// The per-image parameters one camera varies.
    ///
    /// These are the parameters the camera's ``per_image_adjustments`` lists,
    /// and any other one that an image of the camera holds a non-zero value
    /// of: the parameters ``drop_image_adjustments`` changes something for.
    /// Switch Camera Model's drop row, the Camera Intrinsics panel and the
    /// Bundle Adjust dialog's default use the same rule. See
    /// ``specs/core/reconstruction/switch-camera-model.md``.
    ///
    /// Args:
    ///     camera: The camera's index.
    ///
    /// Returns:
    ///     The names, each one of ``"focal"``, ``"principal_point"``,
    ///     ``"radial_bspline"`` and ``"rolling_shutter"``, in that order.
    ///     Raises ``ValueError`` for an unknown camera.
    // A Python docstring, not Rust prose; see `drop_image_adjustments` above.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[pyo3(signature = (camera))]
    fn varied_image_adjustments(&self, camera: usize) -> PyResult<Vec<&'static str>> {
        let image_set = &self.inner().image_set;
        let count = image_set.cameras.len();
        if camera >= count {
            return Err(PyValueError::new_err(
                SwitchCameraModelError::UnknownCamera { camera, count }.to_string(),
            ));
        }
        Ok(core_varied(image_set, camera)
            .into_iter()
            .map(ImageAdjustmentParameter::name)
            .collect())
    }
}
