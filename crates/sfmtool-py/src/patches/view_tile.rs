// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `OrientedPatch.render_view_tile`: one view's `R×R` tile of a patch, with
//! the per-view readings the reference-view rule takes on it.

use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::camera::image::ImageU8Pyramid;
use sfmtool_core::patch::normal_refine::ProjectedImage;
use sfmtool_core::patch::reference_view::render_view_tile;
use sfmtool_core::progress::Progress;

use super::args::parse_sampler;
use super::oriented_patch::PyOrientedPatch;
use super::views::{check_image_matches_camera, pyramid_levels, PyImagePyramidSet};
use crate::flow::warp::extract_image_u8;
use crate::geometry::rigid_transform::PyRigidTransform;
use crate::PyCameraIntrinsics;

#[pymethods]
impl PyOrientedPatch {
    /// Render this patch's ``R×R`` tile in one view, as the bench renders each
    /// observation's tile and as every stored patch bitmap is rendered, and
    /// read the per-view measurements the reference-view rule takes on it.
    ///
    /// The patch is re-anchored on ``keypoint`` first, when one is given, so
    /// its centre projects onto it; the tile is rendered with the sampler the
    /// sampler rule picks from that placement's Jacobian at ``resolution``
    /// (``sampler="per_view"``), or with the one sampler named. A sample the
    /// warp cannot place on the photograph is black and marked invalid.
    ///
    /// Args:
    ///     camera: The view's intrinsics.
    ///     cam_from_world: The view's pose.
    ///     image: The view's photograph, an ``HxW`` or ``HxWxC`` uint8 array
    ///         matching the camera's size, or an :class:`ImagePyramidSet`
    ///         with ``image_index`` naming the view in it.
    ///     image_index: The view's index in ``image`` when that is an
    ///         :class:`ImagePyramidSet`.
    ///     keypoint: The observation's pixel ``(x, y)`` to anchor on, or
    ///         ``None`` to render the patch where it is.
    ///     resolution: ``R``, the tile's side.
    ///     sampler: ``"per_view"`` (default), ``"bilinear_mip"``,
    ///         ``"bilinear"`` or ``"anisotropic"``.
    ///
    /// Returns:
    ///     A dict: ``samples`` the ``(R, R, C)`` uint8 tile (``C`` the
    ///     photograph's channels); ``valid`` the ``(R, R)`` bool array of the
    ///     samples that carry image data; ``sampler`` the sampler's name;
    ///     ``coverage`` the share of valid samples; ``clipped_share`` the share
    ///     of the photograph's pixels inside the tile's outline that are 0 or
    ///     255 in any colour channel (``None`` when no sample lands on the
    ///     photograph); ``viewing_angle_deg`` and ``tilt_direction_deg``, the
    ///     angle between the patch's normal and the direction to the camera
    ///     and the direction in the patch's plane, from ``u`` towards ``v``,
    ///     the ray from the camera leans along (``None`` for a view within
    ///     0.1 degrees of facing the patch); ``jacobian`` the ``(2, 2)`` image
    ///     px per grid px at the tile's centre, or ``None``; and ``placement``
    ///     the :class:`OrientedPatch` the tile was rendered through.
    // This is a Python docstring (rendered by `help()`), not Rust prose: its
    // indented `Args:` / `Returns:` continuation paragraphs read as Markdown
    // indented code blocks, which rustdoc then tries to parse as Rust.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        camera,
        cam_from_world,
        image,
        *,
        image_index=None,
        keypoint=None,
        resolution=24,
        sampler="per_view",
    ))]
    fn render_view_tile<'py>(
        &self,
        py: Python<'py>,
        camera: &PyCameraIntrinsics,
        cam_from_world: &PyRigidTransform,
        image: &Bound<'py, PyAny>,
        image_index: Option<usize>,
        keypoint: Option<[f64; 2]>,
        resolution: usize,
        sampler: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        if resolution < 2 {
            return Err(PyValueError::new_err(format!(
                "resolution must be >= 2, got {resolution}"
            )));
        }
        let choice = parse_sampler(sampler)?;
        let built: ImageU8Pyramid;
        let shared;
        let pyramid: &ImageU8Pyramid = if let Ok(set) = image.cast::<PyImagePyramidSet>() {
            let index = image_index.ok_or_else(|| {
                PyValueError::new_err("an ImagePyramidSet needs image_index to name the view")
            })?;
            shared = std::sync::Arc::clone(&set.get().pyramids);
            shared.get(index).ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "image_index {index} is past the set's {} images",
                    shared.len()
                ))
            })?
        } else {
            let source = extract_image_u8(image)?;
            built = py.detach(|| ImageU8Pyramid::build(&source, pyramid_levels(&source)));
            &built
        };
        let level0 = pyramid.level(0);
        check_image_matches_camera(&camera.inner, 0, level0.width(), level0.height())?;
        let view = ProjectedImage {
            camera: &camera.inner,
            cam_from_world: &cam_from_world.inner,
            pyramid,
        };
        let tile = py.detach(|| {
            render_view_tile(
                &self.inner,
                &view,
                keypoint,
                resolution,
                choice,
                &Progress::none(),
            )
        });
        let d = PyDict::new(py);
        let valid = numpy::ndarray::Array2::from_shape_vec((resolution, resolution), tile.valid)
            .expect("one flag per sample");
        d.set_item("samples", tile.samples.into_pyarray(py))?;
        d.set_item("valid", valid.into_pyarray(py))?;
        d.set_item("sampler", tile.sampler.name())?;
        d.set_item("coverage", tile.coverage)?;
        d.set_item("clipped_share", tile.clipped_share)?;
        d.set_item("viewing_angle_deg", tile.viewing_angle.map(|a| a.angle_deg))?;
        d.set_item(
            "tilt_direction_deg",
            tile.viewing_angle.and_then(|a| a.tilt_direction_deg),
        )?;
        let jacobian: Option<Bound<'py, PyArray2<f64>>> = tile.jacobian.map(|j| {
            numpy::ndarray::Array2::from_shape_fn((2, 2), |(r, c)| j[r][c]).into_pyarray(py)
        });
        d.set_item("jacobian", jacobian)?;
        d.set_item(
            "placement",
            Py::new(
                py,
                PyOrientedPatch {
                    inner: tile.placement,
                },
            )?,
        )?;
        Ok(d)
    }
}
