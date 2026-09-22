// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `PatchCloud.render_bitmaps`: fuse every patch's RGBA bitmap at its stored
//! frame and keypoints, moving nothing.

use numpy::{IntoPyArray, PyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sfmtool_core::patch::keypoint_subpixel::{fuse_patch_cloud_bitmaps, KeypointSubpixelParams};
use sfmtool_core::patch::normal_refine::ProjectedImage;
use sfmtool_core::progress::Progress;

use super::args::parse_sampler;
use super::cloud::PyPatchCloud;
use super::views::{resolve_patch_scene, resolve_pyramids};
use crate::ProgressCounter;

#[pymethods]
impl PyPatchCloud {
    /// Fuse an RGBA bitmap for every patch at the patch's stored frame and each
    /// observation's stored keypoint, **moving nothing**.
    ///
    /// The sub-pixel refiner's own fuse run with no Gauss-Newton step, so a
    /// bitmap here equals what :meth:`refine_keypoints` renders for a patch
    /// whose keypoints it did not move. Each patch is fused from its point's
    /// whole track in ``recon``, at the per-observation keypoints the
    /// reconstruction stores, which is why ``recon`` must be an
    /// ``embedded_patches`` reconstruction.
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from, carrying inline
    ///         keypoints.
    ///     images: One source image (HxWxC uint8 numpy array) per
    ///         reconstruction image, or an :class:`ImagePyramidSet`.
    ///     resolution: The R×R grid of each bitmap.
    ///     sampler: ``"bilinear_mip"`` (default), ``"bilinear"`` or
    ///         ``"anisotropic"``, as :meth:`refine_keypoints` takes it.
    ///     progress: Optional progress counter, bumped once per patch.
    ///
    /// Returns:
    ///     The ``(P, R, R, 4)`` uint8 bitmap column for the reconstruction's
    ///     ``P`` points, as ``clone_with_changes(patch_bitmaps=...)`` takes it.
    ///     A point with no patch, or with fewer than two observations that
    ///     render in frame, gets a zero row.
    // This is a Python docstring (rendered by `help()`), not Rust prose: its
    // indented `Args:` / `Returns:` continuation paragraphs read as Markdown
    // indented code blocks, which rustdoc then tries to parse as Rust.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[pyo3(signature = (recon, images, *, resolution=24, sampler="bilinear_mip", progress=None))]
    fn render_bitmaps<'py>(
        &self,
        py: Python<'py>,
        recon: &Bound<'py, PyAny>,
        images: &Bound<'py, PyAny>,
        resolution: u32,
        sampler: &str,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Bound<'py, PyArray4<u8>>> {
        if resolution < 2 {
            return Err(PyValueError::new_err(format!(
                "resolution must be >= 2, got {resolution}"
            )));
        }
        let (posed, recon_guard, _) =
            resolve_patch_scene(recon, &self.inner, false, "view_sets", "per-patch views")?;
        let recon = recon_guard.as_ref().map(|r| &r.inner).ok_or_else(|| {
            PyValueError::new_err("render_bitmaps needs a reconstruction, not a CameraViews")
        })?;
        if recon.keypoints_xy().is_none() {
            return Err(PyValueError::new_err(
                "render_bitmaps needs the per-observation keypoints an embedded_patches \
                 reconstruction stores (run `sfm xform --to-embedded-patches` first)",
            ));
        }
        let params = KeypointSubpixelParams {
            resolution,
            sampler: parse_sampler(sampler)?,
            ..Default::default()
        };
        let pyramid_set = resolve_pyramids(&posed, images)?;
        let pyramids = pyramid_set.as_slice();
        let views: Vec<Option<ProjectedImage<'_>>> = (0..posed.len())
            .map(|i| {
                Some(ProjectedImage {
                    camera: &posed.cameras[i],
                    cam_from_world: &posed.poses[i],
                    pyramid: &pyramids[i],
                })
            })
            .collect();
        let counter = progress.as_ref().map(|p| p.handle());
        let column = py
            .detach(|| {
                fuse_patch_cloud_bitmaps(
                    &self.inner,
                    recon,
                    &views,
                    &params,
                    counter.as_deref(),
                    &Progress::none(),
                )
            })
            .expect("nothing cancels a fuse given no cancel flag");
        Ok(column.into_pyarray(py))
    }
}
