// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `PatchCloud.render_bitmaps`: render every patch's stored bitmap, its
//! reference view's tile, at its stored frame and keypoints, moving nothing.

use numpy::{IntoPyArray, PyArray1, PyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sfmtool_core::patch::keypoint_subpixel::KeypointSubpixelParams;
use sfmtool_core::patch::normal_refine::ProjectedImage;
use sfmtool_core::patch::stored_bitmap::render_patch_cloud_bitmaps;
use sfmtool_core::progress::Progress;

use super::args::parse_sampler;
use super::cloud::PyPatchCloud;
use super::views::{resolve_patch_scene, resolve_pyramids};
use crate::ProgressCounter;

/// The bitmap column and the reference observation of each point.
type BitmapsAndReferences<'py> = (Bound<'py, PyArray4<u8>>, Bound<'py, PyArray1<i32>>);

#[pymethods]
impl PyPatchCloud {
    /// Render the stored RGBA bitmap of every patch at the patch's stored frame
    /// and each observation's stored keypoint, **moving nothing**.
    ///
    /// Each point's bitmap is the ``R×R`` tile of the observation the
    /// reference-view rule picks among its whole track in ``recon``, rendered
    /// at that observation's stored keypoint, with alpha 255 on the samples on
    /// the photograph and 0 elsewhere. Where the rule picks none, or reaches
    /// its pick only through its last fallback (``"without_any"``), it is the
    /// fused mean of the views and names no observation. A bitmap here equals
    /// what :meth:`refine_keypoints` renders for a patch whose keypoints it did
    /// not move and whose every view passes the refiner's projection gate,
    /// since the refiner runs the rule over the views that pass it.
    /// ``recon`` must carry inline keypoints, as an
    /// ``embedded_patches`` reconstruction does.
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from, carrying inline
    ///         keypoints.
    ///     images: One source image (HxWxC uint8 numpy array) per
    ///         reconstruction image, or an :class:`ImagePyramidSet`.
    ///     resolution: The R×R grid of each bitmap.
    ///     sampler: ``"per_view"`` (default: the sampler rule picks
    ///         ``"anisotropic"`` or ``"bilinear_mip"`` for each view from its
    ///         zoom), or one sampler for every view, ``"bilinear_mip"``,
    ///         ``"bilinear"`` or ``"anisotropic"``, as :meth:`refine_keypoints` takes it.
    ///     progress: Optional progress counter, bumped once per patch.
    ///
    /// Returns:
    ///     ``(bitmaps, reference_observations)``: the ``(P, R, R, 4)`` uint8
    ///     bitmap column for the reconstruction's ``P`` points, and the
    ///     ``(P,)`` int32 index of each point's reference observation within
    ///     its own track, ``-1`` for none, as
    ///     ``clone_with_changes(patch_bitmaps=..., reference_observations=...)``
    ///     takes them. A point with no patch, or with fewer than two
    ///     observations, gets a zero row and ``-1``.
    // This is a Python docstring (rendered by `help()`), not Rust prose: its
    // indented `Args:` / `Returns:` continuation paragraphs read as Markdown
    // indented code blocks, which rustdoc then tries to parse as Rust.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[pyo3(signature = (recon, images, *, resolution=24, sampler="per_view", progress=None))]
    fn render_bitmaps<'py>(
        &self,
        py: Python<'py>,
        recon: &Bound<'py, PyAny>,
        images: &Bound<'py, PyAny>,
        resolution: u32,
        sampler: &str,
        progress: Option<ProgressCounter>,
    ) -> PyResult<BitmapsAndReferences<'py>> {
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
                render_patch_cloud_bitmaps(
                    &self.inner,
                    recon,
                    &views,
                    &params,
                    counter.as_deref(),
                    &Progress::none(),
                )
            })
            .expect("nothing cancels a render given no cancel flag");
        Ok((
            column.bitmaps.into_pyarray(py),
            PyArray1::from_vec(py, column.reference_observations),
        ))
    }
}
