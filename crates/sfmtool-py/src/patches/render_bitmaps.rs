// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `PatchCloud.render_bitmaps`: render every patch's stored bitmap, its
//! reference observation's tile, at its stored frame and keypoints, moving
//! nothing.

use numpy::{IntoPyArray, PyArray1, PyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sfmtool_core::patch::keypoint_subpixel::KeypointSubpixelParams;
use sfmtool_core::patch::normal_refine::ProjectedImage;
use sfmtool_core::patch::stored_bitmap::{render_patch_cloud_bitmaps, UnreferencedPoints};
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
    /// Each point's bitmap is the ``R×R`` tile of its reference observation,
    /// rendered at that observation's stored keypoint, with alpha 255 on the
    /// samples on the photograph and 0 elsewhere. Where ``recon`` stores a
    /// reference observation for the point
    /// (:attr:`SfmrReconstruction.reference_observations` ``>= 0``), that is
    /// the observation rendered, and it is returned unchanged, so a
    /// reconstruction whose bitmaps were dropped renders the same bitmaps
    /// again. Where it stores ``-1``, the reference-view rule picks the
    /// observation among the point's whole track; where the rule picks none,
    /// or reaches its pick only through its last fallback (``"without_any"``),
    /// the bitmap is the fused mean of the views and names no observation. For
    /// a point at ``-1``, a bitmap here equals what :meth:`refine_keypoints`
    /// renders for a patch whose keypoints it did not move and whose every
    /// view passes the refiner's projection gate, since the refiner runs the
    /// rule over the views that pass it.
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
    ///     referenced_only: If true, render only the points with a stored
    ///         reference observation, and give every point at ``-1`` a zero
    ///         row and ``-1`` without running the rule: for a caller that has
    ///         bitmaps for those points already, such as a refinement that
    ///         rendered every point's bitmap by the rule and replaces the
    ///         points that carry a reference with that reference's render.
    ///     progress: Optional progress counter, bumped once per patch.
    ///
    /// Returns:
    ///     ``(bitmaps, reference_observations)``: the ``(P, R, R, 4)`` uint8
    ///     bitmap column for the reconstruction's ``P`` points, and the
    ///     ``(P,)`` int32 index of each point's reference observation within
    ///     its own track, ``-1`` for none, as
    ///     ``clone_with_changes(patch_bitmaps=..., reference_observations=...)``
    ///     takes them. A point with a stored reference whose photograph is
    ///     missing gets a zero row and keeps its reference. A point with no
    ///     patch (a zero patch frame) gets a zero row and keeps the reference
    ///     it stores. A point at ``-1`` with fewer than two observations, or
    ///     any point at ``-1`` with ``referenced_only``, gets a zero row and
    ///     ``-1``.
    // This is a Python docstring (rendered by `help()`), not Rust prose: its
    // indented `Args:` / `Returns:` continuation paragraphs read as Markdown
    // indented code blocks, which rustdoc then tries to parse as Rust.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[pyo3(signature = (
        recon, images, *, resolution=24, sampler="per_view", referenced_only=false, progress=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn render_bitmaps<'py>(
        &self,
        py: Python<'py>,
        recon: &Bound<'py, PyAny>,
        images: &Bound<'py, PyAny>,
        resolution: u32,
        sampler: &str,
        referenced_only: bool,
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
                    if referenced_only {
                        UnreferencedPoints::Skip
                    } else {
                        UnreferencedPoints::Pick
                    },
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
