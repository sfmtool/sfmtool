// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `PatchCloud.validate_member_coherence`: pairwise track-member agreement and
//! the verdict read off it.

use numpy::ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::patch::member_coherence::{
    member_views_from_reconstruction, validate_patch_cloud_member_coherence, MemberCoherenceParams,
    MemberVerdict,
};
use sfmtool_core::patch::normal_refine::{PatchWindow, ProjectedImage, Sampler};

use super::cloud::PyPatchCloud;
use super::views::{resolve_pyramids, resolve_scene};
use crate::ProgressCounter;

/// The `verdict` string a [`MemberVerdict`] reports as.
fn verdict_name(v: MemberVerdict) -> &'static str {
    match v {
        MemberVerdict::KeepAll => "keep_all",
        MemberVerdict::Split => "split",
        MemberVerdict::Retire => "retire",
    }
}

#[pymethods]
impl PyPatchCloud {
    /// Validate each point's track against itself: render every member's patch
    /// through the point's own frame, correlate every **pair** of members, and
    /// read a verdict off that matrix. See
    /// ``specs/core/member-coherence-validation.md``.
    ///
    /// The pairwise matrix sees a member that images a different surface even
    /// when the fused cross-view consensus does not — on a balanced split the
    /// consensus is a compromise blend that flatters both sides.
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from (provides cameras,
    ///         poses, and the per-point member lists via ``point_indexes``),
    ///         **or** a :class:`CameraViews` — which carries no tracks, so
    ///         ``member_views`` becomes required.
    ///     images: One source image (HxWxC uint8 numpy array) per reconstruction
    ///         image, parallel to ``recon`` (index = image index), **or** an
    ///         :class:`ImagePyramidSet` prebuilt from those images.
    ///     bar: Pairwise ZNCC at or above which two members agree (default 0.65;
    ///         calibrated for the default window / sampler / resolution).
    ///     margin_gate: Separation-margin floor a cut must clear (default 0.05);
    ///         below it the track is a drift chain and is kept whole.
    ///     resolution: The R×R patch grid members are rendered and correlated on.
    ///     window: Per-pixel scoring weight — ``"gaussian_disk"`` (default),
    ///         ``"gaussian"``, or ``"uniform"``.
    ///     window_sigma: Window sigma for the gaussian windows.
    ///     sampler: ``"bilinear_mip"`` (default), ``"bilinear"``, or
    ///         ``"anisotropic"``.
    ///     min_valid_fraction: Per-member floor on the window-weighted valid-pixel
    ///         fraction; a member below it is left unscored.
    ///     min_support_pixels: Floor on the **common** support — the pixels valid
    ///         in every scoreable member, which every pairwise ZNCC is computed
    ///         over. A track below it is left entirely unscored (and so kept). The
    ///         default 8 is the floor the shared support builder already enforces,
    ///         so it changes nothing; raise it when vetting wide-baseline tracks,
    ///         where the intersection can shrink to a sliver of the R×R grid.
    ///     point_indexes: If given, validate only the patches with these source
    ///         point ids; ``None`` (default) validates every patch.
    ///     member_views: Optional mapping ``point_index -> [image_index, ...]``
    ///         giving each point's member list — the role the track observations
    ///         play in reconstruction mode. **Required** when the first argument is
    ///         a :class:`CameraViews`; with a reconstruction it *overrides* the
    ///         track-derived list for the points present in the map.
    ///     return_matrix: Also return the per-point ``zncc`` matrix (default
    ///         ``False`` — it is ``k×k`` per point).
    ///
    /// Returns a list of per-patch dicts (in cloud order, patches excluded by
    /// ``point_indexes`` omitted): ``point_index`` (int), ``members`` (1-D uint32
    /// numpy array of image indices, deduplicated first-seen-wins — every other
    /// per-member array is parallel to it), ``verdict`` (``"keep_all"`` /
    /// ``"split"`` / ``"retire"``), ``kept`` (1-D bool numpy array — the members
    /// the point keeps; all-``False`` on a retirement), ``block`` (1-D bool numpy
    /// array — the winning max-support block over the *scored* members;
    /// informational on a retirement), ``scored`` (1-D bool numpy array — which
    /// members carry pairwise evidence), ``support`` (int — the block size, ``0``
    /// when fewer than two members scored), ``n_support`` (int — the common-support
    /// pixel count every pairwise ZNCC was taken over; one number per point,
    /// because the support is frozen per point rather than per pair),
    /// ``margin``, ``min_intra``, ``max_cross`` (floats, NaN where undefined), and
    /// — with ``return_matrix=True`` — ``zncc`` (``k×k`` float64 numpy array, unit
    /// diagonal, NaN for uncorrelatable pairs).
    ///
    /// **Unscored members** — nothing rendered them, or nothing could be
    /// correlated with them — sit outside the decision rule entirely: the block
    /// sweep, the margin and the majority denominator all run over the scored
    /// members only, and an unscored member is passed through in ``kept`` (a
    /// ``"retire"`` still ships nothing: the point itself is refused).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        recon, images, *, bar=0.65, margin_gate=0.05, resolution=24, window="gaussian_disk",
        window_sigma=0.6, sampler="bilinear_mip", min_valid_fraction=0.6, min_support_pixels=8,
        point_indexes=None, member_views=None, return_matrix=false, progress=None
    ))]
    fn validate_member_coherence<'py>(
        &self,
        py: Python<'py>,
        recon: &Bound<'py, PyAny>,
        images: &Bound<'py, PyAny>,
        bar: f64,
        margin_gate: f64,
        resolution: u32,
        window: &str,
        window_sigma: f64,
        sampler: &str,
        min_valid_fraction: f64,
        min_support_pixels: u32,
        point_indexes: Option<Vec<u32>>,
        member_views: Option<std::collections::HashMap<u32, Vec<u32>>>,
        return_matrix: bool,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let (posed, recon_guard) = resolve_scene(recon)?;
        let recon_opt = recon_guard.as_ref().map(|r| &r.inner);
        let n_images = posed.len() as u32;
        if self.inner.point_indexes.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                "patch cloud has no per-patch point_indexes; rebuild it with from_reconstruction",
            ));
        }
        if let Some(recon) = recon_opt {
            if self
                .inner
                .point_indexes
                .iter()
                .any(|&p| p as usize >= recon.points.len())
            {
                return Err(PyValueError::new_err(
                    "patch cloud point_indexes are out of range for this reconstruction \
                     (was the cloud built from a different recon?)",
                ));
            }
        }
        if recon_opt.is_none() && member_views.is_none() {
            return Err(PyValueError::new_err(
                "member_views is required when the first argument is a CameraViews \
                 (there are no tracks to derive per-patch member views from)",
            ));
        }

        let window = match window {
            "uniform" => PatchWindow::Uniform,
            "gaussian" => PatchWindow::Gaussian {
                sigma: window_sigma,
            },
            "gaussian_disk" => PatchWindow::GaussianDisk {
                sigma: window_sigma,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown window: {other:?} (expected uniform|gaussian|gaussian_disk)"
                )))
            }
        };
        let sampler = match sampler {
            "bilinear" => Sampler::Bilinear,
            "bilinear_mip" => Sampler::BilinearMip,
            "anisotropic" => Sampler::Anisotropic,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sampler: {other:?} (expected bilinear|bilinear_mip|anisotropic)"
                )))
            }
        };
        let params = MemberCoherenceParams {
            bar,
            margin_gate,
            resolution,
            window,
            sampler,
            min_valid_fraction,
            min_support_pixels,
        };

        let pyramid_set = resolve_pyramids(&posed, images)?;
        let pyramids = pyramid_set.as_slice();
        let views: Vec<ProjectedImage<'_>> = (0..posed.len())
            .map(|i| ProjectedImage {
                camera: &posed.cameras[i],
                cam_from_world: &posed.poses[i],
                pyramid: &pyramids[i],
            })
            .collect();

        // Per-patch member lists. In reconstruction mode these default to the track
        // observations; `member_views` overrides the listed points (and is the only
        // source in views mode). An empty list makes a patch's validation trivially
        // empty, so `point_indexes` selects a subset by clearing the rest.
        let mut members = match recon_opt {
            Some(recon) => member_views_from_reconstruction(recon, &self.inner),
            None => vec![Vec::new(); self.inner.len()],
        };
        if let Some(map) = &member_views {
            for vs in map.values() {
                if let Some(&bad) = vs.iter().find(|&&i| i >= n_images) {
                    return Err(PyValueError::new_err(format!(
                        "member_views contains image index {bad} out of range for this \
                         scene's {n_images} views"
                    )));
                }
            }
            for (mv, &pid) in members.iter_mut().zip(&self.inner.point_indexes) {
                if let Some(vs) = map.get(&pid) {
                    *mv = vs.clone();
                }
            }
        }
        let selected_mask: Option<std::collections::HashSet<u32>> =
            point_indexes.map(|ids| ids.into_iter().collect());
        if let Some(keep) = &selected_mask {
            for (mv, &pid) in members.iter_mut().zip(&self.inner.point_indexes) {
                if !keep.contains(&pid) {
                    mv.clear();
                }
            }
        }

        let progress_handle = progress.as_ref().map(|p| p.handle());
        let results = py.detach(|| {
            validate_patch_cloud_member_coherence(
                &self.inner,
                &views,
                &members,
                &params,
                progress_handle.as_deref(),
            )
        });

        let mut out = Vec::new();
        for (res, &pid) in results.iter().zip(&self.inner.point_indexes) {
            if let Some(keep) = &selected_mask {
                if !keep.contains(&pid) {
                    continue;
                }
            }
            let k = res.matrix.len();
            let d = PyDict::new(py);
            d.set_item("point_index", pid)?;
            d.set_item("members", res.matrix.members.clone().into_pyarray(py))?;
            d.set_item("verdict", verdict_name(res.decision.verdict))?;
            d.set_item("kept", res.decision.kept.clone().into_pyarray(py))?;
            d.set_item("block", res.decision.block.clone().into_pyarray(py))?;
            d.set_item("scored", res.matrix.scored.clone().into_pyarray(py))?;
            d.set_item("support", res.decision.support)?;
            d.set_item("n_support", res.matrix.n_support)?;
            d.set_item("margin", res.decision.margin)?;
            d.set_item("min_intra", res.decision.min_intra)?;
            d.set_item("max_cross", res.decision.max_cross)?;
            if return_matrix {
                let arr = Array2::from_shape_vec((k, k), res.matrix.zncc.clone())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                d.set_item("zncc", arr.into_pyarray(py))?;
            }
            out.push(d);
        }
        Ok(out)
    }
}
