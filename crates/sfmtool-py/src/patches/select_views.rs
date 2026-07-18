// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `PatchCloud.select_views`: photometric view-set selection.

use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::patch::normal_refine::{
    view_indices_from_reconstruction, PatchWindow, ProjectedImage, Sampler,
};
use sfmtool_core::patch::view_selection::{select_patch_cloud_views, ViewSelectParams};

use super::cloud::PyPatchCloud;
use super::views::{resolve_pyramids, resolve_scene};
use crate::ProgressCounter;

#[pymethods]
impl PyPatchCloud {
    /// Select, per patch, the **view set** ``G`` that photometrically sees it:
    /// the point's track views plus every other image that geometrically sees the
    /// surfel and whose windowed ZNCC to a robust reference appearance (fused from
    /// the track views) clears ``min_relative_zncc`` × the track's own
    /// self-agreement. Track views are always admitted. See
    /// ``specs/core/patch-view-selection.md``.
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from (provides cameras,
    ///         poses, and the per-point track view lists via ``point_indexes``),
    ///         **or** a :class:`CameraViews` — which carries no tracks, so
    ///         ``candidate_views`` becomes required.
    ///     images: One source image (HxWxC uint8 numpy array) per reconstruction
    ///         image, parallel to ``recon`` (index = image index), **or** an
    ///         :class:`ImagePyramidSet` prebuilt from those images (decode the
    ///         pyramids once, share them across kernel calls).
    ///     min_relative_zncc: Admit a candidate whose ZNCC to the reference clears
    ///         this fraction of the track's self-agreement (default 0.7).
    ///     resolution: The R×R patch grid the reference / ZNCC are scored on.
    ///     window: Per-pixel scoring weight — ``"gaussian_disk"`` (default),
    ///         ``"gaussian"``, or ``"uniform"``.
    ///     window_sigma: Window sigma for the gaussian windows.
    ///     sampler: ``"bilinear"`` (default), ``"bilinear_mip"``, or
    ///         ``"anisotropic"``.
    ///     min_valid_fraction: Per-view floor on the window-weighted valid-pixel
    ///         fraction; a view below it does not cover enough of the patch.
    ///     min_track_views: Minimum number of *valid* track views to build a
    ///         reference from; a track below this admits its views verbatim with no
    ///         candidate vetting.
    ///     robust_iters: IRLS passes for the robust reference consensus.
    ///     min_self_agreement: Trust gate on the track's self-agreement (default
    ///         0.3). When the track views' mean ZNCC to the reference is below this,
    ///         there is no trustworthy reference, so the track is admitted verbatim
    ///         with no candidate expansion. At or above it, the admission bar is
    ///         ``min_relative_zncc`` × self-agreement.
    ///     point_indexes: If given, select only for the patches with these source
    ///         point ids; ``None`` (default) selects for every patch.
    ///     candidate_views: Optional mapping ``point_index -> [image_index, ...]``
    ///         giving each point's base (always-admitted) view list — the role the
    ///         track observations play in reconstruction mode. **Required** when the
    ///         first argument is a :class:`CameraViews` (there are no tracks);
    ///         with a reconstruction it *overrides* the track-derived list for the
    ///         points present in the map (points absent keep their track views).
    ///
    /// Returns a list of per-patch dicts (parallel to the cloud's patches, in
    /// cloud order): ``point_index`` (int), ``admitted`` (1-D int32 numpy array of
    /// image indices — the track views first, then the vetted candidates in
    /// ascending order), ``scores`` (1-D float64 numpy array of the per-admitted
    /// ZNCC to the reference, parallel to ``admitted``; NaN where a view could not
    /// be scored), and ``self_agreement`` (float; NaN when no reference could be
    /// built). Patches excluded by ``point_indexes`` are omitted from the list.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        recon, images, *, min_relative_zncc=0.7, resolution=24, window="gaussian_disk",
        window_sigma=0.6, sampler="bilinear", min_valid_fraction=0.6, min_track_views=2,
        robust_iters=3, min_self_agreement=0.3, point_indexes=None, candidate_views=None,
        progress=None
    ))]
    fn select_views<'py>(
        &self,
        py: Python<'py>,
        recon: &Bound<'py, PyAny>,
        images: &Bound<'py, PyAny>,
        min_relative_zncc: f64,
        resolution: u32,
        window: &str,
        window_sigma: f64,
        sampler: &str,
        min_valid_fraction: f64,
        min_track_views: u32,
        robust_iters: u32,
        min_self_agreement: f64,
        point_indexes: Option<Vec<u32>>,
        candidate_views: Option<std::collections::HashMap<u32, Vec<u32>>>,
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
        // Without tracks there is no default candidate list, so `candidate_views`
        // is required. Fail fast before decoding any imagery.
        if recon_opt.is_none() && candidate_views.is_none() {
            return Err(PyValueError::new_err(
                "candidate_views is required when the first argument is a CameraViews \
                 (there are no tracks to derive per-patch candidate views from)",
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
        let params = ViewSelectParams {
            min_relative_zncc,
            resolution,
            window,
            sampler,
            min_valid_fraction,
            min_track_views,
            robust_iters,
            min_self_agreement,
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

        // Per-patch base (always-admitted) view lists. In reconstruction mode these
        // default to the track observations; `candidate_views` overrides the listed
        // points (and is the only source in views mode). An empty list makes a
        // patch's selection trivially empty, so `point_indexes` selects a subset by
        // clearing the rest.
        let mut track_views = match recon_opt {
            Some(recon) => view_indices_from_reconstruction(recon, &self.inner),
            None => vec![Vec::new(); self.inner.len()],
        };
        if let Some(map) = &candidate_views {
            for vs in map.values() {
                if let Some(&bad) = vs.iter().find(|&&i| i >= n_images) {
                    return Err(PyValueError::new_err(format!(
                        "candidate_views contains image index {bad} out of range for this \
                         scene's {n_images} views"
                    )));
                }
            }
            for (tv, &pid) in track_views.iter_mut().zip(&self.inner.point_indexes) {
                if let Some(vs) = map.get(&pid) {
                    *tv = vs.clone();
                }
            }
        }
        let selected_mask: Option<std::collections::HashSet<u32>> =
            point_indexes.map(|ids| ids.into_iter().collect());
        if let Some(keep) = &selected_mask {
            for (tv, &pid) in track_views.iter_mut().zip(&self.inner.point_indexes) {
                if !keep.contains(&pid) {
                    tv.clear();
                }
            }
        }

        let progress_handle = progress.as_ref().map(|p| p.handle());
        let results = py.detach(|| {
            select_patch_cloud_views(
                &self.inner,
                &views,
                &track_views,
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
            let d = PyDict::new(py);
            d.set_item("point_index", pid)?;
            d.set_item("admitted", res.admitted.clone().into_pyarray(py))?;
            d.set_item("scores", res.scores.clone().into_pyarray(py))?;
            d.set_item("self_agreement", res.self_agreement)?;
            out.push(d);
        }
        Ok(out)
    }
}
