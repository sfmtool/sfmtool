// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `PatchCloud.localize_keypoints`: discrete cross-view keypoint search.

use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::patch::keypoint_localize::{
    localize_patch_cloud_keypoints, KeypointLocalizeParams,
    SearchStrategy as LocalizeSearchStrategy,
};
use sfmtool_core::patch::normal_refine::{
    view_indices_from_reconstruction, PatchWindow, ProjectedImage, Sampler,
};

use super::cloud::PyPatchCloud;
use super::views::{resolve_pyramids, resolve_scene};
use crate::ProgressCounter;

#[pymethods]
impl PyPatchCloud {
    /// Refine, per patch, the per-view 2D keypoints by group-wise translation
    /// registration (**congealing**): each round renders every view's patch tile
    /// at its accumulated in-plane offset (a single resample of the source),
    /// builds the robust consensus, and searches each view's residual shift
    /// against the **leave-one-out** consensus of the others, dropping views that
    /// drift too far, leave the frame, or stop agreeing. See
    /// ``specs/core/patch-keypoint-localization.md``.
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from (cameras, poses, and
    ///         the per-point track view lists via ``point_indexes``), **or** a
    ///         :class:`CameraViews` — which carries no tracks, so ``view_sets``
    ///         becomes required.
    ///     images: One source image (HxWxC uint8 numpy array) per reconstruction
    ///         image, parallel to ``recon`` (index = image index), **or** an
    ///         :class:`ImagePyramidSet` prebuilt from those images (decode the
    ///         pyramids once, share them across kernel calls).
    ///     view_sets: Optional mapping ``point_index -> [image_index, ...]`` giving the
    ///         view set to refine per point (typically the output of
    ///         :meth:`select_views`). Points absent from the map fall back to their
    ///         track; ``None`` (default) uses the track for every point.
    ///     max_iters: Max congealing rounds (stops early at convergence).
    ///     search: Max total per-view drift in patch-grid px (bounds runaway; also
    ///         the context-tile margin).
    ///     max_shift_px: Drop a view whose refined keypoint sits more than this many
    ///         source-image px from the point's projection.
    ///     min_relative_zncc: Drop a view whose leave-one-out ZNCC falls below this
    ///         fraction of the views' median leave-one-out ZNCC.
    ///     min_grazing_cos: Grazing cutoff; drop a view whose ray is near-parallel
    ///         to the patch plane (``|d·n|`` below this).
    ///     resolution: The R×R patch grid the consensus / ZNCC are scored on.
    ///     window: ``"gaussian_disk"`` (default), ``"gaussian"``, or ``"uniform"``.
    ///     window_sigma: Window sigma for the gaussian windows.
    ///     sampler: ``"bilinear"`` (default), ``"bilinear_mip"``, or
    ///         ``"anisotropic"``.
    ///     robust_iters: IRLS passes for the robust consensus.
    ///     convergence_px: Stop once a round's mean round-over-round change of
    ///         the per-view refined positions is below this many patch-grid px.
    ///     point_indexes: If given, localize only for the patches with these source
    ///         point ids; ``None`` (default) localizes for every patch.
    ///     search_resolution_multiplier: ``m`` for the discrete cross-view search;
    ///         the search runs at resolution ``R_s = round(m·R)``. ``m = 1.0``
    ///         (default) is the no-op; ``m > 1`` (the supersampled grid) resolves
    ///         sub-pixel offsets directly at a cost that grows ~``m²``. See
    ///         ``specs/core/keypoint-localization-search-cache.md``.
    ///
    /// Returns:
    ///     A list of per-point dicts ``{point_index, views (uint32[K]),
    ///     keypoints (float64[K, 2]), offsets_px (float64[K]),
    ///     loo_zncc (float64[K])}`` over the **kept** views. ``loo_zncc`` is NaN for
    ///     a view no round scored (a lone input view, or a view kept by the two-view
    ///     floor before any consensus was built), so guard before reducing it.
    #[pyo3(signature = (
        recon, images, *, view_sets=None, max_iters=5, search=6.0, max_shift_px=3.0,
        min_relative_zncc=0.7, min_grazing_cos=0.1, resolution=24, window="gaussian_disk",
        window_sigma=0.6, sampler="bilinear", robust_iters=3, convergence_px=0.05,
        point_indexes=None, search_resolution_multiplier=1.0,
        search_strategy="plus_descent", progress=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn localize_keypoints<'py>(
        &self,
        py: Python<'py>,
        recon: &Bound<'py, PyAny>,
        images: &Bound<'py, PyAny>,
        view_sets: Option<std::collections::HashMap<u32, Vec<u32>>>,
        max_iters: u32,
        search: f64,
        max_shift_px: f64,
        min_relative_zncc: f64,
        min_grazing_cos: f64,
        resolution: u32,
        window: &str,
        window_sigma: f64,
        sampler: &str,
        robust_iters: u32,
        convergence_px: f64,
        point_indexes: Option<Vec<u32>>,
        search_resolution_multiplier: f32,
        search_strategy: &str,
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
        // Without tracks there is no default per-patch view list, so `view_sets` is
        // required. Fail fast before decoding any imagery.
        if recon_opt.is_none() && view_sets.is_none() {
            return Err(PyValueError::new_err(
                "view_sets is required when the first argument is a CameraViews \
                 (there are no tracks to derive per-patch views from)",
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
        if !(search_resolution_multiplier.is_finite() && search_resolution_multiplier > 0.0) {
            return Err(PyValueError::new_err(format!(
                "search_resolution_multiplier must be > 0, got {search_resolution_multiplier}"
            )));
        }
        let search_strategy = match search_strategy {
            "exhaustive" => LocalizeSearchStrategy::Exhaustive,
            "plus_descent" => LocalizeSearchStrategy::PlusDescent,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown search_strategy: {other:?} (expected exhaustive|plus_descent)"
                )))
            }
        };
        let params = KeypointLocalizeParams {
            max_iters,
            search,
            max_shift_px,
            min_relative_zncc,
            min_grazing_cos,
            resolution,
            window,
            sampler,
            robust_iters,
            convergence_px,
            search_resolution_multiplier,
            search_strategy,
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

        // Per-patch view sets: the supplied map where present, else the track (in
        // views mode there is no track, so the base is empty and `view_sets` — which
        // is required — supplies every list). An empty view set makes a patch's
        // localization trivially empty, so `point_indexes` selects a subset by
        // clearing the rest.
        let mut sets = match recon_opt {
            Some(recon) => view_indices_from_reconstruction(recon, &self.inner),
            None => vec![Vec::new(); self.inner.len()],
        };
        if let Some(map) = &view_sets {
            // Reject out-of-range image indices up front so the kernel never indexes
            // `views` out of bounds (which would surface as an opaque panic rather
            // than a clean error). The kernel dedups, so duplicates are fine here.
            for vs in map.values() {
                if let Some(&bad) = vs.iter().find(|&&i| i >= n_images) {
                    return Err(PyValueError::new_err(format!(
                        "view_sets contains image index {bad} out of range for this \
                         scene's {n_images} views"
                    )));
                }
            }
            for (set, &pid) in sets.iter_mut().zip(&self.inner.point_indexes) {
                if let Some(vs) = map.get(&pid) {
                    *set = vs.clone();
                }
            }
        }
        let selected_mask: Option<std::collections::HashSet<u32>> =
            point_indexes.map(|ids| ids.into_iter().collect());
        if let Some(keep) = &selected_mask {
            for (set, &pid) in sets.iter_mut().zip(&self.inner.point_indexes) {
                if !keep.contains(&pid) {
                    set.clear();
                }
            }
        }

        let progress_handle = progress.as_ref().map(|p| p.handle());
        let results = py.detach(|| {
            localize_patch_cloud_keypoints(
                &self.inner,
                &views,
                &sets,
                None,
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
            // Flat (K, 2) keypoint array, built with an explicit shape so the
            // no-kept-views case yields a clean (0, 2) array rather than failing
            // column inference.
            let flat: Vec<f64> = res.keypoints.iter().flat_map(|k| [k[0], k[1]]).collect();
            let kpts = ndarray::Array2::from_shape_vec((res.keypoints.len(), 2), flat)
                .expect("keypoints shape matches");
            let d = PyDict::new(py);
            d.set_item("point_index", pid)?;
            d.set_item("views", res.views.clone().into_pyarray(py))?;
            d.set_item("keypoints", kpts.into_pyarray(py))?;
            d.set_item("offsets_px", res.offsets_px.clone().into_pyarray(py))?;
            d.set_item("loo_zncc", res.loo_zncc.clone().into_pyarray(py))?;
            out.push(d);
        }
        Ok(out)
    }
}
