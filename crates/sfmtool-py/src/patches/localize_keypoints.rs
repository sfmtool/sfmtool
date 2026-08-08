// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `PatchCloud.localize_keypoints`: discrete cross-view keypoint search.

use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::patch::keypoint_localize::{
    localize_patch_cloud_keypoints, BasisInputs, BasisPick, KeypointLocalizeParams,
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
    ///     sampler: ``"bilinear_mip"`` (default), ``"bilinear"``, or
    ///         ``"anisotropic"``.
    ///     robust_iters: IRLS passes for the robust consensus.
    ///     convergence_px: Stop once a round's mean round-over-round change of
    ///         the per-view refined positions is below this many patch-grid px.
    ///     point_indexes: If given, localize only for the patches with these source
    ///         point ids; ``None`` (default) localizes for every patch.
    ///     starting_keypoints: Optional explicit per-view seeds:
    ///         ``point_index -> [[x, y], ...]`` in **source-image** pixels,
    ///         parallel to that point's (final) view set — one entry per view, in
    ///         order. Same shape as :meth:`refine_keypoints`'s parameter of the
    ///         same name, with one addition: an entry may be ``None`` instead of
    ///         an ``[x, y]`` pair, seeding **that** view at the point's own
    ///         projection while its siblings keep their explicit seeds. A point
    ///         absent from the map (and every point when this is ``None``, the
    ///         default) seeds each of its views at the point's own projection
    ///         ``project_i(X_p)``, which is exactly today's behaviour — as does an
    ///         all-``None`` list.
    ///
    ///         Seeding localization around the caller's own keypoints rather
    ///         than around the projection starts the search from the appearance
    ///         the caller trusts (the observation that was actually matched)
    ///         rather than from a position carrying the point's reprojection
    ///         residual. The per-view ``None`` is what makes that usable on a
    ///         view set that mixes observed views with expansion candidates: the
    ///         candidates have no observation, hence no keypoint, and take the
    ///         projection.
    ///     search_resolution_multiplier: ``m`` for the discrete cross-view search;
    ///         the search runs at resolution ``R_s = round(m·R)``. ``m = 1.0``
    ///         (default) is the no-op; ``m > 1`` (the supersampled grid) resolves
    ///         sub-pixel offsets directly at a cost that grows ~``m²``. See
    ///         ``specs/core/keypoint-localization-search-cache.md``.
    ///     basis_max_views: Consensus-basis cap ``K`` — at most this many views
    ///         congeal against each other; every remaining view registers **once**
    ///         against the finished basis template. Default ``8``; ``0`` disables
    ///         the cap (bit-identical to the uncapped path, as is any point with
    ///         ``V <= K``). Every observation is still localized and reported; only
    ///         the consensus membership shrinks. Pass ``view_scores`` for the
    ///         validated ZNCC ranking — without them the basis pick falls back to
    ///         grazing angle. See
    ///         ``specs/core/keypoint-localization-consensus-basis.md``.
    ///     basis_force_track_views: Reserve basis seats for the point's track views
    ///         (the leading ``track_view_counts`` entries of its view set) ahead of
    ///         the expansion candidates. Default ``True``.
    ///     basis_pick: How the ranked candidates fill the remaining basis seats —
    ///         ``"top_score"`` (default) or ``"strided"``.
    ///     view_scores: Optional mapping ``point_index -> [score, ...]`` parallel to
    ///         that point's view set: each view's match to the point's starting
    ///         appearance (``select_views``'s ``scores``), ranking the basis pick.
    ///         NaN ranks a view below every scored one. Omitted points (and
    ///         ``None``) fall back to ranking by grazing angle. Each entry must
    ///         be parallel to that point's **full** view set, checked before
    ///         ``point_indexes`` narrows the run — so one map from a whole-cloud
    ///         ``select_views`` can drive chunked localize calls.
    ///     track_view_counts: Optional mapping ``point_index -> t``: how many
    ///         **leading** view-set entries are that point's track views
    ///         (``select_views``'s ``track_view_count``). Omitted points have no
    ///         reserved seats.
    ///
    /// Returns:
    ///     A list of per-point dicts ``{point_index, views (uint32[K]),
    ///     keypoints (float64[K, 2]), offsets_px (float64[K]),
    ///     loo_zncc (float64[K]), is_basis (bool[K])}`` over the **kept** views.
    ///     ``loo_zncc`` is NaN for
    ///     a view no round scored (a lone input view, or a view kept by the two-view
    ///     floor before any consensus was built), so guard before reducing it.
    ///     ``is_basis`` marks the consensus-basis members (all ``True`` unless
    ///     ``basis_max_views`` capped that point's view set).
    // This is a Python docstring (rendered by `help()`), not Rust prose: its
    // indented `Args:` / `Returns:` continuation paragraphs read as Markdown
    // indented code blocks, which rustdoc then tries to parse as Rust.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[pyo3(signature = (
        recon, images, *, view_sets=None, max_iters=5, search=6.0, max_shift_px=3.0,
        min_relative_zncc=0.7, min_grazing_cos=0.1, resolution=24, window="gaussian_disk",
        window_sigma=0.6, sampler="bilinear_mip", robust_iters=3, convergence_px=0.05,
        point_indexes=None, starting_keypoints=None, search_resolution_multiplier=1.0,
        search_strategy="plus_descent", basis_max_views=8, basis_force_track_views=true,
        basis_pick="top_score", view_scores=None, track_view_counts=None, progress=None
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
        starting_keypoints: Option<std::collections::HashMap<u32, Vec<Option<[f64; 2]>>>>,
        search_resolution_multiplier: f32,
        search_strategy: &str,
        basis_max_views: u32,
        basis_force_track_views: bool,
        basis_pick: &str,
        view_scores: Option<std::collections::HashMap<u32, Vec<f64>>>,
        track_view_counts: Option<std::collections::HashMap<u32, u32>>,
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
        let basis_pick = match basis_pick {
            "top_score" => BasisPick::TopScore,
            "strided" => BasisPick::Strided,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown basis_pick: {other:?} (expected top_score|strided)"
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
            basis_max_views,
            basis_force_track_views,
            basis_pick,
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
        // Per-patch consensus-basis evidence, parallel to `sets`. A point absent
        // from `view_scores` gets an empty score list — which the kernel reads as
        // "unscored", falling back to the grazing rank; a length mismatch against
        // the point's view set is a caller bug, so reject it up front rather than
        // silently ranking the tail last.
        //
        // Built BEFORE `point_indexes` clears the unselected sets: the natural
        // "select_views once, then localize in chunks" caller passes the whole
        // score map with a per-chunk `point_indexes`, and validating against
        // already-cleared sets would reject it.
        let scores_per_patch: Option<Vec<Vec<f64>>> = match &view_scores {
            None => None,
            Some(map) => {
                let mut out = Vec::with_capacity(self.inner.len());
                for (set, &pid) in sets.iter().zip(&self.inner.point_indexes) {
                    match map.get(&pid) {
                        Some(s) if s.len() != set.len() => {
                            return Err(PyValueError::new_err(format!(
                                "view_scores[{pid}] has {} entries but the point's view set \
                                 has {} — they must be parallel",
                                s.len(),
                                set.len()
                            )))
                        }
                        Some(s) => out.push(s.clone()),
                        None => out.push(Vec::new()),
                    }
                }
                Some(out)
            }
        };

        let selected_mask: Option<std::collections::HashSet<u32>> =
            point_indexes.map(|ids| ids.into_iter().collect());
        if let Some(keep) = &selected_mask {
            for (set, &pid) in sets.iter_mut().zip(&self.inner.point_indexes) {
                if !keep.contains(&pid) {
                    set.clear();
                }
            }
        }
        // Per-patch explicit seeds, parallel to `sets`. A point absent from the map
        // gets an EMPTY list, which the kernel reads as "unseeded" — that patch's
        // views seed at the projection, i.e. exactly the historical behaviour that
        // `starting_keypoints=None` keeps for the whole cloud; a `None` entry
        // inside a listed point's seeds says the same thing for that one view. A
        // length mismatch against the point's view set would silently mis-pair
        // seeds with views, so reject it up front (mirroring `refine_keypoints`).
        let seeds_per_patch: Option<Vec<Vec<Option<[f64; 2]>>>> = match &starting_keypoints {
            None => None,
            Some(map) => {
                let pid_to_idx: std::collections::HashMap<u32, usize> = self
                    .inner
                    .point_indexes
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (p, i))
                    .collect();
                let mut out = vec![Vec::new(); self.inner.len()];
                for (pid, seeds) in map {
                    let Some(&idx) = pid_to_idx.get(pid) else {
                        return Err(PyValueError::new_err(format!(
                            "starting_keypoints[{pid}] is not a point in this patch cloud",
                        )));
                    };
                    if let Some(keep) = &selected_mask {
                        if !keep.contains(pid) {
                            return Err(PyValueError::new_err(format!(
                                "starting_keypoints[{pid}] is excluded by point_indexes; \
                                 drop the entry or include {pid} in point_indexes",
                            )));
                        }
                    }
                    if seeds.len() != sets[idx].len() {
                        return Err(PyValueError::new_err(format!(
                            "starting_keypoints[{pid}] has {} seeds but the view set has {} views",
                            seeds.len(),
                            sets[idx].len(),
                        )));
                    }
                    out[idx] = seeds.clone();
                }
                Some(out)
            }
        };

        let counts_per_patch: Option<Vec<u32>> = track_view_counts.as_ref().map(|map| {
            self.inner
                .point_indexes
                .iter()
                .map(|pid| map.get(pid).copied().unwrap_or(0))
                .collect()
        });
        let basis_inputs = BasisInputs {
            view_scores: scores_per_patch.as_deref(),
            track_view_counts: counts_per_patch.as_deref(),
        };

        let progress_handle = progress.as_ref().map(|p| p.handle());
        let results = py.detach(|| {
            localize_patch_cloud_keypoints(
                &self.inner,
                &views,
                &sets,
                seeds_per_patch.as_deref(),
                Some(&basis_inputs),
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
            d.set_item("is_basis", res.is_basis.clone())?;
            out.push(d);
        }
        Ok(out)
    }
}
